#code modified for all forms of verb do so

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json

from e2e_bert_coref_master.bert import tokenization

import tensorflow as tf
from e2e_bert_coref_master import util 

import pandas as pd
import numpy as np
import ast
import xlrd

from nltk import Tree
from nltk.tokenize import sent_tokenize
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from stanfordcorenlp import StanfordCoreNLP

#using pretrained bert model 
import math
import torch
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

import coreference_preprocess as cp

class CoreferenceEngine:
    def __init__(self):
        self.config = util.initialize_from_env()
        self.model = util.get_model(self.config)
        saver = tf.train.Saver()
        self.sess = tf.Session()
        with self.sess.as_default() as session:        
            self.model.restore(session)
            self.tokenizer = tokenization.FullTokenizer(vocab_file="./e2e_bert_coref_master/cased_config_vocab/vocab.txt", do_lower_case=False)
        
        # load allennlp constituency parse model and stanford constituency model
        self.predictor_cp = Predictor.from_path("/home/sandhya/coref_docker/Coreference-Module/src/allennlp-models/elmo-constituency-parser-2020.02.10.tar.gz")
        self.nlp = StanfordCoreNLP(r'./stanford-corenlp-4.2.0')

        # Load pre-trained model (weights)
        self.model_gpt = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.model_gpt.eval()
        
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer_gpt = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.coref_preprocess_obj = cp.CoreferencePreprocess()

    def pre_process_data(self, input_text):
        #preprocessed_str = self.coref_preprocess_obj.preprocess_single_input(input_text)
        text = sent_tokenize(input_text)
        #text = sent_tokenize(preprocessed_str)
    
        genre = "wb"
        model_name = "bert_base"

        os.environ['data_dir'] = "."
        os.environ['CHOSEN_MODEL'] = model_name

        data = {
            'doc_key': genre,
            'sentences': [["[CLS]"]],
            'speakers': [["[SPL]"]],
            'clusters': [],
            'sentence_map': [0],
            'subtoken_map': [0],
        }

        # Determine Max Segment
        max_segment = None
        for line in open('e2e_bert_coref_master/experiments.conf'):
            if line.startswith(model_name):
                max_segment = True
            elif line.strip().startswith("max_segment_len"):
                if max_segment:
                    max_segment = int(line.strip().split()[-1])
                    break


        subtoken_num = 0
        for sent_num, line in enumerate(text):
            raw_tokens = line.split()
            tokens = self.tokenizer.tokenize(line)
            if len(tokens) + len(data['sentences'][-1]) >= max_segment:
                data['sentences'][-1].append("[SEP]")
                data['sentences'].append(["[CLS]"])
                data['speakers'][-1].append("[SPL]")
                data['speakers'].append(["[SPL]"])
                data['sentence_map'].append(sent_num - 1)
                data['subtoken_map'].append(subtoken_num - 1)
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)

            ctoken = raw_tokens[0]
            cpos = 0
            for token in tokens:
                data['sentences'][-1].append(token)
                data['speakers'][-1].append("-")
                data['sentence_map'].append(sent_num)
                data['subtoken_map'].append(subtoken_num)

                if token.startswith("##"):
                    token = token[2:]
                if len(ctoken) == len(token):
                    subtoken_num += 1
                    cpos += 1
                    if cpos < len(raw_tokens):
                        ctoken = raw_tokens[cpos]
                else:
                    ctoken = ctoken[len(token):]

        data['sentences'][-1].append("[SEP]")
        data['speakers'][-1].append("[SPL]")
        data['sentence_map'].append(sent_num - 1)
        data['subtoken_map'].append(subtoken_num - 1)
        return json.dumps(data, sort_keys=True)

    def get_coref_for_single_input(self, json_input):
        log_dir = self.config["log_dir"]
        
        with self.sess.as_default() as session:
            example = json.loads(json_input)
            tensorized_example = self.model.tensorize_example(example, is_training=False)
            feed_dict = {i: t for i, t in zip(self.model.input_tensors, tensorized_example)}
            _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.model.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            example["predicted_clusters"], _ = self.model.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                                 predicted_antecedents)
            example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
            example['head_scores'] = []
            return example

    def convert_mention(self, mention, output, comb_text):
        start = output['subtoken_map'][mention[0]]
        end = output['subtoken_map'][mention[1]] + 1
        nmention = (start, end)
        mtext = ''.join(' '.join(comb_text[mention[0]:mention[1] + 1]).split(" ##"))
        return (nmention, mtext)

    def get_coref_clusters(self, dict_input):
        output = dict_input

        comb_text = [word for sentence in output['sentences'] for word in sentence]

        seen = set()
        print('\n\nClusters:')
        cluster_output = []
        for cluster in output['predicted_clusters']:
            mapped = []
            for mention in cluster:
                seen.add(tuple(mention))
                mapped.append(self.convert_mention(mention, output, comb_text))
            cluster_output.append(mapped)
            print(mapped, end = ",\n")

        for mention in output['top_spans']:
            if tuple(mention) in seen:
                continue

        return cluster_output

    def merge_id_token(self, list1, list2):
        # module to convert two lists of same size into a tuple list
        merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
        return merged_list


    def merge_sent_token(self, merge_tokens):
        # module to join split words
        comb_tuples = []
        start_t = merge_tokens[0][0]
        str1 = ""
        flag = False
        for index in range(1, len(merge_tokens)):
            if merge_tokens[index - 1][0] == merge_tokens[index][0]:  # case1: index-1 , index id same, #token merging case
                if (flag == False and len(comb_tuples) == 0):
                    value = merge_tokens[index - 1][1].replace("#", "") + merge_tokens[index][1].replace("#", "")
                    str1 = str1 + value
                    flag = True
                elif (flag == False and len(comb_tuples) > 0):
                    comb_tuples.pop()
                    value = merge_tokens[index - 1][1].replace("#", "") + merge_tokens[index][1].replace("#", "")
                    str1 = str1 + value
                    flag = True
                elif flag == True:
                    value = merge_tokens[index][1].replace("#", "")
                    str1 = str1 + value
                    flag = True
            else:  # case2: index-1, index id not same
                if flag == True:  # case2.1 if index-1 id not same as index
                    str1 = str1.replace('"', '')
                    tup1 = (merge_tokens[index - 1][0], str1)
                    comb_tuples.append(tup1)
                    flag = False
                    str1 = merge_tokens[index][1].replace('"', '')
                    tup1 = (merge_tokens[index][0], str1)
                    comb_tuples.append(tup1)
                    str1 = ""
                else:  # case2.2 if index-1 id not same as index
                    temp_str1 = merge_tokens[index][1].replace('"', '')
                    tup1 = (merge_tokens[index][0], temp_str1)
                    comb_tuples.append(tup1)
                    flag = False

        tup1 = (merge_tokens[index][0], str1.replace('"', ''))
        comb_tuples.append(tup1)
        return comb_tuples


    def map_input_tokens(self, json_input):
        # module to convert input tokens into a tuple list

        comb_text = [word for sentence in json_input['sentences'] for word in sentence]
        token_list = json_input['subtoken_map']
        merge_tokens = self.merge_id_token(token_list, comb_text)
        combined_tuples = self.merge_sent_token(merge_tokens)
        return combined_tuples


    def antecedent_to_substitute_in_original(self, antecedent_str, ana_cluster, org_tokens_list):
        # module to substitute antecedent at anaphora place in original list
        ana_list = []
        
        ana_list.append(ana_cluster[0][0])
        ana_list.append(antecedent_str)
        
        #map list elements 
        mapped_token_list = []
        for element in org_tokens_list:
            if element[0] < ana_cluster[0][0]:
                mapped_token_list.append(element)
            elif element[0] == ana_cluster[0][0]:
                mapped_token_list.append(tuple(ana_list))
            elif element[0] >= ana_cluster[0][1]:
                mapped_token_list.append(element)
        
        return mapped_token_list



    def convert_tup_list_to_str(self, coref_list):
        # convert list to string
        coref_str = ""
        for tup in coref_list:
            coref_str = coref_str + " " + tup[1]

        return coref_str


    def map_paragraph(self, json_input, cluster_input):
    
        # get input text tokens as tuple list
        input_tokens_list = self.map_input_tokens(json_input)

        mapped_str_list = []
        substitution_flag = False

        if (len(cluster_input) > 0):
            #clusters are generated. mapping required
            for cluster in cluster_input:
                antecedent_str = cluster[0][1]
                antecedent_str = antecedent_str.rstrip(".")
                for ind in range(1,len(cluster)):
                    if substitution_flag == False:
                        mapped_str_list = self.antecedent_to_substitute_in_original(antecedent_str, cluster[ind], input_tokens_list)
                        substitution_flag = True
                    else:
                        mapped_str_list = self.antecedent_to_substitute_in_original(antecedent_str, cluster[ind], mapped_str_list)
                        
        else:
            mapped_str_list = input_tokens_list

        #convert the mapped list to string
        
        str_o = ""
        for ele in mapped_str_list:
            str_o = str_o + " "+ele[1]
        str_o = str_o.replace("[CLS]","")
        str_o = str_o.replace("[SEP]","")
        
        merged_coref_output = str_o
        
        return merged_coref_output


    #module to extract phrases from constituency tree
    def ExtractPhrases(self, myTree, phrase):
        myPhrases = []
        
        if (myTree.label() == phrase):
            myPhrases.append( myTree.copy(True) )
        for child in myTree:
            if (type(child) is Tree):
                list_of_phrases = self.ExtractPhrases(child, phrase)
                if (len(list_of_phrases) > 0):
                    myPhrases.extend(list_of_phrases)
        return myPhrases


    #function to score the perplexity 
    def score(self, sentence):
        tokenize_input = self.tokenizer_gpt.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer_gpt.convert_tokens_to_ids(tokenize_input)])
        loss=self.model_gpt(tensor_input, lm_labels=tensor_input)
        return math.exp(loss)

    def r1_failure_to_do_so(self, srch_str, str_list):
        # find index of previous sentence
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx - 1

                
        #get Constituency parse from allennlp         
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) 
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)

        #get all possible verb phrases from each parse tree
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2

        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "
            replace_str = str_list[pre_ind+1].replace("do so ", subs_str)
                    
            candidates.append(replace_str)

        return candidates, pre_ind

    def r2_doing_so(self, srch_str, str_list):
        # find index of previous sentence
        org_str_list = str_list[:]
        print("\n original str list : ", len(org_str_list), org_str_list)
        
        #check for inetrsentential case or intrasentential case
        if len(str_list)>1:
            #intersentential case
            for idx in range(len(str_list)):
                if srch_str in str_list[idx]:
                    pre_ind = idx-1
                    print("\n pre index == ", pre_ind)
        else:
            #intrasentential case
            pre_ind = 0
            print("\n pre index string processing from  :" ,str_list[pre_ind])
            str_subpart = str_list[pre_ind].split("doing ", 1)[0]
                    str_list[pre_ind]=str_subpart
                    print("\n pre index str after slicing : ", str_list[pre_ind])
        
        #get Constituency parse from allennlp           
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) #get allennlp output
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2
        
        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "do not " in subs_str:
                subs_str = subs_str.replace("do not ", " ")

            #replace_str = str_list[pre_ind+1].replace("doing so ", subs_str)       
            #candidates.append(replace_str)

            if pre_ind == 0:
                print("\n sub str to substitute : ", subs_str, "org_str_list : ",org_str_list[pre_ind])
                replace_str = org_str_list[pre_ind].replace("doing so ", subs_str)       
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("doing so ", subs_str)       
                candidates.append(replace_str)
        print("\n all candidates are : ", candidates)
        return candidates, pre_ind

    def r3_do_so(self, srch_str, str_list):
        # find index of previous sentence
        org_str_list = str_list[:]
        print("\n original str list : ", len(org_str_list), org_str_list)
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx-1
                print("\n pre index == ", pre_ind)
                if pre_ind == -1:
                    pre_ind = 0
                    print("\n pre index string  :" ,str_list[pre_ind])
                    str_subpart = str_list[pre_ind].split("do ", 1)[0]
                    str_list[pre_ind]=str_subpart
                    print("\n pre index str after slicing : ", str_list[pre_ind])
        
        #get Constituency parse from allennlp           
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) #get allennlp output
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2
        
        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "do not " in subs_str:
                subs_str = subs_str.replace("do not ", " ")

            if pre_ind == 0:
                print("\n sub str to substitute : ", subs_str, "org_str_list : ",org_str_list[pre_ind])
                replace_str = org_str_list[pre_ind].replace("do so ", subs_str)       
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("do so ", subs_str)       
                candidates.append(replace_str)
        #print("\n all candidates :", candidates)
        return candidates, pre_ind

    def r4_did_so(self, srch_str, str_list):
        # find index of previous sentence
        org_str_list = str_list[:]
        print("\n original str list : ", len(org_str_list), org_str_list)
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx-1
                print("\n pre index == ", pre_ind)
                if pre_ind == -1:
                    pre_ind = 0
                    print("\n pre index string  :" ,str_list[pre_ind])
                    str_subpart = str_list[pre_ind].split("did ", 1)[0]
                    str_list[pre_ind]=str_subpart
                    print("\n pre index str after slicing : ", str_list[pre_ind])
        
        #get Constituency parse from allennlp           
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) #get allennlp output
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2
        
        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "do not " in subs_str:
                subs_str = subs_str.replace("do not ", " ")

            #replace_str = str_list[pre_ind+1].replace("did so ", subs_str)       
            #candidates.append(replace_str)
            if pre_ind == 0:
                print("\n sub str to substitute : ", subs_str, "org_str_list : ",org_str_list[pre_ind])
                replace_str = org_str_list[pre_ind].replace("did so ", subs_str)       
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("did so ", subs_str)       
                candidates.append(replace_str)
        print("\n all candidates :" , candidates)
        return candidates, pre_ind

    def r5_does_so(self, srch_str, str_list):
        # find index of previous sentence
        org_str_list = str_list[:]
        print("\n original str list : ", len(org_str_list), org_str_list)
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx-1
                print("\n pre index == ", pre_ind)
                if pre_ind == -1:
                    pre_ind = 0
                    print("\n pre index string  :" ,str_list[pre_ind])
                    str_subpart = str_list[pre_ind].split("does ", 1)[0]
                    str_list[pre_ind]=str_subpart
                    print("\n pre index str after slicing : ", str_list[pre_ind])
        
        #get Constituency parse from allennlp           
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) #get allennlp output
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2
        
        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "do not " in subs_str:
                subs_str = subs_str.replace("do not ", " ")

            #replace_str = str_list[pre_ind+1].replace("does so ", subs_str)       
            #candidates.append(replace_str)
            if pre_ind == 0:
                print("\n sub str to substitute : ", subs_str, "org_str_list : ",org_str_list[pre_ind])
                replace_str = org_str_list[pre_ind].replace("does so ", subs_str)       
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("does so ", subs_str)       
                candidates.append(replace_str)
        print("\n all candidates: ", candidates)
        return candidates, pre_ind

    def r6_done_so(self, srch_str, str_list):
        # find index of previous sentence
        org_str_list = str_list[:]
        print("\n original str list : ", len(org_str_list), org_str_list)
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx-1
                print("\n pre index == ", pre_ind)
                if pre_ind == -1:
                    pre_ind = 0
                    print("\n pre index string  :" ,str_list[pre_ind])
                    str_subpart = str_list[pre_ind].split("done ", 1)[0]
                    str_list[pre_ind]=str_subpart
                    print("\n pre index str after slicing : ", str_list[pre_ind])
        
        #get Constituency parse from allennlp           
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) #get allennlp output
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford Corenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2
        
        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "do not " in subs_str:
                subs_str = subs_str.replace("do not ", " ")

            #replace_str = str_list[pre_ind+1].replace("done so ", subs_str)       
            #candidates.append(replace_str)
            if pre_ind == 0:
                print("\n sub str to substitute : ", subs_str, "org_str_list : ",org_str_list[pre_ind] )
                replace_str = org_str_list[pre_ind].replace("done so ", subs_str)       
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("done so ", subs_str)       
                candidates.append(replace_str)
        print("\n all candidates: ",candidates)
        return candidates, pre_ind




    def r3_follow_this(self, srch_str, str_list):
        # find index of previous sentence
        for idx in range(len(str_list)):
            if srch_str in str_list[idx]:
                pre_ind = idx-1
        
        #get Constituency parse from AllenNLP  
        output1 = self.predictor_cp.predict(sentence=str_list[pre_ind]) 
        test1 = Tree.fromstring(output1['trees'])
        
        #get Constituency parse from stanford orenlp  
        output2 = self.nlp.parse(str_list[pre_ind])
        test2 = Tree.fromstring(output2)
        
        list_of_verb_phrases1 = self.ExtractPhrases(test1, 'VP')
        list_of_verb_phrases2 = self.ExtractPhrases(test2, 'VP')
        
        vp_list = list_of_verb_phrases1 + list_of_verb_phrases2

        candidates = []
        for phrase in vp_list:
            subs_str = " ".join(phrase.leaves())
            subs_str = subs_str + " "

            if "this " in str_list[pre_ind+1]:
                str_list[pre_ind+1]= str_list[pre_ind+1].replace("this ", " ")
                replace_str = str_list[pre_ind+1].replace("follow ", subs_str)
                candidates.append(replace_str)
            
            elif "these " in str_list[pre_ind+1]:
                str_list[pre_ind+1]= str_list[pre_ind+1].replace("these ", " ")
                replace_str = str_list[pre_ind+1].replace("follow ", subs_str)
                candidates.append(replace_str)
            else:
                replace_str = str_list[pre_ind+1].replace("follow ", subs_str)
                candidates.append(replace_str)

        return candidates, pre_ind



    def ranking_ppl(self, candidate_str, str_list, pre_ind):
        can_scores=[]
        for str1 in candidate_str:
            can_scores.append(self.score(str1))
            
        merged_sc_list = [(can_scores[i], candidate_str[i]) for i in range(0, len(candidate_str))]
        
        min_sc = merged_sc_list[0]
        for each in range(1, len(merged_sc_list)):
            if min_sc[0] > merged_sc_list[each][0]:
                min_sc = merged_sc_list[each]

        if pre_ind== 0:
            str_list[pre_ind]=min_sc[1]
        else:
            str_list[pre_ind+1]=min_sc[1]
        return str_list



    #module for constituency parse based post processing    
    def cp_postprocessing(self, input_st):
        op_str=""
        output_str=""
        in_str_list=sent_tokenize(input_st)

        #rule 0: this symbol cases
        if "symbol" in input_st and "this" in input_st:
            if (len(in_str_list) > 2) and (in_str_list[1]=="<icon>." or in_str_list[1]=="<icon> ."):
                input_st = input_st.replace("this", "<icon>",1)
                output_str = input_st

        #rule 1: failure to do so cases
        
        if 'failure to do so' in input_st:
            sr_str = 'failure to do so'
    
            candidate_str, pre_ind = self.r1_failure_to_do_so(sr_str, in_str_list)
            op_str = self.ranking_ppl(candidate_str, in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])
        
        #rule 2: doing so cases   
        elif 'doing so' in input_st:
             
            sr_str = "doing so"
        
            candidate_str, pre_ind = self.r2_doing_so(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])

        #rule 3: do so cases
        elif 'do so' in input_st:
             
            sr_str = "do so"
        
            candidate_str, pre_ind = self.r3_do_so(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])

        #rule 4: did so cases
        elif 'did so' in input_st:
             
            sr_str = "did so"
        
            candidate_str, pre_ind = self.r4_did_so(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])
        
        #rule 5: does so cases
        elif 'does so' in input_st:
             
            sr_str = "does so"
        
            candidate_str, pre_ind = self.r5_does_so(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])
        
        #rule 6: done so cases
        elif 'done so' in input_st:
             
            sr_str = "done so"
        
            candidate_str, pre_ind = self.r6_done_so(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])
            

        elif 'failure to follow' in input_st:
            
            sr_str = 'failure to follow'
            
            candidate_str, pre_ind = self.r3_follow_this(sr_str, in_str_list)
            op_str= self.ranking_ppl(candidate_str,in_str_list, pre_ind)
            
            output_str = ' '.join([str(line) for line in op_str])
        
        if len(output_str) == 0:
            output_str = input_st
        
        return output_str


    

if __name__ == "__main__":
    obj = CoreferenceEngine()

    print("********************************************8\n ")
    '''
    paragraph1 = " operations. <icon>."\
                "this symbol Never attempt to operate this appliance if it is damaged, malfunctioning, partially disassembled, or has missing or broken parts, including a damaged cord or plug."
    
    paragraph2 = "read all instructions before use ." \
                "your safety and the safety of others are very important ." \
                "we have provided many important safety messages in owner's manual and on your modelid ." \
                "always read and follow all safety messages ."
    
    paragraph3 = "warning ." \
                 "operation ." \
                 "if the electrical supply cord is damaged , it must only be replaced by the manufacturer or its service agent or a similar qualified person in order to avoid a hazard ."
    paragraph4 = "connecting to the power supply . warning . do not damage or cut off the ground prong of the power cord . doing so may cause death , fire , electric shock , or product malfunction ."

    
    paragraph5 = "connecting the water lines ." \
                 "flush out the inlet hoses ." \
                 "after connecting the inlet hoses to the water faucets , turn on the water faucets to flush out foreign substances ( dirt , sand or sawdust ) in the water lines ." \
                 "let water drain into a bucket , and check the water temperature to make sure you've connected the hoses to the correct faucets ."
    '''
    json_input = obj.get_coref_for_single_input(obj.pre_process_data(paragraph1))
    cluster_output = obj.get_coref_clusters(json_input)

     
    coref_output = obj.map_paragraph(json_input, cluster_output)
    print("\n\n coref output before ==  ",coref_output )
    
    coref_output = obj.cp_postprocessing(coref_output)
    print("\n\n coref output after ==  ", coref_output)

