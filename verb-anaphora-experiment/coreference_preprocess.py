#Script to preprocess data for coref model

from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import numpy as np
import os
import sys
import csv
import re
import xlsxwriter
import json
import ast
import time

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

class CoreferencePreprocess:
	def __init__(self):
		self.replace_cnt = 0
		#pos tagger class instance
		self.predictor = Predictor.from_path("/home/sandhya/coref_docker/Coreference-Module/src/allennlp-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")

	def get_pos(self, input_str):
		output=self.predictor.predict(sentence = input_str)
		word_list = output['words']
		pos_list = output['pos']
	
		return (word_list, pos_list)

	def preprocess_single_input(self, input_str):
		search_list = ['washer','refrigerator','tv','television','ac','air','product','appliance','unit','system','apparatus', 'machine']
		replace_str = "modelid"

		word_list, pos_list = self.get_pos(input_str)

		for item in search_list:
			word_ind_list=[]
			for index, wd in enumerate(word_list):
				if wd == item:
					word_ind_list.append(index)

			for word_ind in word_ind_list:
				if word_ind == 0:
					continue
				elif item == 'air' and word_list[word_ind + 1] == 'conditioner':
					new_item = item + ' conditioner'
					
					if (pos_list[word_ind - 1] == "DET") and (word_list[word_ind - 1] == 'this'):
						search_str = "this " + new_item
						input_str = input_str.replace(search_str,replace_str)
						self.replace_cnt = self.replace_cnt + 1
					elif (pos_list[word_ind - 1] == "DET") and (word_list[word_ind - 1] != 'this'):
						search_str = word_list[word_ind - 1] + " " + new_item
						mod_replace_str = word_list[word_ind - 1] + " " + replace_str
						input_str = input_str.replace(search_str,mod_replace_str,1)
						self.replace_cnt = self.replace_cnt + 1

				else:
					if (pos_list[word_ind - 1] == "DET") and (word_list[word_ind - 1] == 'this'):
						search_str = "this " + item
						input_str = input_str.replace(search_str,replace_str)
						self.replace_cnt = self.replace_cnt+1
						
					elif (pos_list[word_ind - 1] == "DET") and (word_list[word_ind - 1] != 'this'):
						search_str = word_list[word_ind - 1] + " " + item
						mod_replace_str = word_list[word_ind - 1] + " " + replace_str
						input_str = input_str.replace(search_str,mod_replace_str,1)
						self.replace_cnt = self.replace_cnt+1

		str_search = "this manual"
		str_replace = "owner's manual"
		input_str = input_str.replace(str_search, str_replace)

		return (input_str)
	'''
	def prepare_batch_coref(df):
		#join the immediate parent to paragraph and 
		#convert each row to a file with sentence tokenized

		#new columns in df
		df['Id'] = ""
		df['SS4_org_sent_no'] = ""
		df['SS5_org_sent_no'] = ""
		df['SS5_merge_sent_no'] = ""
		df['SS5_merge_list'] = ""
		

		for ind in df.index:

			df['Id'][ind] = "P"+str(ind)
			parent_str = " "
			pgp_list = []
			p_flag = False
			gp_flag = False

			if (df['Sub-Section-4'][ind] != "nan."):
				p_flag = True
				pgp_list.append(df['Sub-Section-4'][ind])

			if (df['Sub-Section-3'][ind] != "nan."):
				if p_flag == True:
					gp_flag = True
					pgp_list.append(df['Sub-Section-3'][ind])
				else:
					p_flag = True
					pgp_list.append(df['Sub-Section-3'][ind])

			if (df['Sub-Section-2'][ind] != "nan."):
				if p_flag == False:
					p_flag = True
					pgp_list.append(df['Sub-Section-2'][ind])
				else:
					if gp_flag == False:
						gp_flag = True
						pgp_list.append(df['Sub-Section-2'][ind])

			if (df['Sub-Section-1'][ind] != "nan."):
				if p_flag == False:
					p_flag = True
					pgp_list.append(df['Sub-Section-1'][ind])
				else:
					if gp_flag == False:
						gp_flag = True
						pgp_list.append(df['Sub-Section-1'][ind])

			if (df['Sub-Section'][ind] != "nan."):
				if p_flag == False:
					p_flag = True
					pgp_list.append(df['Sub-Section'][ind])
				else:
					if gp_flag == False:
						gp_flag = True
						pgp_list.append(df['Sub-Section'][ind])

			if (df['Section'][ind] != "nan."):
				if p_flag == False:
					p_flag = True
					pgp_list.append(df['Section'][ind])
				else:
					if gp_flag == False:
						gp_flag = True
						pgp_list.append(df['Section'][ind])					

			for i in reversed(pgp_list):
				parent_str = parent_str+" "+str(i)
			
			#tokenize the sentences and words
			para_str = (parent_str +" "+ df['Sub-Section-5'][ind]).strip()
			merge_sent_list = sent_tokenize(para_str)
			
			#save the tokenize lengths and strings
			if df['Sub-Section-4'][ind] == "nan.":
				df['SS4_org_sent_no'][ind] = 0
			else:
				df['SS4_org_sent_no'][ind] = len(sent_tokenize(df['Sub-Section-4'][ind]))
			
			df['SS5_org_sent_no'][ind] = len(sent_tokenize(df['Sub-Section-5'][ind]))
			df['SS5_merge_sent_no'][ind] = len(merge_sent_list)
			df['SS5_merge_list'][ind] = para_str
		
		return (df)

	'''

if __name__ == "__main__":
	obj = CoreferencePreprocess()

	paragraph1 = "if the appliance electrical supply cord is damaged, it must only be replaced by the manufacturer or its service agent or a similar qualified person in order to avoid a hazard the washer."
	preprocessed_str = obj.preprocess_single_input(paragraph1)
	
