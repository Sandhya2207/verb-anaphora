#Script to run coref model for command line and batch interfaces
#usage for command line input test (Default is command line): python coref_test_suite.py 
#usage for Batch testing an excel file: python coref_test_suite.py --input_filepath <excel file input> --output_filepath <excel file output>


import argparse
import pandas as pd
import sys
import os
import jsonlines
from pandas import ExcelWriter
import time
from nltk.tokenize import sent_tokenize

import coreference_engine as cf

class CorefTester:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method to get the singleton instance"""
        if CorefTester.__instance is None:
            CorefTester()
        return CorefTester.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if CorefTester.__instance is not None:
            raise Exception("CorefTester is not instantiable")
        else:
            CorefTester.__instance = self
        self.initialize()

    def initialize(self):
        """
        function to initialise the Coref engine object that gives Coref output and argument parser object"""
        self.coref_engine_obj = cf.CoreferenceEngine()
        self.arg_parser = argparse.ArgumentParser()
        
    def get_data_excel(self, file_path):
        """
        Read data from excel. TBD: Optimised
        @file_path : Input file to be read
        Returns: 
            list of sentences from input file
        """
        df = pd.read_excel(file_path)
        trainX=df['SS5_merge_list'].values.tolist() 
        return trainX


if __name__ == "__main__":
    #command line input examples

    #eval_text1 = "if the electrical supply cord is damaged, it must only be replaced by the manufacturer or its service agent or a similar qualified person in order to avoid a hazard."
    #eval_text2 = "warning. operation."

    #eval_text1 = "do not store or spill liquid detergents, cleaners, or bleaches (chlorine bleach, oxygen bleach) on the appliance. doing so may result in corrosion, discoloration or damage to the surface of the appliance."
    #eval_text2 = "caution. operation."

    #eval_text1 = "this symbol alerts you to potential hazards that can kill or injure you and others."
    #eval_text2="read all instructions before use. <icon>. "

    #eval_text1= "keep hands and tools out of the ice compartment door and dispenser chute. failure to do so may result in damage or personal injury."
    #eval_text2= "<icon>. caution."

    #eval_text1="do not use solvent-based detergent on the product. doing so can cause corrosion or damage, product failure, electrical shock, or fire."
    #eval_text2="warning. operation."

    #eval_text1="use the jumbo wash/bedding cycle for buoyant or nonabsorbent items such as pillows or comforters. failure to follow this caution can result in leakage."
    #eval_text2="sorting laundry. caution."
    
    start_m = time.time()
    tester_obj = CorefTester.getInstance()
    
    tester_obj.arg_parser.add_argument("--run_time_input",
                        default=False,
                        type=bool,
                        help="True if input_data is passed on runtime through keyboard, only a sample at a time")
    
    tester_obj.arg_parser.add_argument("--input_type",
                        default='SS',
                        type=str,
                        help="Prediction type 'SingleSample(SS)' or 'Batch(B) file'")
    tester_obj.arg_parser.add_argument("--input_filepath",
                        default="input_file.xlsx",
                        help="Path to input file to feed the model ")
    tester_obj.arg_parser.add_argument("--output_filepath",
                        default="output_file.xlsx",
                        help="Path to output file with predictions from the model ")
    tester_obj.arg_parser.add_argument("--json_output",
                        default=False,
                        help="Path to output file with predictions from the model in jsonlines format")

    p_args = tester_obj.arg_parser.parse_args()
    
    if(p_args.run_time_input==False and p_args.input_filepath == None and p_args.output_filepath == None ): 
        print('Neither data object is passed nor data file path; one of these should be passed'); sys.exit(0)

    if p_args.input_type == 'SS':
        flag = "y"
        while flag=="y":
            eval_text1 = input("\n\nEnter the sentence: ")
            eval_text2 = input("\n\n Enter context if any: ")
            context_len=len(sent_tokenize(eval_text2))
            eval_text = eval_text2 + " " + eval_text1
            
            start_s = time.time()

            json_text=tester_obj.coref_engine_obj.get_coref_for_single_input(tester_obj.coref_engine_obj.pre_process_data(eval_text))
            cluster_output=tester_obj.coref_engine_obj.get_coref_clusters(json_text)
            coref_output = tester_obj.coref_engine_obj.map_paragraph(json_text, cluster_output)
            coref_output = tester_obj.coref_engine_obj.cp_postprocessing(coref_output)
            
            #remove context string from output
            coref_out_list = sent_tokenize(coref_output)
            out_str=" "
            for line in range(context_len,len(coref_out_list)):
                out_str = out_str + " "+ coref_out_list[line]


            print("\n\n coref mapped output sentence:  ", out_str)
            end_s = time.time() - start_s
            print("\n\n Time taken for this sentence: ", end_s)
            flag=input("\n\nDo you want to continue ? (y/n) ")
    else:
        if not os.path.exists(p_args.input_filepath):
            print('Input file does not exist');
            sys.exit(0)
        
        start_b = time.time()
        json_for_conll_list=[]   
        df = pd.DataFrame(columns=['Coref-input', 'Coref-output'])
        i =0
        for eval_text in tester_obj.get_data_excel(p_args.input_filepath):
            df.loc[i,'Coref-input']= eval_text

            json_text=tester_obj.coref_engine_obj.get_coref_for_single_input(tester_obj.coref_engine_obj.pre_process_data(eval_text))
            
            if (p_args.json_output != None):
                json_for_conll_list.append(json_text)

            cluster_output=tester_obj.coref_engine_obj.get_coref_clusters(json_text)
            coref_output = tester_obj.coref_engine_obj.map_paragraph(json_text, cluster_output)
            coref_output = tester_obj.coref_engine_obj.cp_postprocessing(coref_output)
            
            df.loc[i,'Coref-output'] = coref_output
            print("\n coref mapped output sentence after postprocess:  ", coref_output)

            i+=1

        if (p_args.json_output != None):
            with jsonlines.open(p_args.json_output, 'w') as writer1:
                for each in json_for_conll_list:
                    writer1.write(each)
            writer1.close()

        with pd.ExcelWriter(p_args.output_filepath) as writer:
            df.to_excel(writer, sheet_name='Coref output')
            writer.save()
            print('Results are written successfully to Excel File.')
        writer.close()
        end_b = time.time() - start_b
        print('Time taken for batch file processing: ', time.strftime("%H:%M:%S", time.gmtime(end_b)))

    tester_obj.coref_engine_obj.nlp.close()
    end_m = time.time() - start_m
    print('Time elapsed in current session: ', time.strftime("%H:%M:%S", time.gmtime(end_m)))



