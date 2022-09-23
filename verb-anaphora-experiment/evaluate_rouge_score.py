#post process and seperate the context sections for rouge score calculations
#usage: python evaluate-rouge-scores.py <step2 output file> < coref output file> <rouge scores output file>

import pandas as pd
import numpy as np
import xlrd
import re
from nltk import sent_tokenize
from rouge import Rouge
import sys
import json

def split_context(df):

	#iterate through all rows in df 
	df['coref_SS5_output']=" "

	for ind in df.index:
		
		coref_op_sent_list = sent_tokenize(df['coref_output'][ind])
		max_len = len(coref_op_sent_list)
		str_ss5 = " "
		
		for line in range(df['context_sent_no'][ind],max_len):
			str_ss5=str_ss5 + " "+ coref_op_sent_list[line]
		
		#print("\n SS5 output ==", str_ss5)
		df['coref_SS5_output'][ind]=str_ss5

	return df  


def calculate_rouge_scores(df):
	#calculate row-wise Rouge scores  for SS5
	
	df['ss5_R1_score_p']=" "
	df['ss5_R1_score_r']=" "
	df['ss5_R1_score_f']=" "
	df['ss5_R2_score_p']=" "
	df['ss5_R2_score_r']=" "
	df['ss5_R2_score_f']=" "
	df['ss5_RL_score_p']=" "
	df['ss5_RL_score_r']=" "
	df['ss5_RL_score_f']=" "

	rouge = Rouge()
	for ind in df.index:
		
		hypo_txt=df['coref_SS5_output'][ind]
		ref_txt=df['Sub-Section-5 O/P'][ind]
		scores = rouge.get_scores(hypo_txt, ref_txt)
		
		df['ss5_R1_score_p'][ind]= scores[0]['rouge-1']['p']
		df['ss5_R1_score_r'][ind]= scores[0]['rouge-1']['r']
		df['ss5_R1_score_f'][ind]= scores[0]['rouge-1']['f']
		df['ss5_R2_score_p'][ind]= scores[0]['rouge-2']['p']
		df['ss5_R2_score_r'][ind]= scores[0]['rouge-2']['r']
		df['ss5_R2_score_f'][ind]= scores[0]['rouge-2']['f']
		df['ss5_RL_score_p'][ind]= scores[0]['rouge-l']['p']
		df['ss5_RL_score_r'][ind]= scores[0]['rouge-l']['r']
		df['ss5_RL_score_f'][ind]= scores[0]['rouge-l']['f']

	print("\n\ndone all rows of  Sub-Section-5\n\n")
	

	hypothesis_list=df['coref_SS5_output'].tolist()
	reference_list=df['Sub-Section-5 O/P'].tolist()
	
	scores_ss = rouge.get_scores(hypothesis_list, reference_list, avg=True)
	with open("Coref_output_ROUGE_scores.json", 'w' ) as outfile:
		json.dump( scores_ss, outfile)
	return df, scores_ss




#main method    
if __name__ == "__main__":

	#read excel sheets into dataframe
	infile_step2=sys.argv[1]
	infile_coref_out=sys.argv[2]
	outfile=sys.argv[3]

	df1 = pd.read_excel(infile_step2)
	df2 = pd.read_excel(infile_coref_out)

	#copy coref output to original df 
	df1['coref_output']=df2['Coref-output']

	df1_out= split_context(df1)
	
	#******** calculate rouge scores *************
	df_out, p_score = calculate_rouge_scores(df1_out)
	
	#save the proceseed df to a file
	# Create a Pandas Excel writer using XlsxWriter as the engine.
	writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

	df_out.to_excel(writer)

	#close file
	writer.save()
	print("\n\n Coref Output average ROUGE score == ", p_score)
	



