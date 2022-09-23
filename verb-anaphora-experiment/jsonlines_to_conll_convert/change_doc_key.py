#change doc_key with filename_to_use for jsonlines file
#usage: python change_doc_key.py <jsonfile output from E2E model> <input excel file to E2E model>


import jsonlines
import pandas as pd
import os
import sys

def merge(list1, list2): 
      
    merged_list = [( list1[i], list2[i] ) for i in range(0, len(list1))] 
    return merged_list

def createList(r1, r2): 
    return list(range(r1, r2))

def create_response_jsonfile(revise_name):
	response_list=[]
	for data in revise_name:
	    new_data={}
	    new_data['doc_key']= data['doc_key']
	    new_data['sentence_map']=data['sentence_map']
	    new_data['sentences']=data['sentences']
	    new_data['speakers']=data['speakers']
	    new_data['subtoken_map']=data['subtoken_map']
	    new_data['top_spans']=data['top_spans']
	    new_data['head_scores']=data['head_scores']
	    new_data['predicted_clusters']=data['predicted_clusters']
	    response_list.append(new_data)
	return(response_list)

def create_gold_jsonfile(revise_name):
	gold_list=[]
	for data in revise_name:
		new_data={}
		new_data['doc_key']= data['doc_key']
		new_data['sentence_map']=data['sentence_map']
		new_data['sentences']=data['sentences']
		new_data['speakers']=data['speakers']
		new_data['subtoken_map']=data['subtoken_map']
		new_data['top_spans']=data['top_spans']
		new_data['head_scores']=data['head_scores']
		new_data['clusters']=data['predicted_clusters']
		gold_list.append(new_data)
	return(gold_list)

if __name__ == "__main__":
    
    j_infile=sys.argv[1]
    x_infile=sys.argv[2]

    #Read CoNLL train data in jsonlines format
    with jsonlines.open(j_infile,'r') as f:
    	lst = [obj for obj in f]

    #read excel input file
    df=pd.read_excel(x_infile)

    fname_lst=df['filename_to_use'].tolist()

    index_l=createList(0, len(fname_lst))
    new_l = merge(index_l,fname_lst)

    revise_lst=[]
    for idx, each in enumerate(lst):
    	each['doc_key'] = new_l[idx][1]
    	revise_lst.append(each)

    response_file = create_response_jsonfile(revise_lst)

    pt, fname = os.path.split(j_infile)
    name, ext = os.path.splitext(fname)
    out_name = name+'_response.jsonlines'


    with jsonlines.open(os.path.join(pt, out_name), 'w') as writer:
    	for each in response_file:
    		writer.write(each)
    writer.close()

    
    #for gold jsonfile
    #uncomment the gold file creation part for creating gold jsonfile for verification
    '''
    
    gold_file = create_gold_jsonfile(revise_lst)

    pt, fname = os.path.split(j_infile)
    name, ext = os.path.splitext(fname)
    out_name = name+'_gold.jsonlines'


    with jsonlines.open(os.path.join(pt, out_name), 'w') as writer:
    	for each in gold_file:
    		writer.write(each)
    writer.close()
    '''
    print("\n done !!!")



