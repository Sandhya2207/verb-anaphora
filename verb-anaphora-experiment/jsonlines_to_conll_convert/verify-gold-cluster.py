#code to verify the gold cluster in jsonlines format
#usage python verify-gold-cluster.py <jsonlines file for cluster verification> > screen-log.txt

import jsonlines
import sys


#function to merge 2 lists into a tuple
def merge2(list1, list2): 
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))] 
    return merged_list


# module to join split words due to tokenization
def merge_sent_token(merge_tokens):
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



def map_input_tokens(json_input):
        # module to convert input tokens into a tuple list

        comb_text = [word for sentence in json_input['sentences'] for word in sentence]
        token_list = json_input['subtoken_map']
        merge_tokens = merge2(token_list, comb_text)
        combined_tuples = merge_sent_token(merge_tokens)
        #print("\n combined tuples == ", combined_tuples)
        return combined_tuples

def createList(r1, r2): 
    return list(range(r1, r2)) 


if __name__ == "__main__":
    #Read file in jsonlines format
    
    infile = sys.argv[1]

    with jsonlines.open(infile,'r') as f:
        lst = [obj for obj in f]


    print("\n no. of files are == ",len(lst))

    for each in lst:
        print("\n**************************\n")
        print("\n filename == ", each['doc_key'])

        mapped_str_list = map_input_tokens(each)
        
        str_o = ""
        for ele in mapped_str_list:
            str_o = str_o + " "+ele[1]
        
        print("\n sentence == ", str_o)
        token_len=len(each['subtoken_map'])
        s_list=[]
        for i in each['sentences']:
            for j in i:
                s_list.append(j)
       
        index_l=createList(0, token_len)

        new_l = merge2(index_l,s_list)
        print("\n token id: \n", new_l)
        print("\n clusters == ", each['clusters'])
