## Preapre data and process with E2E BERT Coref Model.

## Steps in sequence to preprocess data. 

### 1. Merge excel sheets of data in a common format and combine context information

python merge_excel_data_sheets.py  \<input excel file\> \<output excel file\>


### 2. Steps to get the E2E Coref Model working:

1. Clone the repo [E2E Coref](https://github.com/mandarjoshi90/coref).
2. Setup as in repo Readme
3. Download the pre-trained model Bert-Base
4. For prediction, setup the enviroment with minimum packages in requirements.txt file
5. The script has external dependency on the following pre-trained models:
     - Download the stanfordcorenlp models in **\<stanfordcorenlp-4.2.0\>** folder
     - Allennlp model  **\<elmo-constituency-parser-2020.02.10.tar.gz\>** accessed via api
     - Allennlp model **\<biaffine-dependency-parser-ptb-2020.04.06.tar.gz\>** accessed via api
     - Pre-trained openAI GPT Language Model


    **For Command line input predictions :**
    
    python coref_test_suite.py
    
    **For Batch predictions  in text format only :**
    
    python coref_test_suite.py  --input_filepath \<input excel file\> --output_filepath \<Output excel file\>  --input_type B
    
    **For Batch predictions in both text and jsonlines format (required for conll conversion)  :**
    
    python coref_test_suite.py  --input_filepath \<input excel file\> --output_filepath \<Output excel file\>  --input_type B --json_output  \<output jsonlines file\>


### 3. Split the data in original sections and calculate ROUGE scores

   python evaluate_rouge_score.py  \<input excel file1\> \<input excel file2 \> \< output excel file\>

   input file1 : output file of merge_excel_data_sheets script

   input file2 : output file from coref model


### 4. For converting model output to CoNLL format
     
   **For converting model predicted output to CoNLL**
     
   - Save the output in jsonlines format

   - python jsonlines2conll.py \<file in jsonlines format\> -o \<filename to save CoNLL format\>
     
   **For converting gold output to CoNLL**
   
   - Save the output in jsonlines format
     
   - python jsonlines2conll.py -g \<file in jsonlines format\> -o \<filename to save CoNLL format\>

### 5. For evaluating the MUC, B-CUBE, CEAF-E scores

  - perl scorer.pl muc/bcub/ceafe \<gold file in CoNLL format\>   \< model output file in CoNLL format\>






