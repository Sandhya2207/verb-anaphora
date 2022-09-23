#merge excel sheets to standard format for combined processing
#usage: python step0-merge-data-excel-sheets.py <input file in excel format> <output file in excel format>

import pandas as pd
import numpy as np
import xlrd
import re
import sys
from nltk.tokenize import sent_tokenize

def join_sections(df):
    #join the immediate parent to paragraph 

    #new columns in df
    df['context_str'] = ""
    df['context_sent_no'] = ""
    df['SS5_merge_sent_no'] = ""
    df['SS5_merge_list'] = ""
    
    #add '.' after every sentence where it is missing
    for col in ['Section','Sub-Section','Sub-Section O/P', 'Sub-Section-1', 'Sub-Section-1 O/P',
           'Sub-Section-2', 'Sub-Section-2 O/P','Sub-Section-3', 'Sub-Section-3 O/P',
           'Sub-Section-4','Sub-Section-4 O/P','Sub-Section-5','Sub-Section-5 O/P']:
        for ind in df.index:
            df[col][ind] = add_fullstop(df[col][ind])

    for ind in df.index:
        parent_str = " "
        pgp_list = []
        p_flag = False
        gp_flag = False

        if (df['Sub-Section-4'][ind] != "nan ."):
            p_flag = True
            pgp_list.append(df['Sub-Section-4'][ind])

        if (df['Sub-Section-3'][ind] != "nan ."):
            if p_flag == True:
                gp_flag = True
                pgp_list.append(df['Sub-Section-3'][ind])
            else:
                p_flag = True
                pgp_list.append(df['Sub-Section-3'][ind])

        if (df['Sub-Section-2'][ind] != "nan ."):
            if p_flag == False:
                p_flag = True
                pgp_list.append(df['Sub-Section-2'][ind])
            else:
                if gp_flag == False:
                    gp_flag = True
                    pgp_list.append(df['Sub-Section-2'][ind])

        if (df['Sub-Section-1'][ind] != "nan ."):
            if p_flag == False:
                p_flag = True
                pgp_list.append(df['Sub-Section-1'][ind])
            else:
                if gp_flag == False:
                    gp_flag = True
                    pgp_list.append(df['Sub-Section-1'][ind])

        if (df['Sub-Section'][ind] != "nan ."):
            if p_flag == False:
                p_flag = True
                pgp_list.append(df['Sub-Section'][ind])
            else:
                if gp_flag == False:
                    gp_flag = True
                    pgp_list.append(df['Sub-Section'][ind])

        if (df['Section'][ind] != "nan ."):
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
        
        df['context_str'][ind] = parent_str
        

        df['context_sent_no'][ind] = len(sent_tokenize(df['context_str'][ind]))
        df['SS5_merge_sent_no'][ind] = len(merge_sent_list)
        df['SS5_merge_list'][ind] = para_str
    return (df)

#add full stop at the end of sentence wherever missing
def add_fullstop(str):
    str = re.sub('([.,!?()])', r' \1 ', str)
    str = re.sub('\s{2,}', ' ', str)
    str = str.strip()
    if str == "":                    # Don't change empty strings.
        return (str)
    if str[-1] in ["?", ".", "!"]:   # Don't change if already okay.
        return (str)
    if str[-1] == ",":               # Change trailing ',' to '.'.
        return (str[:-1] + " .")
    return (str + " .")

def merge_sheets(df1, df2, df3):
    #rename paragraph sheet to common format
    df1.rename(columns = {'Sub-Section Output':'Sub-Section O/P','Paragraph':'Sub-Section-5', 'Paragraph Output':'Sub-Section-5 O/P','P-Remarks':'SS5-Remarks'}, inplace = True) 
       
    #rename Troubleshooting Questions sheet to common format
    df2.rename(columns={'Question':'Sub-Section-4', 'Question Output':'Sub-Section-4 O/P', 'Q_remarks':'SS4-Remarks', 'Answer':'Sub-Section-5',
           'Answer Output':'Sub-Section-5 O/P', 'A_remarks':'SS5-Remarks'}, inplace=True)

    new_cols=['Sub-Section O/P', 'SS-Remarks','Sub-Section-1 O/P',
           'SS1-Remarks','Sub-Section-2 O/P', 'SS2-Remarks',
           'Sub-Section-3', 'Sub-Section-3 O/P', 'SS3-Remarks']

    df2=df2.reindex(columns=[*df2.columns.tolist(), *new_cols], fill_value=np.nan)

    df2=df2[['Product Category', 'Product Name', 'Section', 'Sub-Section',
           'Sub-Section O/P', 'SS-Remarks', 'Sub-Section-1', 'Sub-Section-1 O/P',
           'SS1-Remarks', 'Sub-Section-2', 'Sub-Section-2 O/P', 'SS2-Remarks',
           'Sub-Section-3', 'Sub-Section-3 O/P', 'SS3-Remarks', 'Sub-Section-4',
           'Sub-Section-4 O/P', 'SS4-Remarks', 'Sub-Section-5',
           'Sub-Section-5 O/P', 'SS5-Remarks']]

    #rename Troubleshooting Tables sheet to common format
    df3.rename(columns={'Sub-Section-3':'Sub-Section-4','Sub-Section-3 O/P':'Sub-Section-4 O/P', 'SS3-Remarks':'SS4-Remarks', 
                         'Sub-Section-4':'Sub-Section-5','Sub-Section-4 O/P':'Sub-Section-5 O/P', 'SS4-Remarks':'SS5-Remarks'}, inplace=True)

    new_cols=['Sub-Section O/P', 'SS-Remarks','Sub-Section-1 O/P',
           'SS1-Remarks','Sub-Section-3', 'Sub-Section-3 O/P', 'SS3-Remarks']

    df3=df3.reindex(columns=[*df3.columns.tolist(), *new_cols], fill_value=np.nan)

    df3=df3[['Product Category', 'Product Name', 'Section', 'Sub-Section',
           'Sub-Section O/P', 'SS-Remarks', 'Sub-Section-1', 'Sub-Section-1 O/P',
           'SS1-Remarks', 'Sub-Section-2', 'Sub-Section-2 O/P', 'SS2-Remarks',
           'Sub-Section-3', 'Sub-Section-3 O/P', 'SS3-Remarks', 'Sub-Section-4',
           'Sub-Section-4 O/P', 'SS4-Remarks', 'Sub-Section-5',
           'Sub-Section-5 O/P', 'SS5-Remarks']]

    # Merge into single df

    m_df=pd.DataFrame()
    m_df=m_df.append(df1,ignore_index = True)
    m_df=m_df.append(df2,ignore_index = True)
    m_df=m_df.append(df3,ignore_index = True)

    m_df = m_df.astype(str)

    #convert df data to lowercase
    for columns in m_df.columns:
        m_df[columns] = m_df[columns].str.lower() 

            
    m_df=m_df.sort_values(by=['Product Category'])
    m_df.reset_index(drop=True, inplace=True)
    
    # fill in the filename
    m_df['filename_to_use']=""
    for ind in m_df.index:
        if (m_df['Product Category'][ind]=='air conditioner') and (m_df['Product Name'][ind]=='lw1517ivsm'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_AC_lw1517ivsm'

        elif (m_df['Product Category'][ind]=='french door refrigerator') and (m_df['Product Name'][ind]=='lrfds3006*/lrfvs3006*/lrfvc2406*/lrfxc2406*/ lrfds3016*/lrfxc2416*/lrfdc2406*'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_FDR_lrfds3006'

        elif (m_df['Product Category'][ind]=='lg oled tv') and (m_df['Product Name'][ind]=='oled65c8pta'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_OTV_oled65c8pta'

        elif (m_df['Product Category'][ind]=='top freezer refrigerator') and (m_df['Product Name'][ind]=='mfl67527912-9'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_TFR_mfl67527912'

        elif (m_df['Product Category'][ind]=='vacuum cleaner') and (m_df['Product Name'][ind]=='mfl69883639'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_VC_mfl69883639'

        elif (m_df['Product Category'][ind]=='washing machine') and (m_df['Product Name'][ind]=='luwm101hwa'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_WM_luwm101hwa'

        elif (m_df['Product Category'][ind]=='washing machine') and (m_df['Product Name'][ind]=='wm9500h*a'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_WM_wm9500h'

        elif (m_df['Product Category'][ind]=='washing machine') and (m_df['Product Name'][ind]=='wt7060c* /wt7100c*/ wt7250c*/ wt7300c*/ wt7305c*/ wt7800c* wt7880h*a/ wt7900h*a'):
            m_df['filename_to_use'][ind]= 'P'+str(ind)+'_WM_wt7060c'            
    
    m_df=m_df[['Product Category', 'Product Name', 'filename_to_use', 'Section', 'Sub-Section',
       'Sub-Section O/P', 'SS-Remarks', 'Sub-Section-1', 'Sub-Section-1 O/P',
       'SS1-Remarks', 'Sub-Section-2', 'Sub-Section-2 O/P', 'SS2-Remarks',
       'Sub-Section-3', 'Sub-Section-3 O/P', 'SS3-Remarks', 'Sub-Section-4',
       'Sub-Section-4 O/P', 'SS4-Remarks', 'Sub-Section-5',
       'Sub-Section-5 O/P', 'SS5-Remarks']]

    return (m_df)

def basic_preprocess(df):
    for col in ['SS5_merge_list','Reference-Output']:
        for ind in df.index:
            df[col][ind] = add_fullstop(df[col][ind])

    #convert df data to lowercase
    for columns in ['SS5_merge_list','Reference-Output']:
        df[columns] = df[columns].str.lower()

    return df 



#main method    
if __name__ == "__main__":
    infile=sys.argv[1]
    outfile=sys.argv[2]

    '''
    dfp=pd.read_excel(infile,sheet_name="Paragraphs")
    dftq=pd.read_excel(infile,sheet_name="Troubleshooting-Questions")
    dftt=pd.read_excel(infile,sheet_name="Troubleshooting-Tables")

    m_df = merge_sheets(dfp, dftq, dftt)
    
    #join sections for context information
    out_df = join_sections(m_df)
    '''

    df_in = pd.read_excel(infile)
    out_df = basic_preprocess(df_in)
    out_df.to_excel(outfile)
    print("\n\n Step-0: Merging input excel sheets to common format done....!!!\n\n")

