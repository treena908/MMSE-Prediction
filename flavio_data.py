from __future__ import with_statement
import os
import re
import pandas as pd
import numpy as np
import pickle
import nltk
from sklearn.metrics import mean_squared_error
from nltk.tokenize import sent_tokenize,word_tokenize
import xlsxwriter 
GET_INV= True
dir_result="./result/"
ft_1=[]
ft_2=[]
ft_3=[]
ft_4=[]
ft_5=[]
ft_6=[]
ft_7=[]
labels=[]
fname=[]
utterances_count=[]
word_count=[]
utterances=[]
def file_tokenization(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []
    for line in input_file:
        # print(line)
        for element in line.split("\n"):
            if "*PAR" in element :
                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                tokenized_list = word_tokenize(cleaned_string)
                # output_list = output_list + tokenized_list
                output_list = output_list + ['\n'] + tokenized_list

    # word = sum(1 for x in output_list if x != '\n')
    # word_count.append(word)
    # print("word %d\n" % word)
    return output_list
def generate_text(input_file):
    text=""
    for line in input_file:
        # print("line")
        # print(line)
        for element in line.split("\n"):
            question=False
            stop=False
            if "*PAR" in element or ["*INV","%mor","%gra"] not in element:
                if "*PAR" in element:
                    str=element
                    if "?" in element:
                        question
                        cleaned_string = element.split('?', 1)[0]
                        # adding punctuation
                        cleaned_string = cleaned_string + "?"
                    elif "." in element:
                        print("line")
                        print(element)
                        cleaned_string = element.split('.', 1)[0]
                        # adding punctuation
                        cleaned_string = cleaned_string + "."
                        print(cleaned_string)

                    # replace par with empty string, deleting the part of the string that starts with PAR
                    cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR', ''))
                    print(cleaned_string)

                    # substitute numerical digits, deleting underscores
                    cleaned_string = re.sub(r'[\d]+', '', cleaned_string.replace('_', ''))
                    print(cleaned_string)

                    text = text + cleaned_string

                if "*PAR" in element:
                    par=True
                else:
                    par=False



    utterances.append(text)

def generate_feature(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    f_1=f_2=f_3=f_4=f_5=f_6=f_7=count=word=0
    file=input_file


    for line in input_file:

        for element in line.split("\n"):

            if "*PAR" in element :

                # if "?" in element:
                #     cleaned_string = element.split('?', 1)[0]
                #     # adding punctuation
                #     cleaned_string = cleaned_string + "?"
                # else:
                #     print("line")
                #     print(element)
                #     cleaned_string = element.split('.', 1)[0]
                #     # adding punctuation
                #     cleaned_string=cleaned_string+"."
                #     print(cleaned_string)



                # replace par with empty string, deleting the part of the string that starts with PAR
                # cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR', ''))
                # print(cleaned_string)
                #
                # # substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+', '', cleaned_string.replace('_', ''))
                # print(cleaned_string)
                #
                # text=text+cleaned_string
                
                #print(element)
                if '(.)' in element:
                    f_1+=element.count('(.)')
                elif '(..)' in element:
                    f_2+=element.count('(..)')
                elif '(...)' in element:
                    f_3+=element.count('(...)')
                elif '[/]' in element:
                    f_4+=element.count('[/]')
                elif '[//]' in element:
                    f_5+=element.count('[//]')
                elif '&uh' in element:
                    f_6+=element.count('&uh')
                elif '+...' in element:
                    f_7+=element.count('+...')
                count+=1



    ft_1.append(f_1)
    ft_2.append(f_2)
    ft_3.append(f_3)
    ft_4.append(f_4)
    ft_5.append(f_5)
    ft_6.append(f_6)
    ft_7.append(f_7)
    utterances_count.append(count)
    # word_count.append(word)
    print(ft_1[0])
    print(ft_4[0])
    print(ft_2[0])
    print(ft_3[0])
    print(ft_5[0])
    print(ft_6[0])
    print(utterances_count[0])
    #return f_1,f_2,f_3,f_4,f_5,f_6,f_7
# def file_tokenization_utterances(input_file):
#     '''
#     :param input_file: single dataset file as readed by Python
#     :return: tokenized string of a single patient interview
#     '''
#     output_list = []
#     #count=0
#     length=0
#     previous="*INV"
#
#     que_list = ["else",
#                 "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
#                 "going on",
#
#                 "is that all", "is that", "what's happening", " happening", "action", "some more", "more"]
#     for line in input_file:
#         for element in line.split("\n"):
#             #if "*PAR" in element or ("*INV" in element and GET_INV):
#             if "*INV" in element:
#
#                 #remove any word after the period.
#                 cleaned_string = element.split('.', 1)[0]
#                 #replace par with empty string, deleting the part of the string that starts with PAR
#                 cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*INV',''))
#                 #substitute numerical digits, deleting underscores
#                 cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
#                 if previous=="*PAR":
#                     for k in range(len(que_list)):
#                         if que_list[k] in cleaned_string:
#                             count+=1
#                             break
#                 tokenized_list = word_tokenize(cleaned_string)
#                 free_tokenized_list = []
#                 for element in tokenized_list:
#                     if element is not '':
#                         free_tokenized_list.append(element)
#                 output_list.append(free_tokenized_list)
#             else:
#                 previous="*PAR"
#     return output_list, count,length
def file_tokenization_both_utterances(input_file,count,filename):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list_inv = []
    output_list_par = []
    output_types=[]
    previous=None
   # print(count)
    length=0
    
    # que_list = ["else",
    #             "anything", "anymore action", "can you","tell me", "mistakes", "how about", "what's going on over here",
    #             "going on",
    #
    #             "is that all", "is that", "what's happening", " happening", "action", "some more", "more","?"]
    # neutral_list=["uhhuh","mhm","okay","alright"]
    utterance=0
    print("filename")
    print(filename)
    for line in input_file:

        for element in line.split("\n"):
            type=None
            tag=None
            if "*INV" in element or "*PAR" in element:
                # with open("./data/" + str(count) + '.txt', 'a') as f:
                    # cleaned_string = element.split('.', 1)[0]

                    # substitute numerical digits,
                    #
                    #
                    # deleting underscores

                    if len(element.split(':')[1])==1:


                        continue
                    else:
                        cleaned_string = re.sub(r'[\d]+', '', element.replace('_', ''))
                        utterances.append(cleaned_string)
                        fname.append(filename)
                        utterances_count.append(utterance)
                        utterance+=1
                    # f.write(cleaned_string + os.linesep)


               # if "*PAR" in element or ("*INV" in element and GET_INV):
                # cleaned_string = element.split('.', 1)[0]
                    # #replace par with empty string, deleting the part of the string that starts with PAR
                # if "*INV" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*INV',''))
                    # tag="*INV"
                # elif "*PAR" in cleaned_string:
                    # #cleaned_string = re.sub(r'[^\w?]+', ' ', cleaned_string.replace('*PAR',''))
                    # tag="*PAR"
                # #substitute numerical digits, deleting underscores
                # cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                # #tokenized_list = word_tokenize(cleaned_string)
                # #free_tokenized_list = []
                # # for element in tokenized_list:
                    # # if element is not '':
                        # # free_tokenized_list.append(element)
                # if tag=="*INV":
                    
                    # #remove any word after the period.
                    
                    # if (previous=="*INV"):
                        # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*PAR: "+ os.linesep)
                        

                    # #output_list_inv.append(free_tokenized_list)
                    # # for k in range(len(neutral_list)):
                        # # if que_list[k] in cleaned_string:
                       
                            # # type="question"
                            # # break
                        
                    # # if type==None:
                        # # if "laughs" in cleaned_string:
                            # # type="imitate"
                        # # else:
                            # # for k in range(len(neutral_list)):
                                # # if neutral_list[k] in cleaned_string:
                                
                                    # # type="neutral"
                                    # # break
                        
                    
                    # previous="*INV"
                    
                    # #output_types.append(type)
                # elif tag =="*PAR" :
                    # #output_list_par.append(free_tokenized_list)
                    # if previous is "*PAR":
                       # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                            # f.write("*INV: "+ os.linesep)
                    # with open(dir_result +str(count) +"_"+label+'.txt', 'a') as f:
                        # f.write(cleaned_string+ os.linesep)
                    # previous="*PAR"
                    # #output_types.append(type)
                
            
    #return output_list_inv, output_list_par,output_types
    #return length
def transcript_to_file():
    file_list=[16,17,20,21,22,26]
    PATH = "./data/script"
    count=51
    for path, dirs, files in os.walk(PATH):
        for filename in files:

            fullpath = os.path.join(path, filename)
            filename = int(filename.split('.')[0])
            if filename in file_list or filename>=37 and filename<=102:
                with open(fullpath, 'r', encoding="utf8",errors='ignore')as input_file:
                    file_tokenization_both_utterances(input_file,count,filename)
                    # count+=1
            else:
                continue
                # dementia_list.append(
                # {'text':tokenized_list,
                # 'label':label}
                # )
                # generate_feature(input_file)
                #
                # labels.append(label)
                # fname.append(filename)
                # break
                # dementia_list.append(
                # {'short_pause_count':f_1,
                # 'long_pause_count':f_2,
                # 'very_long_pause_count':f_3,
                # 'word_repetition_count':f_4,
                # 'retracing_count':f_5,
                # 'filled_pause_count':f_6,
                # 'incomplete_utterance_count':f_7,
                # 'label':label
                # }
                # )


def voting():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df = pd.read_excel("data/adress_acoustic_GP.xlsx", sheet_name='Sheet1')
    dict = {"file": [], "test": [], "result": []}
    df['mmse'] = [None] * len(df)
    count = 0
    for index, row in cc_meta.iterrows():
        j = count
        rows = 0
        vals = 0
        dict["file"].append(row['file'])
        dict["test"].append(row['mmse'])
        for i in range(j, len(df)):
            name = df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            # print(name)
            # print(row['file'].strip())
            if (name.strip() == row['file'].strip()):
                vals += df.iloc[i]["result"]
                count += 1
                rows += 1

            else:
                break
        dict["result"].append(vals / rows)

    df = pd.DataFrame(dict)
    writer = pd.ExcelWriter("data/adress_acoustic_GP_final.xlsx", engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Sheet1', columns=list(df.columns))
    writer.save()
    # with open('data/adress_acoustic_test.pickle', 'wb') as f:
    #     pickle.dump(df, f)
def match_mmse():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df=pd.read_excel("data/ADReSS/acoustic_feature_test.xlsx",sheet_name='Sheet2')

    df['mmse'] = [None] * len(df)
    count=0
    for index, row in cc_meta.iterrows():
        j=count
        for i in range(j,len(df)):
            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            print(name)
            print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse']=row["mmse"]
                print(df.at[i, 'mmse'])
                count+=1

            else:
                break
    writer = pd.ExcelWriter("data/adress_acoustic_test.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    with open('data/adress_acoustic_test.pickle', 'wb') as f:
         pickle.dump(df, f)
def convert_segment():
    # df = pd.read_pickle('data/adress_acoustic_train.pickle')
    df = pd.read_excel('data/adress_acoustic_train.xlsx', sheet_name='Sheet2')
    cc_meta = pd.read_csv("data/ADReSS/meta_all.txt", sep=';')
    # # cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    # df=pd.read_excel("data/adress_acoustic_train.xlsx",sheet_name='Sheet1')
    # length=[]
    #
    df['mmse'] = [None] * len(df)
    rows=[]
    count=0
    for index, row in cc_meta.iterrows():
        # rows.append(row['file'].strip())
        j=count
        # list=[]
        size=0
        for i in range(j,len(df)):

            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            # print(name)
            # print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse'] = row["mmse"]
                rows.append(name.strip())
                size+=1
                # print(df.iloc[i, 2:].values)
                # print(len(df.iloc[i, 2:].values))
                # list.extend(df.iloc[i,2:].values)

                # print(len(list))
                count+=1


            else:
                break

    print(len(rows))
    print(len(df))

    df["id"]   =rows
    writer = pd.ExcelWriter("data/adress_acoustic_train.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    # df=pd.DataFrame(rows)
    # with open('data/adress_acoustic_train_merge.pickle', 'wb') as f:
    #      pickle.dump(df, f)
    # df=pd.read_pickle('data/adress_acoustic_train_merge.pickle')
    # df["file"]=rows

    # print(len(df.iloc[0]))
    # print(len(df.iloc[1]))
    # print(len(df))
    # df.iloc[:,:].fillna(0, inplace = True)
    # with open('data/adress_acoustic_train_merge.pickle', 'wb') as f:
    #      pickle.dump(df, f)
    # df = pd.read_pickle('data/adress_acoustic_train_merge.pickle')
    # print(df.head(1))

def match_mmse():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df=pd.read_excel("data/ADReSS/acoustic_feature_test.xlsx",sheet_name='Sheet2')

    df['mmse'] = [None] * len(df)
    count=0
    for index, row in cc_meta.iterrows():
        j=count
        for i in range(j,len(df)):
            name=df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            print(name)
            print(row['file'].strip())
            if(name.strip()==row['file'].strip()):
                df.at[i, 'mmse']=row["mmse"]
                print(df.at[i, 'mmse'])
                count+=1

            else:
                break
    writer = pd.ExcelWriter("data/adress_acoustic_test.xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1',columns=list(df.columns))
    writer.save()
    with open('data/adress_acoustic_test.pickle', 'wb') as f:
         pickle.dump(df, f)
def generate_test_df():
    cc_meta = pd.read_csv("data/ADReSS/test/meta_data.txt", sep=';')
    # cc_meta['label'] = [0] * len(cc_meta)


    cc_meta['utterances'] = [None] * len(cc_meta)
    # cc_meta['gender'] = np.where(cc_meta.gender == 'male', 0, 1)
    print(cc_meta.head(5))
    print(list(cc_meta))
    print()
    for index, row in cc_meta.iterrows():
        str_id = row['file'].replace(' ', '')

        file_path = "data/ADReSS/test/transcription/" + str_id + '.cha'

        # if (cc_meta.at[index, 'utterances'] == None):
        with open(file_path, 'r', encoding="utf8")as input_file:
        #         # print(str_id)
        #         # output = file_pos_extraction(input_file)
        #         # cc_meta.at[index, 'utterances'] = ''.join(output)
        #     generate_feature(input_file)

            output=file_tokenization(input_file)
            utterances.append(output)
            fname.append(str_id)
            # break




    # return cc_meta

def generate_full_interview_dataframe():
    """
    generates the pandas dataframe containing for each interview its label.
    :return: pandas dataframe.
    """

    dementia_list = []
    for label in ["Control", "Dementia"]:

        PATH = "data/ADReSS/train/transcription/" + label

        for path, dirs, files in os.walk(PATH):
            for filename in files:
                fullpath = os.path.join(path, filename)
                with open(fullpath, 'r',encoding="utf8")as input_file:
                    # tokenized_list = file_tokenization(input_file,label)
                    # dementia_list.append(
                        # {'text':tokenized_list,
                         # 'label':label}
                        # )
                    # generate_text(input_file)
                    name=filename.split('.')[0]
                    fname.append(name)
                    output = file_pos_extraction(input_file,name)
                    labels.append(label)
                    print(output)
                    utterances.append(output)
                    break
                break
                    # print(utterances[0:3])

                    #break
    #     dementia_list.append(
    #         {'short_pause_count':ft_1,
    #         'long_pause_count':ft_2,
    #         'very_long_pause_count':ft_3,
    #         'word_repetition_count':ft_4,
    #         'retracing_count':ft_5,
    #         'filled_pause_count':ft_6,
    #         'incomplete_utterance_count':ft_7,
    #         'label':labels,
    #          'file':fname
    #         }
    #         )
    #
    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
def generate_single_utterances_dataframe():
    
    
    count=0
    
    PATH = "D:/Admission/UIC/conference/BHI/annotation/source" 
    for path, dirs, files in os.walk(PATH):
        for filename in files:
            fullpath = os.path.join(path, filename)
            with open(fullpath, 'r')as input_file:
                
                file_tokenization_both_utterances(input_file,filename)
                
               
                
                
                        

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
# def generate_single_utterances_dataframe():
    
    # length=[]
    
    # ids=[]
    
    # count=0
    # for label in ["Control", "Dementia"]:
        # folders =  ["cookie"]
        # id = 0

        # for folder in folders:
            # PATH = "./Dataset/DementiaBank/Pitt/Pitt/" + label + "/" + folder
            # for path, dirs, files in os.walk(PATH):
                # for filename in files:
                    # fullpath = os.path.join(path, filename)
                    # with open(fullpath, 'r',encoding="utf8")as input_file:
                        # file_tokenization_both_utterances(input_file)
                        
                        # # dementia_list.append(
                        # # {
                        # # 'len':length,
                        # # 'label':label,
                        # # 'id':id
                        # # }
                        # #)
                        
                        # count+=1
                        # if count>25:
                            # break
                        # id = id +1
                        # ids.append(id)
                        # type.append(label)
                        
                # if count>25:
                    # count=1
                    # break    
                        # for element1,element2,element3 in zip(inv,par,type):
                            # dementia_list.append(
                            # {'par': element2,
                            # 'label': label,
                            # 'id':id,
                            # 'inv':element1,
                            # 'type':element3}
                                # )
                        

    # dementia_dataframe = pd.DataFrame(dementia_list)
    # return dementia_dataframe
import re
from nltk.tokenize import word_tokenize

def convert_list_to_string(list_strings):
    return_string = ""
    print("str list")
    print(list_strings)
    for string in list_strings:
        print("str before")
        print(string)
        string = string.replace("*PAR:","")
        string = string.replace("%mor:","")
        string = string.replace("%gra:","")
        string = string.replace("\t","").replace("\n"," ")
        string = string.replace("~"," ")

        string = re.sub(r'[\d]+', '', string)
        string = string.replace('[_]', "")
        string = string.replace('@', "")
        string = string.replace('|', "")
        string = string.replace('_', "")
        string = string.replace('nuk', "")
        string = string.replace('x@n', "")
        string = string.replace('[', "")
        string = string.replace(']', "")
        string = string.replace('(', "")
        string = string.replace(')', "")
        string = string.replace('<', "")
        string = string.replace('>', "")
        string = string.replace(')', "")
        string = string.replace('/', "")
        string = string.replace(':', "")
        string = string.replace('*', "")
        string = string.replace('+...', ".")
        string = string.replace('+', "")

        string = string.replace('xxx', "")
        string = string.replace('exc', "")
        str = string.split('.')
        if len(str) > 1:
            string=str[0]+" . "
        str1=string.split('?')
        if len(str1) > 1:
            string = str1[0] + " ? "
        print("str")
        print(string)
        print("return_str")
        print(return_string)
        return_string += string
    print(return_string)
    return return_string

def convert_mor_to_list(list_string):
    return_list = []
    element_list = list_string.split(" ")
    for element in element_list:
        if "|" in element:
            return_list.append(element.split("|")[0])
    return return_list

def convert_gra_to_list(list_string):
    return_list = []
    element_list = list_string.split(" ")
    for element in element_list:
        if "|" in element:
            return_list.append(element.split("|")[2])
    return return_list

def file_pos_extraction(input_file):
  # with open(fullpath, 'r') as input_file:
  #   id_string = input_file.name.split('/')[-1]
  #   #print(id_string)
  #   result = re.search('(.*).cha',id_string)
  #   id = result.group(1)

    par_output = []
    mor_output = []
    gra_output = []
    str=""
    i = 0
    list_files = list(input_file)
    # print(fname)


    # if (fname in ['S083', 'S114', 'S135', 'S082']):
    #     print(list_files[10])
    while i < len(list_files):
        par_list = []
        mor_list = []
        gra_list = []
        # if (fname in ['S083', 'S114', 'S135', 'S082']):
        #     print(list_files[i])
            # print(len(list_files))
        # if i == 5:
        #     if "|Control|" in list_files[i] or "|ProbableAD|" in list_files[i] or "|PossibleAD|" in list_files[i]:
        #         #age =
        #         pass
        #     else:
        #         return [],[],[], False
        if "*PAR:" in list_files[i]:

            #print(list_files[i])
            while "%mor" not in list_files[i] and "*INV:" not in list_files[i] :
                print("enter")
                # print(list_files[i])
                par_list.append(list_files[i])
                i += 1
                if "@End" in list_files[i]:
                    break
            if "@End" in list_files[i]:

                break

            if "%mor" in list_files[i]:

                #print(list_files[i])
                while "%gra" not in list_files[i]:

                    mor_list.append(list_files[i])
                    i += 1

                if "%gra" in list_files[i]:

                    #print(list_files[i])
                    while "*PAR" not in list_files[i] and "*INV" not in list_files[i] and "@End" not in list_files[i]:
                        gra_list.append(list_files[i])
                        i += 1
        else:
            i += 1
        # w = word_tokenize(convert_list_to_string(par_list))

        w = convert_list_to_string(par_list)
        if '\x15_\x15' in w:
            # w.remove('\x15_\x15')
            w_new=w.replace('\x15_\x15','')
        else:
            w_new = w

        m = convert_mor_to_list(convert_list_to_string(mor_list))
        g = convert_gra_to_list(convert_list_to_string(gra_list))
        # print(w_new)
        # print(len(w_new))

        if len(w_new) > 0:
            par_output += w_new
            str=str+w_new
        if len(m) > 0:
            mor_output.append(m)
        if len(g) > 0:
            gra_output.append(g)
    #return par_output,mor_output,gra_output,id



    return str
def read_annotation():
    dic_dist = [
        'Answer:t3',
        'Answer:t2',
        'Answer:t1',
        'Answer:t4',
        'Answer:t5',
        'Answer:t6',
        'Answer:t7',
        'Answer:t8',
        'Question:General',
        'Question:Reflexive',
        "Answer:Yes",
        "Answer:No",
        "Answer:General",
        "Instruction",
        "Suggestion",
        "Request",
        "Offer",
        "Acknowledgment",
        "Request:Clarification",
        "Feedback:Reflexive",
        "Stalling",
        "Correction",
        "Farewell",
        "Apology",

        "Other"

    ]
    dic_dist = {
        'Answer:t3': [],
        'Answer:t2': [],
        'Answer:t1': [],
        'Answer:t4': [],
        'Answer:t5': [],
        'Answer:t6': [],
        'Answer:t7': [],
        'Answer:t8': [],
        'Question:General': [],
        'Question:Reflexive': [],
        "Answer:Yes": [],
        "Answer:No": [],
        "Answer:General": [],
        "Instruction": [],
        "Suggestion": [],
        "Request": [],
        "Offer": [],
        "Acknowledgment": [],
        "Request:Clarification": [],
        "Feedback:Reflexive": [],
        "Stalling": [],
        "Correction": [],
        "Farewell": [],
        "Apology": [],

        "Other": [],
        "file": []

    }
    df = pd.read_excel('data/DA_annotattion_data.xlsx')
    for index, row in df.iterrows():
        for key in dic_dist:
            dic_dist[key].append(0)




# match_mmse()
# convert_segment()

def plot_hist():
    import matplotlib.pyplot as plt
    df = pd.read_excel('data/ADReSS/meta_data_loso.xlsx')
    # df1 = pd.read_excel('data/ADReSS/rmse_plot.xlsx')
    fig,ax = plt.subplots()


    numBins =12
    (n1,bin1,patches1)=ax.hist(df['control'], bins=np.arange(0, 30 + 2, 2),  alpha=0.5,label='control',edgecolor='black', linewidth=1.2,align='left')
    (n2,bin2,patches2)=ax.hist(df['dementia'], bins=np.arange(0, 30 + 2, 2),  alpha=0.5,label='dementia',edgecolor='black', linewidth=1.2,align='left')
    # plt.xlim([10, 30])
    ax.set_xlabel("MMSE", fontsize=14)

    # set y-axis label
    ax.set_ylabel("count", color="blue", fontsize=14)
    plt.xticks(bin2[:-1])
    plt.legend(loc='center left')
    # twin object for two different y-axis on the sample plot
    # ax2 = ax.twinx()
    # # make a plot with different y-axis using second axis object
    # df1['value'].plot(kind='line', marker='d', ax=ax2)
    # ax2.set_ylabel("rmse", color="blue", fontsize=14)
    plt.show()
    print(n2)
    print(bin2)
    print(n1)
    print(bin1)
    dict={"dem":n2,"con":n1}
    df1=pd.DataFrame(dict)
    return df1
def plot_test():
    import matplotlib.pyplot as plt

    df1 = pd.read_excel('data/ADReSS/rmse_plot_test.xlsx')
    df2 = pd.read_excel('data/ADReSS/rmse_plot_loso.xlsx')
    fig, bar_ax = plt.subplots()
    bar1=bar_ax.bar(df1['bin'], df1['dementia'], color='blue',alpha=0.5)  # plot first y series (line)
    bar2=bar_ax.bar(df1['bin'], df1['control'], color='green', alpha=0.5)  # plot first y series (line)
    bar_ax.set_xlabel('MMSE score')  # label for x axis
    bar_ax.set_ylabel('Count',color='blue')  # label for left y axis
    bar_ax.tick_params('y', colors='blue')  # add color to left y axis
    bar_ax.set_xticklabels(df1.bin, rotation=40)
    # plt.legend(['Dementia', 'Control'], loc='center left')
    line_ax = bar_ax.twinx()
    line,=line_ax.plot(df1['bin'], df1['value'],linestyle='-', marker='o', color='red')  # plot second y series (bar)
    line_ax.set_ylabel('RMSE',color='red')  # label for right y axis
    line_ax.tick_params('y', colors='red')  # add color to right y axis
    plt.legend([bar1,bar2,line],['Dementia', 'Control','Overall RMSE'],loc=9)

    def autolabel(rects,rects2):
        """
        Attach a text label above each bar displaying its height
        """
        i=5
        # for rect in rects:
        for j in range(0,len(rects)-1):
            height = df2['total'][i]
            i+=1
            bar_ax.text(rects[j].get_x() + rects[j].get_width() / 2., 1.05 * rects[j].get_height(),\
            str(round(height,1))+'%',
                    ha='center', va='bottom',fontsize=10)
        print(len(rects2))
        for j in range(9, 10):
            height = df2['total'][i]
            i += 1
            if j==9:
                bar_ax.text(rects2[j].get_x() + rects2[j].get_width() / 2., 1.001 * rects2[j].get_height(), \
                            str(round(height, 1)) + '%',
                            ha='center', va='bottom', fontsize=8)
            else:
                bar_ax.text(rects2[j].get_x() + rects2[j].get_width() / 2., 1.05 * rects2[j].get_height(), \
                            str(round(height, 1)) + '%',
                            ha='center', va='bottom', fontsize=8)



    autolabel(bar1,bar2)
    plt.show()
    fig.savefig('test_plot_num.png')

# act=[[] for i in range(15)]
# pred=[[] for i in range(15)]
# num=[0]*10
# avg=[]*15
# # id1=[]
# # id2=[]
# df = pd.read_excel('data/adress_all_poly_loso.xlsx')
# # print(df.columns)
# for index, row in df.iterrows():
#     print(row['mmse'])
#     print(type(row['mmse']))
#     # test_rmse = np.sqrt(mean_squared_error([row['mmse']], [row['result']]))
#     if row['mmse']>=0 and row['mmse']<2:
#         print("1")
#         act[0].append(row['mmse'])
#         pred[0].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=2 and row['mmse']<4:
#         print("1")
#         act[1].append(row['mmse'])
#         pred[1].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=4 and row['mmse']<6:
#         print("1")
#         act[2].append(row['mmse'])
#         pred[2].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=6 and row['mmse']<8:
#         print("1")
#         act[3].append(row['mmse'])
#         pred[3].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=8 and row['mmse']<10:
#         print("1")
#         act[4].append(row['mmse'])
#         pred[4].append(row['result'])
#         num[0]=num[0]+1
#     elif row['mmse']>=10 and row['mmse']<12:
#         print("1")
#         act[5].append(row['mmse'])
#         pred[5].append(row['result'])
#         num[0]=num[0]+1
#
#     elif row['mmse']>=12 and row['mmse']<14:
#         print("2")
#         act[6].append(row['mmse'])
#         pred[6].append(row['result'])
#         num[1] = num[1] + 1
#     elif row['mmse'] >= 14 and row['mmse'] < 16:
#         print("3")
#         act[7].append(row['mmse'])
#         pred[7].append(row['result'])
#         num[2] = num[2] + 1
#     elif row['mmse'] >= 16 and row['mmse'] < 18:
#         print("4")
#         act[8].append(row['mmse'])
#         pred[8].append(row['result'])
#         num[3] = num[3] + 1
#     elif row['mmse'] >= 18 and row['mmse'] < 20:
#         print("5")
#         act[9].append(row['mmse'])
#         pred[9].append(row['result'])
#         num[4] = num[4] + 1
#     elif row['mmse'] >= 20 and row['mmse'] < 22:
#         print("6")
#         act[10].append(row['mmse'])
#         pred[10].append(row['result'])
#         num[5] = num[5] + 1
#     elif row['mmse'] >= 22 and row['mmse'] < 24:
#         print("7")
#         act[11].append(row['mmse'])
#         pred[11].append(row['result'])
#         num[6] = num[6] + 1
#     elif row['mmse'] >= 24 and row['mmse'] < 26:
#         print("8")
#         act[12].append(row['mmse'])
#         pred[12].append(row['result'])
#         num[7] = num[7] + 1
#     elif row['mmse'] >= 26 and row['mmse'] < 28:
#         print("9")
#         act[13].append(row['mmse'])
#         pred[13].append(row['result'])
#         num[8] = num[8] + 1
#     elif row['mmse'] >= 28 and row['mmse'] <= 30:
#         print("10")
#         act[14].append(row['mmse'])
#         pred[14].append(row['result'])
#         num[9] = num[9] + 1
#
# for i in range(0,15):
#     if len(act[i])>0:
#         print(act[i])
#         print(pred[i])
#         test_rmse = np.sqrt(mean_squared_error(act[i], pred[i]))
#         print(test_rmse)
#         avg.append(test_rmse)
#     elif len(act[i])==0:
#         avg.append(0)
# dic={"value":avg}
# df1=pd.DataFrame(dic)

# dict={"id1":id1,"control":con,"id2":id2,"dementia":dem}
# df1=pd.DataFrame(dict)


#
# # generate_test_df()
# df1 = pd.read_pickle('data/adress_test.pickle')
# # print(df1)
# print(df["gender"])
# print(df1["gender"])
# print(list(df.columns))
# print(list(df1.columns))
# transcript_to_file()
# dementia_dataframe=generate_full_interview_dataframe()
# dementia_dataframe=generate_test_df()
# print(list(dementia_dataframe.columns))
# print(dementia_dataframe.head(5))
# generate_full_interview_dataframe()
# voting()
# from cmath import sqrt
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import scipy.stats
# df1 = pd.read_excel('data/adress_acoustic_GP_final.xlsx',sheet_name='Sheet1')
# y_test=df1.loc[:,"test"]
# y_pred=df1.loc[:,"result"]
# test_rmse = sqrt(mean_squared_error(y_test , y_pred ))
# test_MAE = mean_absolute_error(y_test , y_pred )
# pearson = scipy.stats.pearsonr(y_test , y_pred )
# r2 = r2_score(y_test , y_pred )
# print("rmse " )
# print(test_rmse)
# print("mae  " )
# print(test_MAE)
# print("pearson")
# print(pearson)
# print("r2 %f"%(r2))


# df['word']=df1['word']
# columns={'short_pause_count':ft_1,'long_pause_count':ft_2,'very_long_pause_count':ft_3,
#                             'word_repetition_count':ft_4,
#                             'retracing_count':ft_5,
#                             'filled_pause_count':ft_6,
#                             'incomplete_utterance_count':ft_7,
#                             'label':labels,
#                             'filename':fname,
#                             'utterance_count':utterances_count
#                             }

# df['short_pause_count']=[int(b) / int(m) for b,m in zip(ft_1, word_count)]
# df['long_pause_count']=[int(b) / int(m) for b,m in zip(ft_2, word_count)]
# df['very_long_pause_count']=[int(b) / int(m) for b,m in zip(ft_3, word_count)]
# df['word_repetition_count']=[int(b) / int(m) for b,m in zip(ft_4, word_count)]
# df['retracing_count']=[int(b) / int(m) for b,m in zip(ft_5, word_count)]
# df['filled_pause_count']=[int(b) / int(m) for b,m in zip(ft_6, word_count)]
# df['incomplete_utterance_count']=[int(b) / int(m) for b,m in zip(ft_7, word_count)]
# df['utterance_count']=utterances_count
# df['word']=word_count
# print(df['word_repetition_count'])
# print(df['filled_pause_count'])
# print(len(fname))
# print(len(labels))
# print(len(utterances))
# df1=plot_hist()
# # columns={'file':fname,'label':labels,'utterances':utterances}
# # dementia_dataframe=pd.DataFrame(columns)
# # # # generate_single_utterances_dataframe()
# # #
# # # # dementia_dataframe = pd.DataFrame(dementia_list)
# # # # # print(dataframe)
# with open('data/adress_final_interview.pickle', 'wb') as f:
#      pickle.dump(df, f)
#
# df=pd.read_pickle('data/adress_final_interview.pickle')
# print(list(df.columns))
# print(df.shape)
# print(list(df.columns))
# print(df.head(3))
# writer = pd.ExcelWriter('data/ADReSS/rmse.xlsx', engine='xlsxwriter')
# # # #
# # # # # dementia_dataframe.to_excel(writer, sheet_name='Sheet1',columns=['short_pause_count','long_pause_count',
# # # # # 'very_long_pause_count','word_repetition_count','retracing_count','filled_pause_count','incomplete_utterance_count',
# # # # # 'label','filename','utterance_count'])
# df1.to_excel(writer, sheet_name='Sheet1',columns=['value'])
# writer.save()

plot_test()