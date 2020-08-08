from __future__ import with_statement

import os
import pickle
import re

import pandas as pd
# nltk.download('punkt')
from nltk.tokenize import word_tokenize

# from imblearn.over_sampling import SMOTE
word=[]
GET_INV= False
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Dense, LSTM,Flatten, Dropout, Conv1D, MaxPooling1D, \
#      GlobalMaxPooling1D, concatenate,Bidirectional, Layer,TimeDistributed
# from keras.layers.embeddings import Embedding
# from keras import Input, Model, initializers
# from keras.optimizers import Adagrad
def file_tokenization(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []
    id_string = input_file.name.split('/')[-1]
    print(id_string)
    result = re.search('(.*).cha',id_string)
    id = result.group(1)
    count=0
    for line in input_file:
        for element in line.split("\n"):

            if "*PAR" in element:

                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                tokenized_list = word_tokenize(cleaned_string)
                output_list = output_list+['\n'] + tokenized_list
                count+=1
    return output_list,id,count

def file_tokenization_utterances(input_file):
    '''
    :param input_file: single dataset file as readed by Python
    :return: tokenized string of a single patient interview
    '''
    output_list = []

    # Id selection from file name, for each patient the ID is composed by the number of the patient followed by - and
    # the number of the interview he is substaining.

    id_string = input_file.name.split('/')[-1]
    print(id_string)
    result = re.search('(.*).cha',id_string)
    id = result.group(1)
    for line in input_file:
        for element in line.split("\n"):
            if "@ID:	eng|Pitt_transcripts|PAR|" in element:
                splitted_line = element.split('|')
                age = splitted_line[3]
                if age is not '':
                    result = re.search('(.*);', age)
                    age = int(result.group(1))
                else:
                    age = -1
                if splitted_line[4] == 'female':
                    sex = 0
                elif splitted_line[4] == 'male':
                    sex = 1
                else:
                    sex = -1

            if "*PAR" in element or ("*INV" in element and GET_INV):
                #remove any word after the period.
                cleaned_string = element.split('.', 1)[0]
                #replace par with empty string, deleting the part of the string that starts with PAR
                cleaned_string = re.sub(r'[^\w]+', ' ', cleaned_string.replace('*PAR',''))
                #substitute numerical digits, deleting underscores
                cleaned_string = re.sub(r'[\d]+','',cleaned_string.replace('_',''))
                tokenized_list = word_tokenize(cleaned_string)
                free_tokenized_list = []
                for element in tokenized_list:
                    if element is not '':
                        free_tokenized_list.append(element)
                output_list.append(free_tokenized_list)
    return id, age, sex, output_list

def extact_age_and_sex(file):
    print(file)

    '''
    if sex == 'female':
        sex = 0, 
    elif sex == 'male':
        sex = 1
    '''

def generate_full_interview_dataframe():
    """
    generates the pandas dataframe containing for each interview its label.
    :return: pandas dataframe.
    """
    dementia_list = []
    for label in ["Control", "Dementia"]:
        if label == "Dementia":
            folders = ["cookie", "fluency", "recall", "sentence"]
        else:
            folders = ["cookie"]

        for folder in folders:
            PATH = "data/Pitt_transcripts/" + label + "/" + folder
            for path, dirs, files in os.walk(PATH):
                for filename in files:
                    fullpath = os.path.join(path, filename)
                    with open(fullpath, 'r')as input_file:
                        tokenized_list,id = file_tokenization(input_file)
                        dementia_list.append(
                            {'text':tokenized_list,
                             'label':label,
                             'id':id
                             }
                            )
    dementia_dataframe = pd.DataFrame(dementia_list)
    return dementia_dataframe
def process_mmse_full_interview():
    dictionary={'id':[],'mmse':[],'date':[]}
    df = pd.read_excel('data/mmse_score.xlsx', sheet_name='Sheet1')
    for i in df.index:

        id="%03d" % df['id'][i]
        if df['mmse1'][i]>=0 and df['mmse1'][i]<=30:
            dictionary['mmse'].append(df['mmse1'][i])
            dictionary['id'].append(id+'-1')
            if df['visit1'][i] !=None:
                dictionary['date'].append(df['visit1'][i])
            else:
                dictionary['date'].append('')


        if df['mmse2'][i]>=0 and df['mmse2'][i]<=30:
            dictionary['mmse'].append(df['mmse2'][i])

            dictionary['id'].append(id + '-2')
            if df['visit2'][i] != None:
                dictionary['date'].append(df['visit2'][i])
            else:
                dictionary['date'].append('')
        if df['mmse3'][i]>=0 and df['mmse3'][i]<=30:
            dictionary['mmse'].append(df['mmse3'][i])
            dictionary['id'].append(id+'-3')
            if df['visit3'][i] != None:
                dictionary['date'].append(df['visit3'][i])
            else:
                dictionary['date'].append('')
        if df['mmse4'][i]>=0 and df['mmse4'][i]<=30:
            dictionary['mmse'].append(df['mmse4'][i])
            dictionary['id'].append(id+'-4')
            if df['visit4'][i] != None:
                dictionary['date'].append(df['visit4'][i])
            else:
                dictionary['date'].append('')

        if df['mmse5'][i]>=0 and df['mmse5'][i]<=30:
            dictionary['mmse'].append(df['mmse5'][i])
            dictionary['id'].append(id+'-5')
            if df['visit5'][i] != None:
                dictionary['date'].append(df['visit5'][i])
            else:
                dictionary['date'].append('')

        if df['mmse6'][i]>=0 and df['mmse6'][i]<=30:
            dictionary['mmse'].append(df['mmse6'][i])
            dictionary['id'].append(id+'-6')
            if df['visit6'][i] != None:
                dictionary['date'].append(df['visit6'][i])
            else:
                dictionary['date'].append('')

        if df['mmse7'][i]>=0 and df['mmse7'][i]<=30:
            dictionary['mmse'].append(df['mmse7'][i])
            dictionary['id'].append(id+'-7')
            if df['visit7'][i] != None:
                dictionary['date'].append(df['visit7'][i])
            else:
                dictionary['date'].append('')


    dataframe_mmse=pd.DataFrame(dictionary)
    return dataframe_mmse
def generate_paired_mmse():
    dictionary = {'id': [], 'T=1': [], 'T=2': [],'T=3': [], 'T=4': [],'T=5': [],'AD': [],'HC': []}
    df = pd.read_excel('data/paired_mmse.xlsx', sheet_name='Sheet1')
    for i in df.index:
        sum = 0

        print("old sum %d" % sum)

        if pd.isnull(df.iloc[i]['file1'])==False and pd.isnull(df.iloc[i]['mmse1'])==False:

            sum+=1
        if pd.isnull(df.iloc[i]['file2'])==False and pd.isnull(df.iloc[i]['mmse2'])==False:
            sum+=1
        if pd.isnull(df.iloc[i]['file3'])==False and pd.isnull(df.iloc[i]['mmse3'])==False:
            sum+=1
        if pd.isnull(df.iloc[i]['file4'])==False and pd.isnull(df.iloc[i]['mmse4'])==False:
            sum+=1
        if pd.isnull(df.iloc[i]['file5'])==False and pd.isnull(df.iloc[i]['mmse5'])==False:
            sum+=1
        print("sum %d"%sum)
        if sum==0:
            print("sum %d" % sum)
            continue

        elif sum==1:
            print("sum %d" % sum)
            dictionary['id'].append(df['id'][i])
            dictionary["T=1"].append(1)
            dictionary["T=2"].append(0)
            dictionary["T=3"].append(0)
            dictionary["T=4"].append(0)
            dictionary["T=5"].append(0)

            dictionary["AD"].append(df.iloc[i]['AD'])
            dictionary["HC"].append(df.iloc[i]['HC'])
        elif sum==2:
            print("sum %d" % sum)
            dictionary['id'].append(df['id'][i])
            dictionary["T=1"].append(0)
            dictionary["T=2"].append(1)
            dictionary["T=3"].append(0)
            dictionary["T=4"].append(0)
            dictionary["T=5"].append(0)
            dictionary["AD"].append(df.iloc[i]['AD'])
            dictionary["HC"].append(df.iloc[i]['HC'])
        elif sum == 3:
            print("sum %d" % sum)
            dictionary['id'].append(df['id'][i])
            dictionary["T=1"].append(0)
            dictionary["T=2"].append(0)
            dictionary["T=3"].append(1)
            dictionary["T=4"].append(0)
            dictionary["T=5"].append(0)
            dictionary["AD"].append(df.iloc[i]['AD'])
            dictionary["HC"].append(df.iloc[i]['HC'])
        elif sum == 4:
            print("sum %d" % sum)
            dictionary['id'].append(df['id'][i])
            dictionary["T=1"].append(0)
            dictionary["T=2"].append(0)
            dictionary["T=3"].append(0)
            dictionary["T=4"].append(1)
            dictionary["T=5"].append(0)
            dictionary["AD"].append(df.iloc[i]['AD'])
            dictionary["HC"].append(df.iloc[i]['HC'])
        elif sum==5:
            print("sum %d" % sum)
            dictionary['id'].append(df['id'][i])
            dictionary["T=1"].append(0)
            dictionary["T=2"].append(0)
            dictionary["T=3"].append(0)
            dictionary["T=4"].append(0)
            dictionary["T=5"].append(1)
            dictionary["AD"].append(df.iloc[i]['AD'])
            dictionary["HC"].append(df.iloc[i]['HC'])
    print(len(dictionary["T=1"]))
    print(len(dictionary["T=2"]))
    print(len(dictionary["T=3"]))
    print(len(dictionary["T=4"]))
    print(len(dictionary["T=5"]))
    print(len(dictionary["AD"]))
    print(len(dictionary["HC"]))
    print(len(dictionary["id"]))
    dataframe_mmse = pd.DataFrame(dictionary)
    return dataframe_mmse

# data=df['text']
#






def generate_full_interview_mmse():
    df = pd.read_excel('data/pitt_mmse.xlsx', sheet_name='Sheet1')
    #print(df['id'].head(10))
    dementia_list = []
    for label in ["Control", "Dementia"]:

        PATH = "data/ADReSS/train/transcription/" + label
        for path, dirs, files in os.walk(PATH):

            for filename in files:
                fullpath = os.path.join(path, filename)

                found = df[df['id'].str.contains(filename.split('.')[0])]


                if len(found.index)>0:
                    with open(fullpath, 'r',encoding="utf8")as input_file:
                        tokenized_list, id = file_tokenization(input_file)
                        result=df[df['id'] == filename.split('.')[0]]
                        mmse=result.get_value(result.index[0],'mmse')
                        date=result.get_value(result.index[0],'date')
                        dementia_list.append(
                            {'text': tokenized_list,
                             'label': label,
                             'id': filename.split('.')[0],
                             'mmse':mmse,
                             'date':date

                             }
                        )
                        print(mmse)

    dementia_dataframe = pd.DataFrame(dementia_list)
    return dementia_dataframe


def generate_single_utterances_dataframe():
    dementia_list = []
    id = 0
    for label in ["Control", "Dementia"]:
        # if label == "Dementia":
        #     folders =  ["cookie", "fluency", "recall", "sentence"]
        # else:
        #     folders = ["cookie"]

        # for folder in folders:
            # PATH = "Pitt_transcripts/" + label + "/" + folder
        PATH = "data/ADReSS/train/transcription/"+label
        for path, dirs, files in os.walk(PATH):
            for filename in files:
                fullpath = os.path.join(path, filename)
                with open(fullpath, 'r',encoding="utf8")as input_file:
                    tokenized_list,file,id = file_tokenization(input_file)
                    # print(tokenized_list)
                    # print(tokenized_list[3])
                    # print(len(tokenized_list))
                    # for element in tokenized_list:


                    dementia_list.append(
                        {'text': tokenized_list,
                         'label': label,
                         'id':id,
                         'file':file
                         }
                    )

    dementia_dataframe = pd.DataFrame(dementia_list)
    return dementia_dataframe
def extract_mmse(df):
    for i in df.index:
        st=df.iloc[i]['age']
        st1 = df.iloc[i]['sex']
        if type(st)==int:
            continue

        if(len(st.split(';'))>1):
            df.set_value(i, 'age', st.split(';')[1])
        if (len(st1.split(';')) > 1):
            sex=st1.split(';')[2]
            print(sex.strip())
            print(sex.strip()== 'female')
            if sex.strip()=='male':
                df.set_value(i, 'sex', 0)
            elif sex.strip()=='female' :
                df.set_value(i, 'sex', 1)



    return df




def manual_features_preparation(manual_features):
    len_features = len(manual_features[0])
    feature_name=["f_1","f_2","f_3","f_4","f_5",
             "f_6","f_7","f_8","f_9","f_10",
             "f_11", "f_12", "f_13", "f_14", "f_15",
             "f_16", "f_17"]
    feature={"f_1":[],"f_2":[],"f_3":[],"f_4":[],"f_5":[],
             "f_6":[],"f_7":[],"f_8":[],"f_9":[],"f_10":[],
             "f_11": [], "f_12": [], "f_13": [], "f_14": [], "f_15": [],
             "f_16": [], "f_17": []
             }

    for element in manual_features:
        count=0
        for j in element:
            if type(j)==float or type(j)==int:
                feature[feature_name[count]].append(j)
                count+=1
        #print("count %d"%count)
    return  feature

# df = pd.read_excel('data/adress_data.xlsx', sheet_name='Sheet1')
# dt=df.iloc[0]['text']
# print(dt[0])
# print(type(df.iloc[1]['text']))

# data=df['text']
#

# df=extract_mmse(df)
# # dataframe = generate_paired_mmse()
# df=generate_single_utterances_dataframe()
# data=df['text']
#
# df['word']=[sum(1 for x in w if x != '\n') for w in data]
# print(df['word'])
# # with open('data/adress_data.pickle', 'wb') as f:
# #     pickle.dump(df, f)
anagraphic_data = pd.read_pickle('data/adress_data.pickle')
print(anagraphic_data['word'])

# # df['word']=[sum(1 for x in w if x != '\n') for w in data]
# # # # dict={'text1':[],'text2':[],'mmse1':[],'mmse2':[],'feature1':[],'feature2':[],'id1':[],'id2':[]}
# # # # df = pd.read_pickle('data/pitt_full_interview_features.pickle')
# writer = pd.ExcelWriter("data/adress_data_intermediate.xlsx", engine='xlsxwriter')
# # # # writer = pd.ExcelWriter("karnekal_niu_bansal_2018/results/mmse_smote_result.xlsx", engine='xlsxwriter')
# # # #
# df.to_excel(writer, sheet_name='Sheet1', columns=['text','label','file','id','word'])
# writer.save()
df=pd.read_pickle('data/adress_full_interview_features.pickle')
print(df["features"].values)
# df=pd.read_pickle('data/adress_full_interview_features.pickle')
# print(df["features"].values)
# print(df['mmse'])
# anagraphic_data = pd.read_pickle('data/anagraphic_dataframe.pickle')
# print(anagraphic_data.head(5))
# df=pd.read_pickle('data/pitt_full_interview_features.pickle')
# print(df.columns)
# print(df["features"].values)

# test=[]
# result=[]
# print('result')

# df1 = pd.read_pickle('karnekal_niu_bansal_2018/results/df_result_smote.pickle')
# class_label = []
# discard=[]
# for index, row in df.iterrows():
#     # print("row")
#     # print(row.id)
#     if row['id'].split('-')[0] in discard:
#         continue
#     for i in range(index+1,len(df)):
#         # print("temp")
#
#     # numeric_label.append(row['mmse']/30)
#         temp=df.iloc[i,:]
#         # print(temp.id)
#         if row['id'].split('-')[0]==temp.id.split("-")[0]:
#             dict["text1"].append(row['text'])
#             dict["text2"].append(temp.text)
#             dict["feature1"].append(row['features'])
#             dict["feature2"].append(temp.features)
#             dict["mmse1"].append(row['mmse'])
#             dict["mmse2"].append(temp.mmse)
#             dict["id1"].append(row['id'])
#             dict["id2"].append(temp.id)
#             discard.append(row['id'].split('-')[0])
#             # print("features")
#             # print(type(row['features']))
#             # print(type(dict["feature1"][0]))




# #print(df1.features.values[0])
# print("before")
#
# print(df1.features[0])
# manual_feature=manual_features_preparation(df1.features.values)
# print(len(manual_feature["f_1"]),len(manual_feature["f_2"]),len(manual_feature["f_3"]),
#       len(manual_feature["f_4"]),len(manual_feature["f_5"]),len(manual_feature["f_6"]),len(manual_feature["f_7"]),
#       len(manual_feature["f_8"]),len(manual_feature["f_9"]),len(manual_feature["f_10"]),
#       len(manual_feature["f_11"]), len(manual_feature["f_12"]), len(manual_feature["f_13"]), len(manual_feature["f_14"]),
#       len(manual_feature["f_15"]), len(manual_feature["f_16"]), len(manual_feature["f_17"])
#       )
#print(manual_feature)
# df=pd.read_pickle('data/pitt_full_interview_features.pickle')
# print(df["features"].values)
# df_new_2=pd.DataFrame(dict)
# df1=df_new_2.iloc[0,:]
# print("first")
# print(type(df1.feature1))
# print(df1.feature1[0])
# df_new_array=df_new.values
# df_new_manual_feat=np.array(df_new_array).reshape(len(df_new_array),len(df_new_array[0]),1)
# print("after")
# print(df_new_manual_feat.shape)
# # print(len(df_new[0]))
# sm = SMOTE(random_state=12, ratio=1.0)
# x_train_new, y_train_new = sm.fit_sample(df_new, class_label)
# # # #df2 = pd.read_excel('data/pitt_mmse.xlsx', sheet_name='Sheet1')
# print(len(x_train_new))
#print(len(df1))
#df2 = pd.read_pickle('karnekal_niu_bansal_2018/results/new_ytrain.pickle')

#print(list(df))
#print(len(df.iloc[0]["features"]))
#print(df.iloc[0])
#print(df1.iloc[:,800])
#print(df1.iloc[:,810])

#test.extend(df['test'])
#for item in df['result']:

#     result.extend(item)

#test_mmse=[30*item for item in test]
#result_mmse=[30*item for item in result]

#test_MAE= mean_absolute_error(test, result)
#
#print(test)
#print(result)
#print(test_MAE)
#df=df2[df2[:,0]==0]
#print(df1[df[:,800]])
#data=np.array(df1.iloc[0,:]).reshape((1, 1, 800))
#print(data.shape)
#print(df2.head(-20))
#print(len(df))
# test_MAE= mean_absolute_error(test_mmse, result_mmse)
# print(test_MAE)
# test_MAE_raw=mean_absolute_error(test_mmse, result_mmse, multioutput='raw_values')
# print(test_MAE_raw)
# # print('loss')
# df = pd.read_pickle('karnekal_niu_bansal_2018/results/hist_mmse_1.pickle')
# print(df['loss'])
# print(df['val_loss'])


# df = pd.read_pickle('data/pitt_mmse_features_new.pickle')
# df2=df.iloc[0,:]
# print("second")
# print(type(df2.feature1))
# writer = pd.ExcelWriter("data/pitt_full_mmse_feature_3.xlsx", engine='xlsxwriter')
# # # writer = pd.ExcelWriter("karnekal_niu_bansal_2018/results/mmse_smote_result.xlsx", engine='xlsxwriter')
# # #
# df_new_2.to_excel(writer, sheet_name='Sheet1', columns=['text1','text2','feature1','feature2','mmse1','mmse2','id1','id2'])
# writer.save()
# vocabulary_size=10
# docs = [[['Well', 'done!'],['Good', 'work']],
#
# 		[['Great', 'effort'],
# 		['nice', 'work']]
# 		]
# doc_reshape=docs.resha
# t = Tokenizer(num_words= vocabulary_size)
# # fit the tokenizer on the documents
# t.fit_on_texts(docs)
# train_sequences = t.texts_to_sequences(docs)
#
#
#
# print("count %d"%t.document_count)
# print(train_sequences[0])


# def create_model(time_series_length,news_words,news_embedding_dim,word_cardinality):
#     ## Input of normal time series
#
#     time_series_input = Input(shape=(time_series_length, 1, ), name='time_series')
#
#     ## For every word we have it's own input
#     news_word_inputs = [Input(shape=(time_series_length, ), name='news_word_' + str(i + 1)) for i in range(news_words)]
#
#     ## Shared embedding layer
#     news_word_embedding = Embedding(word_cardinality, news_embedding_dim, input_length=time_series_length)
#     print(np.array(news_word_embedding).shape)
#     ## Repeat this for every word position
#     news_words_embeddings = [news_word_embedding(inp) for inp in news_word_inputs]
#     print(np.array(news_word_embedding).shape)
#
#     ## Concatenate the time series input and the embedding outputs
#     concatenated_inputs = concatenate([time_series_input] + news_words_embeddings, axis=-1)
#     conv1 = Conv1D(TimeDistributed(100, (3), activation='relu'))(concatenated_inputs)
#     ## Feed into LSTM
#     # lstm = LSTM(16)(concatenated_inputs)
#
#     ## Output, in this case single classification
#     output = Dense(time_series_length, activation='sigmoid')(lstm)
#     model = Model(inputs=[time_series_input]+news_word_inputs, outputs=output)
#     # model = Model(inputs=[deep_inputs, pos_tagging], outputs=output)
#     #model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     optimizer = Adagrad(lr=0.001, epsilon=None, decay=0.0)
#     model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
#     model.summary()
#     return model
#
# n_samples = 100
# time_series_length = 5
# news_words = 10
# news_embedding_dim = 16
# word_cardinality = 50
#
# x_time_series = np.random.rand(n_samples, time_series_length, 1)
# x_news_words = np.random.choice(np.arange(50), replace=True, size=(n_samples, time_series_length, news_words))
# print("news-1")
# print(np.array(x_news_words)[0])
# print(np.array(x_news_words).shape)
# x_news_words = [x_news_words[:, :, i] for i in range(news_words)]
# y = np.random.randint(2, size=(n_samples,time_series_length))
# # print("time")
# # print(np.array(x_time_series).shape)
# print("news-2")
# print(np.array(x_news_words).shape)
# print(np.array(x_news_words)[0])
# # print(y)
# model=create_model(time_series_length,news_words,news_embedding_dim,word_cardinality)
# model.fit([x_time_series] + x_news_words, y)
# x_time_series_test = np.random.rand(1, time_series_length, 1)
# x_news_words_test = np.random.choice(np.arange(50), replace=True, size=(1, time_series_length, news_words))
# x_news_words_test = [x_news_words_test[:, :, i] for i in range(news_words)]
# y_test= np.random.randint(2, size=(1,time_series_length))
# result = model.predict([x_time_series_test]+ x_news_words_test)
# print(result)