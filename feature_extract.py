
import pandas as pd 
import numpy as np
import nltk

# nltk.download('stopwords')
# nltk.download('vader_lexicon')
merged_dataframe2 = pd.read_excel('data/adress_test_inter.xlsx',sheet_name='Sheet1')
# data = pd.read_excel('data/full_conversation__mmse.xlsx',sheet_name='Sheet1')
# non_verbal = pd.read_excel('data/nonverbal_feature_mmse.xlsx', sheet_name='Sheet1')
# print(data.head())
# print(len(data))
# #%%
df2 = pd.read_pickle('data/adress_test.pickle')
# df = pd.read_excel('data/pitt_mmse.xlsx', sheet_name='Sheet1')
# print(anagraphic_data.head())
# #%%
# merged_dataframe1 = pd.merge(data, non_verbal, on='id')
# print(len(merged_dataframe1))
# merged_dataframe2 = pd.merge(merged_dataframe1, anagraphic_data, on='id')
# print(merged_dataframe2)
# #%%
# print(len(data))
# print(data.head())

#%%
# noinspection PyUnresolvedReferences
from psycholinguistic import get_psycholinguistic_features
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from tqdm import tqdm
from tqdm import tqdm as tqdm
# nltk.download('averaged_perceptron_tagger')

sid = SentimentIntensityAnalyzer()

new_dataframe = []
for index, row in tqdm(merged_dataframe2.iterrows()):
    single_sentence_list = []

    string = ''
    for token in row.text:
        if token == '\n':
            single_sentence_list.append(string)
            string = ''
        else:
            string += ' ' + token

    counter = 0
    comp_sentiment_sum = 0
    for sentence in single_sentence_list:
        ss = sid.polarity_scores(sentence)
        comp_sentiment_sum += ss['compound']
        counter += 1

    if counter != 0:
        average_sentiment = comp_sentiment_sum/counter
    else:
        average_sentiment = 0

    ## for each interview in the dataset.
    interview = nltk.pos_tag(row.text, lang='eng')

    final_interview = []
    for uttr in interview:
        final_interview.append({'token': uttr[0],'pos':uttr[1]})

    dict = get_psycholinguistic_features(final_interview)

    dict['average_sentiment'] = average_sentiment

    # additional_features = []
    #
    # for  key,value in dict.items():
    #     additional_features.append(value)
    # additional_features.append(row.short_pause_count)
    # additional_features.append(row.long_pause_count)
    # additional_features.append(row.very_long_pause_count)
    # additional_features.append(row.word_repetition_count)
    # additional_features.append(row.retracing_count)
    # additional_features.append(row.filled_pause_count)
    # additional_features.append(row.utterance_count)
    # additional_features.append(row.incomplete_utterance_count)
    # additional_features.append(row.word_count)


    ##Here we take in consideration anagraphic features.

    # anagraphic_features = [row.age,row.education,row.race,row.sex]
    # anagraphic_features = [row.age,  row.sex]

    # dict['features'] = additional_features + anagraphic_features
    # dict['label'] = row.label
    # dict['text'] = row.text
    # dict['mmse'] = row.mmse
    # dict['id'] = row.id
    new_dataframe.append(dict)



#%%
### Word correctness
# final_dataframe = pd.DataFrame(new_dataframe)
df1 = pd.DataFrame(new_dataframe)
# print(final_dataframe.head())
# print(final_dataframe["features"])
# #%%
import pickle

# with open('data/adress_full_interview_features.pickle', 'wb') as f:
#     df=pickle.dump(final_dataframe, f)
# df1=pd.read_pickle('data/adress_full_interview_features.pickle')
# df2 = pd.read_excel('data/adress_text.xlsx',sheet_name='Sheet1')
# print(list(df2.columns))
df2["getAoaScore"]=df1["getAoaScore"]
df2["getSUBTLWordScores"]=df1["getSUBTLWordScores"]
df2["getFamiliarityScore"]=df1["getFamiliarityScore"]
df2["getConcretenessScore"]=df1["getConcretenessScore"]
df2["getImagabilityScore"]=df1["getImagabilityScore"]
df2["average_sentiment"]=df1["average_sentiment"]
with open('data/adress_test.pickle', 'wb') as f:
    df=pickle.dump(df2, f)
df=pd.read_pickle('data/adress_test.pickle')
print(list(df.columns))
print(df.head(3))

# print(df["features"].values)
# print(df["features"].values)
#%%
