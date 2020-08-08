from cmath import sqrt

import  pandas as pd
import numpy as np
# from keras import Sequential
# from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFE, SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import scipy.stats
# import tensorflow as tf

from sklearn.svm import SVR
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import StandardScaler
result_list = {"test":[],"result":[]}
rmse_list={"rmse":[],"MAE":[],"pearson":[],"r2":[]}
def segmented_output():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df=pd.read_excel("data/ADReSS/acoustic_feature_test.xlsx",sheet_name='Sheet2')
    df['mmse'] = [None] * len(cc_meta)
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
def feature_RF(X_train,y_train,vocab):
    imp_feature=[]
    feature_no = []
    features=[]
    gini=[]
    # print(X_train)
    # print(y_train)
    clf=RandomForestRegressor(n_estimators=100)
    clf.fit(X_train, y_train)
    sel = SelectFromModel(clf)
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = X_train.columns[(sel.get_support())]
    feat_labels=list(X_train.columns)
    # print(len(selected_feat))
    for feature in selected_feat:

        features.append(feature)



    for feature,imp in zip(feat_labels, clf.feature_importances_):
        if (type(feature) == int ):
            if(feature in features):

                imp_feature.append(vocab[feature])
                gini.append(imp)
                feature_no.append(feature)
        else:
            if (feature in features):
                    imp_feature.append(feature)
                    gini.append(imp)
                    feature_no.append(feature)

    dict={"feature":imp_feature,"feature_no":feature_no,"gini":gini}
    df=pd.DataFrame(dict)
    final_df = df.sort_values(by=['gini'], ascending=False)
    return final_df

def save_feature(df,name):
    writer = pd.ExcelWriter("./data/" + name + ".xlsx", engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Sheet1', columns=["feature","feature_no","gini"])
    writer.save()

def save_result(name,num):
    if num==1:
        df=pd.DataFrame(result_list)
        cols=['test','result']
    else:
        df = pd.DataFrame(rmse_list)
        cols = ['rmse',"MAE","pearson","r2"]
    writer = pd.ExcelWriter("./data/"+name+".xlsx", engine='xlsxwriter')


    df.to_excel(writer, sheet_name='Sheet1', columns=cols)
    writer.save()


def SVR_classification(x_train,x_test,y_train,y_test):

    X = x_train
    y = y_train.reshape(-1,1)
    # 3 Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler(with_mean=False)
    sc_y = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_y.fit_transform(y)

    # 4 Fitting the Support Vector Regression Model to the dataset
    # Create your support vector regressor here
    # df_train = df_train.iloc[:, 1:]
    # print("GP model")
    # import sklearn.gaussian_process as gp
    # most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
    # regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    regressor=SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)
    regressor.fit(X, y)
    # regressor = tree.DecisionTreeRegressor(max_leaf_nodes=20)
    # regressor.fit(X, y)
    # kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    #
    # regressor = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    # regressor.fit(X, y)
    # params = model.kernel_.get_params()
    # print("gp model fit")
    # 5 Predicting a new result
    # y_pred, std = regressor.predict(x_test, return_std=True)
    # print("GP model pred")
    # x_test=x_test.fillna(x_test.mean())
    # print(x_test)
    # y_pred = regressor.predict(x_test)
    y_pred = sc_y.inverse_transform ((regressor.predict (sc_X.transform(x_test))))
    test_rmse = sqrt(mean_squared_error(y_test * 30, y_pred * 30))
    test_MAE = mean_absolute_error(y_test * 30, y_pred * 30)
    # pearson=scipy.stats.pearsonr(y_test * 30, y_pred * 30)
    pearson=0
    r2=r2_score(y_test * 30, y_pred * 30)
    for val in y_pred:
        result_list["result"].append(val*30)
    for val in y_test:
        result_list["test"].append(val*30)
        # print("MAE")
        # print(test_MAE)

    return test_rmse,test_MAE,pearson,r2
# plot vovabulary feature
def prt_features(vocab,cols,rfe,rank):
    count=0
    for item in rfe:
        if item == True:
            if count>612:
                print(cols[count])
            else:
                print(cols[count])




def wrapper(X,Y,vocab):





    y = Y  # Target Variable
    # print(list(X.columns))
    # # most important SVR parameter is Kernel type. It can be #linear,polynomial or gaussian SVR. We have a non-linear condition #so we can select polynomial or gaussian but here we select RBF(a #gaussian type) kernel.
    # # regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    # model = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    model = LinearRegression()
    #
    # Initializing RFE model
    rfe = RFE(model, 30)
    # Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y)
    # Fitting the data to model
    model.fit(X_rfe, y)
    cols=list(X.columns)
    prt_features(vocab,cols,rfe.support_,rfe.ranking_)
    print(rfe.support_)
    print(rfe.ranking_)

def backward_elimination(df,Y):
    print(list(df.columns))
    X = df  # Feature Matrix
    y = Y  # Target Variable
    cols = list(X.columns)

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # model = LinearRegression()
    # model.fit(X, y)

    # print(model.score(X_test, Y_test))
    # print(y)
    # print(X)
    # Adding constant column of ones, mandatory for sm.OLS model

    X_1 = sm.add_constant(X)
    # Fitting sm.OLS model
    model = sm.OLS(y, X_1).fit()
    print(model.summary())
    print(model.pvalues)
    # Backward Elimination

    pmax = 1
    while (len(cols) > 0):
        p = []
        X_1 = X[cols]
        X_1 = sm.add_constant(X_1)
        model = sm.OLS(y, X_1).fit()
        p = pd.Series(model.pvalues.values[1:], index=cols)
        pmax = max(p)
        feature_with_p_max = p.idxmax()
        if (pmax > 0.05):
            cols.remove(feature_with_p_max)
        else:
            break
    selected_features_BE = cols
    print(selected_features_BE)

def filter_method(df):

    print(list(df.columns))
    # Using Pearson Correlation
    # plt.figure(figsize=(150, 150))
    cor = df.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()
    cor_target = abs(cor["mmse"])
    # Selecting highly correlated features
    relevant_features1 = cor_target[cor_target >= 1.5]
    # relevant_features2 = cor_target[cor_target <= -0.4]
    # print(relevant_features1.index)
    print(relevant_features1)
    # print(relevant_features2)
    # print(df[['word_repetition_count', 'incomplete_utterance_count']].corr())
    # print(df[['utterance_count', 'incomplete_utterance_count']].corr())
    # print(df[['utterance_count', 'word']].corr())
    # print(df[['utterance_count', 'word_repetition_count']].corr())
    # print(df[['utterance_count', 94]].corr())

# def NN_Regression(x_train,x_test,y_train,y_test):
#     from sklearn.preprocessing import StandardScaler
#     sc_X = StandardScaler(with_mean=False)
#     sc_y = StandardScaler()
#     y = y_train.reshape(-1, 1)
#     X_train = sc_X.fit_transform(x_train)
#     Y_train = sc_y.fit_transform(y)
#     model = Sequential()
#     model.add(Dense(X_train.shape[1], activation="relu", input_dim=X_train.shape[1]))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal',activation='linear'))
#     model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
#
#     model.fit(X_train, Y_train,
#               epochs=20,
#               batch_size=128)
#
#     y_pred = sc_y.inverse_transform((model.predict(sc_X.transform(x_test))))
#     test_rmse = sqrt(mean_squared_error(y_test * 30, y_pred * 30))
#     test_MAE = mean_absolute_error(y_test * 30, y_pred * 30)
#     # pearson = scipy.stats.pearsonr(y_test * 30, y_pred * 30)
#     pearson=0
#     r2 = r2_score(y_test * 30, y_pred * 30)
#     for val in y_pred:
#         result_list["result"].append(val * 30)
#     for val in y_test:
#         result_list["test"].append(val * 30)
#         # print("MAE")
#         # print(test_MAE)
#     return test_rmse, test_MAE, pearson, r2
def plot_vocab(count_occur_df):
    g = count_occur_df.nlargest(columns="Count", n=50)
    plt.figure(figsize=(30, 35))
    ax = sns.barplot(data=g, x="Count", y="Word")
    ax.set(xlabel='Count (appeared atleast 10 docs and atmost 50 docs, no punctuation considered)')
    plt.show()


def voting():
    # cc_meta = pd.read_csv("data/meta_all.txt", sep=';')
    cc_meta = pd.read_excel("data/adress_test.xlsx", sheet_name='Sheet1')
    df = pd.read_excel("data/adress_fs_audio_poly.xlsx", sheet_name='Sheet1')
    dict={"file":[],"test":[],"result":[]}
    df['mmse'] = [None] * len(df)
    count = 0
    for index, row in cc_meta.iterrows():
        j = count
        rows=0
        vals=0
        dict["file"].append(row['file'])
        dict["test"].append(row['mmse'])
        for i in range(j, len(df)):
            name = df.iloc[i]["file"].split("-")[0]
            # name = df.iloc[i]["file"]
            # print(name)
            # print(row['file'].strip())
            if (name.strip() == row['file'].strip()):
                vals+=df.iloc[i]["result"]
                count += 1
                rows+=1

            else:
                break
        dict["result"].append(vals/rows)

    r2 = r2_score(df["test"], df["result"])
    test_rmse = sqrt(mean_squared_error(df["test"], df["result"]))
    # test_MAE = mean_absolute_error(df["test"], df["result"])
    print("final r2 %f" %(r2))
    print("final rmse " )
    print( test_rmse)



    df=pd.DataFrame(dict)
    writer = pd.ExcelWriter("data/adress_fs_acoustic_poly_final.xlsx", engine='xlsxwriter')

    df.to_excel(writer, sheet_name='Sheet1', columns=list(df.columns))
    writer.save()
    with open('data/adress_acoustic_test.pickle', 'wb') as f:
        pickle.dump(df, f)
def generate_ngrams(df_train, df_test):

    # print("testing..")
    # print(type(df))
    # vectorizer = CountVectorizer(min_df=10,ngram_range=(1, 3))


    vectorizer = TfidfVectorizer(sublinear_tf=True,min_df=5,max_df=60, ngram_range=(1, 3),norm='l2')
    train_text=df_train["utterances"]
    test_text=df_test["utterances"]




    x_train = vectorizer.fit_transform(train_text)


    x_test = vectorizer.transform(test_text)

    vocab=vectorizer.get_feature_names()
    count_list = x_train.toarray().sum(axis=0)
    test_count_list=x_test.toarray().sum(axis=0)
    ngram_feature=x_train.toarray()

    count_occur_df = pd.DataFrame({'Word':vocab,'Count':count_list})
    count_occur_df_test = pd.DataFrame({'Word': vocab, 'Count': test_count_list})
    count_occur_df.columns = ['Word', 'Count']
    count_occur_df.sort_values('Count', ascending=False, inplace=True)
    count_occur_df_test.sort_values('Count', ascending=False, inplace=True)

    # print("vocab")
    # print(vocab)
    # print(len(vocab))
    # print(ngram_feature.shape)
    # print('train')
    # print(count_occur_df.head)
    # print('test')
    # print(count_occur_df_test.head)
    # print(count_occur_df.head(20))
    # print(count_occur_df_test.head(20))
    # plot_vocab(count_occur_df)
    # plot_vocab(count_occur_df_test)
    return x_train,x_test,vocab
    # return y_test
def linguistic_model():
    df = pd.read_pickle('data/adress_final_interview.pickle')
    df_test = pd.read_pickle('data/adress_test.pickle')
    numeric_label = []
    numeric_label_test = []
    for index, row in df.iterrows():
        numeric_label.append(int(row['mmse']) / 30)
    for index, row in df_test.iterrows():
        numeric_label_test.append(int(row['mmse']) / 30)
    y_train = np.array(numeric_label)
    y_tests = np.array(numeric_label_test)
    # # generate ngrams
    df_train=df
    x_train,x_test,vocab=generate_ngrams(df_train, df_test)
    train = pd.DataFrame(x_train.toarray())
    # # # demographic
    train_feature_1=df_train.iloc[:,4:5]
    # # # nonverbal
    train_feature_2 = df_train.iloc[:,7:14]
    # # # # psycholinguistic
    train_feature_3 = df_train.iloc[:, 15:]


    test = pd.DataFrame(x_test.toarray())
    test_feature_1 = df_test.iloc[:,2:3]
    test_feature_2 = df_test.iloc[:,7:14]
    test_feature_3 = df_test.iloc[:, 15:]

    # train_feature_1=np.array(train_feature_1)
    # train_feature_2 = np.array(train_feature_2)
    # train_feature_3 = np.array(train_feature_3)
    #
    # test_feature_1 = np.array(test_feature_1)
    # test_feature_2 = np.array(test_feature_2)
    # test_feature_3 = np.array(test_feature_3)

    x_train = pd.concat((train,train_feature_2,train_feature_3), axis=1)
    x_test = pd.concat((test,test_feature_2,test_feature_3), axis=1)
    # #feature selection
    #
    x_train=x_train.fillna(x_train.mean())
    vocab=list(x_train.columns)
    df_feature = feature_RF(x_train, y_train, vocab)

    df_feature_column = df_feature["feature_no"].tolist()
    x_train=x_train.loc[:,df_feature_column]
    x_test=x_test.loc[:,df_feature_column]

    save_feature(df_feature,"important_audio_feature")
    rmse,mae,pearson,r2=SVR_classification(x_train,x_test,y_train,y_tests)
    print("rmse")
    print(rmse)
    print("MAE")
    print(mae)
    print("pearson")
    print(pearson)
    print("R2")
    print(r2)
    rmse_list["rmse"].append(rmse)
    rmse_list["MAE"].append(mae)
    rmse_list["pearson"].append(pearson)
    rmse_list["r2"].append(r2)

    save_result("adress_fs_poly",1)
    save_result("adress_fs_2_poly",2)
def acoustic_model():
    df = pd.read_excel('data/adress_acoustic_train.xlsx', sheet_name='Sheet1')
    # cc_meta = pd.read_csv("data/ADReSS/meta_all.txt", sep=';')


    # df_audio = pd.read_excel('data/acoustic_feature_adress.xlsx', sheet_name='Sheet2')
    # df_feature = pd.read_excel('data/important_audio_feature.xlsx', sheet_name='Sheet1')
    # df_feature_column=list(df_feature.loc[:,"feature_no"])

    df_test = pd.read_excel('data/adress_acoustic_test.xlsx', sheet_name='Sheet1')

    numeric_label = []
    numeric_label_test = []
    for index, row in df.iterrows():
        numeric_label.append(int(row['mmse']) / 30)
    for index, row in df_test.iterrows():
        numeric_label_test.append(int(row['mmse']) / 30)
    y_train = np.array(numeric_label)
    y_tests = np.array(numeric_label_test)

    x_train = df.iloc[:, 4:]
    x_test=df_test.iloc[:, 2:]

    rmse,mae,pearson,r2=SVR_classification(x_train,x_test,y_train,y_tests)

    rmse_list["rmse"].append(rmse)
    rmse_list["MAE"].append(mae)
    rmse_list["pearson"].append(pearson)
    rmse_list["r2"].append(r2)


    print("rmse")
    print(rmse)
    print("MAE")
    print(mae)
    print("pearson")
    print(pearson)
    print("R2")
    print(r2)

    save_result("adress_fs_audio_poly",1)
    save_result("adress_fs_audio_2_poly",2)
    # prepare final rmse score on transcript level averaging rmse value for each segment of the audio for
    # the transcript
    voting()


linguistic_model()
# acoustic_model()




# kf = KFold(n_splits=5,shuffle=True)

# count=0
# final_rmse=0
# final_mae=0
# loo=LeaveOneOut()
# for train_index, test_index in kf.split(df):
# for train_index, test_index in loo.split(cc_meta.file):
# print("iteration %d"%(count) )
#
# # train_index=cc_meta.loc[train_index,'file']
# # test_index = cc_meta.loc[test_index, 'file']
# # df_train, df_test = df.loc[df.index.isin(train_index)], df.loc[df.index.isin(test_index)]


