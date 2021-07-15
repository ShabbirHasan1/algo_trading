import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")

df = pd.read_csv("Final_500_Data.csv")
df_latest = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
df = df[df['Date']!=df['Date'].max()]
df_latest2 = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
dic_prc = df_latest.set_index('Symbol').to_dict()
df_latest2['Pr'] = df_latest2['Symbol'].map(dic_prc['Adj Close'])
df_latest2.dropna(inplace = True)
df_latest2['ret'] = df_latest2['Pr']/df_latest2['Adj Close']
dic_ret = df_latest2[['Symbol','ret']].set_index('Symbol').to_dict()

df = df.drop(['uptrend','Open','High','Low','Close'], axis = 1) 
df.fillna(method ='bfill', axis=0, inplace=True)
tickers = df['Symbol'].unique().tolist()

def trg_reg_1(DF):
    X = DF.copy()
    #X['reg_tgt_1'] = X['Adj Close'].shift(-1)
    X['reg_tgt_1'] = X['Adj Close'].shift(-1)/X['Adj Close']
    return X#[:-5]

dict ={}
for ticker in tickers:
    dict[ticker] = trg_reg_1(df[df['Symbol']==ticker])
    
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)


models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
# models.append(('EN', ElasticNet()))
# models.append(('KNN', KNeighborsRegressor()))
# models.append(('CART', DecisionTreeRegressor()))
# models.append(('SVR', SVR()))
# models.append(('MLP', MLPRegressor()))
# Boosting methods
# models.append(('ABR', AdaBoostRegressor()))
# models.append(('GBR', GradientBoostingRegressor()))
# Bagging methods
# models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

def reg1(DF,tick,mod):
    dataset = DF.copy()
    dataset = dataset.drop(['Symbol'], axis=1).set_index('Date')
    X = dataset[:-1].iloc[:, :-1]
    y = dataset[:-1].iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    n = int(.80*len(X))
    bestfeatures = SelectKBest(k=5, score_func=f_regression)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    features = list(featureScores.nlargest(20,'Score')['Specs'])  #print 10 best features
    # correlations = np.abs(X_train.corrwith(y_train))
    # features =  list(correlations.sort_values(ascending=False)[0:15].index)
    X_train = X_train[features].values
    X_test = X_test[features].values
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    classifier = mod[1]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    X_nul = dataset[-1:].iloc[:, :-1][features].values
    X_nul = sc.transform(X_nul)
    y_pred_nul = classifier.predict(X_nul)
    k1 = 100*(y_pred_nul/dataset['Adj Close'][-1])[0]
    dataset[f'reg1_{mod[0]}'] = k1
    dataset['Symbol'] = tick
    dataset= dataset[[f'reg1_{mod[0]}','Symbol']]
    dic = {}
    dic[dataset.columns[0]] = k1
    dic[dataset.columns[1]] = tick
    return 100*y_pred_nul #k1

dict ={}
for ticker in tickers[:]:
    dict[ticker] = []
    for j in range(0,len(models)):
        try:
            dict[ticker].append(reg1(df[df["Symbol"]==ticker],ticker,models[j]))
        except:
            pass
cols = []
for i in range(0,len(models)):
    cols.append("Reg_"+models[i][0])


Regression = pd.DataFrame(dict.items())

for j in range(0,len(cols)):
    Regression[f"{cols[j]}"] = ""
    
for i in range(0,len(Regression)):
    for j in range(0,len(cols)):
        if len(Regression[1][i])>0:
            Regression[f"{cols[j]}"].iloc[i]= Regression[1][i][j][0]

Regression.iloc[:,2:] = Regression.iloc[:,2:].apply(pd.to_numeric, errors='coerce')
Regression.rename(columns = {0:'Symbol'}, inplace = True)
dic_prc = df_latest.set_index('Symbol').to_dict()
Regression['Pr'] = Regression['Symbol'].map(dic_prc['Adj Close'])
Regression['ret'] = Regression['Symbol'].map(dic_ret['ret'])
Regression.dropna(inplace = True)
Regression['Hard'] = (Regression.iloc[:,2:5]>100).sum(axis=1)
Regression['Metric'] = Regression['Reg_LR']*Regression['Reg_LASSO']*Regression['Reg_ETR']
Regression = Regression[Regression['Hard']==3]
Regression = Regression.sort_values(by = ['Metric'], ascending = [False])




# def reg1(DF,tick,mod):
#     dataset = DF.copy()
#     dataset = dataset.drop(['Symbol'], axis=1).set_index('Date')
#     X = dataset[:-1].iloc[:, :-1]
#     y = dataset[:-1].iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#     n = int(.80*len(X))
#     bestfeatures = SelectKBest(k=5, score_func=f_regression)
#     fit = bestfeatures.fit(X,y)
#     dfscores = pd.DataFrame(fit.scores_)
#     dfcolumns = pd.DataFrame(X.columns)
#     featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#     featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#     features = list(featureScores.nlargest(12,'Score')['Specs'])  #print 10 best features
#     # correlations = np.abs(X_train.corrwith(y_train))
#     # features =  list(correlations.sort_values(ascending=False)[0:15].index)
#     X_train = X_train[features].values
#     X_test = X_test[features].values
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     classifier = mod[1]
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     X_nul = dataset[-1:].iloc[:, :-1][features].values
#     X_nul = sc.transform(X_nul)
#     y_pred_nul = classifier.predict(X_nul)
#     k1 = 100*(y_pred_nul/dataset['Adj Close'][-1])[0]
#     dataset[f'reg1_{mod[0]}'] = k1
#     dataset['Symbol'] = tick
#     dataset= dataset[[f'reg1_{mod[0]}','Symbol']]
#     dic = {}
#     dic[dataset.columns[0]] = k1
#     dic[dataset.columns[1]] = tick
#     return y_pred_nul #k1

# # DF = df[df['Symbol']=="RELIANCE.NS"]


# dict ={}
# for ticker in tickers[:]:
#     dict[ticker] = []
#     for j in range(0,len(models)):
#         try:
#             dict[ticker].append(reg1(df[df["Symbol"]==ticker],ticker,models[j]))
#         except:
#             pass
# cols = []
# for i in range(0,len(models)):
#     cols.append("Reg_"+models[i][0])


# Regression = pd.DataFrame(dict.items())
# for j in range(0,len(cols)):
#     Regression[f"{cols[j]}"] = ""
# for i in range(0,len(Regression)):
#     for j in range(0,len(cols)):
#         if len(Regression[1][i])>0:
#             Regression[f"{cols[j]}"].iloc[i]= Regression[1][i][j][0]

# Regression.iloc[:,2:] = Regression.iloc[:,2:].apply(pd.to_numeric, errors='coerce')
# Regression.rename(columns = {0:'Symbol'}, inplace = True)
# dic_prc = df_latest.set_index('Symbol').to_dict()
# Regression['Pr'] = Regression['Symbol'].map(dic_prc['Adj Close'])
# Regression.dropna(inplace = True)


# err = []
# for i in range(0,len(cols)):
#     err.append(mean_absolute_error(Regression['Pr'],Regression[cols[i]]))
# df.reset_index(inplace=True)


# def reg2(DF,tick):
#     dataset = DF.copy()
#     t = dataset['reg1'].unique()[0]
#     dataset = dataset.drop(['Symbol','reg1'], axis=1).set_index('Date')
#     X = dataset[:-1].iloc[:, :-1]
#     y = dataset[:-1].iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
#     n = int(.85*len(X))
#     correlations = np.abs(X_train.corrwith(y_train))
#     features =  list(correlations.sort_values(ascending=False)[0:15].index)
#     X_train = X_train[features].values
#     X_test = X_test[features].values
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     classifier = GradientBoostingRegressor()
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     X_nul = dataset[-1:].iloc[:, :-1][features].values
#     X_nul = sc.transform(X_nul)
#     y_pred_nul = classifier.predict(X_nul)
#     # cm = confusion_matrix(y_test, y_pred)
#     # acc = accuracy_score(y_test, y_pred)
#     # dataset = dataset.iloc[n:,:]
#     k1 = 100*(y_pred_nul/dataset['Adj Close'][-1])[0]
#     dataset['reg1'] = t
#     dataset['reg2'] = k1
#     dataset['Symbol'] = tick
#     return dataset

# # DF = df[df['Symbol']=="BHEL.NS"]
# dict ={}
# for ticker in tickers[:]:
#     dict[ticker] = reg2(df[df["Symbol"]==ticker],ticker)

# df = pd.concat([dict[ticker] for ticker in list(dict.keys())],axis=0)
# df.reset_index(inplace=True)



# def trg_log_1(DF):
#     X = DF.copy()
#     X['trg_log_1'] = np.where(X['Adj Close'].shift(-1)/X['Adj Close'] >1.04,1,0)
#     return X#[:-5]

# dict ={}
# for ticker in tickers:
#     dict[ticker] = trg_log_1(df[df['Symbol']==ticker])

# df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

# def log1(DF,tick):
#     dataset = DF.copy()
#     t = dataset['reg1'].unique()[0]
#     dataset = dataset.drop(['Symbol','reg1','reg_tgt_1'], axis=1).set_index('Date')
#     X = dataset[:-1].iloc[:, :-1]
#     y = dataset[:-1].iloc[:, -1]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
#     n = int(.85*len(X))
#     correlations = np.abs(X_train.corrwith(y_train))
#     features =  list(correlations.sort_values(ascending=False)[0:15].index)
#     X_train = X_train[features].values
#     X_test = X_test[features].values
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
#     classifier = LogisticRegression(random_state = 0)
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict_proba(X_test)[:,1]
#     X_nul = dataset[-1:].iloc[:, :-1][features].values
#     X_nul = sc.transform(X_nul)
#     y_pred_nul = classifier.predict_proba(X_nul)[:,1][0]
#     dataset['log1'] = y_pred_nul
#     dataset['Symbol'] = tick
#     dataset['reg1'] = t
#     return dataset

# dict ={}
# for ticker in tickers:
#     try:
#         dict[ticker] = log1(df[df['Symbol']==ticker],ticker)
#     except:
#         pass

# df = pd.concat([dict[ticker] for ticker in list(dict.keys())],axis=0)

# df.reset_index(inplace=True)
# final = df[df['Date']==df['Date'].max()]


