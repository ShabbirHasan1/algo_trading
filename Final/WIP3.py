from kiteconnect import KiteConnect
import pandas_datareader as pdr
import pandas as pd
import datetime
import math
from finta import TA
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
from datetime import datetime
from stocktrends import Renko
import warnings
cwd = os.chdir(r"C:\Users\Manish\Desktop\Final_Codes\Final")
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)


today = datetime.now().day
sym = pd.read_csv("NSE_Symbols_Pink.csv", dtype={'Instrument': int})[['Symbol','Instrument']]
dic1 = sym.set_index('Instrument').to_dict()
tokens = sym['Instrument'].tolist()[:]

strike = pd.read_csv("Strike_Size.csv")
dic2 = strike.set_index('Symbol').to_dict()


def fetchOHLC(ticker,interval,duration):
    """extracts historical data and outputs in the form of dataframe"""
    data = pd.DataFrame(kite.historical_data(ticker,dt.date.today()-dt.timedelta(duration), dt.date.today(),interval))
    data.date =data.date.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    return data

ticker = tokens[2]
# nifty = fetchOHLC(256265,"3minute",4)
data = fetchOHLC(ticker,"3minute",4)[:-5]

# nifty['sp_percent_change'] = nifty['close'].pct_change(periods=1).astype(float)
# data = data.merge(nifty['sp_percent_change'], left_index=True, right_index=True)
data['candle'] = data['close'] - data['open']

# data['relative_change'] = data['percent_change'] - data['sp_percent_change']
# data.reset_index(inplace=True)
data.columns = [x.lower() for x in data.columns]

indicators = ['SMA', 'SMM', 'SSMA', 'EMA', 'DEMA', 'TEMA', 'TRIMA', 'TRIX', 'VAMA', 'ER', 'KAMA', 'ZLEMA',
              'WMA', 'HMA', 'EVWMA', 'VWAP', 'SMMA', 'MACD', 'PPO', 'VW_MACD', 'EV_MACD', 'MOM', 'ROC', 'RSI',
              'IFT_RSI']
broken_indicators = ['SAR', 'TMF', 'VR', 'QSTICK']

for indicator in indicators:
      if indicator not in broken_indicators:
          df = None
          # Using python's eval function to create a method from a string instead of having every method defined
          df = eval('TA.' + indicator + '(data)')
          # Some method return series, so we can check to convert here
          if not isinstance(df, pd.DataFrame):
              df = df.to_frame()
          # Appropriate labels on each column
          df = df.add_prefix(indicator + '_')
          # Join merge dataframes based on the date
          data = data.merge(df, left_index=True, right_index=True)
  # Fix labels
data.columns = data.columns.str.replace(' ', '_')
stocks = data.copy().set_index('date')
candle = abs(stocks['close'] - stocks['open']).median()
bins=[-50*candle, -4*candle, -1*candle, candle, 4*candle, 50*candle]
# group_names = ['strong sell', 'sell', 'hold', 'buy', 'strong buy']
group_names = [0, 1, 2, 3, 4]
# stocks['short_result'] = pd.cut(stocks['candle'], bins=bins, labels=group_names)
stocks['short_result_new'] = pd.cut(stocks['candle'], bins=bins, labels=group_names).shift(-5)
stocks.fillna(0, inplace = True)


X = stocks[:-5].iloc[:, :-1]
y = stocks[:-5].iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
n = int(.75*len(X))
bestfeatures = SelectKBest(k=5, score_func=f_classif)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
features = list(featureScores.nlargest(20,'Score')['Specs']) 
X_train = X_train[features].values
X_test = X_test[features].values
X_nul = stocks[features][-5:].values





sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

X_nul = sc.transform(X_nul)
pred_rfc_nul = sum(rfc.predict(X_nul)>=3)

# print(classification_report(y_test, pred_rfc))
# print(confusion_matrix(y_test, pred_rfc))


