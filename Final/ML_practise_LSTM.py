from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import pandas as pd
import numpy as np
import os
os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("Final_500_Data.csv")
df_latest = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
# df = df[df['Date']!=df['Date'].max()]
# df_latest2 = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
# dic_prc = df_latest.set_index('Symbol').to_dict()
# df_latest2['Pr'] = df_latest2['Symbol'].map(dic_prc['Adj Close'])
# df_latest2.dropna(inplace = True)
# df_latest2['ret'] = df_latest2['Pr']/df_latest2['Adj Close']
# dic_ret = df_latest2[['Symbol','ret']].set_index('Symbol').to_dict()

df = df.drop(['uptrend','Open','High','Low','Close'], axis = 1) 
df.fillna(method ='bfill', axis=0, inplace=True)
tickers = df['Symbol'].unique().tolist()


DF = df[df['Symbol']=="RELIANCE.NS"]
DF = DF.drop(['Symbol'], axis=1).set_index('Date')
DF_train = DF[:-5]['Adj Close']
DF_test = DF[-5:]['Adj Close']

scaler = MinMaxScaler()
scaler.fit(DF_train)
scaled_train = scaler.transform(DF_train)
scaled_test = scaler.transform(DF_test)


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step)]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step])
	return np.array(dataX), np.array(dataY)

dataset = DF_train
time_step = 3
X_train, y_train = create_dataset(DF_train, time_step)
X_test, ytest = create_dataset(DF_test, time_step)