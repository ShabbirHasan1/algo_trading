from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
from datetime import datetime
from stocktrends import Renko
import warnings
cwd = os.chdir("C:\Desktop\Learning\Courses\zerodha")
import time
from sklearn.preprocessing import MinMaxScaler

access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)


today = datetime.now().day
sym = pd.read_csv("NSE_Symbols.csv", dtype={'Instrument': int})[['Symbol','Instrument']]
dic1 = sym.set_index('Instrument').to_dict()
tokens = sym['Instrument'].tolist()[:250]

sym = pd.read_csv("NSE_Symbols.csv", dtype={'Instrument': int})[['Symbol','Instrument']]
dic1 = sym.set_index('Instrument').to_dict()


#os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")

def fetchOHLC(ticker,interval ="2minute",duration = 3):
    """extracts historical data and outputs in the form of dataframe"""
    data = pd.DataFrame(kite.historical_data(ticker,dt.date.today()-dt.timedelta(duration), dt.date.today(),interval))
    data.date =data.date.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    return data


#df = fetchOHLC(tokens[0])

warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def AR1(DF):
    x = DF.copy()
    fitted_model = ExponentialSmoothing(x['close'],trend='mul',seasonal='mul',seasonal_periods=20).fit()
    pred1 = fitted_model.forecast(1).iloc[0]
    return pred1/x['close'].iloc[-1]

AR1(fetchOHLC(tokens[0]))

warnings.filterwarnings('ignore')
from statsmodels.tsa.ar_model import AR,ARResults
def AR2(DF):
    x = DF.copy()
    model = AR(x['close'])
    AR1fit = model.fit(ic = 't-stat',maxlag=4)
    pred2 = AR1fit.predict(start=len(x), end=len(x), dynamic=False).iloc[0]
    return pred2/x['close'].iloc[-1]

AR2(fetchOHLC(tokens[0]))


from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
def AR3(DF):
    x = DF.copy()
    model = ARMA(x['close'],order=(3,2))
    results = model.fit()
    pred3 = results.predict(start=len(x), end=len(x)).iloc[0]
    return pred3/x['close'].iloc[-1]

AR3(fetchOHLC(tokens[0]))


warnings.filterwarnings('ignore')
def AR4(DF):
    x = DF.copy()
    model = ARIMA(x['close'],order=(5,1,3))
    results = model.fit()
    pred4 = results.predict(start=len(x)-5, end=len(x), dynamic=False, typ='levels').iloc[1]
    return pred4/x['close'].iloc[-1]

AR4(fetchOHLC(tokens[0]))


warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
def AR5(DF):
    x = DF.copy()
    model = SARIMAX(x['close'],order=(5,1,0),seasonal_order=(1,0,1,21),enforce_invertibility=False)
    results = model.fit()
    pred5 = results.predict(start=len(x), end=len(x), dynamic=False, typ='levels').iloc[0]
    return pred5/x['close'].iloc[-1]

AR5(fetchOHLC(tokens[0]))


