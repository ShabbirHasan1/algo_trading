import requests
import pandas as pd
import os
import datetime as dt
from datetime import datetime
import time
import json
import numpy as np
from selenium import webdriver
import random
import yfinance as yf

df = {}

strike =50


headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
     'accept-language':'en-US,en;q=0.9,bn;q=0.8','accept-encoding':'gzip, deflate, br'}

url_oc = "https://www.nseindia.com/option-chain"
url = f"https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"

def oc():
    try:
        session = requests.Session()
        request = session.get(url_oc, headers=headers, timeout=8)
        cookies = dict(request.cookies)
        response = session.get(url, headers=headers, timeout=5, cookies=cookies).json()
        data = response['filtered']['data']
        data=pd.DataFrame(data)
        data['Call_OI'] = data['CE'].apply(lambda x: x['openInterest'] if x==x else "")
        data['Call_changein_OI'] = data['CE'].apply(lambda x: x['changeinOpenInterest'] if x==x else "")
        data['Put_OI'] = data['PE'].apply(lambda x: x['openInterest'] if x==x else "")
        data['Put_changein_OI'] = data['PE'].apply(lambda x: x['changeinOpenInterest'] if x==x else "")
        data = data.rename(columns={'strikePrice': 'StrikePrice'})
        data = data[['StrikePrice','Call_OI','Call_changein_OI','Put_OI','Put_changein_OI']]
        data = data.apply(pd.to_numeric,errors='coerce')
        symbol = "NIFTY"
        ohlc5 = yf.download(tickers = "^NSEI",period = "5d",interval = "5m")
        # ohlc1 = yf.download(tickers = "^NSEI",period = "5d",interval = "1m")
        close = ohlc5['Close'].iloc[-1]
        time_data = ohlc5.reset_index()['Datetime'].iloc[-1]
        up = data[data['StrikePrice']>=close][:6]
        res = up['StrikePrice'].loc[up['Call_OI'].idxmax()]
        call_chng_oi_up = sum(up['Call_changein_OI'])
        put_chng_oi_up = sum(up['Put_changein_OI'])
        down = data[data['StrikePrice']<close][-6:]
        sup = down['StrikePrice'].loc[down['Put_OI'].idxmax()]
        call_chng_oi_down = sum(down['Call_changein_OI'])
        put_chng_oi_down = sum(down['Put_changein_OI'])
        calls = call_chng_oi_up + call_chng_oi_down
        puts = put_chng_oi_up + put_chng_oi_down
        Action = ""
        if puts - calls > 0:
            Action = "Buy"
        else:
            Action = "Sell"
        cols = ['index','Resistance','Support','Call_chng','Put_chng','diff','time','Action']
        d = [symbol,res,sup,calls,puts,puts-calls,time_data,Action]
        df[i] = pd.DataFrame([d])
        df[i].columns = cols
        print(f"Call Change:{calls}, Put Change:{puts}, Diff:{puts-calls}, Res:{res},Supp:{sup}")
        return df[i]
    
    except:
        pass


i=1
while True:
    t1 = datetime.now()
    final = oc()
    # data = pd.concat(df[symbol] for symbol in df.keys())
    # print(data)
    i = i +1
    t2 = datetime.now()
    # print(f"Loop Number------>{i}")
    #time.sleep(max(1,2*60 - (t2-t1).seconds))
    time.sleep(120 - (t2-t1).seconds)

