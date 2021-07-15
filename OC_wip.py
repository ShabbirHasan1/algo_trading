#https://www.nseindia.com/api/option-chain-equities?symbol=BAJAJ-AUTO
from kiteconnect import KiteConnect
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

cwd = os.chdir("C:\Desktop\Learning\Courses\zerodha")

access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)

sym = pd.read_csv("NSE_Symbols_Pink.csv", dtype={'Instrument': int})[['Symbol','Instrument','Pink']]
dic1 = sym.set_index('Symbol').to_dict()
symbols = sym['Symbol'].tolist()[:-2]

strike = pd.read_csv("Strike_Size.csv")
# strike['Brick'] = round(10*strike['Brick'].astype(int)/10,1)
dic2 = strike.set_index('Symbol').to_dict()

cwd = os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")

def fetchOHLC(ticker,interval = "minute",duration=4):
    """extracts historical data and outputs in the form of dataframe"""
    data = pd.DataFrame(kite.historical_data(ticker,dt.date.today()-dt.timedelta(duration), dt.date.today(),interval))
    data.date =data.date.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    return data


headers={'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
     'accept-language':'en-US,en;q=0.9,bn;q=0.8','accept-encoding':'gzip, deflate, br'}

url_oc = "https://www.nseindia.com/option-chain"
url = f"https://www.nseindia.com/api/option-chain-equities?symbol="


def main():
    oc = {}
    for symbol in symbols[:]:
        try:
            session = requests.Session()
            request = session.get(url_oc, headers=headers, timeout=8)
            cookies = dict(request.cookies)
            response = session.get(url +symbol[:-3], headers=headers, timeout=5, cookies=cookies).json()
            data = response['filtered']['data']
        except:
            continue
        data=pd.DataFrame(data)
        data['Call_OI'] = data['CE'].apply(lambda x: x['openInterest'] if x==x else "")
        data['Call_changein_OI'] = data['CE'].apply(lambda x: x['changeinOpenInterest'] if x==x else "")
        data['Put_OI'] = data['PE'].apply(lambda x: x['openInterest'] if x==x else "")
        data['Put_changein_OI'] = data['PE'].apply(lambda x: x['changeinOpenInterest'] if x==x else "")
        data = data.rename(columns={'strikePrice': 'StrikePrice'})
        data = data[['StrikePrice','Call_OI','Call_changein_OI','Put_OI','Put_changein_OI']]
        data = data.apply(pd.to_numeric,errors='coerce')
        try:
            d = fetchOHLC(dic1['Instrument'][symbol])
        except:
            continue
        close = d['close'].iloc[-1]
        time_data = t1
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
        df = pd.DataFrame([d])
        oc[symbol]=df
        time.sleep(random.randint(10,16))
        print(symbol[:-3] + " is Done")
    
    df = pd.concat(oc[symbol] for symbol in oc.keys())
    cols = ['index','Resistance','Support','Call_chng','Put_chng','diff','time','Action']
    df.columns = cols
    try:
        df1 = pd.read_csv("OC.csv")
    except:
        df1 = pd.DataFrame()
    try:
        df3 = pd.read_csv("OC_trend.csv").set_index("index")
        df3 = df3.loc[:, df3.columns != 'Action']
    except:
        df3 = pd.DataFrame()        
    df2 = df1.append(df)
    df2['time'] = pd.to_datetime(df2['time'])
    df2 = df2.sort_values(by = ['time'], ascending = [True])
    df2 = df2.drop_duplicates(['index'], keep='last')
    try:
        k = pd.pivot_table(df2,index =['index'],columns = ['time'], values = 'diff', aggfunc = np.sum)
        k2 = pd.concat([df3,k], axis=1)
        k2 = k2.loc[:, k2.isnull().mean() < .16].iloc[:,-4:]
        # k2['Sell']= np.where((k2.iloc[:,-4] > k2.iloc[:,-3]) & (k2.iloc[:,-3] > k2.iloc[:,-2]) & (k2.iloc[:,-2] > k2.iloc[:,-1]),"Sell","")
        # k2['Buy']= np.where((k2.iloc[:,-5] < k2.iloc[:,-4]) & (k2.iloc[:,-4] < k2.iloc[:,-3]) & (k2.iloc[:,-3] < k2.iloc[:,-2]),"Buy","")
        # k2['Action'] = k2['Sell'] + k2['Buy']
        # k2 = k2.drop(['Sell','Buy'], axis = 1) 
        k2.to_csv("OC_trend.csv")
    except:
        pass
    # print("Buy........>")
    # print(list(k2[k2['Action']=="Buy"].index))
    # print("Sell........>")
    # print(list(k2[k2['Action']=="Sell"].index))
    df2 = df2[['index','Resistance','Support','Call_chng','Put_chng','diff','time','Action']]
    df2.to_csv("OC.csv",index = False)
    return df2


i = 0
while True:
    t1 = fetchOHLC(dic1['Instrument'][symbols[0]])['date'].iloc[-1]
    final = main()
    i = i +1
    t2 = datetime.now()
    print(f"Loop Number------>{i}")
    #time.sleep(max(1,2*60 - (t2-t1).seconds))
    time.sleep(random.randint(12*60,15*60))

