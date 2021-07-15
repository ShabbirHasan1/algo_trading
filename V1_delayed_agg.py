# from kiteconnect import KiteConnect
# from selenium import webdriver
# import time
# import os


# cwd = os.chdir("C:\Desktop\Learning\Courses\zerodha")

# def autologin():
#     token_path = "api_key.txt"
#     key_secret = open(token_path,'r').read().split()
#     kite = KiteConnect(api_key=key_secret[0])
#     service = webdriver.chrome.service.Service('./chromedriver')
#     service.start()
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options = options.to_capabilities()
#     driver = webdriver.Remote(service.service_url, options)
#     driver.get(kite.login_url())
#     driver.implicitly_wait(10)
#     username = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[1]/input')
#     password = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/input')
#     username.send_keys(key_secret[2])
#     password.send_keys(key_secret[3])
#     driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[4]/button').click()
#     pin = driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[2]/div/input')
#     pin.send_keys(key_secret[4])
#     driver.find_element_by_xpath('/html/body/div[1]/div/div[2]/div[1]/div/div/div[2]/form/div[3]/button').click()
#     time.sleep(10)
#     request_token=driver.current_url.split('request_token=')[1].split('&action')[0]
#     with open('request_token.txt', 'w') as the_file:
#         the_file.write(request_token)
#     driver.quit()

# autologin()

# #generating and storing access token - valid till 6 am the next day
# request_token = open("request_token.txt",'r').read()
# key_secret = open("api_key.txt",'r').read().split()
# kite = KiteConnect(api_key=key_secret[0])
# data = kite.generate_session(request_token, api_secret=key_secret[1])
# with open('access_token.txt', 'w') as file:
#         file.write(data["access_token"])



from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from stocktrends import Renko
import warnings
cwd = os.chdir("C:\Desktop\Learning\Courses\zerodha")
import time
 

access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)


today = datetime.now().day
month = datetime.now().month
# sym = pd.read_csv("NSE_Symbols_1.csv", dtype={'Instrument': int})[['Symbol','Instrument']]
sym = pd.read_csv("NSE_Symbols_Pink.csv", dtype={'Instrument': int})[['Symbol','Instrument','Pink']]
dic1 = sym.set_index('Instrument').to_dict()
tokens = sym['Instrument'].tolist()[:]

strike = pd.read_csv("Strike_Size.csv")
dic2 = strike.set_index('Symbol').to_dict()

os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")

mon = ""
if (month ==1) & (today < 26):
    mon = "21JAN"
elif ((month ==1) & (today >= 26) | (month ==2) & (today < 23)):
    mon = "21FEB"
else:
    mon = "21MAR"

def fetchOHLC(ticker,interval,duration):
    """extracts historical data and outputs in the form of dataframe"""
    data = pd.DataFrame(kite.historical_data(ticker,dt.date.today()-dt.timedelta(duration), dt.date.today(),interval))
    data.date =data.date.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
    return data

def suppo(DF):
    x = DF.copy()
    levels = []
    for i in range(2,x.shape[0]-2):
      if isSupport(x,i):
        l = x['low'][i]
        s =  np.mean(x['high'] - x['low'])
        if np.sum([abs(l-x) < s  for x in levels]) == 0:
          levels.append((i,l))
    return levels[-1][1],levels[-2][1]
    
def ress(DF):
     x = DF.copy()
     levels = []
     for i in range(2,x.shape[0]-2):
      if isResistance(x,i):
        l = x['high'][i]
        s =  np.mean(x['high'] - x['low'])
        if np.sum([abs(l-x) < s  for x in levels]) == 0:
          levels.append((i,l))
     return levels[-1][1],levels[-2][1]

def isSupport(df,i):
  support = df['low'][i] < df['low'][i-1]  and df['low'][i] < df['low'][i+1] and df['low'][i+1] < df['low'][i+2] and df['low'][i-1] < df['low'][i-2]
  return support

def isResistance(df,i):
  resistance = df['high'][i] > df['high'][i-1]  and df['high'][i] > df['high'][i+1] and df['high'][i+1] > df['high'][i+2] and df['high'][i-1] > df['high'][i-2]
  return resistance

def trend(ohlc_df,n=5):
    "function to assess the trend by analyzing each candle"
    df = ohlc_df.copy()
    df["up"] = np.where(df["low"]>=df["low"].shift(1),1,0)
    df["dn"] = np.where(df["high"]<=df["high"].shift(1),1,0)
    if df["close"].iloc[-1] > df["open"].iloc[-1]:
        if df["up"][-1*n:].sum() >= int(0.7*n):
            return "uptrend"
    elif df["open"].iloc[-1] > df["close"].iloc[-1]:
        if df["dn"][-1*n:].sum() >= int(0.7*n):
            return "downtrend"
    else:
        return None

def doji(ohlc_df):    
    """returns dataframe with doji candle column"""
    df = ohlc_df.copy()
    avg_candle_size = abs(df["close"] - df["open"]).median()
    df["doji"] = abs(df["close"] - df["open"]) <=  (0.05 * avg_candle_size)
    return df


def hammer(ohlc_df):    
    """returns dataframe with hammer candle column"""
    df = ohlc_df.copy()
    df["hammer"] = (((df["high"] - df["low"])>3*(df["open"] - df["close"])) & \
                   ((df["close"] - df["low"])/(.001 + df["high"] - df["low"]) > 0.6) & \
                   ((df["open"] - df["low"])/(.001 + df["high"] - df["low"]) > 0.6)) & \
                   (abs(df["close"] - df["open"]) > 0.1* (df["high"] - df["low"]))
    return df

def shooting_star(ohlc_df):    
    """returns dataframe with shooting star candle column"""
    df = ohlc_df.copy()
    df["sstar"] = (((df["high"] - df["low"])>3*(df["open"] - df["close"])) & \
                   ((df["high"] - df["close"])/(.001 + df["high"] - df["low"]) > 0.6) & \
                   ((df["high"] - df["open"])/(.001 + df["high"] - df["low"]) > 0.6)) & \
                   (abs(df["close"] - df["open"]) > 0.1* (df["high"] - df["low"]))
    return df


def maru_bozu(ohlc_df):    
    """returns dataframe with maru bozu candle column"""
    df = ohlc_df.copy()
    avg_candle_size = abs(df["close"] - df["open"]).median()
    df["h-c"] = df["high"]-df["close"]
    df["l-o"] = df["low"]-df["open"]
    df["h-o"] = df["high"]-df["open"]
    df["l-c"] = df["low"]-df["close"]
    df["maru_bozu"] = np.where((df["close"] - df["open"] > 2.3*avg_candle_size) & \
                               (df[["h-c","l-o"]].max(axis=1) < 0.003*avg_candle_size),"maru_bozu_green",
                               np.where((df["open"] - df["close"] > 2.3*avg_candle_size) & \
                               (abs(df[["h-o","l-c"]]).max(axis=1) < 0.003*avg_candle_size),"maru_bozu_red",False))
    df.drop(["h-c","l-o","h-o","l-c"],axis=1,inplace=True)
    return df

def candle_type(ohlc_df):    
    """returns the candle type of the last candle of an OHLC DF"""
    candle = None
    if doji(ohlc_df)["doji"].iloc[-1] == True:
        candle = "doji"    
    if maru_bozu(ohlc_df)["maru_bozu"].iloc[-1] == "maru_bozu_green":
        candle = "maru_bozu_green"       
    if maru_bozu(ohlc_df)["maru_bozu"].iloc[-1] == "maru_bozu_red":
        candle = "maru_bozu_red"        
    if shooting_star(ohlc_df)["sstar"].iloc[-1] == True:
        candle = "shooting_star"        
    if hammer(ohlc_df)["hammer"].iloc[-1] == True:
        candle = "hammer"       
    return candle


def candle_pattern(ohlc_df):    
    """returns the candle pattern identified"""
    pattern = None
    signi = "low"
    avg_candle_size = abs(ohlc_df["close"] - ohlc_df["open"]).mean()
    res= ress(ohlc_df)[0]
    sup= suppo(ohlc_df)[0]
    res_top = (res + 1.8*avg_candle_size)
    res_bottom = (res - 1.8*avg_candle_size)
    sup_bottom = (sup - 1.8*avg_candle_size)
    sup_top = (sup + 1.8*avg_candle_size)
    
    if (sup - 1.8*avg_candle_size) < ohlc_df["close"].iloc[-1] < (sup + 1.8*avg_candle_size):
        signi = "Near Support"
        
    if (res - 1.8*avg_candle_size) < ohlc_df["close"].iloc[-1] < (res + 1.8*avg_candle_size):
        signi = "Near Resistance"
    
    if candle_type(ohlc_df) == 'doji' \
        and ohlc_df["close"].iloc[-1] > ohlc_df["close"].iloc[-2] \
        and ohlc_df["close"].iloc[-1] > ohlc_df["open"].iloc[-1]:
            pattern = "doji_bullish"
    
    if candle_type(ohlc_df) == 'doji' \
        and ohlc_df["close"].iloc[-1] < ohlc_df["close"].iloc[-2] \
        and ohlc_df["close"].iloc[-1] < ohlc_df["open"].iloc[-1]:
            pattern = "doji_bearish" 
            
    if candle_type(ohlc_df) == "maru_bozu_green":
        pattern = "maru_bozu_bullish"
    
    if candle_type(ohlc_df) == "maru_bozu_red":
        pattern = "maru_bozu_bearish"
        
    if trend(ohlc_df,5) == "uptrend" and candle_type(ohlc_df) == "hammer":
        pattern = "hanging_man_bearish"
        
    if trend(ohlc_df,5) == "downtrend" and candle_type(ohlc_df) == "hammer":
        pattern = "hammer_bullish"
        
    if trend(ohlc_df,5) == "uptrend" and candle_type(ohlc_df) == "shooting_star":
        pattern = "shooting_star_bearish"
        
    if trend(ohlc_df,5) == "uptrend" \
        and candle_type(ohlc_df) == "doji" \
        and ohlc_df["high"].iloc[-1] < ohlc_df["close"].iloc[-2] \
        and ohlc_df["low"].iloc[-1] > ohlc_df["open"].iloc[-2]:
        pattern = "harami_cross_bearish"
        
    if trend(ohlc_df,5) == "downtrend" \
        and candle_type(ohlc_df) == "doji" \
        and ohlc_df["high"].iloc[-1] < ohlc_df["open"].iloc[-2] \
        and ohlc_df["low"].iloc[-1] > ohlc_df["close"].iloc[-2]:
        pattern = "harami_cross_bullish"
        
    if trend(ohlc_df,5) == "uptrend" \
        and candle_type(ohlc_df) != "doji" \
        and ohlc_df["open"].iloc[-1] > .9999*ohlc_df["high"].iloc[-2] \
        and .9999*ohlc_df["close"].iloc[-1] < ohlc_df["low"].iloc[-2]:
        pattern = "engulfing_bearish"
        
    if trend(ohlc_df,5) == "downtrend" \
        and candle_type(ohlc_df) != "doji" \
        and ohlc_df["close"].iloc[-1] > .9999*ohlc_df["high"].iloc[-2] \
        and .9999*ohlc_df["open"].iloc[-1] < ohlc_df["low"].iloc[-2]:
        pattern = "engulfing_bullish"
       
    return (signi,pattern,res_top,sup_bottom,res_bottom,sup_top)

def vol_spike(DF, n=20):
    df = DF.copy()
    df['MA_Vol'] = df["volume"].ewm(span=n,min_periods=n).mean()
    df['vol_spike'] = np.where(df['volume']>2.4*df['MA_Vol'],1,0)
    return df

def MACD(DF,a=12,b=26,c=9):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.drop(['MA_Fast','MA_Slow'], axis = 1, inplace=True)
    # df.dropna(inplace=True)
    return df

def bigger_candles(DF):
    # 4.6 can be changed to either higher or lower
    x = DF.copy()
    avg_candle_size = abs(x["close"] - x["open"]).median()
    b1 = 1*(abs(x["close"].iloc[-1] - x["open"].iloc[-1]) >=  (5.5 * avg_candle_size))
    b2 = 1*(abs(x["close"].iloc[-2] - x["open"].iloc[-2]) >=  (5.6 * avg_candle_size))
    b3 = 1*(abs(x["close"].iloc[-3] - x["open"].iloc[-3]) >=  (5.7 * avg_candle_size))
    b4 = 1*(abs(x["close"].iloc[-4] - x["open"].iloc[-4]) >=  (5.8 * avg_candle_size))
    b5 = 1*(abs(x["close"].iloc[-5] - x["open"].iloc[-5]) >=  (5.9 * avg_candle_size))
    return (b1+b2+b3+b4+b5)

def last_candle(DF):
    # 4.6 can be changed to either higher or lower
    x = DF.copy()
    avg_candle_size = abs(x["close"] - x["open"]).median()
    b1 = abs(x["close"].iloc[-1] - x["open"].iloc[-1]) / (avg_candle_size)
    return b1

def gapup(DF):
    x = DF.copy()
    x['date'] = pd.to_datetime(x['date'])
    today = x['date'].dt.day == x['date'].dt.day.unique()[-1]
    yes = x['date'].dt.day == x['date'].dt.day.unique()[-2]
    yes_close = x[yes]['close'].iloc[-1]
    today_open = x[today]['close'].iloc[0]
    per = 100*((today_open -yes_close )/yes_close)
    return per
    
    
def vol_blast(DF):
    x = DF.copy()
    x['diff'] = np.where(x['volume']>=x['volume'].nlargest(2).iloc[-1],1,0)
    c = pd.to_datetime(x['date']).dt.day == today
    x = x[c]
    x['ver'] = x['diff'].diff()
    days=1
    if len(x.loc[x['ver']==1])>0:
        days = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).days
    if (sum(x['volume'])>0) & (days ==0):
        t=(datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).seconds/60
    else:
        t=999
    if (x['diff'].iloc[0]==1) & (sum(x['diff'])) ==1:
        t = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.iloc[0]['date'],'%Y-%m-%d %H:%M')).seconds/60
    return t

def grn(DF):
    x = DF.copy()
    t = 1*((x['close'].iloc[-1] - x['open'].iloc[-1]) >0)
    return t

def fifteen(DF):
    x = DF.copy()
    #x.reset_index(inplace=True)
    x['date'] = pd.to_datetime(x['date'])
    c1 = x['date'].dt.hour == 9
    c2 = x['date'].dt.minute <= 30
    c3 = x['date'].dt.day == today
    c = c1 & c2 & c3
    upper = x[c]['high'].max()
    # cp = candle_pattern(x)
    # upper = max(upper, cp[2])
    lower = x[c]['low'].min()
    range_cond = 1*(100*((upper - lower)/lower)<1)
    x['diff'] = np.where((x['close'] - upper)>0,1,0)
    x['ver'] = x['diff'].diff()
    x = x[c3]
    days = 1
    if len(x.loc[x['ver']==1])>0:
        days = (x['date'].iloc[-1] - x.loc[x['ver']==1].iloc[-1]['date']).days
    if (len(x.loc[x['ver']==1])>0) & (days==0)>0:
        t=((x['date'].iloc[-1] - x.loc[x['ver']==1].iloc[-1]['date']).seconds)/60
    else:
        t=999
    return t,range_cond

def fifteen_down(DF):
    x = DF.copy()
    #x.reset_index(inplace=True)
    x['date'] = pd.to_datetime(x['date'])
    c1 = x['date'].dt.hour == 9
    c2 = x['date'].dt.minute <= 30
    c3 = x['date'].dt.day == today
    c = c1 & c2 & c3
    upper = x[c]['high'].max()
    # cp = candle_pattern(x)
    # upper = max(upper, cp[2])
    lower = x[c]['low'].min()
    # lower = min(lower,cp[3])
    range_cond = 1*(100*((upper - lower)/lower)<1)
    x['diff'] = np.where((lower -x['close'])>0,1,0)
    x['ver'] = x['diff'].diff()
    x = x[c3]
    days = 1
    if len(x.loc[x['ver']==1])>0:
        days = (x['date'].iloc[-1] - x.loc[x['ver']==1].iloc[-1]['date']).days
    if (len(x.loc[x['ver']==1])>0) & (days==0)>0:
        t=((x['date'].iloc[-1] - x.loc[x['ver']==1].iloc[-1]['date']).seconds)/60
    else:
        t=999
    return t,range_cond

def vwap(DF):
    x = DF.copy()
    c = pd.to_datetime(x['date']).dt.day == today
    x = x[c]
    x['vwap'] = (x['volume']*(x['high']+x['low']+x['close'])/3).cumsum()/x['volume'].cumsum()
    # k = 1*((x['close'].iloc[-4]<x['vwap'].iloc[-4]) & (x['close'].iloc[-3]<x['vwap'].iloc[-3]) & (x['close'].iloc[-2]<x['vwap'].iloc[-2]) & (x['close'].iloc[-1]>x['vwap'].iloc[-1]))
    x['diff'] = np.where(x['close']-x['vwap']>0,1,0)
    x['ver'] = x['diff'].diff()
    days = 1
    if len(x.loc[x['ver']==1])>0:
        days = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).days
    if (len(x.loc[x['ver']==1])>0) & (days==0):
        t= (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).seconds/60
    else:
        t=999
    if (x['diff'].iloc[0]==1) & (sum(x['diff'])==1):
        t = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.iloc[0]['date'],'%Y-%m-%d %H:%M')).seconds/60
    return t

def vwap_down(DF):
    x = DF.copy()
    c = pd.to_datetime(x['date']).dt.day == today
    x = x[c]
    x['vwap'] = (x['volume']*(x['high']+x['low']+x['close'])/3).cumsum()/x['volume'].cumsum()
    # k = 1*((x['close'].iloc[-4]<x['vwap'].iloc[-4]) & (x['close'].iloc[-3]<x['vwap'].iloc[-3]) & (x['close'].iloc[-2]<x['vwap'].iloc[-2]) & (x['close'].iloc[-1]>x['vwap'].iloc[-1]))
    x['diff'] = np.where(x['vwap'] - x['close']>0,1,0)
    x['ver'] = x['diff'].diff()
    days = 1
    if len(x.loc[x['ver']==1])>0:
        days = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).days
    if (len(x.loc[x['ver']==1])>0) & (days==0):
        t= (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.loc[x['ver']==1].iloc[-1]['date'],'%Y-%m-%d %H:%M')).seconds/60
    else:
        t=999
    if (x['diff'].iloc[0]==1) & (sum(x['diff'])==1):
        t = (datetime.strptime(x['date'].iloc[-1],"%Y-%m-%d %H:%M") - datetime.strptime(x.iloc[0]['date'],'%Y-%m-%d %H:%M')).seconds/60
    return t

def bollBnd(DF,n=25):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['close'].rolling(n).mean()
    #df["MA"] = df['close'].ewm(span=n,min_periods=n).mean()
    df["BB_up"] = df["MA"] + 2*df['close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    return df

def bollCond(DF, n=20):
    df = DF.copy()
    df['MA_Boll'] = df["BB_width"].ewm(span=n,min_periods=n).mean()
    df['Boll_spike'] = np.where(df['BB_width']>1.88*df['MA_Boll'],1,0)
    return df

def rsi(df, n=14):
    "function to calculate RSI"
    delta = df["close"].diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[n-1]] = np.mean( u[:n]) # first value is average of gains
    u = u.drop(u.index[:(n-1)])
    d[d.index[n-1]] = np.mean( d[:n]) # first value is average of losses
    d = d.drop(d.index[:(n-1)])
    rs = u.ewm(com=n,min_periods=n).mean()/d.ewm(com=n,min_periods=n).mean()
    return 100 - 100 / (1+rs)
 
def main(): 
    res1_dic = {}
    sup1_dic = {}
    res2_dic = {}
    sup2_dic = {}
    pattern_dic = {}
    signi_dic = {}
    res_top_dic = {}
    res_bottom_dic = {}
    sup_bottom_dic = {}
    sup_top_dic = {}
    vol_cond_dic = {}
    Macd_buy_dic= {}
    vol_blast_dic = {}
    grn_dic = {}
    gapup_dic = {}
    close = {}
    bigger_candles_dic = {}
    fifteen_dic = {}
    fifteen_down_dic = {}
    range_cond = {}
    vwap_dic = {}
    vwap_down_dic = {}
    boll_cond_dic = {}
    trend_dic = {}
    rsi_dic = {}
    last_candle_dic = {}
    for ticker in tokens:
        try:
            ohlc = fetchOHLC(ticker,"3minute",6)[:-1]
            # name = datetime.strptime(ohlc['date'].iloc[-1],"%Y-%m-%d %H:%M").strftime('%H_%M')
            if vol_spike(ohlc)['vol_spike'].iloc[-1]==0:
                continue
            vol_cond_dic[dic1['Symbol'][ticker]] =vol_spike(ohlc)['vol_spike'].iloc[-1]
            vwap_dic[dic1['Symbol'][ticker]] = vwap(ohlc)
            last_candle_dic[dic1['Symbol'][ticker]] = last_candle(ohlc)
            rsi_dic[dic1['Symbol'][ticker]] = rsi(ohlc).iloc[-1]
            vwap_down_dic[dic1['Symbol'][ticker]] = vwap_down(ohlc)
            sup1_dic[dic1['Symbol'][ticker]],sup2_dic[dic1['Symbol'][ticker]] = suppo(ohlc)
            res1_dic[dic1['Symbol'][ticker]],res2_dic[dic1['Symbol'][ticker]] = ress(ohlc)
            vol_blast_dic[dic1['Symbol'][ticker]] = vol_blast(ohlc)
            grn_dic[dic1['Symbol'][ticker]] = grn(ohlc)
            gapup_dic[dic1['Symbol'][ticker]] = gapup(ohlc)
            close[dic1['Symbol'][ticker]] = ohlc['close'].iloc[-1]
            bigger_candles_dic[dic1['Symbol'][ticker]] = bigger_candles(ohlc)
            cp = candle_pattern(ohlc)
            signi_dic[dic1['Symbol'][ticker]] = cp[0]
            pattern_dic[dic1['Symbol'][ticker]] = cp[1]
            res_top_dic[dic1['Symbol'][ticker]] = cp[2]
            sup_bottom_dic[dic1['Symbol'][ticker]] = cp[3]
            res_bottom_dic[dic1['Symbol'][ticker]] = cp[4]
            sup_top_dic[dic1['Symbol'][ticker]] = cp[5]
            f = fifteen(ohlc) 
            f_down = fifteen_down(ohlc)
            fifteen_dic[dic1['Symbol'][ticker]]= f[0]
            fifteen_down_dic[dic1['Symbol'][ticker]]= f_down[0]
            range_cond[dic1['Symbol'][ticker]] = f[1]
            Macd_buy_dic[dic1['Symbol'][ticker]] = 1*(MACD(ohlc)['MACD'].iloc[-1]>MACD(ohlc)['Signal'].iloc[-1])
            trend_dic[dic1['Symbol'][ticker]] = trend(ohlc)
            ohlc = bollBnd(ohlc)
            boll_cond_dic[dic1['Symbol'][ticker]] =bollCond(ohlc)['Boll_spike'].iloc[-1]
        except:
            pass
    final = pd.DataFrame(vol_cond_dic.items())
    try:
        df = pd.read_csv("OC.csv")
    except:
        pass
    try:
        news = pd.read_csv("News.csv")
    except:
        pass
    if len(final)>0:
        final.columns = ['index','Vol_spike']
        final['Macd_buy'] = final['index'].map(Macd_buy_dic)
        final['last_candle'] = final['index'].map(last_candle_dic)
        final['Boll_spike'] = final['index'].map(boll_cond_dic)
        final['trend'] = final['index'].map(trend_dic)
        final['rsi'] = final['index'].map(rsi_dic)
        final['resistance'] = final['index'].map(res1_dic)
        final['support'] = final['index'].map(sup1_dic)
        final['resistance_top'] = final['index'].map(res_top_dic)
        final['resistance_bottm'] = final['index'].map(res_bottom_dic)
        final['support_top'] = final['index'].map(sup_top_dic)
        final['support_bottm'] = final['index'].map(sup_bottom_dic)
        final['pattern'] = final['index'].map(pattern_dic)
        final['significance'] = final['index'].map(signi_dic)
        final['green'] = final['index'].map(grn_dic)
        final['range_cond'] = final['index'].map(range_cond)
        final['gap_up'] = final['index'].map(gapup_dic)
        final['green'] = final['index'].map(grn_dic)
        final['vol_blast'] = final['index'].map(vol_blast_dic)
        final['vwap_cross'] = final['index'].map(vwap_dic)
        final['vwap_cross_down'] = final['index'].map(vwap_down_dic)
        final['fifteen_high_cross'] = final['index'].map(fifteen_dic)
        final['fifteen_down_cross'] = final['index'].map(fifteen_down_dic)
        final['close'] = final['index'].map(close)
        final['bigger_candles'] = final['index'].map(bigger_candles_dic)
        final['brick'] = final['index'].map(dic2['Brick'])
        final['lower'] = (final['close']-(final['close'] % final['brick'])).astype("Float32")
        final['upper'] = final['lower']+ final['brick']
        final['lower'] = final['lower'].apply(lambda x: str(x))
        final['upper'] = final['upper'].apply(lambda x: str(x))
        final['lower'] = np.where(final['lower'].apply(lambda x: x[-1])=="5",final['lower'].apply(lambda x: x[:-2])+".5",final['lower'].apply(lambda x: x[:-2]))
        final['upper'] = np.where(final['upper'].apply(lambda x: x[-1])=="5",final['upper'].apply(lambda x: x[:-2])+".5",final['upper'].apply(lambda x: x[:-2]))
        final['tick_lower'] = np.where((1-final['brick'].isnull()), final['index'].apply(lambda x: x[:-3]) + mon + final['lower']+"PE",0)
        final['tick_upper'] = np.where((1-final['brick'].isnull()), final['index'].apply(lambda x: x[:-3]) + mon + final['upper']+"CE",0)
        final['Buy'] = np.where((final['green']==1) & (final['Macd_buy']==1) & (final['trend']=='uptrend') & ((final['close']<final['resistance_bottm']) | (final['close']>final['resistance_top'])),1,0)
        final['Sell'] = np.where((final['green']==0) & (final['Macd_buy']==0) & (final['trend']=='downtrend') & ((final['close']>final['support_top']) | (final['close']<final['support_bottm'])),1,0)
        final['res_break'] = np.where((final['close']>final['resistance_top']),1,0)
        final['sup_break'] = np.where((final['close']<final['support_bottm']),1,0)
        final = final[final['bigger_candles']<=1]
        buy = final[final['Buy']==1]
        sell = final[final['Sell']==1]
        buy = buy[['index','gap_up','vol_blast','vwap_cross','fifteen_high_cross','Boll_spike','pattern','significance','close','bigger_candles','tick_upper','res_break','Macd_buy','rsi','last_candle']]
        sell = sell[['index','gap_up','vol_blast','vwap_cross_down','fifteen_down_cross','Boll_spike','pattern','significance','close','bigger_candles','tick_lower','sup_break','Macd_buy','rsi','last_candle']]
        buy.rename(columns = {'vwap_cross':'vwap','fifteen_high_cross':'fifteen_cross','tick_upper':'tick','res_break':'level_break'}, inplace = True) 
        sell.rename(columns = {'vwap_cross_down':'vwap','fifteen_down_cross':'fifteen_cross','tick_lower':'tick','sup_break':'level_break'}, inplace = True)
        buy['Action'] = "Buy"
        sell['Action'] = "Sell"
        final = buy.append(sell)
        news_dic = news[['Symbol','News']].set_index('Symbol').to_dict()
        ult_res = df[['index','Resistance']].set_index('index').to_dict()
        ult_sup = df[['index','Support']].set_index('index').to_dict()
        ult_act = df[['index','Action']].set_index('index').to_dict()
        final['OC_Resistance'] = final['index'].map(ult_res['Resistance'])
        final['OC_Support'] = final['index'].map(ult_sup['Support'])
        final['OC_Action'] = final['index'].map(ult_act['Action'])
        final['OC_len'] = .16*(final['OC_Resistance'] - final['OC_Support'])
        final['Near_OC_Resistance_S'] = np.where((final['OC_Resistance'] - final['OC_len']) <=final['close'],1,0)
        final['Near_OC_Support_B'] = np.where(final['close']<= (final['OC_Support'] + final['OC_len']),1,0)
        final['dyn_resistance'] = final['index'].map(res1_dic)
        final['dyn_support'] = final['index'].map(sup1_dic)
        final['dyn_len'] = .16*abs((final['dyn_resistance'] - final['dyn_support']))
        final = final[((final['Action'] == "Buy") & ((final['close']<final['OC_Resistance'] - final['OC_len']) | (final['close']>final['OC_Resistance'] + final['OC_len']))) | ((final['Action'] == "Sell") & ((final['close']>final['OC_Support'] + final['OC_len']) | (final['close']<final['OC_Support'] - final['OC_len'])))]
        final = final[((final['Action'] == "Buy") & ((final['close']<final['dyn_resistance'] - final['dyn_len']) | (final['close']>final['dyn_resistance'] + final['dyn_len']))) | ((final['Action'] == "Sell") & ((final['close']>final['dyn_support'] + final['dyn_len']) | (final['close']<final['dyn_support'] - final['dyn_len'])))]
        final = final[['index','Action','gap_up','vol_blast','vwap','fifteen_cross','Boll_spike','pattern','close','OC_Resistance','dyn_resistance','OC_Support','dyn_support','Near_OC_Resistance_S','Near_OC_Support_B','rsi','last_candle','tick']]
        if (t1.minute < 30) & (t1.hour == 9):
            final = final[final['vol_blast']<=0]
        else:
            final = final[(final['vol_blast']<=0) | (final['vwap']<=0) | (final['fifteen_cross']<=0)]
        name = "V1_delayed_"+str(t1.strftime('%H_%M')) + ".csv"
        ind1 = final[(final['Action']=="Buy") & (final['Near_OC_Resistance_S']==1)].index
        ind2 = final[(final['Action']=="Sell") & (final['Near_OC_Support_B']==1)].index
        final.drop(ind1, inplace = True)
        final.drop(ind2, inplace = True)
        # final['News'] = final['index'].map(news_dic['News'])
        if len(final) >0:
            final.to_csv(name, index = False)
    return final

i = 1
while True:
    t1 = datetime.now()
    final = main()
    i = i +1
    t2 = datetime.now()
    print(f"Loop Number------>{i}")
    #time.sleep(max(1,2*60 - (t2-t1).seconds))
    time.sleep(1)

