import bitmex
import pandas as pd
import numpy as np
import math
import os.path
import time
from bitmex import bitmex
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib

os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")

bitmex_api_key = 'DzMrq4INMxS-r6lJEyfCExSu'    #Enter your own API-key here
bitmex_api_secret = 'QCcxM3lHW4FtVWzc4WkCuhfOkdxb54erp3kxtQJz6SUMkN2U' #Enter your own API-secret here

binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750

bitmex_client = bitmex(test=False, api_key=bitmex_api_key, api_secret=bitmex_api_secret)



tokens = ["XBTUSD","ETHUSD","XRPUSD","DOGE","BCHUSD","DOT","ADA"]
start=pd.Timestamp("2021-02-09")
end=pd.Timestamp.now()
number_of_minutes_needed = 250

today = datetime.now().day
month = datetime.now().month


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
            past_minutes_data = bitmex_client.Trade.Trade_getBucketed(binSize='1m', count=number_of_minutes_needed, symbol=ticker,startTime=pd.Timestamp("2021-01-01"),endTime=end,reverse=True).result()[0]
            past_minutes_data = reversed(past_minutes_data)
            past_minutes_data_list = []
            for past_minute_data in past_minutes_data:
                processed_min_data = {}
                timestamp_minute = str(past_minute_data["timestamp"]).split(':')[0] + ":" + str(past_minute_data["timestamp"]).split(':')[1] + ":00"
                processed_min_data["index"] = past_minute_data["symbol"]
                processed_min_data["date"] =  datetime.strptime(timestamp_minute, '%Y-%m-%d %H:%M:%S') + timedelta(hours=5, minutes=30)
                processed_min_data["date"] = processed_min_data["date"].strftime("%Y-%m-%d %H:%M")
                processed_min_data["open"] = past_minute_data["open"]
                processed_min_data["close"] = past_minute_data["close"]
                processed_min_data["volume"] = past_minute_data["volume"]
                processed_min_data["high"] = past_minute_data["high"]
                processed_min_data["low"] = past_minute_data["low"]
                past_minutes_data_list.append(processed_min_data)
            ohlc = pd.DataFrame(past_minutes_data_list)
            vol_cond_dic[ticker] =vol_spike(ohlc)['vol_spike'].iloc[-1]
            vwap_dic[ticker] = vwap(ohlc)
            last_candle_dic[ticker] = last_candle(ohlc)
            rsi_dic[ticker] = rsi(ohlc).iloc[-1]
            vwap_down_dic[ticker] = vwap_down(ohlc)
            sup1_dic[ticker],sup2_dic[ticker] = suppo(ohlc)
            res1_dic[ticker],res2_dic[ticker] = ress(ohlc)
            vol_blast_dic[ticker] = vol_blast(ohlc)
            grn_dic[ticker] = grn(ohlc)
            gapup_dic[ticker] = gapup(ohlc)
            close[ticker] = ohlc['close'].iloc[-1]
            bigger_candles_dic[ticker] = bigger_candles(ohlc)
            cp = candle_pattern(ohlc)
            signi_dic[ticker] = cp[0]
            pattern_dic[ticker] = cp[1]
            res_top_dic[ticker] = cp[2]
            sup_bottom_dic[ticker] = cp[3]
            res_bottom_dic[ticker] = cp[4]
            sup_top_dic[ticker] = cp[5]
            f = fifteen(ohlc) 
            f_down = fifteen_down(ohlc)
            fifteen_dic[ticker]= f[0]
            fifteen_down_dic[ticker]= f_down[0]
            range_cond[ticker] = f[1]
            Macd_buy_dic[ticker] = 1*(MACD(ohlc)['MACD'].iloc[-1]>MACD(ohlc)['Signal'].iloc[-1])
            trend_dic[ticker] = trend(ohlc)
            ohlc = bollBnd(ohlc)
            boll_cond_dic[ticker] =bollCond(ohlc)['Boll_spike'].iloc[-1]
        except:
            pass
    final = pd.DataFrame(vol_cond_dic.items())
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
        final['Buy'] = np.where((final['green']==1) & (final['Macd_buy']==1) & (final['trend']=='uptrend') & ((final['close']<final['resistance_bottm']) | (final['close']>final['resistance_top'])),1,0)
        final['Sell'] = np.where((final['green']==0) & (final['Macd_buy']==0) & (final['trend']=='downtrend') & ((final['close']>final['support_top']) | (final['close']<final['support_bottm'])),1,0)
        final['res_break'] = np.where((final['close']>final['resistance_top']),1,0)
        final['sup_break'] = np.where((final['close']<final['support_bottm']),1,0)
        final = final[final['bigger_candles']<=1]
        buy = final[final['Buy']==1]
        sell = final[final['Sell']==1]
        buy = buy[['index','gap_up','vol_blast','vwap_cross','fifteen_high_cross','Boll_spike','pattern','significance','close','bigger_candles','res_break','Macd_buy','rsi','last_candle']]
        sell = sell[['index','gap_up','vol_blast','vwap_cross_down','fifteen_down_cross','Boll_spike','pattern','significance','close','bigger_candles','sup_break','Macd_buy','rsi','last_candle']]
        buy.rename(columns = {'vwap_cross':'vwap','fifteen_high_cross':'fifteen_cross','res_break':'level_break'}, inplace = True) 
        sell.rename(columns = {'vwap_cross_down':'vwap','fifteen_down_cross':'fifteen_cross','sup_break':'level_break'}, inplace = True)
        buy['Action'] = "Buy"
        sell['Action'] = "Sell"
        final = buy.append(sell)
        # final['dyn_resistance'] = final['index'].map(res1_dic)
        # final['dyn_support'] = final['index'].map(sup1_dic)
        # final['dyn_len'] = .16*abs((final['dyn_resistance'] - final['dyn_support']))
        # final = final[((final['Action'] == "Buy") & ((final['close']<final['dyn_resistance'] - final['dyn_len']) | (final['close']>final['dyn_resistance'] + final['dyn_len']))) | ((final['Action'] == "Sell") & ((final['close']>final['dyn_support'] + final['dyn_len']) | (final['close']<final['dyn_support'] - final['dyn_len'])))]
        # final = final[['index','Action','gap_up','vol_blast','vwap','fifteen_cross','Boll_spike','pattern','close','dyn_resistance','dyn_support','rsi','last_candle','tick']]
        # final = final[final['vol_blast']<=0]
        if len(final)>0:
            name = "bit_"+str(t1.strftime('%H_%M')) + ".csv"
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
    time.sleep(15)
