import datetime as dt
import yfinance as yf
import pandas as pd
import os
os.chdir(f"C:\Desktop\Learning\Courses\Algo_Final")
import time
import random

sym = pd.read_csv("NSE_Symbols.csv")['Symbol'].tolist()
stocks = sym[:700]
start = dt.datetime.today()-dt.timedelta(750)
end =dt.datetime.today()
cl_price = pd.DataFrame() # empty dataframe which will be filled with closing prices of each stock
ohlcv_data = {} # empty dictionary which will be filled with ohlcv dataframe for each ticker
for ticker in stocks:
    print(ticker)
    ohlcv_data[ticker] = yf.download(ticker,start,end)
    ohlcv_data[ticker]['Symbol'] = ticker
    time.sleep(.01*random.randint(100,150))
data = pd.concat([ohlcv_data[ticker] for ticker in stocks],axis=0)
data.to_csv("Nifty_500_comp_price.csv")  

time.sleep(10)

import numpy as np
from stocktrends import Renko
import statsmodels.api as sm
import talib
import datetime
df = pd.read_csv("Nifty_500_comp_price.csv", index_col = 0)


tickers = df['Symbol'].unique().tolist()


def MACD(DF,a=12,b=26,c=9):
    """function to calculate MACD
       typical values a = 12; b =26, c =9"""
    df = DF.copy()
    df["MA_Fast"]=df["Adj Close"].ewm(span=a,min_periods=a).mean()
    df["MA_Slow"]=df["Adj Close"].ewm(span=b,min_periods=b).mean()
    df["MACD"]=df["MA_Fast"]-df["MA_Slow"]
    df["Signal"]=df["MACD"].ewm(span=c,min_periods=c).mean()
    df.drop(['MA_Fast','MA_Slow'], axis = 1, inplace=True)
    # df.dropna(inplace=True)
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = MACD(df[df['Symbol']==ticker])

df = pd.concat([dict[ticker] for ticker in tickers],axis=0)
df['MACD_Action'] = 1*(df['MACD'] > 1.35*df['Signal'])

def Mac_Buy(DF):
    df = DF.copy()
    df['MACD_Buy'] = np.where(df['MACD_Action'].rolling(15).sum()>14,1,0)
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = Mac_Buy(df[df['Symbol']==ticker])

df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

def ATR(DF,n=14):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

dict ={}
for ticker in tickers:
    dict[ticker] = ATR(df[df['Symbol']==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)


def BollBnd(DF,n=20):
    "function to calculate Bollinger Band"
    df = DF.copy()
    df["MA"] = df['Adj Close'].rolling(n).mean()
    df["BB_up"] = df["MA"] + 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_dn"] = df["MA"] - 2*df['Adj Close'].rolling(n).std(ddof=0) #ddof=0 is required since we want to take the standard deviation of the population and not sample
    df["BB_width"] = df["BB_up"] - df["BB_dn"]
    df["BB_width_per"] = 100*(df["BB_width"]/df["MA"])
    # df.dropna(inplace=True)
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = BollBnd(df[df['Symbol']==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

def RSI(DF,n=14):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Adj Close'] - df['Adj Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(n).mean().tolist()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    df.drop(['delta','gain','loss','avg_gain','avg_loss','RS'],axis=1, inplace = True)
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = RSI(df[df['Symbol']==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

df['RSI_Signal'] = np.where(df['RSI']<=30,1,0)


def ADX(DF,n=14):
    "function to calculate ADX"
    df2 = DF.copy()
    df2['TR'] = ATR(df2,n)['TR'] #the period parameter of ATR function does not matter because period does not influence TR calculation
    df2['DMplus']=np.where((df2['High']-df2['High'].shift(1))>(df2['Low'].shift(1)-df2['Low']),df2['High']-df2['High'].shift(1),0)
    df2['DMplus']=np.where(df2['DMplus']<0,0,df2['DMplus'])
    df2['DMminus']=np.where((df2['Low'].shift(1)-df2['Low'])>(df2['High']-df2['High'].shift(1)),df2['Low'].shift(1)-df2['Low'],0)
    df2['DMminus']=np.where(df2['DMminus']<0,0,df2['DMminus'])
    TRn = []
    DMplusN = []
    DMminusN = []
    TR = df2['TR'].tolist()
    DMplus = df2['DMplus'].tolist()
    DMminus = df2['DMminus'].tolist()
    for i in range(len(df2)):
        if i < n:
            TRn.append(np.NaN)
            DMplusN.append(np.NaN)
            DMminusN.append(np.NaN)
        elif i == n:
            TRn.append(df2['TR'].rolling(n).sum().tolist()[n])
            DMplusN.append(df2['DMplus'].rolling(n).sum().tolist()[n])
            DMminusN.append(df2['DMminus'].rolling(n).sum().tolist()[n])
        elif i > n:
            TRn.append(TRn[i-1] - (TRn[i-1]/n) + TR[i])
            DMplusN.append(DMplusN[i-1] - (DMplusN[i-1]/n) + DMplus[i])
            DMminusN.append(DMminusN[i-1] - (DMminusN[i-1]/n) + DMminus[i])
    df2['TRn'] = np.array(TRn)
    df2['DMplusN'] = np.array(DMplusN)
    df2['DMminusN'] = np.array(DMminusN)
    df2['DIplusN']=100*(df2['DMplusN']/df2['TRn'])
    df2['DIminusN']=100*(df2['DMminusN']/df2['TRn'])
    df2['DIdiff']=abs(df2['DIplusN']-df2['DIminusN'])
    df2['DIsum']=df2['DIplusN']+df2['DIminusN']
    df2['DX_New'] = df2['DIplusN']/df2['DIminusN']
    df2['DX']=100*(df2['DIdiff']/df2['DIsum'])
    ADX = []
    DX = df2['DX'].tolist()
    for j in range(len(df2)):
        if j < 2*n-1:
            ADX.append(np.NaN)
        elif j == 2*n-1:
            ADX.append(df2['DX'][j-n+1:j+1].mean())
        elif j > 2*n-1:
            ADX.append(((n-1)*ADX[j-1] + DX[j])/n)
    df2['ADX']=np.array(ADX)
    df2.drop(['TR','DMplus','DMminus','TRn','DMplusN','DMminusN','DIdiff','DIsum','DX','DIminusN','DIplusN'],axis=1, inplace=True)
    return df2

dict ={}
for ticker in tickers:
    dict[ticker] = ADX(df[df['Symbol']==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

df["ADX_Signal"] = np.where(df['ADX']>60,1,0)


def renko_DF(DF):
    "function to convert ohlc data into renko bricks"
    df = DF.copy()
    df.reset_index(inplace=True)
    df = df.iloc[:,[0,1,2,3,5,6]]
    df.rename(columns = {"Date" : "date", "High" : "high","Low" : "low", "Open" : "open","Adj Close" : "close", "Volume" : "volume"}, inplace = True)
    df2 = Renko(df)
    df2.brick_size = round(2*ATR(DF,14)["ATR"][-1],0)
    renko_df = df2.get_ohlc_data() #if using older version of the library please use get_bricks() instead
    return renko_df


dict ={}
for ticker in tickers:
    try:
        dict[ticker] = renko_DF(df[df['Symbol']==ticker])
        dict[ticker]['Symbol'] = ticker
    except:
        pass

df = df.reset_index()

renko_master = pd.concat([dict[ticker] for ticker in list(dict.keys())],axis=0).rename(columns = {"date":"Date"})
renko_master["bar_num"] = np.where(renko_master["uptrend"]==True,1,np.where(renko_master["uptrend"]==False,-1,0))

def renko_clean(DF):
    df = DF.copy()
    for i in range(1,len(df["bar_num"])):
        if df["bar_num"][i]>0 and df["bar_num"][i-1]>0:
            df["bar_num"][i]+=df["bar_num"][i-1]
        elif df["bar_num"][i]<0 and df["bar_num"][i-1]<0:
            df["bar_num"][i]+=df["bar_num"][i-1]
    df.drop_duplicates(subset="Date",keep="last",inplace=True)
    return df
    

dict ={}
for ticker in tickers:
    dict[ticker] = renko_clean(renko_master[renko_master['Symbol']==ticker])
renko_master = pd.concat([dict[ticker] for ticker in tickers],axis=0)

final= pd.merge(df,renko_master[['Date','Symbol','uptrend','bar_num']],how='left', on =['Date','Symbol'])
final.set_index('Date', inplace = True)
final["bar_name"] = final["bar_num"].fillna(method='ffill')


def EMA(DF):
    df = DF.copy()
    df['EMA_20'] = df["Adj Close"].ewm(span=20,min_periods=20).mean()
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = EMA(final[final['Symbol']==ticker])

final = pd.concat([dict[ticker] for ticker in tickers],axis=0)


#signal for Renko
final['Renko_Signal_1'] = np.where((final["Adj Close"]-1.1*final['EMA_20']>0) & (final["bar_name"]>=2),1,0)



def OBV(DF):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df['Adj Close'].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df['direction'][0] = 0
    df['vol_adj'] = df['Volume'] * df['direction']
    df['OBV'] = df['vol_adj'].cumsum()
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = OBV(final[final['Symbol']==ticker])
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

def slope(DF,col='Adj Close',n=5):
    "function to calculate the slope of regression line for n consecutive points on a plot"
    ser = DF[col]
    ser = (ser - ser.min())/(ser.max() - ser.min())
    x = np.array(range(len(ser)))
    x = (x - x.min())/(x.max() - x.min())
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y_scaled = ser[i-n:i]
        x_scaled = x[:n]
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    DF[f'slope_{col}'] = np.array(slope_angle)
    return DF

dict ={}
for ticker in tickers:
    dict[ticker] = slope(final[final['Symbol']==ticker],n=5)
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

dict ={}
for ticker in tickers:
    dict[ticker] = slope(final[final['Symbol']==ticker],col ='OBV',n=5)
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

final["Close_Pr_Signal_HardRule"] = np.where((final['slope_Adj Close']>30) & (final['slope_Adj Close']<=75),1,0)

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    daily_ret = DF["Adj Close"].pct_change()
    cum_return = (1 + daily_ret).cumprod()
    cum_roll_max = cum_return.cummax()
    drawdown = cum_roll_max - cum_return.min()
    drawdown_pct= drawdown/cum_roll_max
    df["max_dd"] = drawdown_pct.max()
    return df

dict ={}
for ticker in tickers:
    dict[ticker] = max_dd(final[final['Symbol']==ticker])
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)
# final['calmar'] = final['cagr']/final['max_dd']

def strategy1_data(DF):
    df = DF.copy()
    df["roll_max_cp"] = df["High"].rolling(14).max()
    df["roll_min_cp"] = df["Low"].rolling(14).min()
    df["roll_max_vol"] = df["Volume"].rolling(14).max()
    return df

dict ={}
for ticker in tickers:
        dict[ticker] = strategy1_data(final[final['Symbol']==ticker])

final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

# Strategy 1 = Resistance Breakout
def strategy1(DF):
    df = DF.copy()
    c1 = df['High']/df["roll_max_cp"]>=1
    c2 = df['Volume']/(1.5*df["roll_max_vol"].shift(1))>=1
    c = 1*c1*c2
    df['startegy1_signal'] = np.where(c==1,1,0)
    return df


dict ={}
for ticker in tickers:
        dict[ticker] = strategy1(final[final['Symbol']==ticker])

final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

# y = talib.get_function_groups()

#Strategy2 = Renko & OBV
final['strategy2_signal'] = np.where((final['slope_OBV']>80) & (final['bar_name']>=3),1,0)

dict ={}
for ticker in tickers:
    dict[ticker] = slope(final[final['Symbol']==ticker],col ='MACD',n=10)
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

dict ={}
for ticker in tickers:
    dict[ticker] = slope(final[final['Symbol']==ticker],col ='Signal',n=10)
final = pd.concat([dict[ticker] for ticker in tickers],axis=0)

#Strategy3 = 1. MACD>Signal, 2. renko >=2 and 3. MACD Slope of 5 period > Signal slope of 5 period
final['strategy3_signal'] = np.where((final['MACD']>1.3*final['Signal']) & (final['bar_name']>=3) & (final['slope_MACD']>1.2*final['slope_Signal']),1,0)

final.to_csv("Final_500_Data.csv")

df1 = final[['Adj Close','Symbol','MACD_Buy','RSI_Signal','ADX_Signal','Renko_Signal_1','Close_Pr_Signal_HardRule','startegy1_signal','strategy2_signal','strategy3_signal']]
df1 = df1[df1["Close_Pr_Signal_HardRule"]==1]

df1['Score'] = df1.iloc[:,2:10].sum(axis=1)

df1 = df1.rename_axis('Date').sort_values(by = ['Date','Score'], ascending = [False, False])
df1 = df1.drop(['Close_Pr_Signal_HardRule'], axis = 1)
df1['Score'] = df1['Score'] - 1
df1 = df1[df1['Score']>=2]#.iloc[:,[0,1,10]]

df1.to_csv("Final_Buy.csv")

time.sleep(10)



from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Final_500_Data.csv")
df = df.drop(['uptrend'], axis = 1) 
df.fillna(method ='bfill', axis=0, inplace=True)
tickers = df['Symbol'].unique().tolist()


def AR(DF):
    x = DF.copy()
    from statsmodels.tsa.ar_model import AR
    model = AR(x['Adj Close'])
    ARfit = model.fit()
    fcast = ARfit.predict(start=len(x), end=len(x), dynamic=False).rename('Forecast').iloc[0]
    last = x['Adj Close'].iloc[-1]
    x['AR'] = 100*((fcast/last)-1)
    return x



dict ={}
for ticker in tickers[:]:
    dict[ticker] = AR(df[df["Symbol"]==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)


def trg(DF):
    X = DF.copy()
    X['Target'] = np.where(X['Adj Close'].shift(-5)/X['Adj Close'] >1.04,1,0)
    return X#[:-5]

dict ={}
for ticker in tickers:
    dict[ticker] = trg(df[df['Symbol']==ticker])
    
from sklearn.ensemble import AdaBoostClassifier

df = pd.concat([dict[ticker] for ticker in tickers],axis=0)
def logi(DF,tick):
    dataset = DF.copy()
    dataset = dataset.drop(['Symbol'], axis=1).set_index('Date')
    X = dataset[:-5].iloc[:, :-1].values
    y = dataset[:-5].iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    n = int(.75*len(X))
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #classifier = LogisticRegression(random_state = 0)
    classifier = AdaBoostClassifier(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    X_nul = dataset[-5:].iloc[:, :-1].values
    X_nul = sc.transform(X_nul)
    y_pred_nul = classifier.predict(X_nul)
    # cm = confusion_matrix(y_test, y_pred)
    # acc = accuracy_score(y_test, y_pred)
    dataset = dataset.iloc[n:,:]
    dataset['Pred'] = y_pred.tolist() + y_pred_nul.tolist()
    dataset['Symbol'] = tick
    return dataset

dict ={}
for ticker in tickers[:]:
    try:
        dict[ticker] = logi(df[df["Symbol"]==ticker],ticker)
    except:
        pass
df = pd.concat([dict[ticker] for ticker in list(dict.keys())],axis=0)

def per(DF):
    x = DF.copy()
    x['Pred1_Count'] = x['Pred'].rolling(6).sum()
    return x

dict ={}
for ticker in tickers[:]:
    dict[ticker] = per(df[df["Symbol"]==ticker])
df = pd.concat([dict[ticker] for ticker in tickers],axis=0)

df.reset_index(inplace = True)

final = df[df['Date']==df['Date'].max()]
# final = final[final['Pred1_Count']>=2]

dic1 = final[['Symbol','Pred1_Count']].set_index('Symbol').to_dict()
dic2 = final[['Symbol','AR']].set_index('Symbol').to_dict()

df1 = pd.read_csv("Final_Buy.csv")
df1['Pred1_Count'] = df1['Symbol'].map(dic1['Pred1_Count'])
df1['AR'] = df1['Symbol'].map(dic2['AR'])
df1 = df1.sort_values(by = ['Date','Score','Pred1_Count','AR'], ascending = [False, False,False,False])


df1.to_csv("Final_Buy_ML.csv", index = False)
