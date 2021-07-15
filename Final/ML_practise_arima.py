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
from sklearn.metrics import mean_squared_error,mean_absolute_error
os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")

df = pd.read_csv("Final_500_Data.csv")
comp = pd.read_csv("comp.csv")
df_latest = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
# df = df[df['Date']!=df['Date'].max()]
# df_latest2 = df[df['Date']==df['Date'].max()][['Symbol','Adj Close']]
dic_prc = df_latest.set_index('Symbol').to_dict()
# df_latest2['Pr'] = df_latest2['Symbol'].map(dic_prc['Adj Close'])
# df_latest2.dropna(inplace = True)
# df_latest2['ret'] = df_latest2['Pr']/df_latest2['Adj Close']
# dic_ret = df_latest2[['Symbol','ret']].set_index('Symbol').to_dict()

df = df.drop(['uptrend','Open','High','Low','Close'], axis = 1) 
df.fillna(method ='bfill', axis=0, inplace=True)
tickers = df['Symbol'].unique().tolist()
tickers1 = comp[comp['Best']=="AR1"]['index'].tolist()
tickers2 = comp[comp['Best']=="AR2"]['index'].tolist()
tickers3 = comp[comp['Best']=="AR3"]['index'].tolist()
tickers4 = comp[comp['Best']=="AR4"]['index'].tolist()
tickers5 = comp[comp['Best']=="AR5"]['index'].tolist()



warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing
def AR1(DF):
    x = DF.copy()
    fitted_model = ExponentialSmoothing(x['Adj Close'],trend='add',seasonal='add',seasonal_periods=20).fit()
    pred1 = fitted_model.forecast(1).iloc[0]
    return pred1

dict ={}
for ticker in tickers1[:]:
    try:
        dict[ticker] = AR1(df[df["Symbol"]==ticker])
    except:
        pass

ar1 = pd.DataFrame(dict.items())
ar1.columns = ['Symbol', 'Pred']
ar1.set_index('Symbol', inplace=True)



warnings.filterwarnings('ignore')
from statsmodels.tsa.ar_model import AR,ARResults
def AR2(DF):
    x = DF.copy()
    model = AR(x['Adj Close'])
    AR1fit = model.fit(ic = 't-stat',maxlag=4)
    pred2 = AR1fit.predict(start=len(x), end=len(x), dynamic=False).iloc[0]
    return pred2


dict ={}
for ticker in tickers2[:]:
    try:
        dict[ticker] = AR2(df[df["Symbol"]==ticker])
    except:
        pass

ar2 = pd.DataFrame(dict.items())
ar2.columns = ['Symbol', 'Pred']
ar2.set_index('Symbol', inplace=True)

DF = df[df['Symbol']=="INDIGO.NS"]
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
def AR3(DF):
    x = DF.copy()
    model = ARMA(x['Adj Close'],order=(3,2))
    results = model.fit()
    pred3 = results.predict(start=len(x), end=len(x)).iloc[0]
    return pred3

dict ={}
for ticker in tickers3[:]:
    try:
        dict[ticker] = AR3(df[df["Symbol"]==ticker])
    except:
        pass

ar3 = pd.DataFrame(dict.items())
ar3.columns = ['Symbol', 'Pred']
ar3.set_index('Symbol', inplace=True)


warnings.filterwarnings('ignore')
def AR4(DF):
    x = DF.copy()
    model = ARIMA(x['Adj Close'],order=(5,1,3))
    results = model.fit()
    pred4 = results.predict(start=len(x)-5, end=len(x), dynamic=False, typ='levels').iloc[1]
    return pred4


dict ={}
for ticker in tickers4[:]:
    try:
      dict[ticker] = AR4(df[df["Symbol"]==ticker])
    except:
        pass

ar4 = pd.DataFrame(dict.items())
ar4.columns = ['Symbol', 'Pred']
ar4.set_index('Symbol', inplace=True)


# DF = df[df['Symbol']=="INFY.NS"]
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
def AR5(DF):
    x = DF.copy()
    model = SARIMAX(x['Adj Close'],order=(5,1,0),seasonal_order=(1,0,1,21),enforce_invertibility=False)
    results = model.fit()
    pred5 = results.predict(start=len(x), end=len(x), dynamic=False, typ='levels').iloc[0]
    return pred5


dict ={}
for ticker in tickers5[:]:
    try:
        dict[ticker] = AR5(df[df["Symbol"]==ticker])
    except:
        pass

ar5 = pd.DataFrame(dict.items())
ar5.columns = ['Symbol', 'Pred']
ar5.set_index('Symbol', inplace=True)

final =pd.concat([ar1,ar2,ar3,ar4,ar5],axis=1)
final['Predicted']= final.sum(axis=1)
final.rename_axis('Symbol',inplace = True)
final.reset_index(inplace=True)
final = final[['Symbol','Predicted']]
final['last_Price'] = final['Symbol'].map(dic_prc['Adj Close'])
final['Ret'] = final['Predicted']/final['last_Price']
final.sort_values(by = ['Ret'],ascending = [False],inplace = True)
final.to_csv("final.csv")


# warnings.filterwarnings('ignore')
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# def AR1(DF):
#     x = DF.copy()
#     x_train = x[:-5]
#     x_test = x[-5:]
#     start = len(x_train)
#     end = len(x_train) +len(x_test)-1
#     fitted_model = ExponentialSmoothing(x_train['Adj Close'],trend='add',seasonal='add',seasonal_periods=20).fit()
#     pred1 = fitted_model.forecast(5).rename('HW Forecast')
#     return np.sqrt(mean_squared_error(x_test['Adj Close'],pred1))

# dict ={}
# for ticker in tickers[:]:
#     try:
#         dict[ticker] = AR1(df[df["Symbol"]==ticker])
#     except:
#         pass

# ar1 = pd.DataFrame(dict.items())
# ar1.columns = ['Symbol', 'AR1']
# ar1.set_index('Symbol', inplace=True)

# warnings.filterwarnings('ignore')
# from statsmodels.tsa.ar_model import AR,ARResults
# def AR2(DF):
#     x = DF.copy()
#     x_train = x[:-5]
#     x_test = x[-5:]
#     start = len(x_train)
#     end = len(x_train) +len(x_test)-1
#     model = AR(x_train['Adj Close'])
#     AR1fit = model.fit(ic = 't-stat',maxlag=3)
#     pred2 = AR1fit.predict(start=start, end=end, dynamic=False)
#     return np.sqrt(mean_squared_error(x_test['Adj Close'],pred2))

# dict ={}
# for ticker in tickers[:]:
#     try:
#         dict[ticker] = AR2(df[df["Symbol"]==ticker])
#     except:
#         pass

# ar2 = pd.DataFrame(dict.items())
# ar2.columns = ['Symbol', 'AR2']
# ar2.set_index('Symbol', inplace=True)

# warnings.filterwarnings('ignore')
# from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
# def AR3(DF):
#     x = DF.copy()
#     x_train = x[:-5]
#     x_test = x[-5:]
#     start = len(x_train)
#     end = len(x_train) +len(x_test)-1
#     model = ARMA(x_train['Adj Close'],order=(3,2))
#     results = model.fit()
#     pred3 = results.predict(start=start, end=end)
#     return np.sqrt(mean_squared_error(x_test['Adj Close'],pred3))

# dict ={}
# for ticker in tickers[:]:
#     try:
#         dict[ticker] = AR3(df[df["Symbol"]==ticker])
#     except:
#         pass

# ar3 = pd.DataFrame(dict.items())
# ar3.columns = ['Symbol', 'AR3']
# ar3.set_index('Symbol', inplace=True)


# warnings.filterwarnings('ignore')
# def AR4(DF):
#     x = DF.copy()
#     x_train = x[:-5]
#     x_test = x[-5:]
#     start = len(x_train)
#     end = len(x_train) +len(x_test)-1
#     model = ARIMA(x_train['Adj Close'],order=(5,1,3))
#     results = model.fit()
#     pred4 = results.predict(start=start-5, end=end-5, dynamic=False, typ='levels')
#     return np.sqrt(mean_squared_error(x_test['Adj Close'],pred4))


# dict ={}
# for ticker in tickers[:]:
#     try:
#      dict[ticker] = AR4(df[df["Symbol"]==ticker])
#     except:
#         pass

# ar4 = pd.DataFrame(dict.items())
# ar4.columns = ['Symbol', 'AR4']
# ar4.set_index('Symbol', inplace=True)

# warnings.filterwarnings('ignore')
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# def AR5(DF):
#     x = DF.copy()
#     x_train = x[:-5]
#     x_test = x[-5:]
#     start = len(x_train)
#     end = len(x_train) +len(x_test)-1
#     model = SARIMAX(x_train['Adj Close'],order=(5,1,0),seasonal_order=(1,0,1,21),enforce_invertibility=False)
#     results = model.fit()
#     pred5 = results.predict(start=start, end=end, dynamic=False, typ='levels')
#     return np.sqrt(mean_squared_error(x_test['Adj Close'],pred5))


# dict ={}
# for ticker in tickers[:]:
#     try:
#         dict[ticker] = AR5(df[df["Symbol"]==ticker])
#     except:
#         pass

# ar5 = pd.DataFrame(dict.items())
# ar5.columns = ['Symbol', 'AR5']
# ar5.set_index('Symbol', inplace=True)

# comp = pd.concat([ar1,ar2,ar3,ar4,ar5],axis=1)
# comp['Best'] = comp.idxmin(axis=1)
# comp.reset_index(inplace=True)
# comp.to_csv("comp.csv")



# def AR(DF):
#     x = DF.copy()
#     #from statsmodels.tsa.ar_model import AR
#     from statsmodels.tsa.arima_model import ARIMA
#     #model = AR(x['Adj Close'])
#     model = ARIMA(x['Adj Close'],order = (5,0,0))
#     ARfit = model.fit()
#     fcast = ARfit.predict(start=len(x), end=len(x)).rename('Forecast').iloc[0]
#     last = x['Adj Close'].iloc[-1]
#     # x['AR'] = 100*((fcast/last))
#     return 100*((fcast/last))


# DF = df[df['Symbol']=="HDFC.NS"]
# dict ={}
# for ticker in tickers[:]:
#     dict[ticker] = AR(df[df["Symbol"]==ticker])
    
# ar1 = pd.DataFrame(dict.items())
# ar1.columns = ['Symbol', 'Pred']
# ar1['ret'] = ar1['Symbol'].map(dic_ret['ret'])


