from kiteconnect import KiteConnect
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

short_window = 6
long_window = 20
ticker = tokens[62]
indicators = fetchOHLC(ticker,"3minute",5)[['date','close']]
ohlc = indicators.copy()
indicators['signal'] = 0.0
indicators['short_avg'] = indicators['close'].ewm(span=short_window, adjust=False).mean()
indicators['long_avg'] = indicators['close'].ewm(span=long_window, adjust=False).mean()
indicators['signal'][short_window:] = np.where(indicators['short_avg'][short_window:] > indicators['long_avg'][short_window:], 1.0, 0.0)
indicators['positions'] = indicators['signal'].diff()
indicators.loc[indicators['positions'] ** 2 == 1]

fig = plt.figure(figsize=(13, 10))

# Labels for plot
ax1 = fig.add_subplot(111, ylabel='Price')

# Plot stock price over time
ohlc['close'].plot(ax=ax1, color='black', lw=2.)

# Plot the the short and long moving averages
indicators[['short_avg', 'long_avg']].plot(ax=ax1, lw=2.)

# Plot where to buy indicators
ax1.plot(indicators.loc[indicators.positions == 1.0].index,
         indicators.short_avg[indicators.positions == 1.0],
         '^', markersize=10, color='g')

# Plots where to sell indicators
ax1.plot(indicators.loc[indicators.positions == -1.0].index,
         indicators.short_avg[indicators.positions == -1.0],
         'v', markersize=10, color='r')
plt.xticks(indicators.date)

# Show the plot
plt.show()


num_buy = len(indicators.loc[indicators.positions == 1.0])
num_sell = len(indicators.loc[indicators.positions == -1.0])
total_buy = sum(indicators.loc[indicators.positions == 1.0]['close'][:-1])
total_sell = sum(indicators.loc[indicators.positions == -1.0]['close'])
if num_buy != num_sell:
    total_buy += ohlc['close'].iloc[-1]
total_profit_trading = total_sell - total_buy + (num_buy - num_sell)
total_profit_holding = ohlc['close'].iloc[-1] - ohlc['close'].iloc[0]
percent_increase = 100 * (total_profit_trading - total_profit_holding) / total_profit_holding





