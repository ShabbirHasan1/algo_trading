import numpy as np
import pandas as pd
import os
from fbprophet import Prophet

os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")
df = pd.read_csv("Final_500_Data.csv")
df = df[['Date','Adj Close','Symbol']]
df.columns=  ['ds','y','Symbol']
df['ds'] = pd.to_datetime(df['ds'])
df = df[df['Symbol']=="FLUOROCHEM.NS"]
df.drop(['Symbol'], axis=1, inplace = True)

df.plot(x='ds', y='y')

len(df)

train = df.iloc[:504]
test = df.iloc[504:]

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods = 1)
forecast = m.predict(future)


###############################################################################

os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")
df = pd.read_csv("Final_500_Data.csv")
df = df[['Date','Adj Close','Symbol']]
df = df[df['Symbol']=="ASIANPAINT.NS"]
df.drop(['Symbol'], axis=1, inplace = True)
df.set_index("Date", inplace = True)
df.plot()

# from statsmodels.tsa.seasonal import seasonal_decompose
# result = seasonal_decompose(df['Adj Close'],model='multiplicative', period = 1)



import numpy as np
import pandas as pd
import os


os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")
df = pd.read_csv("Final_500_Data.csv")
df = df[['Date','Adj Close','Symbol']]
from statsmodels.tsa.ar_model import AR,ARResults

df = df[df['Symbol']=="RELIANCE.NS"]

# train = df.iloc[:490]
# test = df.iloc[490:]

# model = AR(train['Adj Close'])
# AR1fit = model.fit(maxlag=3,method='mle')
# AR1fit.aic
# print(f'Lag: {AR1fit.k_ar}')
# print(f'Coefficients:\n{AR1fit.params}')

# start=len(train)
# end=len(train)+len(test)-1
# predictions1 = AR1fit.predict(start=start, end=end, dynamic=False).rename('AR(1) Predictions')


model = AR(df['Adj Close'])
ARfit = model.fit()
fcast = ARfit.predict(start=len(df), end=len(df), dynamic=False).rename('Forecast')


#####################################3
#################################3333

# Load specific forecasting tools
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from pmdarima import auto_arima # for determining ARIMA orders

import numpy as np
import pandas as pd
import os


os.chdir(r"C:\Desktop\Learning\Courses\Algo_Final")
df = pd.read_csv("Final_500_Data.csv")
df = df[['Date','Adj Close','Symbol']]

df = df[df['Symbol']=="RELIANCE.NS"]

