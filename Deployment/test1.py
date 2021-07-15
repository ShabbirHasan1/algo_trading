from kiteconnect import KiteConnect
import logging
import os
import datetime as dt
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import warnings
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib
cwd = os.getcwd()
import time 

access_token = open("access_token.txt",'r').read()
key_secret = open("api_key.txt",'r').read().split()
kite = KiteConnect(api_key=key_secret[0])
kite.set_access_token(access_token)
d = pd.DataFrame()



today = datetime.now().day
month = datetime.now().month
# sym = pd.read_csv("NSE_Symbols_1.csv", dtype={'Instrument': int})[['Symbol','Instrument']]
sym = pd.read_csv("NSE_Symbols_1.csv", dtype={'Instrument': int})[['Symbol','Instrument','Pink']]


def mailing(DF):
    x = DF.copy()
    s = smtplib.SMTP('smtp.gmail.com', 587)
    recipients = ['ttips2021@gmail.com']
    emaillist = [elem.strip().split(',') for elem in recipients]
    s.starttls()
    s.login("ttips2021@gmail.com", "trade_2021")
    msg = MIMEMultipart()
    msg['Subject'] = f"{str((t1 +dt.timedelta(hours=5,minutes=30)).strftime('%H_%M'))} - Data"
    msg['From'] = 'ttips2021@gmail.com'
    html = """\
    <html>
      <head></head>
      <body>
        {0}
      </body>
    </html>
    """.format(x.to_html())
    part1 = MIMEText(html, 'html')
    msg.attach(part1)
    s.sendmail(msg['From'],emaillist, msg.as_string())
    s.quit()
        
t1 = datetime.now()
if (t1.hour>=3) & (t1.hour < 10):
        print(t1.minute)
if (t1.hour >= 10 and t1.minute >= 1):
        print("off hours")
mailing(sym)
