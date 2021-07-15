import os
import pandas as pd
from datetime import date, timedelta
from datetime import datetime
from GoogleNews import GoogleNews
import time
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


cwd = os.chdir("C:\Desktop\Learning\Courses\zerodha")
sym = pd.read_csv("NSE_Symbols_Pink.csv", dtype={'Instrument': int})[['Symbol','Instrument','Name']]
dic1 = sym.set_index('Name').to_dict()
tokens = sym['Name'].tolist()[:]
end = datetime.now()
start = datetime.now() - timedelta(1)
end = end.strftime("%m-%d-%Y")
start = start.strftime("%m-%d-%Y")


cwd = os.chdir("C:/Desktop/Learning/Courses/zerodha/exp")

analyser = SentimentIntensityAnalyzer()

def comp_score(text):
   return analyser.polarity_scores(text)["compound"]   

def main():
    news = {}
    for ticker in tokens:
        try:
            time.sleep(random.randint(5,12))
            googlenews=GoogleNews(start=start,end=end)
            googlenews.search(ticker)
            articles=googlenews.result(sort=True)
            body = []
            for article in articles[:min(3,len(articles))]:
                body.append(article['title'])
                cont = ". ".join(body)
            news[dic1['Symbol'][ticker]] = cont
        except:
            continue
    final = pd.DataFrame(news.items())
    final.columns = ['Symbol','News']
    final['time'] = str(t1.strftime('%H_%M'))
    try:
        df1 = pd.read_csv("News.csv")
    except:
        df1= pd.DataFrame()
    df2 = final.append(df1)
    df2 = df2.sort_values(by = ['time'], ascending = [True])
    df2 = df2.drop_duplicates(['Symbol'], keep='last')
    df2["sentiment"] = df2["News"].apply(comp_score)
    df2.to_csv("News.csv", index = False)
    return df2
    
i = 0
while True:
    t1 = datetime.now()
    final = main()
    i = i +1
    t2 = datetime.now()
    print(f"Loop Number------>{i}")
    #time.sleep(max(1,2*60 - (t2-t1).seconds))
    time.sleep(random.randint(25*60,30*60))