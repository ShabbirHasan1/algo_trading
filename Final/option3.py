import requests
import json
import pandas as pd

new_url = 'https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY'

header = {
          "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
          "X-Requested-With": "XMLHttpRequest"
        }
page = requests.get(new_url,headers=header)
dajs = json.loads(page.text)


def fetch_oi(expiry_dt):
    ce_values = [data['CE'] for data in dajs['records']['data'] if "CE" in data and data['expiryDate'] == expiry_dt]
    pe_values = [data['PE'] for data in dajs['records']['data'] if "PE" in data and data['expiryDate'] == expiry_dt]

    ce_dt = pd.DataFrame(ce_values).sort_values(['strikePrice'])
    pe_dt = pd.DataFrame(pe_values).sort_values(['strikePrice'])
    
    print(ce_dt[['strikePrice','lastPrice']])

def main():
    
    expiry_dt = '27-Aug-2020'
    fetch_oi(expiry_dt)

if __name__ == '__main__':
    main()