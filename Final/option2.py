import pandas as pd
import requests
from datetime import datetime

# Colunn at which strike price is listed in NSE option chain table
strike_price_column_index = 11

# Encapuslate NSE option data and function
class OptionChain:

    # static variable to hold current running expiry date
    expiry = ''

    # Common Utility to find maxinum value in the column, its index and then return the strike price with respect to index value
    def find_max_strike_price(self, df, option_type, column_name):

        # Covert "-" as 0 so that all data can be treated as integer
        temp_df = df[option_type].replace("-", "0")
        # Delete the last row which will have summation of all the data
        temp_df = temp_df[:-1]
        # Set the specific column as integer (by default it is string)
        temp_df[column_name]=temp_df[column_name].astype(int)
        # Dind the index value where the max value exists
        max_at_index = temp_df[column_name].idxmax()
        # Return the strike price which is available in the index value
        return(int(df.iloc[max_at_index, strike_price_column_index]))

    # Get strike prices where max OI /change in OI for CE and PE
    def get_max_OI_data(self, symbol, expiry):

        url ="https://www.nseindia.com/live_market/dynaContent/live_watch/option_chain/optionKeys.jsp?symbol=" + symbol + "&date" + expiry
        header = {
          "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
          "X-Requested-With": "XMLHttpRequest"
        }

        # Pull NSE option chain
        r = requests.get(url, headers=header)
        # Convert html page as Table and read the first table which has option data
        df = pd.read_html(r.text)[1]

        # Get all max OI data and store to local object variables
        self.max_high_oi_ce = self.find_max_strike_price(df, "CALLS", "OI")
        self.max_change_oi_ce = self.find_max_strike_price(df, "CALLS", "Chng in OI")
        self.max_high_oi_pe = self.find_max_strike_price(df, "PUTS", "OI")
        self.max_change_oi_pe = self.find_max_strike_price(df, "PUTS", "Chng in OI")

    # Constructor for OptionChain which will run for every script like Nifty and Banknifty
    def __init__(self, symbol):

        # Store the symbol
        self.symbol = symbol
        # Find out the expiry for which we need to pull the details

        if not OptionChain.expiry: # If expiry is not yet found
            # List the expiry details and read the first expiry
            base_url = 'https://www.capitalzone.in/test.php?symbol=' + symbol
            page_output = str(requests.get(base_url).content)
            expiry_list = page_output.split(",")
            OptionChain.expiry = expiry_list[0].split("\"")[1]
            expirydate = datetime.strptime(OptionChain.expiry, '%d%b%Y').date()
            todaydate = datetime.today().date()

            # If today is greater than expiry, then it is expired. Choose the next expiry in the list
            if todaydate > expirydate:
                NSEOption.expiry = expiry_list[1].split("\"")[1]

        # Form the actual URL from which we can pull PCR and max pain
        base_url = 'https://www.capitalzone.in/test.php?symbol=' + symbol + "&expiry=" + OptionChain.expiry

        # Load the page and sent to HTML parse
        page_output = str(requests.get(base_url).content)
        # Locate the string where max pain string is available
        loc = page_output.find("\"max_pain\"")
        data = page_output[loc:].split("\"")
        # Get max pain and convert from float to integer
        self.max_pain = int(float(data[3]))
        # Get max pain and convert from string to float
        self.pcr = float(data[11])
        self.get_max_OI_data(symbol, OptionChain.expiry)

    # Tracing utility to display elements of this class if needed (For debugging purpose)
    def display_all(self):

        print(self.max_pain, self.pcr);