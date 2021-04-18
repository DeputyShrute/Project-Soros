 #!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
import os
import requests
from pathlib import Path
from scripts.data_validation import data_validation
from scripts.model_start import Models
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter("ignore", DeprecationWarning)



class get_data:

    def get_url():
        '''
        Array lists all available currency pairs
        In a loop the pair is then concat into a specific URL which pulls 10 years worth of data for that currency
        Time period is 01/01/10 - 01/01/2020
        Error handling used to check the link is accessible
        The data is then downloaded into the Data folder an named as the currency pair (x)
        '''
        currency_pair = ["EURUSD", "EURJPY", "USDCHF", "GBPUSD", "USDJPY", "USDCAD", "AUDUSD", "EURGBP", "EURAUD", "EURCHF", "GBPCHF", "CADJPY", "GBPJPY",
                         "AUDNZD", "AUDCAD", "AUDCHF", "AUDJPY", "CHFJPY", "EURNZD", "EURCAD", "CADCHF", "NZDJPY", "NZDUSD", "GBPAUD", "GBPNZD", "NZDCAD", "NZDCHF"]
        currency_pairs = sorted(currency_pair)
        for x in currency_pairs:
            try:
                Forex_URL = "https://query1.finance.yahoo.com/v7/finance/download/" + \
                    x + "=X?period1=1262304000&period2=1611273600&interval=1d&events=history&includeAdjustedClose=true"
                weburl = requests.get(Forex_URL)
                print(x, weburl.status_code)
            except ConnectionError:
                conn = False
                while conn == False:
                    Forex_URL = "https://query1.finance.yahoo.com/v7/finance/download/" + \
                        x + "=X?period1=1262304000&period2=1611273600&interval=1d&events=history&includeAdjustedClose=true"
                    weburl = requests.get(Forex_URL)
                    print(x, weburl.status_code)
                    if ConnectionError == False:
                        conn = True
            if weburl.status_code == 200:
                download_link = "https://query1.finance.yahoo.com/v7/finance/download/" + \
                    x + "=X?period1=1262304000&period2=1611273600&interval=1d&events=history&includeAdjustedClose=true"
                file_download = requests.get(
                    download_link, allow_redirects=True)
                path = 'Data' + x + '.csv'
                if os.path.exists(path):
                    continue
                else:
                    try:
                        with open('scripts/Finance_Data/'+x+'.csv', 'wb') as f:
                            f.write(file_download.content)
                    except FileNotFoundError as e:
                        print("Not found: ", x, e)
            else:
                print("Failed to access: ", x)
                continue
            data_validation(x)


if __name__ == "__main__":
    get_data.get_url()
