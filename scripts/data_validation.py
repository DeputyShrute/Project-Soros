 #!/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd
from pandas import Series, DataFrame
from scripts.candlestick import create_candlestick, candle_identifier


def data_validation(x):
    '''
    Loads up the data file for the given currency (x)
    Removes the Volume column as it is always zero
    Check the file for any missing values
    If values are missing, the cell is filled with a mean of the value before and after
    Writes new data frame back to the CSV
    '''
    data_location = 'scripts/Finance_Data/Raw_Data' + x + '.csv'
    data = pd.read_csv(data_location, index_col=False)
    data.drop('Volume', axis=1, inplace=True)
    data.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close"]
    print(data.isnull().sum())
    data = data.interpolate()
    #data.reset_index(drop=True, inplace=True)
    print(data.isnull().sum())
    try:
        data.to_csv('scripts/Finance_Data/Raw_Data'+x+'.csv',sep=',', index=False)
        #candle_identifier(x)
    except FileNotFoundError as e:
                    print("Not found: ", x, e)
