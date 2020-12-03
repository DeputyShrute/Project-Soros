import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def data_validation(x):
    '''
    Loads up the data file for the given currency (x)
    Removes the Volume column as it is always zero
    Check the file for any missing values
    If values are missing, the cell is filled with a mean of the value before and after
    Writes new data frame back to the CSV
    '''
    data_location = 'Data/' + x + '.csv'
    data = pd.read_csv(data_location)
    data.drop('Volume', axis=1, inplace=True)
    data.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close"]
    print(data.isnull().sum())
    data = data.interpolate()
    print(data.isnull().sum())
    try:
        data.to_csv('Data/'+x+'.csv', sep=',')
    except FileNotFoundError as e:
                    print("Not found: ", x, e)
