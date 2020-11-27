import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def data_validation(x):
    data_location = 'Data/' + x + '.csv'
    data = pd.read_csv(data_location)
    data.drop('Volume', axis=1, inplace=True)
    data.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close"]
    data = data.interpolate()
    print(data.isnull().sum())
    try:
        data.to_csv('Data/'+x+'.csv', sep=',')
    except FileNotFoundError as e:
                    print("Not found: ", x, e)
