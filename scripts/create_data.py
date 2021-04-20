 #!/usr/bin/env python -W ignore::DeprecationWarning
import matplotlib.dates as mpl_dates
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import os


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class make_chart:

    def load(select, currency_pairs):
        X = currency_pairs[select]
        # Loads the CSV data
        file_loc = '/home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/Raw_Data/' + X
        data = pd.read_csv(
            file_loc, index_col=0, parse_dates=True)
        # Prints the head
        print(data)
        csv_rows = data.shape[0]
        data.index.name = 'Date'
        make_chart.create(data, csv_rows, X)

    def create(data, csv_rows, X):
        n = csv_rows - 30
        try:
            mpf.plot(data[n:csv_rows], type='candle', show_nontrading=False,
                     savefig=dict(fname='/home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/Chart_Snapshot/{name}_test.jpg'.format(name=X)), style='yahoo', axisoff=True)
            os.chdir('/home/ryan/Documents/Python/Project-Soros/darknet/data')
            file = open('test.txt', 'w+')
            file.write('//home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/Chart_Snapshot/{name}_test.jpg'.format(name=X))
            file.close
        except FileNotFoundError:
            print('exception')
            return
