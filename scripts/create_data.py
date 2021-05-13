#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
import matplotlib.dates as mpl_dates
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from scripts.part_launch import launch
import os


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class make_chart:

    def load(select, currency_pairs):
        X = currency_pairs[select]
        # Loads the CSV data
        directo = os.path.dirname(__file__)
        path1 = os.path.join(directo, 'Finance_Data/Raw_Data/')
        file_loc = path1 + X
        data = pd.read_csv(
            file_loc, index_col=0, parse_dates=True)
        # Prints the head
        print(data)
        csv_rows = data.shape[0]
        data.index.name = 'Date'
        make_chart.create(data, csv_rows, X)

    def create(data, csv_rows, X):
        n = csv_rows - 30
        directo = os.path.dirname(__file__)
        path1 = os.path.join(directo, 'Finance_Data/Chart_Snapshot/')
        mpf.plot(data[n:csv_rows], type='candle', show_nontrading=False,
                 savefig=dict(fname=path1+'{name}_test.jpg'.format(name=X)), style='yahoo', axisoff=True)
        launch.darknet(X)


class make_chart_test:

    def load_data(currency_pairs):

        # Loads the CSV data
        file_loc = 'scripts/Finance_Data/Raw_Data/' + currency_pairs + '.csv'
        data = pd.read_csv(
            file_loc, index_col=0, parse_dates=True)
        # Prints the head
        print(data)
        csv_rows = data.shape[0]
        data.index.name = 'Date'
        make_chart_test.create(data, csv_rows, currency_pairs)

    def create(data, csv_rows, X):
        n = csv_rows - 22
        q = 0
        w = 50
        num = 1
        while w < n:
            
            
            s  = mpf.make_mpf_style(base_mpf_style='yahoo', gridcolor='white', figcolor='white', edgecolor='white', facecolor='white')
            mpf.plot(data[q:w], type='candle', show_nontrading=False,
                     savefig=dict(fname='scripts/Finance_Data/Chart_Snapshot/TEST/operation/{name}_{number}test.jpg'.format(name=X, number=num)), style=s, axisoff=False)
            f = open(
                "darknet/data/test.txt", 'a')
            f.write(
                "\nscripts/Finance_Data/Chart_Snapshot/TEST/operation/{name}_{number}test.jpg".format(name=X, number=num))
            q += 50
            w += 50
            num += 1
            # launch.darknet(X)


if __name__ == "__main__":
    make_chart_test.load_data("test_data")
