import csv
import json
import pandas as pd
import numpy as np
import talib
import candle_ranking
from itertools import compress


def dataCandle(pair):
    # Assigns the available patterns
    candle_names = talib.get_function_groups()['Pattern Recognition']
    # Reads in the data from a CSV
    df = pd.read_csv(
        "C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Finance_Data/" + pair + ".csv")

    # extract OHLC
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    # create columns for each pattern
    for candle in candle_names:
        # Takes each column from above and compares
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)
    # Drops the empty index
    df = df.drop("Unnamed: 0", axis=1)
    
    # Adds two columns NaN 
    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan

    df.to_csv(r"C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Finance_Data/Candlestick_patterns/" + pair +"_full.csv")

    print(df)

    for index, row in df.iterrows():
        # no pattern found
        if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index, 'candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:

            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(
                ), row[candle_names].values != 0))[0] + '_Bull'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(
                ), row[candle_names].values != 0))[0] + '_Bear'
                df.loc[index, 'candlestick_pattern'] = pattern
                df.loc[index, 'candlestick_match_count'] = 1
        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(
                compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                else:
                    container.append(pattern + '_Bear')
            rank_list = [candle_ranking.candle_rankings[p] for p in container]
            if len(rank_list) == len(container):
                rank_index_best = rank_list.index(min(rank_list))
                df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                df.loc[index, 'candlestick_match_count'] = len(container)
    # clean up candle columns
    df.drop(candle_names, axis=1, inplace=True)

    i = 0
    for index in df['candlestick_pattern']:
        if 'Bear' in index:
            df.at[i, 'candlestick_pattern_direction'] = 1 # Down
        elif 'Bull' in index:
            df.at[i, 'candlestick_pattern_direction'] = 2 # Up
        else:
            df.at[i, 'candlestick_pattern_direction'] = 0
        i +=1


    print(df.head(20))
    df.to_csv(r"C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Finance_Data/Candlestick_patterns/" + pair +"_class.csv")


dataCandle('EURUSD')
