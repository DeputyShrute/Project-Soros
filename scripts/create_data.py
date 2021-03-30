import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import matplotlib.dates as mpl_dates

data = pd.read_csv(
    "/home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/GBPUSD.csv", index_col=0, parse_dates=True)
print(data)
data.index.name = 'Date'

n = 1
m = 200

for i in range(14):

    mpf.plot(data[n:m], type='candle', show_nontrading=False,
             savefig=dict(fname='EURGBP0{num}.jpg'.format(num=i)), style='yahoo', axisoff=True)
    
    n += 200
    m += 200
