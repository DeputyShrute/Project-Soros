import pickle
from numpy.core.numeric import NaN
from numpy.core.shape_base import hstack
import matplotlib.pyplot as plt
from numpy import array, newaxis
import pandas as pd
#from script.models import CNN
from tensorflow.python.keras.saving.model_config import model_from_json
from scripts.model_start import Models
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image
import numpy as np
import mplfinance as mpf
import os
import csv

class launch:

    def darknet(X):
        try:
            os.chdir('/home/ryan/Documents/Python/Project-Soros/darknet/data')
            file = open('test.txt', 'w+')
            file.write(
                '//home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/Chart_Snapshot/{name}_test.jpg'.format(name=X))
            file.close
            os.chdir("/home/ryan/Documents/Python/Project-Soros/darknet")
            os.system("./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -dont_show -ext_output < data/test.txt > data/result.txt")
            predictions = Image.open('predictions.jpg')
            predictions.show()
            os.wait()
            launch.predictions(X)
        except FileNotFoundError:
            print('exception')
            return

    def predictions(X):

        # Loads saved model so retraining isn't needed
        # json_file = open('saved_models/MLP.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # model.load_weights('saved_models/MLP.h5')

        X = 'test_data'

        # Creates arrays to be used to specify each column
        open_col, high_col, low_col, clos_col, raw_seq = [], [], [], [], array([
        ])
        # Read input from CSV
        with open('scripts/Finance_Data/Raw_Data/' + X + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            # Assignes each column within CSV to appropriate Array
            for lines in csv_reader:
                if 'null' in lines:
                    continue
                else:
                    # Index is +1 due to CSV indexing
                    if lines[1] != 'null':
                        open_col.append(float(lines[1]))
                    if lines[2] != 'null':
                        high_col.append(float(lines[2]))
                    if lines[3] != 'null':
                        low_col.append(float(lines[3]))
                    if lines[4] != 'null':
                        clos_col.append(float(lines[4]))

        # Converts list to a Numpy array
        open_col = array(open_col)
        high_col = array(high_col)
        low_col = array(low_col)
        clos_col = array(clos_col)

        # Reshape the array to columns and rows
        open_col = open_col.reshape((len(open_col), 1))
        high_col = high_col.reshape((len(high_col), 1))
        low_col = low_col.reshape((len(low_col), 1))
        clos_col = clos_col.reshape((len(clos_col), 1))

        raw_seq = hstack((open_col, high_col, low_col, clos_col))

        tfn = int(input('Val: '))
        val = int(input('val2: '))
        tf=val
        #tfn = 0

        loaded_model = pickle.load(open('saved_models/KNN_file', 'rb'))

        res = [] 
        print(raw_seq[0])
        #res.append([1.1212,1.12218,1.12106,1.12188])
        #res.append([0.0, 0.0, 0.0, 0.0])
        for i in raw_seq:
            i = [i]
            result = loaded_model.predict(i)
            result=result.flatten()
            result=result.tolist()
            res.append(result)
        print(res[1])
        print('\n')
        no = [NaN, NaN, NaN, NaN]
        res.insert(0,no)
        result = pd.DataFrame(res)
        
        print(result)

        file_loc = 'scripts/Finance_Data/Raw_Data/' + X + '.csv'
        data = pd.read_csv(
            file_loc, parse_dates=False, usecols=['Date'])
        data = data.values.tolist()

        data = np.array(data)
        data = data.flatten()

        raw_seq = pd.DataFrame(raw_seq)
        raw_seq['Date'] = data
        raw_seq.index = pd.DatetimeIndex(raw_seq['Date'])
        raw_seq.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        raw_seq['Volume'] = 0
        print(raw_seq)

        # data = np.delete(data, 0)
        data = np.append(data,'2021-05-10')
        
        result['Date'] = data

        result.index = pd.DatetimeIndex(result['Date'])
        result.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        result['Volume'] = 0

        print(result)


        #print(len(result))
        s = mpf.make_mpf_style(base_mpl_style='seaborn',rc={'axes.grid':False})
        fig = mpf.figure(style=s,figsize=(7.5,5.75))
        ax1 = fig.subplot()
        ax2 = ax1.twinx()
        mpf.plot(result[tfn:tf], ax=ax1, type='candle')
        mpf.plot(raw_seq[tfn:tf], ax=ax2, type='candle', style='yahoo')
        plt.show()


if __name__ == "__main__":
    run = launch()
    run.predictions()