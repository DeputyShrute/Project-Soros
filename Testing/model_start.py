# from scripts.models import LSTM
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from tensorflow.python.keras.callbacks import History
from numpy.core.shape_base import hstack
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from numpy import array
#from models import CNN, MLP, KNN, LSTMs
import csv
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Models:

    def __init__(self, symbol, timestep, model_type, layers, verbose=0):
        print('Constructor Initialised')
        self.symbol = symbol.upper()
        self.timestep = timestep
        self.model_type = model_type.upper()
        self.verbose = verbose
        self.layers = layers

    def __str__(self):
        return self.model_type

    def split_sequence(raw_seq, n_steps):
        X, y = list(), list()
        for i in range(len(raw_seq)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(raw_seq)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = raw_seq[i:end_ix, :], raw_seq[end_ix, :]
            X.append(seq_x)
            y.append(seq_y)

        # raw_seq = [float(i)/max(raw_seq) for i in raw_seq]

        return array(X), array(y)

    def data(self):
        # Read input from CSV
        open_col, high_col, low_col, clos_col, raw_seq = [], [], [], [], array([
        ])

        with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Finance_Data/' + self.symbol + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for lines in csv_reader:
                if lines[2] != 'null':
                    #print(lines[2])
                    open_col.append(float(lines[2]))
                if lines[3] != 'null':
                    high_col.append(float(lines[3]))
                if lines[4] != 'null':
                    low_col.append(float(lines[4]))
                if lines[5] != 'null':
                    clos_col.append(float(lines[5]))

        open_col = array(open_col)
        high_col = array(high_col)
        low_col = array(low_col)
        clos_col = array(clos_col)

        #raw_seq = array([open_col[i]+high_col[i]+low_col[i]+clos_col[i]
                         #for i in range(len(open_col))])

        Models.data_prep(self, open_col, high_col, low_col, clos_col, raw_seq)

    def split_data(raw_seq, n_steps, size):

        # split into samples
        X, y = Models.split_sequence(raw_seq, n_steps)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size)
        return X_train, X_test, y_train, y_test

    def data_prep(self, open_col, high_col, low_col, clos_col, raw_seq):

        # Reshape the array to columns and rows
        open_col = open_col.reshape((len(open_col), 1))
        high_col = high_col.reshape((len(high_col), 1))
        low_col = low_col.reshape((len(low_col), 1))
        clos_col = clos_col.reshape((len(clos_col), 1))

        # Stacks arrays side by side in one array
        raw_seq = hstack((open_col, high_col, low_col, clos_col))

        # Splits the data into test and train (data, windows, size of test)
        X_train, X_test, y_train, y_test = Models.split_data(
            raw_seq, self.timestep, 0.2)

        # Splits the data into test and val (data, windows, size of val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2)

        Models.check_model(self, X_train, X_val, y_train,
                          y_val, X_test, y_test, raw_seq)

    def plotting(history):
        # Plot accuracy metrics
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.legend()
        plt.show()

        # plt.title('Accuracy')
        # plt.plot(history.history['accuracy'], label='Train')
        # plt.plot(history.history['val_accuracy'], label='Val')
        # plt.legend()
        # plt.show()

        return

    def accuracy(yhat, y_test, X_test):
        columns = ['Open', 'High', 'Low', 'Close']
        for i in range(0,4):
            print(columns[i])
            print("Mean absolute error =", round(
                sm.mean_absolute_error(y_test[:,i], yhat[:,i]), 4))
            print("Mean squared error =", round(
                sm.mean_squared_error(y_test[:,i], yhat[:,i]), 4))
            print("Explain variance score =", round(
                sm.explained_variance_score(y_test[:,i], yhat[:,i]), 4))
            print("R2 score =", round(sm.r2_score(y_test[:,i], yhat[:,i]), 4))
        print("R2 score =", round(sm.r2_score(y_test, yhat), 4))

    def direction(yhat):
        if yhat >= 0.5:
            print('UP')
        if yhat < 0.5:
            print('Down')

    def check_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):

        if self.model_type == 'CNN':
            history, model = CNN.CNN_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Models.plotting(history)
            yhat = CNN.CNN_test_model(X_test, model, self.verbose, y_test)
            Models.accuracy(yhat, y_test, X_test)

        if self.model_type == 'MLP':
            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = MLP.data_format( self, X_train, X_val, y_train, y_val, self.verbose, X_test, y_test, self.layers)
            history, model = MLP.MLP_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            Models.plotting(history)
            yhat = MLP.MLP_test_model(X_test, model, self.verbose, y_test)
            Models.accuracy(yhat, y_test, X_test)

        if self.model_type == 'KNN':
            X_train, X_val, y_train, X_test= KNN.data_format(X_train, X_val, y_train, X_test)
            yhat = KNN.KNN_train_model(
                self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq)
            Models.accuracy(yhat, y_test, X_test)

        if self.model_type == 'LSTM':
            
            history, model = LSTMs.LSTM_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Models.plotting(history)
            yhat = LSTMs.LSTM_test_model(X_test, model, self.verbose)
            Models.accuracy(yhat, y_test, X_test)


class CNN:

    def CNN_train_model(self, X_train, X_val, y_train, y_val, verbose):
        features = X_train.shape[2]
        neuron_Val = 50
        # define model
        model = Sequential()
        # Conveluted layer
        model.add(Conv1D(filters=2, kernel_size=2,
                         activation='relu', input_shape=(self.timestep, features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(neuron_Val, activation='relu'))
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mean_squared_error'])
        # model.summary()

        # fit model
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=verbose, shuffle=True)

        return history, model

    def CNN_test_model(X_test, model, verbose, y_test):

        yhat = model.predict(X_test, verbose=verbose)
        return yhat


class MLP:
    def data_format(self, X_train, X_val, y_train, y_val, verbose, X_test, y_test, layers):
        #Assigns input size dynamic
        n_input = X_train.shape[1] * X_train.shape[2]
        # Reshapes input data
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_val = X_val.reshape((X_val.shape[0], n_input))
        # Defines each output
        ytrain1 = y_train[:,0].reshape((y_train.shape[0], 1))
        ytrain2 = y_train[:,1].reshape((y_train.shape[0], 1))
        ytrain3 = y_train[:,2].reshape((y_train.shape[0], 1))
        ytrain4 = y_train[:,3].reshape((y_train.shape[0], 1))

        n_output = y_train.shape[1]

        r2 = 0
        mae = 0
        mse = 0

        #return X_train, X_val, y_train, n_input, n_output

    #def MLP_train_model(self, X_train, X_val, y_train, y_val, verbose, n_input, n_output):
        for i in range(1,5):
            print('loop:',i)
            visible = Input(shape=(n_input,))
            dense = Dense(layers, activation='relu')(visible)

            open_out = Dense(1)(dense)
            high_out = Dense(1)(dense)
            low_out = Dense(1)(dense)
            close_out = Dense(1)(dense)

            model = Model(inputs=visible, outputs=[open_out, high_out, low_out, close_out])
            model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])


            # fit model
            history = model.fit(X_train, [ytrain1, ytrain2, ytrain3, ytrain4], validation_data=(
                X_val, y_val), epochs=1000, verbose=verbose)
            #model.summary()

            #return history, model

        #def MLP_test_model(X_test, model, verbose, y_test):



            yhat = model.predict(X_test, verbose=verbose)

            X = (round(sm.mean_absolute_error(y_test, yhat), 20))
            y = (round(sm.mean_squared_error(y_test, yhat), 20))
            q = (round(sm.r2_score(y_test, yhat), 20))

            if r2 == 0:
                r2 = q
            elif r2 > q:
                r2 = q

            if mae == 0:
                mae = X
            elif mae > X:
                mae = X

            if mse == 0:
                mse = y
            elif mse > y:
                mse = y
        

        with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/data.csv', 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
            #for i in range(0, 4):
            csvwriter.writerow([mae, mse, r2])

        #return yhat


class KNN:

    def data_format(self, X_train, y_train, X_test, y_test):
        n_input = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_test = X_test.reshape((X_test.shape[0], n_input))
        n_output = y_train.shape[1]

        r2 = 0
        mae = 0
        mse = 0
        
        #return X_train, X_val, y_train, X_test

    #def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        #kval = 1000
        for i in range(1,5):
            print('loop:', i)
            classifier = KNeighborsRegressor(n_neighbors = self.layers)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            X = (round(sm.mean_absolute_error(y_test, y_pred), 20))
            y = (round(sm.mean_squared_error(y_test, y_pred), 20))
            q = (round(sm.r2_score(y_test, y_pred), 20))

            print(y_pred)

            print(q)

            if r2 == 0:
                r2 = q
            elif r2 > q:
                r2 = q

            if mae == 0:
                mae = X
            elif mae > X:
                mae = X

            if mse == 0:
                mse = y
            elif mse > y:
                mse = y
        

        with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/data.csv', 'a+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
            #for i in range(0, 4):
            csvwriter.writerow([mae, mse, r2])

        
        #return y_pred


class LSTMs:

    def LSTM_train_model(self, X_train, X_val, y_train, y_val, verbose):

        features = X_train.shape[2]

        model = Sequential()
        model.add(LSTM(50, activation='relu',
                       return_sequences=True, input_shape=(self.timestep, features)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(4))
        model.compile(optimizer='adam',
                      loss='mse', metrics=['mean_squared_error'])
        model.summary()

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=200, verbose=2)

        return history, model

    def LSTM_test_model(X_test, model, verbose):

        yhat = model.predict(X_test, verbose=verbose)

        print(yhat)

        return yhat


if __name__ == "__main__":
    timeframe = [1,20,50,100,250,500,1000]
    for i in timeframe:
        window = i
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 1)
        Open.data()
    for i in timeframe:
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 10)
        Open.data()
    for i in timeframe:
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 100)
        Open.data()
    for i in timeframe:
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 250)
        Open.data()
    for i in timeframe:
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 500)
        Open.data()
    for i in timeframe:
        print('Window:', i)
        Open = Models('EURUSD', window, 'MLP', 1000)
        Open.data()
