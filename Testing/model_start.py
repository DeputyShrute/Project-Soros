# from scripts.models import LSTM
from numpy.core.fromnumeric import shape
from numpy.core.shape_base import hstack
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import csv
import sklearn.metrics as sm
from tensorflow.python.keras.callbacks import History
from numpy.core.shape_base import hstack
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from numpy import array
from models import CNN, MLP, KNN, LSTMs
import csv
import os
import time
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Models:

    def __init__(self, symbol, timestep, model_type, filters, verbose=0):
        # Creates paramters when class initialised
        print('Constructor Initialised')
        self.symbol = symbol.upper()
        self.timestep = timestep
        self.model_type = model_type.upper()
        self.verbose = verbose
        self.filters = filters

    def __str__(self):
        # Used to compare self as a string
        return self.model_type

    def split_sequence(raw_seq, n_steps):
        # Splits the data up into the windows specified by n
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

        return array(X), array(y)

    def data(self):
        # Creates arrays to be used to specify each column
        open_col, high_col, low_col, clos_col, raw_seq = [], [], [], [], array([
        ])
        # Read input from CSV
        with open('Finance_Data/' + self.symbol + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            # Assignes each column within CSV to appropriate Array
            for lines in csv_reader:
                if lines[2] != 'null':
                    open_col.append(float(lines[2]))
                if lines[3] != 'null':
                    high_col.append(float(lines[3]))
                if lines[4] != 'null':
                    low_col.append(float(lines[4]))
                if lines[5] != 'null':
                    clos_col.append(float(lines[5]))

        # Converts list to a Numpy array
        open_col = array(open_col)
        high_col = array(high_col)
        low_col = array(low_col)
        clos_col = array(clos_col)
        # Call data prep
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

        return

    def accuracy(yhat, y_test, X_test):
        columns = ['Open', 'High', 'Low', 'Close']
        for i in range(0, 4):
            print(columns[i])
            print("Mean absolute error =", round(
                sm.mean_absolute_error(y_test[:, i], yhat[:, i]), 4))
            print("Mean squared error =", round(
                sm.mean_squared_error(y_test[:, i], yhat[:, i]), 4))
            print("Explain variance score =", round(
                sm.explained_variance_score(y_test[:, i], yhat[:, i]), 4))
            print("R2 score =", round(
                sm.r2_score(y_test[:, i], yhat[:, i]), 4))
        print("R2 score =", round(sm.r2_score(y_test, yhat), 4))

    def direction(yhat):
        if yhat >= 0.5:
            print('UP')
        if yhat < 0.5:
            print('Down')

    def check_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        """ Funtion used to navigate to the specific model. The is defined when initialising the class.
            Reads the self.model_type 
            Each statement does the following:
                - Calls function to format data for the model
                - Calls funtion to train the model
                - Calls funtion to plot the MSE graph
                - Calls funtion to test the model
                - Returns the accuarcy as R2 score"""

        if self.model_type == 'CNN':

            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = CNN.data_format(
                X_train, X_val, y_train)
            history, model = CNN.CNN_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4, self.filters)
            #Models.plotting(history)
            yhat = CNN.CNN_test_model(
                self, X_test, model, self.verbose, y_test)
            #Models.accuracy(yhat, y_test, X_test)

        if self.model_type == 'MLP':

            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = MLP.data_format(
                X_train, X_val, y_train)
            history, model = MLP.MLP_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            #Models.plotting(history)
            yhat = MLP.MLP_test_model(X_test, model, self.verbose, y_test)
            #Models.accuracy(yhat, y_test, X_test)

        if self.model_type == 'KNN':

            X_train, X_val, y_train, X_test = KNN.data_format(
                X_train, X_val, y_train, X_test)
            yhat = KNN.KNN_train_model(
                self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq)
            #Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'LSTM':

            history, model = LSTMs.LSTM_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Models.plotting(history)
            yhat = LSTMs.LSTM_test_model(X_test, model, self.verbose, y_test)
            Models.accuracy(yhat, y_test, X_test)

class CNN:
    def data_format(X_train, X_val, y_train):

        # Assigns input size dynamic
        n_input = X_train.shape[1] * X_train.shape[2]
        # Defines each output
        ytrain1 = y_train[:, 0].reshape((y_train.shape[0], 1))
        ytrain2 = y_train[:, 1].reshape((y_train.shape[0], 1))
        ytrain3 = y_train[:, 2].reshape((y_train.shape[0], 1))
        ytrain4 = y_train[:, 3].reshape((y_train.shape[0], 1))

        n_output = y_train.shape[1]

        return X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4

    def CNN_train_model(self, X_train, X_val, y_train, y_val, verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4, filter):
        #features = X_train.shape[2]
        # define model

        visible = Input(shape=(self.timestep, 4))
        cnn = Conv1D(filters=filter, kernel_size=2, activation='relu')(visible)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(250, activation='relu')(cnn)

        open_out = Dense(1)(cnn)
        high_out = Dense(1)(cnn)
        low_out = Dense(1)(cnn)
        close_out = Dense(1)(cnn)

        model = Model(inputs=visible, outputs=[
                      open_out, high_out, low_out, close_out])
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mean_squared_error'])

        history = model.fit(X_train, [ytrain1, ytrain2, ytrain3, ytrain4], validation_data=(
            X_val, y_val), epochs=500, verbose=self.verbose)

        return history, model

    def CNN_test_model(self, X_test, model, verbose, y_test):
        #X_test = X_test.reshape((1, self.timestep, 4 ))
        yhat = model.predict(X_test, verbose=verbose)

        yhat = np.concatenate((yhat), axis=1)
        print('Test:', X_test)
        print('Actual:', y_test)
        print('Predicted:', yhat)

        columns = ['Open', 'High', 'Low', 'Close']
        files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        for i in range(0,4):
            mae = round(sm.mean_absolute_error(y_test[:,i], yhat[:,i]), 20)
            mse = round(sm.mean_squared_error(y_test[:,i], yhat[:,i]), 20)
            r2 =round(sm.r2_score(y_test[:,i], yhat[:,i]), 20)
            with open("../Testing/" + files[i], "a+", newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
                #for i in range(0, 4):
                csvwriter.writerow([mae, mse, r2])

        return yhat


class MLP:
    def data_format(X_train, X_val, y_train):
        # Assigns input size dynamic
        n_input = X_train.shape[1] * X_train.shape[2]
        # Reshapes input data
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_val = X_val.reshape((X_val.shape[0], n_input))
        # Defines each output
        ytrain1 = y_train[:, 0].reshape((y_train.shape[0], 1))
        ytrain2 = y_train[:, 1].reshape((y_train.shape[0], 1))
        ytrain3 = y_train[:, 2].reshape((y_train.shape[0], 1))
        ytrain4 = y_train[:, 3].reshape((y_train.shape[0], 1))

        n_output = y_train.shape[1]

        return X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4

    def MLP_train_model(self, X_train, X_val, y_train, y_val, verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4):
        
        with open('config/MLP.json', 'r') as params:
            json_param = params.read()
        
        obj = json.loads(json_param)
        
        visible = Input(shape=(n_input,))
        dense = Dense(obj['neuron_val'], activation=str(obj['activate']))(visible)

        open_out = Dense(1)(dense)
        high_out = Dense(1)(dense)
        low_out = Dense(1)(dense)
        close_out = Dense(1)(dense)

        model = Model(inputs=visible, outputs=[
                      open_out, high_out, low_out, close_out])
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mean_squared_error'])

        # fit model
        history = model.fit(X_train, [ytrain1, ytrain2, ytrain3, ytrain4], validation_data=(
            X_val, y_val), epochs=obj['epochs'], verbose=verbose)

        return history, model

    def MLP_test_model(X_test, model, verbose, y_test):

        n_input = X_test.shape[1] * X_test.shape[2]
        X_test = X_test.reshape((X_test.shape[0], n_input))
        print(type(X_test))
        yhat = model.predict(X_test, verbose=verbose)

        print(np.shape(X_test))

        yhat = np.concatenate((yhat), axis=1)
        print('Test:\n', X_test)
        print('Actual:\n', y_test)
        print('Predicted:\n', yhat)

        # columns = ['Open', 'High', 'Low', 'Close']
        # files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        # for i in range(0,4):
        #     mae = round(sm.mean_absolute_error(y_test[:,i], yhat[:,i]), 20)
        #     mse = round(sm.mean_squared_error(y_test[:,i], yhat[:,i]), 20)
        #     r2 =round(sm.r2_score(y_test[:,i], yhat[:,i]), 20)
        #     with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/' + files[i], 'a+', newline='') as csvfile:
        #         csvwriter = csv.writer(csvfile)
        #         #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
        #         #for i in range(0, 4):
        #         csvwriter.writerow([mae, mse, r2])

        return yhat


class KNN:

    def data_format(X_train, X_val, y_train, X_test):
        n_input = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_val = X_val.reshape((X_val.shape[0], n_input))
        X_test = X_test.reshape((X_test.shape[0], n_input))
        n_output = y_train.shape[1]

        return X_train, X_val, y_train, X_test

    def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        k=1000
        classifier = KNeighborsRegressor(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        print('X_test:\n', X_test)
        print('y_test:\n', y_test)
        #print('y_test:\n', y_test[:,3])
        print('Y_pred:\n', y_pred)


        columns = ['Open', 'High', 'Low', 'Close']
        files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        for i in range(0,4):
            mae = round(sm.mean_absolute_error(y_test[:,i], y_pred[:,i]), 20)
            mse = round(sm.mean_squared_error(y_test[:,i], y_pred[:,i]), 20)
            r2 =round(sm.r2_score(y_test[:,i], y_pred[:,i]), 20)
            with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/' + files[i], 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
                #for i in range(0, 4):
                csvwriter.writerow([mae, mse, r2])


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

    def LSTM_test_model(X_test, model, verbose, y_test):

        yhat = model.predict(X_test, verbose=verbose)

        print('Test:', X_test)
        print('Actual:', y_test)
        print('Predicted:', yhat)

        return yhat

if __name__ == "__main__":

    Open = Models('AUDCAD', 1000, 'KNN', 1, 2)
    Open.data()

    # filters = [1,50,100,250,500,1000]
    # for i in filters:
    #     Open = Models('EURUSD', 500, 'CNN', i)
    #     print('500 ', i)
    #     Open.data()
    # for i in filters:
    #     Open = Models('EURUSD', 1000, 'CNN', i)
    #     print('1000 ', i)
    #     Open.data()



