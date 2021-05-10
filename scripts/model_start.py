#!/usr/bin/env python -W ignore::DeprecationWarning
import logging
from sklearn.preprocessing import StandardScaler
from numpy.core.shape_base import hstack
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from numpy import array
from keras.backend import clear_session
from models import CNN, MLP, KNN, LSTMs, BaseLine
import csv
import os
import time
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Models:

    def __init__(self, symbol, timestep, model_type, verbose=0):
        # Creates paramters when class initialised
        print('Constructor Initialised')
        self.symbol = symbol.upper()
        self.timestep = timestep
        self.model_type = model_type.upper()
        self.verbose = verbose
        self.logger = logging.getLogger('myLogger')
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler('runs.log')
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

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
        with open('scripts/Finance_Data/Raw_Data/' + self.symbol + '.csv', 'r') as csv_file:
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

        # scaler = StandardScaler()
        # scaler.fit(raw_seq)#
        # scaler.transform(raw_seq)

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
        print('Modsel')
        plt.show()

        return

    def accuracy(self, yhat, y_test, final_cols, model_type):
        print("=====================")
        print("Accuracy Results")
        print("=====================\n")
        print(str(model_type))
        columns = ['Open', 'High', 'Low', 'Close']
        for i in range(0, 4):
            print(columns[i])
            print("Mean absolute error =", round(
                sm.mean_absolute_error(y_test[:, i], yhat[:, i]), 4))
            print("Mean squared error =", round(
                sm.mean_squared_error(y_test[:, i], yhat[:, i], squared=True), 4))
            print("Explain variance score =", round(
                sm.explained_variance_score(y_test[:, i], yhat[:, i]), 4))
            print("RMSE =", round(
                sm.mean_squared_error(y_test[:, i], yhat[:, i], squared=False), 4))
            print("R2 score =", round(
                sm.r2_score(y_test[:, i], yhat[:, i]), 4))
        print("\nOverall Accuracy")
        print("Mean absolute error =", round(
            sm.mean_absolute_error(y_test, yhat), 4))
        print("Mean squared error =", round(
            sm.mean_squared_error(y_test, yhat, squared=True), 4))
        print("Explain variance score =", round(
            sm.explained_variance_score(y_test, yhat), 4))
        print("RMSE =", round(
            sm.mean_squared_error(y_test, yhat, squared=False), 4))
        print("R2 score =", round(
            sm.r2_score(y_test, yhat), 4))
        print("R2 score =", round(sm.r2_score(y_test, yhat), 4))

        if model_type == 'MLP':
            MLP.MLP_analyse(y_test, yhat, final_cols)
        if model_type == 'BASELINE':
            MLP.MLP_analyse(y_test, yhat, final_cols)
        if model_type == 'KNN':
            MLP.MLP_analyse(y_test, yhat, final_cols)
        if model_type == 'CNN':
            LSTMs.LSTM_analyse(self, y_test, yhat, final_cols)
        if model_type == 'LSTM':
            LSTMs.LSTM_analyse(self, y_test, yhat, final_cols)
        # with open('model_config/' + model_type + '.json', 'r') as params:
        #     json_param = params.read()
        # obj = json.loads(json_param)

        self.logger.info(model_type)
        for i in range(0, 4):
            self.logger.info(columns[i])
            self.logger.info("Mean absolute error =%s", round(
                sm.mean_absolute_error(y_test[:, i], yhat[:, i]), 4))
            self.logger.info("Mean squared error =%s", round(
                sm.mean_squared_error(y_test[:, i], yhat[:, i], squared=True), 9))
            self.logger.info("Explain variance score =%s", round(
                sm.explained_variance_score(y_test[:, i], yhat[:, i]), 4))
            self.logger.info("RMSE =%s", round(
                sm.mean_squared_error(y_test[:, i], yhat[:, i], squared=False), 4))
            self.logger.info("R2 score =%s", round(
                sm.r2_score(y_test[:, i], yhat[:, i]), 4))
        self.logger.info("R2 score =%s", round(sm.r2_score(y_test, yhat), 4))
        self.logger.info("End\n")

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
            history = CNN.CNN_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            Models.plotting(history)
            yhat = CNN.CNN_test_model(
                self, X_test, self.verbose, y_test)
            Models.accuracy(self, yhat, y_test, X_test, self.model_type)

        if self.model_type == 'MLP':

            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = MLP.data_format(
                X_train, X_val, y_train)
            history = MLP.MLP_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            # Models.plotting(history)
            yhat, final_cols = MLP.MLP_test_model(X_test, self.verbose, y_test)
            Models.accuracy(self, yhat, y_test, final_cols, self.model_type)

        if self.model_type == 'KNN':

            X_train, X_val, y_train, X_test = KNN.data_format(
                X_train, X_val, y_train, X_test)
            yhat, final_cols = KNN.KNN_train_model(
                self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq)
            Models.accuracy(self, yhat, y_test, final_cols, self.model_type)

        if self.model_type == 'LSTM':

            history, model = LSTMs.LSTM_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Models.plotting(history)
            yhat = LSTMs.LSTM_test_model(X_test, model, self.verbose, y_test)
            Models.accuracy(self, yhat, y_test, X_test, self.model_type)

        if self.model_type == 'BASELINE':
            n_input, X_train, n_output = BaseLine.data_format(X_train, y_train)
            model = BaseLine.baseline_train(
                self, X_train, y_train, n_input, n_output)
            yhat, final_cols = BaseLine.baseline_test(X_test, n_input, model)
            Models.accuracy(self, yhat, y_test, final_cols, self.model_type)


if __name__ == "__main__":
    clear_session()
    Open = Models('EURUSD', 1, 'KNN', 2)
    Open.data()
