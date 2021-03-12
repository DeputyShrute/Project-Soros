# from scripts.models import LSTM
from numpy.core.shape_base import hstack
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from numpy import array
from models import CNN, MLP, KNN, LSTMs
import csv
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:

    def __init__(self, symbol, timestep, model_type, verbose=0):
        print('Constructor Initialised')
        self.symbol = symbol.upper()
        self.timestep = timestep
        self.model_type = model_type.upper()
        self.verbose = verbose

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

        Model.data_prep(self, open_col, high_col, low_col, clos_col, raw_seq)

    def split_data(raw_seq, n_steps, size):

        # split into samples
        X, y = Model.split_sequence(raw_seq, n_steps)

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
        X_train, X_test, y_train, y_test = Model.split_data(
            raw_seq, self.timestep, 0.2)

        # Splits the data into test and val (data, windows, size of val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2)

        Model.check_model(self, X_train, X_val, y_train,
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
            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = CNN.data_format(X_train, X_val, y_train)
            history, model = CNN.CNN_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            Model.plotting(history)
            yhat = CNN.CNN_test_model(X_test, model, self.verbose, y_test)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'MLP':
            X_train, X_val, y_train, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4 = MLP.data_format(X_train, X_val, y_train)
            history, model = MLP.MLP_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4)
            Model.plotting(history)
            yhat = MLP.MLP_test_model(X_test, model, self.verbose, y_test)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'KNN':
            X_train, X_val, y_train, X_test= KNN.data_format(X_train, X_val, y_train, X_test)
            yhat = KNN.KNN_train_model(
                self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'LSTM':
            
            history, model = LSTMs.LSTM_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Model.plotting(history)
            yhat = LSTMs.LSTM_test_model(X_test, model, self.verbose)
            Model.accuracy(yhat, y_test, X_test)


if __name__ == "__main__":
    Open = Model('EURUSD', 1000, 'CNN')
    Open.data()
