# from scripts.models import LSTM
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from numpy import array
from models import CNN, MLP, KNN, LSTMs
import statistics
import csv
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:

    def __init__(self, symbol, timestep, column, model_type, verbose=0):
        print('Constructor Initialised')
        self.symbol = symbol.upper()
        self.timestep = timestep
        self.column = column.upper()
        if self.column == 'OPEN':
            self.column = 2
        if self.column == 'HIGH':
            self.column = 3
        if self.column == 'LOW':
            self.column = 4
        if self.column == 'CLOSE':
            self.column = 5
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
            seq_x, seq_y = raw_seq[i:end_ix], raw_seq[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        # raw_seq = [float(i)/max(raw_seq) for i in raw_seq]

        return array(X), array(y)

    def data(self):
        # Read input from CSV
        raw_seq = []
        with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Data/' + self.symbol + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for lines in csv_reader:
                if lines[self.column] != 'null':
                    raw_seq.append(float(lines[self.column]))

        Model.train_test(self, raw_seq)

    def split_data(raw_seq, n_steps, size):

        # split into samples
        X, y = Model.split_sequence(raw_seq, n_steps)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size)
        return X_train, X_test, y_train, y_test

    def train_test(self, raw_seq, ):
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
        print("Mean absolute error =", round(sm.mean_absolute_error(y_test, yhat), 2)) 
        print("Mean squared error =", round(sm.mean_squared_error(y_test, yhat), 2)) 
        print("Median absolute error =", round(sm.median_absolute_error(y_test, yhat), 2)) 
        print("Explain variance score =", round(sm.explained_variance_score(y_test, yhat), 2)) 
        print("R2 score =", round(sm.r2_score(y_test, yhat), 2))

    def direction(yhat):
        if yhat >= 0.5:
            print('UP')
        if yhat < 0.5:
            print('Down')

    def check_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):

        if self.model_type == 'CNN':
            history, model = CNN.CNN_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, X_test, y_test)
            Model.plotting(history)
            yhat = CNN.CNN_test_model(X_test, model, self.verbose, y_test)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'MLP':
            history, model = MLP.MLP_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose)
            Model.plotting(history)
            yhat = MLP.MLP_test_model(X_test, model, self.verbose, y_test)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'KNN':
            yhat = KNN.KNN_train_model(
                self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq)
            Model.accuracy(yhat, y_test, X_test)

        if self.model_type == 'LSTM':
            history, model = LSTMs.LSTM_train_model(
                self, X_train, X_val, y_train, y_val, self.verbose, X_test, y_test)
            Model.plotting(history)
            yhat = LSTMs.LSTM_test_model(X_test, model, self.verbose, y_test)
            Model.accuracy(yhat, y_test, X_test)


if __name__ == "__main__":
    Open = Model('EURUSD', 20, 'open', 'MLP', 0)
    Open.data()
