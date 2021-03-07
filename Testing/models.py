from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from tensorflow.python.keras.callbacks import History


class CNN:

    def CNN_train_model(self, X_train, X_val, y_train, y_val, verbose):
        features = X_train.shape[2]
        # define model
        model = Sequential()
        # Conveluted layer
        model.add(Conv1D(filters=64, kernel_size=2,
                         activation='relu', input_shape=(self.timestep, features)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(4))
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
        #model.summary()

        # fit model
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=1000, verbose=verbose, shuffle=True)

        return history, model

    def CNN_test_model(X_test, model, verbose, y_test):

        yhat = model.predict(X_test, verbose=verbose)
        return yhat


class MLP:
    def data_format(X_train, X_val, y_train):
        n_input = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_val = X_val.reshape((X_val.shape[0], n_input))
        n_output = y_train.shape[1]

        return X_train, X_val, y_train, n_input, n_output


    def MLP_train_model(self, X_train, X_val, y_train, y_val, verbose, n_input, n_output ):

        model = Sequential()
        model.add(Dense(200, activation='relu', input_dim=(n_input)))
        #model.add(Dense(2, activation='relu'))
        model.add(Dense(n_output))
        model.compile(optimizer='adam', loss='mse', metrics=['mean_squared_error'])
        model.summary()

        # fit model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=1000, verbose=verbose)

        return history, model

    def MLP_test_model(X_test, model, verbose, y_test):

        n_input = X_test.shape[1] * X_test.shape[2]
        X_test = X_test.reshape((X_test.shape[0], n_input))
        yhat = model.predict(X_test, verbose=verbose)

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
        for m in range(1, 148):
            classifier = KNeighborsRegressor(n_neighbors=m)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            columns = ['Open', 'High', 'Low', 'Close']
            for i in range(0,4):
                print(columns[i])
                print("Mean absolute error =", round(
                    sm.mean_absolute_error(y_test[i], y_pred[i]), 20))
                print("Mean squared error =", round(
                    sm.mean_squared_error(y_test[i], y_pred[i]), 20))
                print("Median absolute error =", round(
                    sm.median_absolute_error(y_test[i], y_pred[i]), 20))
                print("Explain variance score =", round(
                    sm.explained_variance_score(y_test[i], y_pred[i]), 20))
                print("R2 score =", round(sm.r2_score(y_test[i], y_pred[i]), 20))



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
