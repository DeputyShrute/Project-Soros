from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
import statistics
from tensorflow.python.keras.callbacks import History


class CNN:

    def CNN_train_model(self, X_train, X_val, y_train, y_val, verbose, X_test, y_test):

        # Reshapes the data for input dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        # define model
        model = Sequential()
        # Conveluted layer
        model.add(Conv1D(filters=2, kernel_size=self.timestep-1,
                         activation='relu', input_shape=(self.timestep, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()

        # fit model
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=verbose, shuffle=True)

        return history, model

    def CNN_test_model(X_test, model, verbose, y_test):

        #new_seq = array(raw_seq[-27:])
        # print(new_seq)
        #new_seq = new_seq.reshape((1, self.timestep, 1))

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        yhat = model.predict(X_test, verbose=verbose)

        # print(yhat)
        return yhat


class MLP:

    def MLP_train_model(self, X_train, X_val, y_train, y_val, verbose):

        model = Sequential()
        model.add(Dense(200, activation='relu', input_dim=(self.timestep)))
        #model.add(Dense(2, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()

        # fit model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=1000, verbose=verbose)

        return history, model

    def MLP_test_model(X_test, model, verbose, y_test):

        yhat = model.predict(X_test, verbose=verbose)
        score = mean_absolute_error(y_test, yhat)
        print('MAE: %.3f' % score)

        return yhat


class KNN:

    def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        X_test = [[1.211534,1.216619,1.216856,1.214329,1.216323,1.210478,1.212136,1.212283,1.206768,1.203905,1.204181,1.196745,1.204935,1.205255,1.211695,1.211945,1.213003,1.212106,1.213239,1.2088540]]
        for m in range (1,148):
            classifier = KNeighborsRegressor(n_neighbors=m)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            print(y_pred)
            #print(y_test)
            accuracy = []
            for i in range(len(y_pred)):
                acheived = float(y_pred[i])
                #print('Acheived: ',acheived)
                actual = float(y_test[i])
                #print('Actual: ', actual)
                val = (acheived/actual)*100
                accuracy.append(val)
            # print(accuracy)
            new_acc = []
            for j in range(len(accuracy)):
                if accuracy[j] < 100:
                    val = float(accuracy[j])
                    new_acc.append(100-val)

                if accuracy[j] > 100:
                    val = float(accuracy[j])
                    new_acc.append(val - 100)

            total = (statistics.mean(new_acc)*100)

            print('K:',m, total)


class LSTMs:

    def LSTM_train_model(self, X_train, X_val, y_train, y_val, verbose, X_test, y_test):

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='relu',
                       return_sequences=True, input_shape=(self.timestep, 1)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam',
                      loss='mse', metrics=['accuracy'])
        model.summary()

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=200, verbose=2)

        return history, model

    def LSTM_test_model(X_test, model, verbose, y_test):

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        yhat = model.predict(X_test, verbose=verbose)

        print(yhat)

        return yhat
