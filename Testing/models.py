#!/usr/bin/env python -W ignore::DeprecationWarning
from tensorflow.python.saved_model.function_deserialization import load_function_def_library
from tensorflow.python.keras.saving.model_config import model_from_json
from tensorflow.python.keras.callbacks import History
import sklearn.metrics as sm
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from re import VERBOSE
from numpy.core.fromnumeric import shape, size
from numpy.core.shape_base import hstack
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from keras.models import save_model, load_model
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


class BaseLine:
    def data_format(X_train, y_train):
        n_input = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], n_input))
        n_output = y_train.shape[1]
        return n_input, X_train, n_output

    def baseline_train(self, X_train, y_train, n_input, n_output):
        model = Sequential()
        model.add(Dense(1000, activation='relu', input_dim=n_input))
        #model.add(Dense(1000, activation='relu', input_dim=n_input))
        model.add(Dense(n_output))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        model.fit(X_train, y_train, epochs=100, verbose=2)

        return model

    def baseline_test(X_test, n_input, model):
        n_input = X_test.shape[1] * X_test.shape[2]
        X_test = X_test.reshape((X_test.shape[0], n_input))
        yhat = model.predict(X_test, verbose=0)

        close = X_test[:, -1]
        low = X_test[:, -2]
        high = X_test[:, -3]
        opens = X_test[:, -4]

        final_cols = np.column_stack((opens, high, low, close))
        


        return yhat, final_cols


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

    def CNN_train_model(self, X_train, X_val, y_train, y_val, verbose, n_input, n_output, ytrain1, ytrain2, ytrain3, ytrain4):
        #features = X_train.shape[2]
        # define model

        with open('model_config/CNN.json', 'r') as params:
            json_param = params.read()

        obj = json.loads(json_param)

        visible = Input(shape=(self.timestep, 4))
        cnn = Conv1D(filters=obj['filters'], kernel_size=obj['kernel_size'],
                     activation=obj['activation_conv1d'])(visible)
        cnn = MaxPooling1D(pool_size=obj['pool_size'])(cnn)
        cnn = Flatten()(cnn)
        cnn = Dense(obj['Dense_layer_1'],
                    activation=obj['activation_dense'])(cnn)

        open_out = Dense(1)(cnn)
        high_out = Dense(1)(cnn)
        low_out = Dense(1)(cnn)
        close_out = Dense(1)(cnn)

        model = Model(inputs=visible, outputs=[
                      open_out, high_out, low_out, close_out])

        model.compile(optimizer='adam', loss='mse',
                      metrics=['mean_squared_error'])

        history = model.fit(X_train, [ytrain1, ytrain2, ytrain3, ytrain4], validation_data=(
            X_val, y_val), epochs=obj['epochs'], verbose=self.verbose)

        # Saved the model as JSON and .h5
        model_json = model.to_json()
        with open('saved_models/CNN.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('saved_models/CNN.h5')

        return history

    def CNN_test_model(self, X_test, verbose, y_test):
        #X_test = X_test.reshape((1, self.timestep, 4 ))

        # Loads saved model so retraining isn't needed
        json_file = open('saved_models/CNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('saved_models/CNN.h5')

        print(X_test)
        print(X_test.shape)

        # Predicts on trained model
        yhat = model.predict(X_test, verbose=verbose)

        # Prints the results
        yhat = np.concatenate((yhat), axis=1)
        print('Test:', X_test)
        print('Next Actual:\n', y_test)
        print('Next Predicted:\n', yhat)

        # columns = ['Open', 'High', 'Low', 'Close']
        # files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        # for i in range(0,4):
        # with open("../Testing/" + files[i], "a+", newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
        #     #for i in range(0, 4):
        #     csvwriter.writerow([mae, mse, r2])

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

        with open('model_config/MLP.json', 'r') as params:
            json_param = params.read()

        obj = json.loads(json_param)

        visible = Input(shape=(n_input,))
        dense = Dense(obj['neuron_val'], activation=str(
            obj['activate']))(visible)

        open_out = Dense(1)(dense)
        high_out = Dense(1)(dense)
        low_out = Dense(1)(dense)
        close_out = Dense(1)(dense)

        model = Model(inputs=visible, outputs=[
                      open_out, high_out, low_out, close_out])
        model.compile(optimizer='adam', loss='mse',
                      metrics=['mean_squared_error'])
        model.summary()

        # fit model
        history = model.fit(X_train, [ytrain1, ytrain2, ytrain3, ytrain4], validation_data=(
            X_val, y_val), epochs=obj['epochs'], verbose=verbose)

        # Saved the model as JSON and .h5
        model_json = model.to_json()
        with open('saved_models/MLP.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('saved_models/MLP.h5')

        return history, model

    def MLP_test_model(X_test, verbose, y_test):

        # Loads saved model so retraining isn't needed
        json_file = open('saved_models/MLP.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('saved_models/MLP.h5')

        n_input = X_test.shape[1] * X_test.shape[2]
        X_test = X_test.reshape((X_test.shape[0], n_input))
        print(type(X_test))
        yhat = model.predict(X_test, verbose=verbose)

        yhat = np.concatenate((yhat), axis=1)
        print('Test:\n', X_test)

        close = X_test[:, -1]
        low = X_test[:, -2]
        high = X_test[:, -3]
        opens = X_test[:, -4]

        final_cols = np.column_stack((opens, high, low, close))

        print('Current:\n Open   High    Low    Close\n', final_cols)

        print('Next:\n', y_test)
        print('Predicted:\n', yhat)

        columns = ['Open', 'High', 'Low', 'Close']
        files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        for i in range(0, 4):
            mae = round(sm.mean_absolute_error(y_test[:, i], yhat[:, i]), 20)
            mse = round(sm.mean_squared_error(y_test[:, i], yhat[:, i]), 20)
            r2 = round(sm.r2_score(y_test[:, i], yhat[:, i]), 20)
        #     with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/' + files[i], 'a+', newline='') as csvfile:
        #         csvwriter = csv.writer(csvfile)
        #         #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
        #         #for i in range(0, 4):
        #         csvwriter.writerow([mae, mse, r2])

        return yhat, final_cols

    def MLP_analyse(y_test, yhat, final_cols):
        # Loop checks each value and whether it moves in the correct direction
        # This loop looks at all columns
        correct_dir = []
        for i in range(len(final_cols)):
            for j in range(len(final_cols[i])):
                if y_test[i][j] < final_cols[i][j]:
                    if yhat[i][j] < final_cols[i][j]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)
                elif y_test[i][j] > final_cols[i][j]:
                    if yhat[i][j] > final_cols[i][j]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)

        correct = sum(correct_dir)
        total = correct / (size(correct_dir))
        print("Number of total values", size(correct_dir))
        print("All values correct: ", correct)
        print("Total percentage of correct:", total)

        # This loops applies the same logic but applies to open and close column as they are the more important
        correct_dir = []
        for i in range(len(final_cols)):
            if y_test[i][0] < final_cols[i][0]:
                if yhat[i][0] < final_cols[i][0]:
                    correct_dir.append(1)
                    # print("========")
                    # print("Current: ",final_cols[i][0])
                    # print("Next: ",y_test[i][0])
                    # print("Predicted: ", round(yhat[i][0],5))
                else:
                    correct_dir.append(0)
            elif y_test[i][0] > final_cols[i][0]:
                if yhat[i][0] > final_cols[i][0]:
                    correct_dir.append(1)
                    # print("========")
                    # print("Current: ",final_cols[i][0])
                    # print("Next: ",y_test[i][0])
                    # print("Predicted: ", round(yhat[i][0],5))
                else:
                    correct_dir.append(0)
            elif y_test[i][3] < final_cols[i][3]:
                if yhat[i][3] < final_cols[i][3]:
                    correct_dir.append(1)
                    # print("========")
                    # print("Current: ",final_cols[i][3])
                    # print("Next: ",y_test[i][3])
                    # print("Predicted: ", round(yhat[i][3],5))
                else:
                    correct_dir.append(0)
            elif y_test[i][3] > final_cols[i][3]:
                if yhat[i][3] > final_cols[i][3]:
                    correct_dir.append(1)
                    # print("========")
                    # print("Current: ",final_cols[i][3])
                    # print("Next: ",y_test[i][3])
                    # print("Predicted: ", round(yhat[i][3],5))
                else:
                    correct_dir.append(0)

        correct = sum(correct_dir)
        total = correct / (size(correct_dir))
        print("Number of total values", size(correct_dir))
        print("All values correct: ", correct)
        print("Total percentage of correct:", total)


class KNN:

    def data_format(X_train, X_val, y_train, X_test):
        n_input = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], n_input))
        X_val = X_val.reshape((X_val.shape[0], n_input))
        X_test = X_test.reshape((X_test.shape[0], n_input))
        n_output = y_train.shape[1]

        return X_train, X_val, y_train, X_test

    def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        k = 5
        classifier = KNeighborsRegressor(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        close = X_test[:, -1]
        low = X_test[:, -2]
        high = X_test[:, -3]
        opens = X_test[:, -4]

        final_cols = np.column_stack((opens, high, low, close))

        print('X_test:\n', final_cols)
        print('y_test:\n', y_test)
        #print('y_test:\n', y_test[:,3])
        print('Y_pred:\n', y_pred)

        return y_pred, final_cols
        # columns = ['Open', 'High', 'Low', 'Close']
        # files = ['open.csv', 'high.csv', 'low.csv', 'close.csv']
        # for i in range(0,4):
        #     mae = round(sm.mean_absolute_error(y_test[:,i], y_pred[:,i]), 20)
        #     mse = round(sm.mean_squared_error(y_test[:,i], y_pred[:,i]), 20)
        #     r2 =round(sm.r2_score(y_test[:,i], y_pred[:,i]), 20)
        #     with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Testing/' + files[i], 'a+', newline='') as csvfile:
        #         csvwriter = csv.writer(csvfile)
        #         #csvwriter.writerow(['Column', 'Mean absolute error', 'Mean squared error', 'Explain variance score', 'R2 score', kval])
        #         #for i in range(0, 4):
        #         csvwriter.writerow([mae, mse, r2])


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
            X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=2)

        # Saved the model as JSON and .h5
        model_json = model.to_json()
        with open('saved_models/LSTM.json', 'w') as json_file:
            json_file.write(model_json)
        model.save_weights('saved_models/MLP.h5')

        return history, model

    def LSTM_test_model(X_test, model, verbose, y_test):

        # # Loads saved model so retraining isn't needed
        # json_file = open('saved_models/MLP.json', 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # model = model_from_json(loaded_model_json)
        # model.load_weights('saved_models/MLP.h5')

        yhat = model.predict(X_test, verbose=verbose)

        print('Test:', X_test)
        print('Actual:', y_test)
        print('Predicted:', yhat)

        return yhat

    def LSTM_analyse(self, y_test, yhat, final_cols):
        correct_dir = []
        for i in range(len(final_cols)):
            for j in range(len(final_cols[i][self.timestep-1])):
                if y_test[i][j] < final_cols[i][self.timestep-1][j]:
                    if yhat[i][j] < final_cols[i][self.timestep-1][j]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)
                elif y_test[i][j] > final_cols[i][self.timestep-1][j]:
                    if yhat[i][j] > final_cols[i][self.timestep-1][j]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)

        correct = sum(correct_dir)
        total = correct / (size(correct_dir))
        print("Number of total values", size(correct_dir))
        print("All values correct: ", correct)
        print("Total percentage of correct:", total)

        correct_dir = []
        for i in range(len(final_cols)):
            for j in range(len(final_cols[i][self.timestep-1])):
                if y_test[i][0] < final_cols[i][self.timestep-1][0]:
                    if yhat[i][0] < final_cols[i][self.timestep-1][0]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)
                elif y_test[i][0] > final_cols[i][self.timestep-1][0]:
                    if yhat[i][0] > final_cols[i][self.timestep-1][0]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)
                elif y_test[i][3] < final_cols[i][self.timestep-1][3]:
                    if yhat[i][3] < final_cols[i][self.timestep-1][3]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)
                elif y_test[i][3] > final_cols[i][self.timestep-1][3]:
                    if yhat[i][3] > final_cols[i][self.timestep-1][3]:
                        correct_dir.append(1)
                    else:
                        correct_dir.append(0)

        correct = sum(correct_dir)
        total = correct / (size(correct_dir))
        print("Number of total values", size(correct_dir))
        print("All values correct: ", correct)
        print("Total percentage of correct:", total)
