from sklearn.model_selection import train_test_split
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot
import tensorflow as tf
from numpy import array
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:

    def __init__(self, symbol, timestep, column, loop):
        print('Constructor Initialised')
        self.symbol = symbol
        self.timestep = timestep
        self.column = column.upper()
        if self.column == 'OPEN':
            self.column = 2
        if self.column == 'HIGH':
            self.column = 3
        if self.column == 'LOW':
            self.column = 4
        self.loop = loop

    def split_sequence(raw_seq, new_seq, n_steps):
        X, y = list(), list()
        for i in range(len(raw_seq)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(new_seq)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = raw_seq[i:end_ix], new_seq[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def data(self):
        # define input sequence
        raw_seq = []
        with open('C:/Users/Ryan Easter/OneDrive - University of Lincoln/University/Year 4 (Final)/Project/Artefact/Project-Soros/Data/' + self.symbol + '.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for lines in csv_reader:
                if lines[self.column] != 'null':
                    raw_seq.append(float(lines[self.column]))
        new_seq = []
        new_seq.append(0)
        j = len(raw_seq)
        for i in range(1, j):
            if raw_seq[i] > raw_seq[i-1]:
                new_seq.append(1) # UP
            if raw_seq[i] < raw_seq[i-1]:
                new_seq.append(0)# DOWN
            # if raw_seq[i] == raw_seq[i-1]:
            #     new_seq.append(0) # SIDE

        Model.model(self, raw_seq, new_seq)

    def split_data(raw_seq, new_seq ,n_steps, size):
        # split into samples
        X, y = Model.split_sequence(raw_seq, new_seq, n_steps)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size)
        return X_train, X_test, y_train, y_test

    def model(self, raw_seq, new_seq):
        average = []
        for i in range(self.loop):

            # Splits the data into test and train (data, windows, size of test)
            X_train, X_test, y_train, y_test = Model.split_data(
                raw_seq, new_seq, self.timestep, 0.2)

            # Splits the data into test and val (data, windows, size of val)
            X_train, X_val, y_train, y_val = Model.split_data(
                raw_seq, new_seq, self.timestep, 0.2)

            # Reshapes the data for input dimension
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            # define model
            model = Sequential()
            #Conveluted layer
            model.add(Conv1D(filters=128, kernel_size=2,
                             activation='relu', input_shape=(self.timestep, 1)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()

            # fit model
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val), epochs=500, verbose=2, shuffle=True)

            # Plot accuracy metrics
            pyplot.title('Loss / Mean Squared Error')
            pyplot.plot(history.history['loss'], label='Train')
            pyplot.plot(history.history['val_loss'], label='Val')
            pyplot.legend()
            pyplot.show()

            pyplot.title('Accuarcy')
            pyplot.plot(history.history['accuracy'], label='Train')
            pyplot.plot(history.history['val_accuracy'], label='Val')
            pyplot.legend()
            pyplot.show()


            #Test model
            #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            new_seq = array(raw_seq[-27:])
            new_seq = new_seq.reshape((1, self.timestep, 1))
            yhat = model.predict(new_seq, verbose=0)
            # Print and log output
            print("----------------")
            print(yhat)
            Model.direction(yhat)
            average.append(yhat)
            Model.log(yhat, i)

        # tot_avg = (sum(average)/self.loop)
        # print(tot_avg)

    def log(yhat, iteration):
        outF = open('output.txt', 'a')
        for i in yhat:
            output = "\nIteration: %d\n" % (iteration)
            column = "Open Price\n"
            outF.write(column)
            outF.write(str(output))
            outF.write(str(i).strip("[]"))
            outF.write('\n')
            outF.write('------------------------')
            outF.write('\n')
        outF.close()

    def direction(yhat):
        if yhat >= 0.5:
            print('UP')
        if yhat < 0.5:
            print('Down')

if __name__ == "__main__":
    Open = Model('AUDCAD', 27, 'open', 1)
    Open.data()
