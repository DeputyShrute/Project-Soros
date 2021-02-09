import csv
from re import VERBOSE
import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def data():
	# define input sequence
	raw_seq = []
	with open('/home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/AUDCAD.csv', 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		for lines in csv_reader: 
			if lines[1] != 'null':
				raw_seq.append(float(lines[1]))
	raw_seq.pop(0)  # remove column header
	#Normalise the data
	#norm = [float(i)/sum(raw_seq) for i in raw_seq]
	#print(raw_seq)
	model(raw_seq)


def split_data(raw_seq, n_steps):
	# split into samples
	X, y = split_sequence(raw_seq, n_steps)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	return X_train, X_test, y_train, y_test


def model(raw_seq):
	n_steps = 27
	for i in range(3):

		X_train, X_test, y_train, y_test = split_data(raw_seq, n_steps)

		X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

		# define model
		model = Sequential()
		model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, 1)))
		model.add(MaxPooling1D(pool_size=2))
		model.add(Flatten())
		model.add(Dense(50, activation='relu'))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')
		model.summary()

		# fit model
		model.fit(X_train, y_train, epochs=1000, verbose=0)

		# demonstrate prediction
		#test = np.asarray(raw_seq[-n_steps:])

		#norm = [float(i)/sum(test)for i in test]
		X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
		yhat = model.predict(X_test, verbose=0)

		plt.plot(yhat, X_train)
		plt.show()

		print("----------------")
		print(yhat)
		log(yhat, i)


def log(yhat, iteration):
	outF = open('output.txt', 'a')
	for i in yhat:
		output = "\nIteration: %d\n" %(iteration)
		column = "Open Price\n"
		outF.write(column)
		outF.write(str(output))
		outF.write(str(i).strip("[]"))
		outF.write('\n')
		outF.write('------------------------')
		outF.write('\n')
	outF.close()


if __name__ == "__main__":
	data()
