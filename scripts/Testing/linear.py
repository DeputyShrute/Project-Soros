# univariate mlp example
from re import VERBOSE
import numpy as np
from numpy import array
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from numpy.core.fromnumeric import size

# split a univariate sequence into samples


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

		print(size(X_train))

		# define model
		model = Sequential()
		model.add(Dense(100, activation='relu', input_dim=n_steps))
		model.add(Dense(1))
		model.compile(optimizer='adam', loss='mse')
		model.summary()

		# fit model
		model.fit(X_train, y_train, epochs=1000, verbose=0)

		train_results = model.evaluate(X_train, y_train, verbose=0)
		print(f'RMSE TRAIN: {round(np.sqrt(train_results), 2)}')

		# demonstrate prediction
		test = array([0.94551, 0.9393, 0.9498, 0.9473, 0.9489, 0.9486, 0.9564, 0.9604, 0.9565, 0.9524, 0.9526, 0.9489, 0.9501, 0.9515, 0.952, 0.9495, 0.9547, 0.9559, 0.9556, 0.9524, 0.9522, 0.946, 0.9463, 0.9376, 0.9362, 0.9307, 0.9271])
		test = test.reshape((1, n_steps))
		yhat = model.predict(test, verbose=0)

		print("----------------")
		print(yhat)
		log(yhat, i)


def log(yhat, iteration):
	outF = open('output.txt', 'a')
	for i in yhat:
		output = "\nIteration: %d\n" %(iteration)
		outF.write(str(output))
		outF.write(str(i).strip("[]"))
		outF.write('\n')
		outF.write('------------------------')
		outF.write('\n')
	outF.close()


if __name__ == "__main__":
	data()
