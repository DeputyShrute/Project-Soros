# univariate mlp example
from numpy import array
import csv
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
 
# define input sequence
raw_seq = []
with open('/home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/AUDCAD.csv', 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	for lines in csv_reader:
		if lines[1] != 'null':
			raw_seq.append(float(lines[1]))

raw_seq.pop(0) # remove column header
print(size(raw_seq))

# choose a number of time steps
n_steps = 4
for i in range(3):
	
	# split into samples
	X, y = split_sequence(raw_seq, n_steps)

	# define model
	model = Sequential()
	model.add(Dense(800, activation='relu', input_dim=n_steps))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	model.summary()

	# fit model
	model.fit(X, y, epochs=1000, verbose=0)

	# demonstrate prediction
	x_input = array([0.9570, 0.9497, 0.9493, 0.9495])
	x_input = x_input.reshape((1, n_steps))
	yhat = model.predict(x_input, verbose=0)
	print("----------------")
	print(yhat)
	outF = open('output.txt', 'a')
	for i in yhat:
		outF.write(str(i))
		outF.write('\n')
	outF.close()