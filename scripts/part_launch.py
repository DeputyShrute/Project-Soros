from numpy.core.shape_base import hstack
from matplotlib.pyplot import xlim
from numpy import array, newaxis
from models import CNN
from tensorflow.python.keras.saving.model_config import model_from_json
from model_start import Models
from PIL import Image
import os
import csv

class launch:

    def darknet(X):
        try:
            os.chdir('/home/ryan/Documents/Python/Project-Soros/darknet/data')
            file = open('test.txt', 'w+')
            file.write(
                '//home/ryan/Documents/Python/Project-Soros/scripts/Finance_Data/Chart_Snapshot/{name}_test.jpg'.format(name=X))
            file.close
            os.chdir("/home/ryan/Documents/Python/Project-Soros/darknet")
            os.system("./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -dont_show -ext_output < data/test.txt > data/result.txt")
            predictions = Image.open('predictions.jpg')
            predictions.show()
            os.wait()
            launch.predictions(X)
        except FileNotFoundError:
            print('exception')
            return

    def predictions(X):

        # Loads saved model so retraining isn't needed
        json_file = open('saved_models/CNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('saved_models/CNN.h5')

        X = 'EURUSD'

        # Creates arrays to be used to specify each column
        open_col, high_col, low_col, clos_col, raw_seq = [], [], [], [], array([
        ])
        # Read input from CSV
        with open('scripts/Finance_Data/Raw_Data/' + X + '.csv', 'r') as csv_file:
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

        # Reshape the array to columns and rows
        open_col = open_col.reshape((len(open_col), 1))
        high_col = high_col.reshape((len(high_col), 1))
        low_col = low_col.reshape((len(low_col), 1))
        clos_col = clos_col.reshape((len(clos_col), 1))

        raw_seq = hstack((open_col, high_col, low_col, clos_col))

        # raw_seq.reshape(1, 4, 4472)
        # test = raw_seq[newaxis,:,:]
        # print(test)
        # print(test.shape)

        test = raw_seq.reshape(1,raw_seq.shape[0], raw_seq.shape[1])

        print(test.shape)

        yhat = model.predict(test, verbose=2)

if __name__ == "__main__":
    run = launch()
    run.predictions()