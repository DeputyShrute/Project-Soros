import os
from PIL import Image


class launch:

    def darkent(X):
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
            launch.predictions()
        except FileNotFoundError:
            print('exception')
            return

    def predictions():

