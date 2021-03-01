from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report
from sklearn import metrics
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
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
        model.add(Dense(100, activation='relu'))
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

        yhat_new = []
        acc = []
        for i in yhat:
            if i > 0.5:
                yhat_new.append(1)
            if i < 0.5:
                yhat_new.append(0)

        result = metrics.confusion_matrix(y_test, yhat_new)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, yhat_new)
        print("Classification Report:",)
        print(result1)
        print(accuracy_score(y_test, yhat_new))

        return yhat


class MLP:

    def MLP_train_model(self, X_train, X_val, y_train, y_val, verbose):

        model = Sequential()
        model.add(Dense(1000, activation='relu', input_dim=(self.timestep)))
        #model.add(Dense(2, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()

        # fit model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=5000, verbose=verbose)

        return history, model

    def MLP_test_model(X_test, model, verbose, y_test):

        yhat = model.predict(X_test, verbose=verbose)
        score = mean_absolute_error(y_test, yhat)
        print('MAE: %.3f' % score)

        yhat_new = []
        acc = []
        for i in yhat:
            if i > 0.5:
                yhat_new.append(1)
            if i < 0.5:
                yhat_new.append(0)

        result = metrics.confusion_matrix(y_test, yhat_new)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, yhat_new)
        print("Classification Report:",)
        print(result1)
        print(accuracy_score(y_test, yhat_new))

        return yhat


class KNN:

    def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        Range_k = 1
        scores = {}
        scores_list = []
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # print(y_pred)

        scores = metrics.accuracy_score(y_test, y_pred)
        scores_list.append(metrics.accuracy_score(y_test, y_pred))
        result = metrics.confusion_matrix(y_test, y_pred)

        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, y_pred)
        print("Classification Report:",)
        print(result1)
        print(accuracy_score(y_test, y_pred))

        plt.plot(Range_k, scores_list)
        plt.title('Model Accuarcy / K Value')
        plt.xlabel("Value of K")
        plt.ylabel("Accuracy")
        plt.show()


class LSTMs:

    def LSTM_train_model(self, X_train, X_val, y_train, y_val, verbose, X_test, y_test):
        
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, activation='sigmoid',return_sequences=True, input_shape=(self.timestep, 1)))
        model.add(LSTM(50, activation='sigmoid'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val), epochs=200, verbose=2)

        return history, model

    def LSTM_test_model(X_test, model, verbose, y_test):

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        yhat = model.predict(X_test, verbose=verbose)

        yhat_new = []
        acc = []
        for i in yhat:
            if i > 0.5:
                yhat_new.append(1)
            if i < 0.5:
                yhat_new.append(0)

        result = metrics.confusion_matrix(y_test, yhat_new)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, yhat_new)
        print("Classification Report:",)
        print(result1)
        print(accuracy_score(y_test, yhat_new))

        return yhat
