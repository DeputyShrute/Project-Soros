from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import History


class CNN:

    def CNN_train_model(self, X_train, X_val, y_train, y_val):

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
            X_train, y_train, validation_data=(X_val, y_val), epochs=500, verbose=2, shuffle=True)

        return history, model

    def CNN_test_model(X_test, model):

        #new_seq = array(raw_seq[-27:])
        # print(new_seq)
        #new_seq = new_seq.reshape((1, self.timestep, 1))

        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        yhat = model.predict(X_test, verbose=0)

        return yhat


class MLP:

    def MLP_train_model(self, X_train, X_val, y_train, y_val):

        model = Sequential()
        model.add(Dense(800, activation='relu', input_dim=(self.timestep)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        model.summary()

        # fit model
        history = model.fit(X_train, y_train, validation_data=(
            X_val, y_val), epochs=1000, verbose=2)

        return history, model

    def MLP_test_model(X_test, model):

        yhat = model.predict(X_test, verbose=2)

        return yhat


class KNN:

    def KNN_train_model(self, X_train, X_val, y_train, y_val, X_test, y_test, raw_seq):
        Range_k = range(1,20)
        scores = {}
        scores_list = []
        for k in Range_k:
            classifier = KNeighborsClassifier(n_neighbors=k)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            #print(y_pred)
            scores[k] = metrics.accuracy_score(y_test, y_pred)
            scores_list.append(metrics.accuracy_score(y_test, y_pred))
        result = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, y_pred)
        print("Classification Report:",)
        print(result1)

        plt.plot(Range_k, scores_list)
        plt.title('Model Accuarcy / K Value')
        plt.xlabel("Value of K")
        plt.ylabel("Accuracy")
        plt.show()