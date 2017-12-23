from sklearn import svm as sk_svm
from sklearn.neighbors import KNeighborsClassifier as sk_knn
import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class Classifier(object):

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train(self):
        pass

    def test(self):
        pass

    def check_accuracy(self):
        pass

class SVM_Classifier(Classifier):

    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def train(self):
        """
        train - function to run the training on the svm

        args    self.train_data - data set for training (in numpy array form)
                self.train_labels - labels for the data set in int form

        returns svm - svm to test and run
        """
        #set up svm
        svm = sk_svm.SVC(kernel='linear', probability=True)
        
        #reshape train data for compatibility with svm
        reshaped_train_data = self.data_reshape(self.train_data)

        #train svm
        print("Starting to train svm...")
        svm.fit(reshaped_train_data, self.train_labels)
        print("SVM training finished...")

        return svm

    def test(self, svm):
        """
        test - function to test the trained svm

        args    self.test_data - test data set
                svm - trained svm
                

        returns test_response
        """
        #reshape test data for compatibility with svm
        reshaped_test_data = self.data_reshape(self.test_data)
        print("Starting to test svm...")
        test_response = svm.predict(reshaped_test_data)

        return test_response

   
    def check_accuracy(self, svm, test_response):
        """
        check_accuracy - function to check the accuracy of the test responses

        args    self.test_labels - original array of labels in int form
                svm - trained svm
                test_response - array of predicted labels from the svm test

        returns accuracy - percentage accuracy
        """

        #reshape train data for compatibility with svm
        reshaped_test_data = self.data_reshape(self.test_data)
        predictions = svm.predict_proba(reshaped_test_data)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        print(self.test_labels)
        class_names = []
        prev_class = 1
        for cls in self.test_labels:
            if prev_class != cls:
                class_names.append(cls)
                prev_class = cls

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, self.test_labels))
        print('Accuracy: %.3f' % accuracy)

        return accuracy

    def data_reshape(self, data):
        """
        data_reshape - function to reshape the data to work with libsvm

        args    data - dataset to reshape

        return reshaped data - reshaped dataset
        """
        data_size = len(data)
        reshaped_data = data.reshape(data_size, -1)
        print("DATA SHAPE")
        print(reshaped_data.shape)
        return reshaped_data

class KNNClassifier(Classifier):
    """
    An implementation of K Nearest Neighbours classifier from scikit learn.
    """

    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def train(self):
        """
        train - function to run to train the model

        args    self.train_data - data set for training (in numpy array form)
                self.train_labels - labels for the data set in int form

        returns model - model to test and run
        """
        #set up svm
        model = sk_knn()

        #reshape train data for compatibility with svm
        reshaped_train_data = self.data_reshape(self.train_data)

        #train svm
        print("Starting to train KNN model...")
        model.fit(reshaped_train_data, self.train_labels)
        print("Training finished...")

        return model

    def test(self, model):
        """
        test - function to test the trained model

        args    self.test_data - test data set
                model - trained model

        returns test_response
        """
        #reshape test data for compatibility with svm
        reshaped_test_data = self.data_reshape(self.test_data)
        print("Starting to test model...")
        test_response = model.predict(reshaped_test_data)

        return test_response

    def check_accuracy(self, model, test_response):
        """
        check_accuracy - function to check the accuracy of the test responses

        args    self.test_labels - original array of labels in int form
                model - trained model
                test_response - array of predicted labels from the svm test

        returns accuracy - percentage accuracy
        """

        #reshape train data for compatibility with svm
        reshaped_test_data = self.data_reshape(self.test_data)
        predictions = model.predict_proba(reshaped_test_data)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        print(self.test_labels)
        class_names = []
        prev_class = 1
        for cls in self.test_labels:
            if prev_class != cls:
                class_names.append(cls)
                prev_class = cls

        for i in range(len(best_class_indices)):
            print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

        accuracy = np.mean(np.equal(best_class_indices, self.test_labels))
        print('Accuracy: %.3f' % accuracy)

        return accuracy

    def data_reshape(self, data):
        """
        data_reshape - function to reshape the data to work with libsvm

        args    data - dataset to reshape

        return reshaped data - reshaped dataset
        """
        data_size = len(data)
        reshaped_data = data.reshape(data_size, -1)
        print("DATA SHAPE")
        print(reshaped_data.shape)
        return reshaped_data

class Neural_Classifier(Classifier):
    
    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def train(self):
         epochs = 25
         batch_size = 25
         input_shape = self.train_data.shape
         print(input_shape)

         model = Sequential()
         model.add(Dense(units=20, input_dim=128, activation='sigmoid'))

         model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

         model.fit(self.train_data, self.train_labels, epochs=epochs, batch_size=batch_size)

         return model

    def test(self):
        return 0

    def check_accuracy(self, model):
        loss_and_metrics = model.evaluate(self.test_data, self.test_labels, batch_size = batch_size)
        return loss_and_metrics
