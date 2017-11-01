from sklearn import svm as sk_svm
import numpy as np

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
        svm = sk_svm.SVC(C=18)
        
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

   
    def check_accuracy(self, test_response):
        """
        check_accuracy - function to check the accuracy of the test responses

        args    self.test_labels - original array of labels in int form
                test_response - array of predicted labels from the svm test

        returns accuracy - percentage accuracy
        """
        success_check = self.test_labels == test_response

        success = 0
        for result in success_check:
            if result == True:
                success += 1

        accuracy = float(success) / len(self.test_labels)
        print("num labels %d" % len(self.test_labels))
        print("num predicts %d" % len(test_response))

        return accuracy

    def data_reshape(self, data):
        """
        data_reshape - function to reshape the data to work with libsvm

        args    data - dataset to reshape

        return reshaped data - reshaped dataset
        """
        data_size = len(data)
        reshaped_data = data.reshape(data_size, -1)

        return reshaped_data

class Neural_Classifier(Classifier):
    
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def check_accuracy(self):
        pass
