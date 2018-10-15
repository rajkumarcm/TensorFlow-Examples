"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Digit Recognition using Deep Learning
--------------------------------------------"""
from Simple_TF import Simple_TF
import numpy as np
from data import load_data
import tensorflow as tf
from accuracy import get_accuracy
import os

class MNIST:

    digit_rec = None

    def get_data(self, lim=None):
        """
        Use this function to load data that can be further split into training, and validation set
        :param lim: (Optional) Int that controls the size of data returned
        :return: X (feature data), y (target)
        """
        X = None
        y = None

        try:
            print("Attempting to load file")
            temp = np.load('mnist.npy')
            N, M = temp.shape
            X = temp[:, :M - 1]
            y = temp[:, M - 1]

        except Exception:
            print("File not found")
            print("Processing data")
            X, y = load_data("train.csv", "\n", ",", target_col=0, numeric_target=True)
            temp = np.hstack((X, y.reshape([-1, 1])))
            print("Saving processed data")
            np.save('mnist.npy', temp)

        N, M = X.shape
        if lim is not None:
            if lim <= N:
                X = X[:lim,:]
                y = y[:lim]
            else:
                raise ValueError("limit cannot be larger than the total size")

        return X, y

    def partition_data(self, X, y, tr_size, shuffle=False):
        """
        Splits the data into training, and validation set
        :param X: Array must be in shape: [Samples,Features]
        :param y: Array must be in shape: [Samples]
        :param tr_size: Int that represent the size of training set
        :param shuffle: (Optional) Boolean flag that allow the data to be shuffled prior to partitioning
        :return: X_tr, y_tr, X_vl, y_vl after shuffling (if enabled), and partitioning
        """
        N, M = X.shape
        if shuffle:
            indices = np.arange(0,N)
            np.random.shuffle(indices)
            X = X[indices,:]
            y = y[indices]

        lim = tr_size
        X_tr = X[:lim, :]
        y_tr = y[:lim]
        X_vl = X[lim:, :]
        y_vl = y[lim:]
        return X_tr, y_tr, X_vl, y_vl

    def get_test_data(self):
        """
        Use this function strictly after training since we don't
        the data that would be returned by this function to lie around
        while training network occupying memory
        :return: Training data input X, Training data target y
        """
        X, y = load_data("data/test.csv", "\n", ",", target_col=0, numeric_target=True)

        return X, y

    def get_sample(self, X, y, size=1):
        """
        As the name suggests, this function simply returns samples
        from given data
        :param X: Array must be of shape: [Samples, Features]
        :param y: Array must be of shape: [Samples]
        :param size: (Optional) Int that controls the number of samples returned
        :return: Sample input X, Sample target y
        """
        N, M = X.shape
        indices = np.random.randint(0,N,size)
        X = X[indices,:]
        y = y[indices]
        return X, y

    def train(self):

        X, y = self.get_data()
        N, _ = X.shape
        size = 31000
        X_tr, y_tr, X_vl, y_vl = self.partition_data(X, y, tr_size=size, shuffle=True)
        N, M = X_tr.shape
        batch_size = 1000
        steps = int(N / batch_size)

        dr = Simple_TF(lr=1e-2, \
                       layers=['conv', 'maxpool', 'batchnorm', 'conv', 'maxpool', 'batchnorm', 'dense', \
                               'batchnorm', 'dense'], \
                       neurons=[32, None, None, 64, None, None, 100, None, 11], \
                       activations=[tf.nn.relu, None, None, tf.nn.relu, None, None, tf.nn.relu, None, None], \
                       n_class=10,
                       epochs=30,
                       batch_size=batch_size,
                       steps=steps,
                       train=True,
                       checkpoint='model_final')

        H = int(np.sqrt(M))
        W = H
        N_tr = X_tr.shape[0]
        N_vl = X_vl.shape[0]
        X_tr = np.reshape(X_tr, [N_tr, H, W, 1])  # Let's just assume we only have one channel for the moment
        y_tr = np.reshape(y_tr, [-1])
        X_vl = np.reshape(X_vl, [N_vl, H, W, 1])  # Let's just assume we only have one channel for the moment
        y_vl = np.reshape(y_vl, [-1])

        """//////////////////////////////////////////////////////////////////////////////////////
        Increase the output labels by 1 so that 0th class label is considered 1, which in turn
        prevents cross entropy from returning nan
        //////////////////////////////////////////////////////////////////////////////////////"""
        dr.main(X_tr, y_tr+1, X_vl, y_vl+1)

        self.digit_rec = dr

    def test(self):
        print("Loading test data")
        #X_test, label = self.get_test_data()

        # Remove the code after debugging--------
        X_test, y = load_data("data/test.csv", "\n", ",", target_col=0, numeric_target=True)
        # X_test, y = self.get_sample(X_test, y, size=10)
        X_test = np.hstack((y.reshape([-1,1]), X_test))
        del y
        #----------------------------------------

        N, M = X_test.shape
        H = int(np.sqrt(M))
        W = H
        X_test = np.reshape(X_test,[-1,H,W,1])
        # label = np.reshape(label,[-1])
        predicted = None

        if self.digit_rec is None:
            # Check whether checkpoint is present
            if os.path.isfile('tmp/model_final.meta'):
                # If a checkpoint is present, then we can perhaps restore
                dr = Simple_TF(lr=1e-2, \
                               layers=['conv', 'maxpool', 'batchnorm', 'conv', 'maxpool', 'batchnorm', 'dense', \
                                       'batchnorm', 'dense'], \
                               neurons=[32, None, None, 64, None, None, 100, None, 11], \
                               activations=[tf.nn.relu, None, None, tf.nn.relu, None, None, tf.nn.relu, None, None], \
                               n_class=10,
                               epochs=30,
                               batch_size=None,
                               steps=None,
                               train=False,
                               checkpoint='model_final.meta')
                predicted = dr.main(X_test, y=None, vl_input=None, vl_y=None)
            else:
                """
                1. Instance of Digit Recognition is not found
                2. No trained model were found
                So the only left is to train the model if this method test is called directly.
                """
                self.train()
                self.test()
        else:
            """
            If Digit Recognition instance is up running, then make use of it.
            """
            self.digit_rec.train = False
            predicted = self.digit_rec.main(X=X_test, y=None, vl_input=None, vl_y=False)

        # conf_mat, precision, recall, f1_score = get_accuracy(label, predicted, n_class=10)
        # print("Confusion Matrix: \n{}\n".format(conf_mat))
        # print("Precision: \n{}".format(precision))
        # print("Recall: \n{}".format(recall))
        # print("F1 Score: \n{}".format(f1_score))
        # print("Average F1 Score: %d" % (int(np.mean(f1_score))))
        np.save("prediction_submission.npy",predicted)


if __name__ == '__main__':
    mnist = MNIST()
    mnist.train()
    # mnist.test()
    mnist.digit_rec.close_session()