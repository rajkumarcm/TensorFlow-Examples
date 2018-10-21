"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Digit Recognition using Deep Learning
--------------------------------------------"""
from Simple_TF import Simple_TF
import numpy as np
from data import load_data
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import time
import progressbar

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
        # with tf.Session() as sess:
        print("Starting training phase...")
        X, y = self.get_data()
        N, _ = X.shape
        size = 31000
        X_tr, y_tr, X_vl, y_vl = self.partition_data(X, y, tr_size=size, shuffle=True)
        N, M = X_tr.shape
        batch_size = 1000
        steps = int(N / batch_size)
        epochs = 30
        lr = 1e-2

        H = int(np.sqrt(M))
        W = H
        N_tr = X_tr.shape[0]
        N_vl = X_vl.shape[0]
        X_tr = np.reshape(X_tr, [N_tr, H, W, 1])  # Let's just assume we only have one channel for the moment
        y_tr = np.reshape(y_tr, [-1])
        X_vl = np.reshape(X_vl, [N_vl, H, W, 1])  # Let's just assume we only have one channel for the moment
        y_vl = np.reshape(y_vl, [-1])

        sample = X_tr[0,:,:,0].reshape([1,H,W,1])

        dr = Simple_TF(sample=sample,
                       lr=lr, \
                       layers=['conv', 'maxpool', 'batchnorm', 'conv', 'maxpool', 'batchnorm', 'dense', \
                               'batchnorm', 'dense'], \
                       neurons=[32, None, None, 64, None, None, 100, None, 11], \
                       activations=[tf.nn.relu, None, None, tf.nn.relu, None, None, tf.nn.relu, None, None], \
                       n_class=10,
                       epochs=epochs,
                       steps=steps,
                       restore=False,
                       checkpoint='model_final')


        plt.figure()
        plt.grid(True)
        plt.title("Digit Recognition Training Performance")
        plt.xlabel("Step")
        plt.ylabel("Cross Entropy")
        plt.tight_layout()

        """//////////////////////////////////////////////////////////////////////////////////////
        Increase the output labels by 1 so that 0th class label is considered 1, which in turn
        prevents cross entropy from returning nan
        //////////////////////////////////////////////////////////////////////////////////////"""
        si = 0
        ei = batch_size
        for epoch in range(epochs):
            for step in range(steps):
                global_count = (epoch * step) + step
                vl_indices = np.random.randint(0,N_vl,batch_size)
                tr_cost, vl_cost, tr_accuracy, vl_accuracy = dr.optimize(X_tr[si:ei,:,:,0].reshape([-1,H,W,1]),\
                                                                         y_tr[si:ei]+1, \
                                                                         X_vl[vl_indices,:,:,0].reshape([-1,H,W,1]), \
                                                                         y_vl[vl_indices]+1, \
                                                                         lr, global_count)
                print("epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f, tr_accuracy: %d%%, vl_accuracy: %d%%" % \
                      (epoch, step, tr_cost, vl_cost, tr_accuracy, vl_accuracy))

                if (epoch == 0 and step >= 10) or (epoch >= 1):
                    plt.scatter(global_count, tr_cost, c='b', marker='o', linewidths=0.5, alpha=0.7, \
                                label="Training Error")
                    plt.scatter(global_count, vl_cost, c='r', marker='^', linewidths=0.5, alpha=0.7, \
                                label="Validation Error")
                    plt.pause(0.001)

                if epoch == 0 and step == 10:
                    plt.legend()

                si = ei
                ei += batch_size

            if (epoch != 0) and (epoch % 10 == 0):
                lr /= 10

        plt.show()
        self.digit_rec = dr

    def test(self):
        print("Loading test data")
        #X_test, label = self.get_test_data()

        try:
            X_test = np.load("mnist_test.npy")
        except Exception:
            # Remove the code after debugging--------
            X_test, y = load_data("data/test.csv", "\n", ",", target_col=0, numeric_target=True)

            # Uncomment this only for testing the accuracy of prediction visually
            # X_test, y = self.get_sample(X_test, y, size=10)
            X_test = np.hstack((y.reshape([-1,1]), X_test))
            del y
            np.save("mnist_test.npy",X_test)
        #----------------------------------------
        batch_size = 1000
        N, M = X_test.shape
        H = int(np.sqrt(M))
        W = H
        X_test = np.reshape(X_test,[N,H,W,1])
        steps = int(np.floor(N / batch_size))
        predicted = []

        si = 0
        ei = batch_size
        # with tf.Session() as sess:
        with progressbar.ProgressBar(max_value=steps) as bar:
            for step in range(steps):
                if self.digit_rec is not None:
                    predicted.append(self.digit_rec.predict(X=X_test[si:ei,:,:,:].reshape([-1,H,W,1]), \
                                                            feed_dict=None))
                elif os.path.isfile('tmp/model_final.meta'):
                    sample = X_test[0,:,:,0].reshape([1,H,W,1])
                    self.digit_rec = Simple_TF(sample=sample,
                                               lr=1e-2, \
                                               layers=['conv', 'maxpool', 'batchnorm', 'conv', 'maxpool', \
                                                        'batchnorm', 'dense', \
                                                        'batchnorm', 'dense'], \
                                               neurons=[32, None, None, 64, None, None, 100, None, 11], \
                                               activations=[tf.nn.relu, None, None, tf.nn.relu, None, None, \
                                                            tf.nn.relu, None, None], \
                                               n_class=10,
                                               epochs=None,
                                               steps=None,
                                               restore=True,
                                               checkpoint='model-2.meta')
                    predicted.append(self.digit_rec.predict(X=X_test[si:ei,:,:,:].reshape([-1,H,W,1]), \
                                                            feed_dict=None))
                else:
                    raise Exception("Neither Digit Recognition instance with tuned parameter nor "
                                    "valid checkpoint found to make predictions")
                si = ei
                ei = si + batch_size
                bar.update(step)
                time.sleep(0.01)

        predicted = np.concatenate(predicted, axis=0)
        predicted = np.reshape(predicted, [-1, 1])

        # conf_mat, precision, recall, f1_score = get_accuracy(label, predicted, n_class=10)
        # print("Confusion Matrix: \n{}\n".format(conf_mat))
        # print("Precision: \n{}".format(precision))
        # print("Recall: \n{}".format(recall))
        # print("F1 Score: \n{}".format(f1_score))
        # print("Average F1 Score: %d" % (int(np.mean(f1_score))))
        np.save("prediction_submission.npy",predicted)

if __name__ == '__main__':
    mnist = MNIST()
    # mnist.train()
    mnist.test()