"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Digit Recognition using Deep Learning
--------------------------------------------"""
from Simple_TF import Simple_TF
import numpy as np
from data import Data
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
import time
import progressbar
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class MNIST:

    digit_rec = None
    proj_dir = None
    data = Data()
    model = None

    def __init__(self, model):
        self.proj_dir = self.get_proj_dir()
        self.model = model

    def get_proj_dir(self):
        path = os.path.abspath("")
        dir_parts = path.split("/")
        curr_dir = dir_parts[::-1][0]
        if curr_dir == "TensorFlow_Examples":
            path = os.path.abspath("DigitRecognition")
        return path

    def get_data(self, lim=None):
        """
        Use this function to load data that can be further split into training, and validation set
        :param lim: (Optional) Int that controls the size of data returned
        :return: X (feature data), y (target)
        """
        path = ""
        try:
            path = self.proj_dir
            print("Attempting to load file")
            temp = np.load("mnist.npy")
            N, M = temp.shape
            X = temp[:, :M - 1]
            y = temp[:, M - 1]

        except Exception:
            print("File not found")
            print("Processing data")
            X, y = self.data.load_csv("%s/data/train.csv"%path, "\n", ",", target_col=0, numeric_target=True)
            temp = np.hstack((X, y.reshape([-1, 1])))
            print("Saving processed data")
            np.save(os.path.abspath("mnist.npy"), temp)

        N, M = X.shape
        if lim is not None:
            if lim <= N:
                X = X[:lim,:]
                y = y[:lim]
            else:
                raise ValueError("limit cannot be larger than the total size")
        return X, y

    def get_test_data(self):
        """
        Use this function strictly after training since we don't
        the data that would be returned by this function to lie around
        while training network occupying memory
        :return: Training data input X, Training data target y
        """
        X, y = self.data.load_csv("data/test.csv", "\n", ",", target_col=0, numeric_target=True)
        return X, y

    def train(self, lim=None, partition_size=31000, epochs=30, save_model=True):
        X, y = self.get_data(lim)
        N, _ = X.shape
        size = partition_size
        X_tr, y_tr, X_vl, y_vl = self.data.partition_data(X, y, tr_size=size, shuffle=True)
        N, M = X_tr.shape
        batch_size = 1000
        steps = int(N / batch_size)
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
        restore = False

        basedir = self.proj_dir
        model = self.model

        dr = Simple_TF(proj_dir = basedir,
                       sample = sample,
                       lr = lr,
                       layers = model["layers"] ,
                       neurons = model["neurons"],
                       activations = model["activations"],
                       n_class = 10,
                       epochs = epochs,
                       steps = steps,
                       restore = restore,
                       checkpoint = "model_final")

        plt.figure()
        plt.grid(True)
        plt.title("Digit Recognition Training Performance %s"%model["id"])
        plt.xlabel("Step")
        plt.ylabel("Cross Entropy")
        plt.tight_layout()

        """//////////////////////////////////////////////////////////////////////////////////////
        Increase the output labels by 1 so that 0th class label is considered 1, which in turn
        prevents cross entropy from returning nan
        //////////////////////////////////////////////////////////////////////////////////////"""
        print("Starting training phase...")
        for epoch in range(epochs):
            si = 0
            ei = batch_size
            for step in range(steps):
                global_count = (epoch * steps) + step
                vl_indices = np.random.randint(0,N_vl,batch_size)
                tr_cost, vl_cost, tr_accuracy, vl_accuracy = dr.optimize(X_tr[si:ei,:,:,0].reshape([-1,H,W,1]),
                                                                         y_tr[si:ei]+1,
                                                                         X_vl[vl_indices,:,:,0].reshape([-1,H,W,1]),
                                                                         y_vl[vl_indices]+1,
                                                                         lr, global_count)
                print("count: %d, epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f, tr_accuracy: %d%%, "
                      "vl_accuracy: %d%%" %
                      (global_count, epoch, step, tr_cost, vl_cost, tr_accuracy, vl_accuracy))

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
        if save_model:
            self.digit_rec.save_model()

    def test(self):
        print("Loading test data")
        #X_test, label = self.get_test_data()
        path = self.proj_dir
        try:
            X_test = np.load("%s/mnist_test.npy"%path)
        except Exception:
            # Remove the code after debugging--------
            X_test, y = self.data.load_csv("%s/data/test.csv"%path, "\n", ",", target_col=0, numeric_target=True)

            # Uncomment this only for testing the accuracy of prediction visually
            # X_test, y = self.get_sample(X_test, y, size=10)
            X_test = np.hstack((y.reshape([-1,1]), X_test))
            del y
            np.save(os.path.abspath("%s/mnist_test.npy"%path), X_test)
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
        model = self.model
        # with tf.Session() as sess:
        with progressbar.ProgressBar(max_value=steps) as bar:
            for step in range(steps):
                if self.digit_rec is not None:
                    predicted.append(self.digit_rec.predict(X=X_test[si:ei,:,:,:].reshape([-1,H,W,1]), \
                                                            feed_dict=None))
                elif os.path.isfile('tmp/model_final.meta'):
                    sample = X_test[0,:,:,0].reshape([1,H,W,1])
                    self.digit_rec = Simple_TF(sample=sample,
                                               lr=model["lr"],
                                               layers=model["layers"],
                                               neurons=model["neurons"],
                                               activations=model["activations"],
                                               n_class=10,
                                               epochs=None,
                                               steps=None,
                                               restore=True,
                                               checkpoint='model_final.meta')
                    predicted.append(self.digit_rec.predict(X=X_test[si:ei,:,:,:].reshape([-1,H,W,1]), feed_dict=None))
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
        np.save("%s/prediction_submission.npy"%path,predicted)

if __name__ == '__main__':

    lr = 1e-2
    layers=['conv', 'maxpool', 'batchnorm', 'conv', 'maxpool', \
            'batchnorm', 'dense', 'batchnorm', 'dense']
    neurons=[32, None, None, 64, None, None, 100, None, 11]
    activations=[tf.nn.relu, None, None, tf.nn.relu, None, None, \
                 tf.nn.relu, None, None]
    model = {"id":0, "lr":lr, "layers":layers, "neurons":neurons, "activations":activations}
    mnist = MNIST(model)
    mnist.train()
    # mnist.test()
