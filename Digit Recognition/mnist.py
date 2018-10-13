import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from accuracy import get_accuracy

class Digit_Recognition:
    hidden_layers = None
    neurons = None
    lr = None
    n_labels = None
    activations = None
    epochs = None
    batch_size = None

    def __init__(self, lr, layers, neurons, activations, n_class, epochs, batch_size, steps):
        self.lr = lr
        self.layers = layers
        self.neurons = neurons
        self.activations = activations
        self.n_labels = n_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = steps

    def conv2d(self, input, kernel, strides, padding, local_i):
        return tf.nn.conv2d(input, kernel, strides, padding, name='Convolution%d' % (local_i + 1))

    def add_conv(self, input, global_i, local_i, stride=[1,1,1,1], padding='SAME'):
        output_channels = self.neurons[global_i]
        input_channels = input.shape[::-1][0].value
        weight = tf.Variable(tf.truncated_normal(shape=[5,5,input_channels,output_channels], \
                                                 name="Kernel%d" % (local_i + 1)))
        activation = self.activations[global_i]
        bias = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), \
                               shape=[], trainable=True, name="ConvBias%d"%local_i)
        return activation(self.conv2d(input, weight, stride, padding, local_i) + bias)

    def max_pool(self, local_i, input, k_size=[1,2,2,1], strides=[1,1,1,1], padding='SAME'):
        return tf.nn.max_pool(input, k_size, strides, padding, name="Max_Pool%d" % (local_i + 1))

    def add_dense(self, input, global_i, local_i, activation=tf.nn.relu):
        shape = input.get_shape().as_list()
        input_shape = 1
        for i in range(1,len(shape)):
            input_shape = input_shape * shape[i]

        flat_input = tf.reshape(input, [-1, input_shape])
        weight = tf.Variable(tf.truncated_normal(shape= [input_shape, self.neurons[global_i]]), \
                             name="Dense_Layer_%d" % (local_i))
        bias = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), \
                               shape=[self.neurons[global_i]], trainable=True, name='DenseBias%d'%local_i)
        temp = tf.matmul(flat_input, weight) + bias
        activation = self.activations[global_i]
        if activation is not None:
            return activation(temp)
        else:
            return temp

    def predict(self, sess, raw_output, feed_dict):
        return sess.run(tf.argmax(tf.nn.softmax(raw_output), axis=1), feed_dict)

    def main(self, input, y, vl_input, vl_y):
        _,H,W,C = input.shape
        output = tf.placeholder(dtype=tf.float32, shape=[None,H,W,C])
        actual = tf.placeholder(dtype=tf.int32, shape=[None])
        output_original = output

        conv_count = 0
        dense_count = 0
        maxpool_count = 0
        lr = self.lr

        #graph = tf.get_default_graph
        #with graph:
        with tf.Session() as sess:

            for global_i, layer in zip(range(len(self.layers)),self.layers):

                if layer == 'conv':
                    output = self.add_conv(output, global_i, conv_count)
                    conv_count += 1

                elif layer == 'maxpool':
                    output = self.max_pool(maxpool_count, output)
                    maxpool_count += 1

                elif layer == 'dense':
                    output = self.add_dense(output, global_i, dense_count)
                    dense_count += 1

                else:
                    raise ValueError("Invalid layer...")

            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actual, logits=output, name="CrossEntropy")
            cost = tf.reduce_mean(cost)
            optimize = tf.train.MomentumOptimizer(lr,momentum=0.8).minimize(cost)
            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            fig, ax = plt.subplots(1,2)
            fig.set_figheight(8)
            fig.set_figwidth(16)

            plt.grid(True)
            ax[0].set_title("Digit Recognition Training Performance")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Cross Entropy")
            ax[0].set_xlim([0,self.epochs*self.steps])

            plt.grid(True)
            ax[1].set_title("Digit Recognition Training Performance")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("F1 score in %")
            ax[1].set_xlim([0, self.epochs*self.steps])

            plt.tight_layout()
            global_count = 0
            for epoch in range(self.epochs):
                si = 0
                ei = self.batch_size
                for step in range(self.steps):
                    tr_feed = {output_original: input[si:ei,:,:,:].reshape([-1,H,W,C]), actual: y[si:ei]}
                    vl_feed = {output_original: vl_input[si:ei,:,:,:].reshape([-1,H,W,C]), actual: vl_y[si:ei]}
                    tr_cost, _ = sess.run([cost, optimize], tr_feed)
                    tr_cost = np.mean(tr_cost)

                    pred_output = self.predict(sess,output,tr_feed)
                    tr_conf_mat, tr_precision, tr_recall, tr_f1_score = get_accuracy(y[si:ei],pred_output,self.n_labels)
                    tr_accuracy = int(np.mean(tr_f1_score)*100)

                    vl_cost, _ = sess.run([cost, optimize], vl_feed)
                    vl_cost = np.mean(vl_cost)
                    pred_output = self.predict(sess, output, vl_feed)
                    vl_conf_mat, vl_precision, vl_recall, vl_f1_score = get_accuracy(vl_y[si:ei], pred_output, \
                                                                                     self.n_labels)
                    vl_accuracy = int(np.mean(vl_f1_score)*100)

                    print("epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f, tr_accuracy: %d%%, vl_accuracy: %d%%" %\
                         (epoch,step,tr_cost,vl_cost,tr_accuracy,vl_accuracy))

		    """
                    if epoch > 0:
                        ax[0].scatter(global_count,tr_cost,c='b',marker='o',linewidths=0.5,alpha=0.7,label="Training Error")
                        ax[0].scatter(global_count,vl_cost,c='r',marker='^',linewidths=0.5,alpha=0.7,label="Validation Error")

                        ax[1].scatter(global_count,tr_accuracy,c='b',marker='o',linewidths=0.5,alpha=0.7,\
                                      label="Accuracy on training data")
                        ax[1].scatter(global_count, vl_accuracy, c='r', marker='^', linewidths=0.5, alpha=0.7, \
                                      label="Accuracy on validation data")
                        plt.pause(0.001)
                        # plt.draw_all()

                        if epoch == 0 and step == 11:
                            plt.legend()
                            # plt.show()
                    """
                    si = ei
                    ei = ei + self.batch_size
                    global_count += 1

            plt.show()

            # Save the model
            save_path = saver.save(sess, "tmp/model.cpkt",global_step=epoch)
            print("Model saved in path: %s"%save_path)

if __name__ == '__main__':
    #(self, lr, layers, neurons, n_class)

    X = None
    y = None
    try:
        print("Attempting to load file")
        temp = np.load('mnist.npy')
        N,M = temp.shape
        X = temp[:,:M-1]
        y = temp[:,M-1]
    except Exception:
        print("File not found")
        print("Processing data")
        X, y = load_data("train.csv", "\n", ",", target_col=0, numeric_target=True)
        temp = np.hstack((X,y.reshape([-1,1])))
        print("Saving processed data")
        np.save('mnist.npy',temp)

    del temp
    print("Loading graph")
    N,M = X.shape
    lim = int(np.floor(N/2))
    X_tr = X[:lim,:]
    y_tr = y[:lim]
    X_vl = X[lim:,:]
    y_vl = y[lim:]

    batch_size = 1000
    steps = int(X_tr.shape[0]/batch_size)

    dr = Digit_Recognition(lr=1e-2, \
                           layers=['conv', 'maxpool', 'conv', 'maxpool', 'dense', 'dense'], \
                           neurons=[32, None, 64, None, 100, 10], \
                           activations=[tf.nn.relu, None, tf.nn.relu, None, tf.nn.relu, None], \
                           n_class=10,
                           epochs=500,
                           batch_size = batch_size,
                           steps=steps)


    # Uncomment for Debugging
    # X_tr = X[:100, :]
    # y_tr = y[:100]
    # X_vl = X[100:200, :]
    # y_vl = y[100:200]
    #------------------------


    H = int(np.sqrt(M))
    W = H
    N_tr = X_tr.shape[0]
    N_vl = X_vl.shape[0]
    X_tr = np.reshape(X_tr,[N_tr,H,W,1]) # Let's just assume we only have one channel for the moment
    y_tr = np.reshape(y_tr,[-1])
    X_vl = np.reshape(X_vl, [N_vl, H, W, 1])  # Let's just assume we only have one channel for the moment
    y_vl = np.reshape(y_vl, [-1])
    dr.main(X_tr,y_tr,X_vl,y_vl)
    print("debug...")
