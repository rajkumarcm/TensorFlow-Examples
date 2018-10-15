import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from accuracy import get_accuracy
import sys
import math

class Simple_TF:
    hidden_layers = None
    neurons = None
    lr = None
    n_labels = None
    activations = None
    epochs = None
    batch_size = None
    train = None
    checkpoint = None
    sess = None

    def __init__(self, lr, layers, neurons, activations, n_class, epochs, batch_size, steps, train, checkpoint=None):
        self.lr = lr
        self.layers = layers
        self.neurons = neurons
        self.activations = activations
        self.n_labels = n_class + 1 # 0th class represent blank label
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps = steps
        self.train = train
        self.checkpoint = checkpoint
        if train is False and checkpoint is None:
            raise ValueError("Inference cannot happen without checkpoint file for the parameters to be restored")

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

    def add_batchnorm(self, input, local_i):
        # BATCH NORMALIZATION
        input_channels = input.shape[::-1][0].value
        mean, variance = tf.nn.moments(input, axes=[0])
        beta = tf.Variable(tf.constant(0.0, shape=[input_channels]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[input_channels]), name='gamma', trainable=True)
        return tf.nn.batch_normalization(input, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-5, \
                                         name="BatchNormalization%d"%local_i)

    def predict(self, sess, raw_output, feed_dict):
        #-1 to shift the labels one down since we account for an empty label in 0th position
        return sess.run(tf.argmax(tf.nn.softmax(raw_output), axis=1) -1, feed_dict)

    def main(self, X, y, vl_input, vl_y):
        tf.reset_default_graph()
        _,H,W,C = X.shape
        output = tf.placeholder(dtype=tf.float32, shape=[None,H,W,C])
        actual = tf.placeholder(dtype=tf.int32, shape=[None])
        output_original = output

        conv_count = 0
        dense_count = 0
        maxpool_count = 0
        batchnorm_count = 0
        lr = tf.placeholder(dtype=tf.float32,shape=[])
        lr_new = self.lr

        print("Loading graph")
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

                elif layer == 'batchnorm':
                    output = self.add_batchnorm(output, batchnorm_count)
                    batchnorm_count += 1

                else:
                    raise ValueError("Invalid layer...")

            saver = tf.train.Saver()
            if self.train:

                cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actual, logits=output, name="CrossEntropy")
                cost = tf.reduce_mean(cost)

                # output = tf.nn.softmax(output)
                # cost = -tf.reduce_sum((actual * tf.log(output)),axis=1)
                optimize = tf.train.MomentumOptimizer(lr,momentum=0.9).minimize(cost)

                sess.run(tf.global_variables_initializer())
                fig = plt.figure()
                # fig.set_figheight(8)
                # fig.set_figwidth(16)

                plt.grid(True)
                plt.title("Digit Recognition Training Performance")
                plt.xlabel("Step")
                plt.ylabel("Cross Entropy")
                # plt.set_xlim([0,self.epochs*self.steps])
                plt.tight_layout()

                global_count = 0
                total_steps = self.epochs * self.steps
                try:
                    for epoch in range(self.epochs):
                        si = 0
                        ei = self.batch_size
                        for step in range(self.steps):
                            tr_feed = {output_original: X[si:ei, :, :, :].reshape([-1, H, W, C]), \
                                       actual: y[si:ei], lr:lr_new}
                            vl_indices = np.random.randint(0,vl_input.shape[0],size=self.batch_size)
                            vl_feed = {output_original: vl_input[vl_indices,:,:,:].reshape([-1,H,W,C]), \
                                       actual: vl_y[vl_indices]}
                            tr_cost, _ = sess.run([cost, optimize], tr_feed)

                            pred_output = self.predict(sess,output,tr_feed)
                            tr_conf_mat, tr_precision, tr_recall, tr_f1_score = get_accuracy(y[si:ei]-1,\
                                                                                             pred_output,\
                                                                                             self.n_labels)
                            tr_f1_score = tr_f1_score[1:] # We simply ignore the score for blank node
                            tr_accuracy = int(np.mean(tr_f1_score)*100)

                            vl_cost = sess.run(cost, vl_feed)
                            if math.isnan(vl_cost):
                                print("debug...")
                            pred_output = self.predict(sess, output, vl_feed)
                            vl_conf_mat, vl_precision, vl_recall, vl_f1_score = get_accuracy(vl_y[vl_indices]-1, \
                                                                                             pred_output, \
                                                                                             self.n_labels)
                            vl_f1_score = vl_f1_score[1:] # We simply ignore the score for blank node
                            vl_accuracy = int(np.mean(vl_f1_score)*100)

                            print("epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f, tr_accuracy: %d%%, vl_accuracy: %d%%" %\
                                 (epoch,step,tr_cost,vl_cost,tr_accuracy,vl_accuracy))

                            if (epoch == 0 and step >= 10) or (epoch >= 1):
                                plt.scatter(global_count,tr_cost,c='b',marker='o',linewidths=0.5,alpha=0.7,\
                                            label="Training Error")
                                plt.scatter(global_count,vl_cost,c='r',marker='^',linewidths=0.5,alpha=0.7,\
                                            label="Validation Error")
                                plt.pause(0.001)

                                if epoch == 0 and step == 10:
                                    plt.legend()

                            si = ei
                            ei = ei + self.batch_size
                            global_count += 1

                        if (epoch != 0) and (epoch%10 == 0):
                            lr_new /= 10

                    plt.show()
                    # Save the model
                    save_path = saver.save(sess, "tmp/%s"%self.checkpoint)
                    print("Model saved in path: %s"%save_path)
                    self.sess = sess

                except KeyboardInterrupt:
                    save = raw_input("Keyboard interrupt received. Do you want to save the training? (y/n)\n")
                    if save == "y":
                        checkpoint = raw_input("Enter the checkpoint name: (Leave blank for default)\n")
                        if checkpoint == '':
                            checkpoint = self.checkpoint
                        save_path = saver.save(sess, "tmp/%s" % checkpoint, global_step=epoch)
                        print("Model saved in path: %s" % save_path)
                    else:
                        print("Model not saved")
                        print("Program terminating...")
                        sys.exit()
            else:
                #model.cpkt-29.data-00000-of-00001
                imported_meta = tf.train.import_meta_graph("tmp/%s"%self.checkpoint)
                imported_meta.restore(sess,tf.train.latest_checkpoint('tmp/'))
                # saver.restore(sess,'tmp/%s'%self.checkpoint)
                predicted = self.predict(sess, output, feed_dict={output_original:X})
                return predicted

    def close_session(self):
        self.sess.close()
        self.sess = None