"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Simple Interface for TensorFlow
-------------------------------------------"""

import tensorflow as tf
import numpy as np
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
    saver = None
    output_original = None
    output = None
    cost = None
    lr_ = None
    proj_dir = None

    def __init__(self, proj_dir, sample, lr, layers, neurons, activations, n_class,
                 epochs, steps, restore, batch_size=None, output_shape=None, checkpoint=None):
        self.proj_dir = proj_dir
        self.lr = lr
        self.layers = layers
        self.neurons = neurons
        self.activations = activations
        self.n_labels = n_class + 1 # 0th class represent blank label
        self.epochs = epochs
        self.steps = steps
        if (restore) and (not checkpoint):
            raise ValueError("Checkpoint cannot be None when restore is expected")
        self.checkpoint = checkpoint
        _, H, W, C = sample.shape
        self.output = tf.placeholder(dtype=tf.float32, shape=[batch_size, H, W, C])
        self.output_original = self.output
        self.actual = tf.placeholder(dtype=tf.int32, shape=output_shape) # y
        self.lr_ = tf.placeholder(dtype=tf.float32, shape=[])
        self.output, self.cost, self.train = self.def_graph(self.lr_, self.output, self.actual)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if restore:
            print("Restoring model...")
            imported_meta = tf.train.import_meta_graph("%s/tmp/%s" % (proj_dir,checkpoint))
            imported_meta.restore(self.sess, tf.train.latest_checkpoint('%s/tmp/'%proj_dir))
        else:
            self.sess.run(tf.global_variables_initializer())

    def get_bilinear_filter(self, filter_shape, upscale_factor):
        ##filter_shape is [width, height, num_in_channels, num_out_channels]
        kernel_size = filter_shape[1]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(filter_shape[0]):
            for y in range(filter_shape[1]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location)/ upscale_factor)) *\
                        (1 - abs((y - centre_location)/ upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return init
        # return tf.convert_to_tensor(weights, dtype=tf.float32)

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

    def add_deconv(self, input, global_i, local_i, strides=[2,2], padding="same", upscale_factor=2):
        out_channels = self.neurons[global_i]
        in_channels = input.shape[::-1][0].value
        # deconvolution aka upsampling makes no changes to the depth dimension so we output the
        # same number of channels being input so 3,3,in_channels,in_channels
        filter_init = self.get_bilinear_filter(filter_shape=[3,3,in_channels,in_channels], \
                                               upscale_factor=upscale_factor)
        deconv = tf.keras.layers.Conv2DTranspose(filters=in_channels, kernel_size=[3,3], \
                                                 strides=strides, padding=padding, \
                                                 use_bias=True, kernel_initializer=filter_init,
                                                 name="Deconvolution%d"%local_i)
        return deconv(input)

    def max_pool(self, local_i, input, k_size=[1,2,2,1], strides=[1,1,1,1], padding='SAME'):
        return tf.nn.max_pool(input, k_size, strides, padding, name="Max_Pool%d" % (local_i + 1))

    def add_dense(self, input, global_i, local_i):
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
    def def_graph(self, lr, output, actual):
        conv_count = 0
        deconv_count = 0
        dense_count = 0
        maxpool_count = 0
        batchnorm_count = 0

        print("Loading graph..")
        for global_i, layer in zip(range(len(self.layers)), self.layers):

            if layer == 'conv':
                output = self.add_conv(output, global_i, conv_count)
                conv_count += 1

            elif layer == "deconv":
                output = self.add_deconv(output, global_i, deconv_count)
                deconv_count += 1

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

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actual, logits=output, name="CrossEntropy")
        cost = tf.reduce_mean(cost)
        train = tf.train.MomentumOptimizer(lr,momentum=0.9).minimize(cost)
        return output, cost, train

    def optimize(self, X, y, vl_input, vl_y, lr, global_count):
        _,H,W,C = X.shape
        try:
            tr_feed = {self.output_original: X, self.actual: y, self.lr_: lr}
            vl_feed = {self.output_original: vl_input, self.actual: vl_y}

            tr_cost, _ = self.sess.run([self.cost, self.train], tr_feed)
            pred_output = self.predict(X=None, feed_dict=tr_feed)
            tr_conf_mat, tr_precision, tr_recall, tr_f1_score = get_accuracy(y - 1, \
                                                                             pred_output, \
                                                                             self.n_labels)
            tr_f1_score = tr_f1_score[1:]  # We simply ignore the score for blank node
            tr_accuracy = int(np.mean(tr_f1_score) * 100)

            vl_cost = self.sess.run(self.cost, vl_feed)
            if math.isnan(vl_cost):
                print("debug...")
            pred_output = self.predict(X=None, feed_dict=vl_feed)
            vl_conf_mat, vl_precision, vl_recall, vl_f1_score = get_accuracy(vl_y - 1, \
                                                                             pred_output, \
                                                                             self.n_labels)
            vl_f1_score = vl_f1_score[1:]  # We simply ignore the score for blank node
            vl_accuracy = int(np.mean(vl_f1_score) * 100)

            return tr_cost, vl_cost, tr_accuracy, vl_accuracy

        except KeyboardInterrupt:
            save = input("Keyboard interrupt received. Do you want to save the training? (y/n)\n")
            if save == "y":
                save_path = self.saver.save(self.sess, "%s/tmp/model"%(self.proj_dir), global_step=global_count)
                print("Model saved in path: %s" % save_path)
            else:
                print("Model not saved")
            print("Program terminating...")
            sys.exit()

    def predict(self, X, feed_dict):
        #-1 to shift the labels one down since we account for an empty label in 0th position
        if feed_dict is None:
            feed_dict = {self.output_original: X}
        return self.sess.run(tf.argmax(tf.nn.softmax(self.output), axis=1) -1, feed_dict)

    def save_model(self, filename=None):
        # Save the model
        if filename is None:
            filename = self.checkpoint

        save_path = self.saver.save(self.sess, "%s/tmp/%s" %(self.proj_dir,filename))
        print("Model saved in path: %s" % save_path)
