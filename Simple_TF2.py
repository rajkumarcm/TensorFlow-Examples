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
    loss = None
    device = None

    def __init__(self, proj_dir, sample, lr, layers, neurons, activations, loss,
                 epochs, steps, restore, device, batch_size=None, output_shape=None, checkpoint=None):
        self.proj_dir = proj_dir
        self.device = device
        self.lr = lr
        self.layers = layers
        self.neurons = neurons
        self.activations = activations
        self.loss = loss
        self.epochs = epochs
        self.steps = steps
        if (restore) and (not checkpoint):
            raise ValueError("Checkpoint cannot be None when restore is expected")
        self.checkpoint = checkpoint
        tmp_shape = sample.shape
        self.output = None
        if len(tmp_shape) == 4:
            _, H, W, C = sample.shape
            self.output = tf.placeholder(dtype=tf.float32, shape=[batch_size, H, W, C])
        else:
            _, D, H, W, C = sample.shape
            self.output = tf.placeholder(dtype=tf.float32, shape=[batch_size, D, H, W, C])
        self.output_original = self.output
        self.actual = tf.placeholder(dtype=tf.float32, shape=output_shape) # y  CHANGED FROM INT32 TO FLOAT32
        self.lr_ = tf.placeholder(dtype=tf.float32, shape=[])
        self.output, self.cost, self.train = self.def_graph(self.lr_, self.output, self.actual, self.loss)

        config = None
        if self.device == "gpu":
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(intra_op_parallelism_threads=16,
                                    inter_op_parallelism_threads=16)
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
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

    def get_bilinear_filter3d(self, filter_shape, upscale_factor):
        ##filter_shape is [depth, height, width, num_in_channels, num_out_channels]
        kernel_size = filter_shape[2]
        ### Centre location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            centre_location = upscale_factor - 1
        else:
            centre_location = upscale_factor - 0.5

        bilinear = np.zeros([filter_shape[1], filter_shape[2]])
        for x in range(filter_shape[1]):
            for y in range(filter_shape[2]):
                ##Interpolation Calculation
                value = (1 - abs((x - centre_location) / upscale_factor)) * \
                        (1 - abs((y - centre_location) / upscale_factor))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for d in range(filter_shape[0]):
            for i in range(filter_shape[3]):
                weights[d, :, :, i, i] = bilinear
        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        return init

    def conv2d(self, input_x, kernel, strides, padding, local_i):
        return tf.nn.conv2d(input_x, kernel, strides, padding, name='Convolution%d' % (local_i + 1))

    def conv3d(self, input_x, kernel, strides, padding, local_i):
        return tf.nn.conv3d(input_x, kernel, strides, padding, name='Convolution%d' % (local_i + 1))

    def add_conv(self, input_x, global_i, local_i, stride=[1, 1, 1, 1], padding='SAME'):
        output_channels = self.neurons[global_i]
        input_channels = input_x.shape[::-1][0].value
        weight = tf.Variable(tf.truncated_normal(shape=[5,5,input_channels,output_channels], \
                                                 name="Kernel%d" % (local_i + 1)))
        activation = self.activations[global_i]
        bias = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), \
                               shape=[], trainable=True, name="ConvBias%d"%local_i)
        return activation(self.conv2d(input_x, weight, stride, padding, local_i) + bias)

    def add_conv3d(self, input_x, global_i, local_i, stride=[1, 1, 1, 1, 1], padding='SAME'):
        output_channels = self.neurons[global_i]
        input_channels = input_x.shape[::-1][0].value
        weight = tf.Variable(tf.truncated_normal(shape=[1,5,5,input_channels,output_channels], \
                                                 name="Kernel%d" % (local_i + 1)))
        activation = self.activations[global_i]
        bias = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), \
                               shape=[], trainable=True, name="Conv3dBias%d"%local_i)
        return activation(self.conv3d(input_x, weight, stride, padding, local_i) + bias)

    def add_deconv(self, input_x, local_i, strides=[2, 2], padding="same", upscale_factor=2):
        in_channels = input_x.shape[::-1][0].value
        # deconvolution aka upsampling makes no changes to the depth dimension so we output the
        # same number of channels being input_x so 3,3,in_channels,in_channels
        filter_init = self.get_bilinear_filter(filter_shape=[3,3,in_channels,in_channels], \
                                               upscale_factor=upscale_factor)
        deconv = tf.keras.layers.Conv2DTranspose(filters=in_channels, kernel_size=[3,3], \
                                                 strides=strides, padding=padding, \
                                                 use_bias=True, kernel_initializer=filter_init,
                                                 name="Deconvolution%d"%local_i)
        return deconv(input_x)

    def add_deconv3d(self, input_x, local_i, strides=[1, 2, 2], padding="same", upscale_factor=2):
        in_channels = 1
        # deconvolution aka upsampling makes no changes to the depth dimension so we output the
        # same number of channels being input so 3,3,in_channels,in_channels
        filter_init = self.get_bilinear_filter3d(filter_shape=[1, 3, 3, in_channels, in_channels],
                                                 upscale_factor=upscale_factor)

        deconv = tf.keras.layers.Conv3DTranspose(filters=in_channels,
                                                 kernel_size=[1, 3, 3],
                                                 strides=strides,
                                                 padding=padding,
                                                 use_bias=True,
                                                 kernel_initializer=filter_init,
                                                 name="Deconvolution%d" % local_i)
        return deconv(input_x)

    def max_pool(self, local_i, input_x, k_size=[1, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 1]):
        return tf.nn.max_pool(input_x, k_size, strides, padding=padding, name="Max_Pool%d" % (local_i + 1))

    def max_pool3d(self, local_i, input_x, k_size=[1, 2, 2], padding="VALID", strides=[1, 2, 2]):
        maxpool = tf.keras.layers.MaxPool3D(pool_size=k_size, strides=strides,
                                            padding=padding, name="Max_Pool%d" % (local_i + 1))
        return maxpool(input_x)

    def add_dense(self, input_x, global_i, local_i):
        shape = input_x.get_shape().as_list()
        input_shape = 1
        for i in range(1,len(shape)):
            input_shape = input_shape * shape[i]

        flat_input = tf.reshape(input_x, [-1, input_shape])
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

    def add_batchnorm(self, input_x, local_i):
        # BATCH NORMALIZATION
        input_channels = input_x.shape[::-1][0].value
        mean, variance = tf.nn.moments(input_x, axes=[0])
        beta = tf.Variable(tf.constant(0.0, shape=[input_channels]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[input_channels]), name='gamma', trainable=True)
        return tf.nn.batch_normalization(input_x, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-5, \
                                         name="BatchNormalization%d"%local_i)

    def def_graph(self, lr, output, actual, loss):
        conv_count = 0
        conv3d_count = 0
        deconv_count = 0
        deconv3d_count = 0
        dense_count = 0
        maxpool_count = 0
        maxpool3d_count = 0
        batchnorm_count = 0

        print("Loading graph..")
        for global_i, layer in zip(range(len(self.layers)), self.layers):
            if layer == 'conv':
                output = self.add_conv(output, global_i, conv_count)
                conv_count += 1

            elif layer == 'conv3d':
                output = self.add_conv3d(output, global_i, conv3d_count)
                conv3d_count += 1

            elif layer == "deconv":
                output = self.add_deconv(input_x=output, local_i=deconv_count)
                deconv_count += 1

            elif layer == "deconv3d":
                output = self.add_deconv3d(input_x=output, local_i=deconv3d_count)
                deconv3d_count += 1

            elif layer == 'maxpool':
                output = self.max_pool(maxpool_count, output)
                maxpool_count += 1

            elif layer == 'maxpool3d':
                output = self.max_pool3d(maxpool3d_count, output)
                maxpool3d_count += 1

            elif layer == 'dense':
                output = self.add_dense(output, global_i, dense_count)
                dense_count += 1

            elif layer == 'batchnorm':
                output = self.add_batchnorm(output, batchnorm_count)
                batchnorm_count += 1

            else:
                raise ValueError("Invalid layer...")

        cost = None
        if loss == "sparse_softmax_cross_entropy":
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actual, logits=output, name="CrossEntropy")
        elif loss == "mse":
            cost = tf.losses.mean_squared_error(labels=actual, predictions=output)
        elif loss == "cross_entropy":
            # Hard code
            # output.shape=[batch_size, D, H, W, C]
            # actual.shape=[batch_size, D, H, W, C]
            if len(output.shape) == 5:
                cost = tf.math.reduce_sum(actual * tf.log(output), axis=[4])
                cost = -1 * tf.math.reduce_mean(cost, axis=[1, 2, 3])
            else:
                # output.shape=[batch_size, H, W, C]
                # actual.shape=[batch_size, H, W, C]
                cost = tf.math.reduce_sum(actual * tf.log(output+1e-9), axis=[3])
                # cost.shape = [batch_size, H, W]
                cost = -1 * tf.math.reduce_mean(cost, axis=[1, 2])

        cost = tf.reduce_mean(cost)
        train = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(cost)

        return output, cost, train

    def optimize(self, X, y, vl_input, vl_y, lr, global_count):
        H = None
        W = None
        C = None
        D = None
        if len(X.shape) == 4:
            _, H, W, C = X.shape
        else:
            _, D, H, W, C = X.shape
        try:
            if lr is None:
                lr = self.lr
            tr_feed = {self.output_original: X, self.actual: y, self.lr_: lr}
            vl_feed = {self.output_original: vl_input, self.actual: vl_y}

            tr_cost, _ = self.sess.run([self.cost, self.train], tr_feed)
            vl_cost = self.sess.run(self.cost, vl_feed)

            if math.isnan(vl_cost):
                print("debug...")

            return tr_cost, vl_cost

        except KeyboardInterrupt:
            save = input("Keyboard interrupt received. Do you want to save the training? (y/n)\n")
            if save == "y":
                save_path = self.saver.save(self.sess, "%s/tmp/model"%(self.proj_dir), global_step=global_count)
                print("Model saved in path: %s" % save_path)
            else:
                print("Model not saved")
                self.sess.close()
            print("Program terminating...")
            sys.exit()

    def predict(self, X, feed_dict):
        #-1 to shift the labels one down since we account for an empty label in 0th position
        if feed_dict is None:
            feed_dict = {self.output_original: X}
        # return self.sess.run(tf.argmax(tf.nn.softmax(self.output), axis=1) -1, feed_dict)
        return self.sess.run(self.output, feed_dict=feed_dict)

    def save_model(self, filename=None):
        # Save the model
        if filename is None:
            filename = self.checkpoint

        save_path = self.saver.save(self.sess, "%s/tmp/%s" %(self.proj_dir,filename))
        print("Model saved in path: %s" % save_path)
