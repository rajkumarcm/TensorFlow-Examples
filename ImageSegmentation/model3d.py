"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: 3D Image Segmentation using Deep Learning
--------------------------------------------"""

from os import listdir, environ
from os.path import join, abspath
environ["PYTHONPATH"] = '/home/rajkumarcm/Documents/TensorFlow-Examples/'
environ["CUDA_VISIBLE_DEVICES"]="-1"

from Simple_TF2 import Simple_TF
import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt

class Seg:

    proj_dir = None
    files = None
    tr_files = None
    vl_files = None
    test_files = None
    tr_count = 0
    vl_count = 0
    test_count = 0
    model = None
    device = None
    data_path = "/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data"
    if environ["CUDA_VISIBLE_DEVICES"] == "-1":
        device = "cpu"
    else:
        device = "gpu"

    batch_size = 1
    n_classes = 3
    steps = None
    #------------------------------------------------

    def __init__(self, model, partition_size=[90, 30, 20]):
        self.proj_dir = self.get_proj_dir()
        self.model = model
        tr_path = join(self.data_path, "aff_imgs")
        self.files = listdir(tr_path)
        self.n_files = len(self.files)
        tmp_path = join(tr_path, self.files[0])
        sample = nib.load(tmp_path).get_data()
        self.height, self.width, self.depth, _ = sample.shape
        """DEBUGGING----------------------------------"""
        self.height = 112
        self.width = 112
        #-------------------------------------------------
        self.target_shape = [self.batch_size, self.depth, self.height, self.width, model["neurons"][::-1][0]]
        self.steps = int(self.n_files/self.batch_size)
        self.tr_files = self.files[:partition_size[0]]
        self.vl_files = self.files[partition_size[0]:partition_size[0]+partition_size[1]]
        last_index = partition_size[0] + partition_size[1]
        self.test_files = self.files[last_index:last_index+partition_size[2]]

    def get_proj_dir(self):
        path = abspath("")
        dir_parts = path.split("/")
        curr_dir = dir_parts[::-1][0]
        if curr_dir == "TensorFlow-Examples":
            path = abspath("ImageSegmentation")
        return path

    def get_data(self, data_type, step, dense_rep=True):
        fname = ''
        if data_type == 'training':
            fname = self.tr_files[step]
        elif data_type == 'validation':
            if self.vl_count >= len(self.vl_files):
                self.vl_count = 0
            fname = self.vl_files[self.vl_count]
        else:
            if self.test_count >= len(self.test_files):
                self.test_count = 0
            fname = self.test_files[self.test_count]
        img_path = join(self.data_path, "aff_imgs")
        img_path = join(img_path, fname)
        lbl_path = join(self.data_path, "aff_lbls")
        lbl_path = join(lbl_path, fname)

        img = nib.load(img_path).get_data()
        height, width, _, _ = img.shape
        img = np.transpose(img, [2, 0, 1, 3]).reshape([1, self.depth, height, width, 1])
        lbl = nib.load(lbl_path).get_data()
        lbl = np.transpose(lbl, [2, 0, 1, 3]).reshape([1, self.depth, height, width, 1])

        tmp = np.zeros([1, self.depth, height, width, self.n_classes])
        tmp_2d_background = np.zeros([height, width])
        tmp_2d_kidney = np.zeros([height, width])
        tmp_2d_liver = np.zeros([height, width])
        if dense_rep:
            for d in range(self.depth):
                tmp_lbl = lbl[0, d, :, :, 0]
                indices = tmp_lbl == 0
                tmp_2d_background[indices] = 1

                indices = tmp_lbl == 3
                tmp_2d_kidney[indices] = 1

                indices = tmp_lbl == 4
                tmp_2d_liver[indices] = 1

                # WARNING BATCH SIZE HARD-CODED
                tmp[0, d, :, :, 0] = tmp_2d_background
                tmp[0, d, :, :, 1] = tmp_2d_kidney
                tmp[0, d, :, :, 2] = tmp_2d_liver

            del tmp_2d_background
            del tmp_2d_kidney
            del tmp_2d_liver
            """DEBUGGING----------------------------------"""
            img = img[0, :, :self.height, :self.width, :].reshape([1, 371, self.height, self.width, 1])
            tmp = tmp[0, :, :self.height, :self.width, :].reshape([1, 371, self.height, self.width, self.n_classes])
            #------------------------------------------------
            return [img, tmp]
        else:
            return [img, lbl]


    def train(self, epochs=30):
        base_dir = self.proj_dir
        sample, _ = self.get_data(data_type='training', step=0)
        seg = Simple_TF(proj_dir=base_dir,
                        sample=sample,
                        lr=self.model["lr"],
                        layers=self.model["layers"],
                        neurons=self.model["neurons"],
                        activations=self.model["activations"],
                        loss=self.model["loss"],
                        epochs=epochs,
                        steps=self.steps,
                        restore=False,
                        device=self.device,
                        batch_size=self.batch_size,
                        output_shape=self.target_shape,
                        checkpoint="model_final")

        plt.figure()
        tmp_img = None
        tmp_lbl = None
        for epoch in range(epochs):
            avg_tr_cost = 0
            avg_vl_cost = 0
            for step in range(self.steps):
                global_count = (epoch * self.steps) + step
                tr_img, tr_lbl = self.get_data(data_type='training', step=step)
                vl_img, vl_lbl = self.get_data(data_type='validation', step=step)
                tr_cost, vl_cost = seg.optimize(X=tr_img,
                                                y=tr_lbl,
                                                vl_input=vl_img,
                                                vl_y=vl_lbl,
                                                lr=None,
                                                global_count=global_count)
                avg_tr_cost += tr_cost
                avg_vl_cost += vl_cost
                print("count: %d, epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f" %
                      (global_count, epoch, step, tr_cost, vl_cost))
            avg_tr_cost /= self.steps
            avg_vl_cost /= self.steps
            plt.scatter(epoch, avg_tr_cost, c='b', marker='.', label="Training Error")
            plt.scatter(epoch, avg_vl_cost, c='r', marker='^', label="Validation Error")
            plt.pause(1e-5)
            if epoch == 0:
                plt.legend()
        """Testing-------------------------------------------"""
        tmp_img, tmp_lbl = self.get_vl_data(step=10, dense_rep=False)
        prediction = seg.predict(X=tmp_img, feed_dict=None)
        prediction = np.argmax(prediction[0], axis=2)
        indices = prediction == 1
        prediction[indices] = 3
        indices = prediction == 2
        prediction[indices] = 4
        """---------------------------------------------------"""
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(tmp_img[0,:,:,0], cmap="bone")
        axes[1].imshow(tmp_lbl[0,:,:,0], cmap="bone")
        axes[2].imshow(prediction, cmap="bone")
        plt.show()
        response = input('Save the model?\n')
        if response == "y":
            seg.save_model()
        else:
            print("Program terminating...")
            exit(0)


if __name__ == '__main__':
    layers = ['conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', \
              'deconv3d', 'conv3d', 'deconv3d', 'conv3d', 'deconv3d', 'conv3d', 'deconv3d', 'conv3d']
    neurons = [32, None, 64, None, 128, None, 256, None, None, 128, None, 64, None, 32, None, 3]
    activations = [tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.softmax, None, \
                   None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax]
    loss= "mse"
    model = {"id":0, "lr":1e-4, "layers":layers, "neurons":neurons, "activations":activations, "loss":"mse"}
    seg = Seg(model)
    seg.train()
