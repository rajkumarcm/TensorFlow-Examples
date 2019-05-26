"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: Image Segmentation using Deep Learning
--------------------------------------------"""

import os
os.environ["PYTHONPATH"] = '/home/rajkumarcm/Documents/TensorFlow-Examples/'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from Simple_TF2 import Simple_TF
import numpy as np
import nibabel as nib
import tensorflow as tf
from matplotlib import pyplot as plt

class Seg:

    proj_dir = None
    model = None
    batch_size = 1
    device = None
    if os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
        device = "cpu"
    else:
        device = "gpu"

    # Temp variables---------------------------------
    tr_imgs = None
    tr_lbls = None
    vl_imgs = None
    vl_lbls = None
    shape = None
    target_shape = None
    steps = None
    #------------------------------------------------

    def __init__(self, model):
        self.proj_dir = self.get_proj_dir()
        self.model = model

        """Data for training------------------------------------------------------------------------"""
        self.tr_imgs = nib.load("/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data/aff_imgs/nusurgery007.512.nii.gz")
        self.tr_imgs = self.tr_imgs.get_data()
        self.tr_lbls = nib.load("/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data/aff_lbls/nusurgery007.512.nii.gz")
        self.tr_lbls = self.tr_lbls.get_data()

        X, Y, Z, _ = self.tr_imgs.shape
        self.shape = [1, X, Y, 1]
        self.target_shape = [1, X, Y, 3]
        self.steps = Z

        """Convert to Float to avoid numerical errors---------------------------------"""
        self.tr_imgs = self.tr_imgs.astype(np.float32)
        tmp = np.zeros([Z, X, Y])
        for z in range(Z):
            tmp[z] = self.tr_imgs[:,:,z,0] / np.max(self.tr_imgs[:,:,z,0])
        self.tr_imgs = np.copy(tmp)
        del tmp

        self.tr_lbls = np.transpose(self.tr_lbls, [2, 0, 1, 3])
        self.tr_lbls = self.tr_lbls.reshape([Z, X, Y])
        self.tr_lbls = self.tr_lbls.astype(np.uint8)

        """Data for validation-----------------------------------------------------------"""
        self.vl_imgs = nib.load("/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data/aff_imgs/nusurgery007.512.nii.gz")
        self.vl_imgs = self.vl_imgs.get_data()
        self.vl_lbls = nib.load("/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data/aff_lbls/nusurgery007.512.nii.gz")
        self.vl_lbls = self.vl_lbls.get_data()

        """Convert to float validation set----------------------------------------------"""
        self.vl_imgs = self.vl_imgs.astype(np.float32)
        tmp = np.zeros([Z, X, Y])
        for z in range(Z):
            tmp[z] = self.vl_imgs[:,:,z,0] / np.max(self.vl_imgs[:,:,z,0])
        self.vl_imgs = tmp
        del tmp

        self.vl_lbls = np.transpose(self.vl_lbls, [2, 0, 1, 3])
        self.vl_lbls = np.reshape(self.vl_lbls, [Z, X, Y])

        #----------------------------------------------------------------------------------------------------------------------------

    def get_proj_dir(self):
        path = os.path.abspath("")
        dir_parts = path.split("/")
        curr_dir = dir_parts[::-1][0]
        if curr_dir == "TensorFlow-Examples":
            path = os.path.abspath("ImageSegmentation")
        return path

    def get_tr_data(self, step, dense_rep=True):
        if dense_rep:
            tmp1 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.tr_lbls[step] == 0
            tmp1[indices] = 1

            tmp2 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.tr_lbls[step] == 3
            tmp2[indices] = 1

            tmp3 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.tr_lbls[step] == 4
            tmp3[indices] = 1

            tmp_lbl = np.zeros(self.target_shape)
            tmp_lbl[:, :, :, 0] = tmp1
            tmp_lbl[:, :, :, 1] = tmp2
            tmp_lbl[:, :, :, 2] = tmp3
            del tmp1
            del tmp2
            del tmp3

            return [self.tr_imgs[step].reshape(self.shape), tmp_lbl]
        else:
            return [self.tr_imgs[step].reshape(self.shape),
                    self.tr_lbls[step].reshape(self.shape)]

    def get_vl_data(self, step, dense_rep=True):
        if dense_rep:
            tmp1 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.vl_lbls[step] == 0
            tmp1[indices] = 1

            tmp2 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.vl_lbls[step] == 3
            tmp2[indices] = 1

            tmp3 = np.zeros([self.shape[1], self.shape[2]])  # WARNING OF HARD-CODING-----------------------
            indices = self.vl_lbls[step] == 4
            tmp3[indices] = 1

            tmp_lbl = np.zeros(self.target_shape)
            tmp_lbl[:,:,:,0] = tmp1
            tmp_lbl[:,:,:,1] = tmp2
            tmp_lbl[:,:,:,2] = tmp3
            del tmp1
            del tmp2
            del tmp3

            return [self.vl_imgs[step].reshape(self.shape), tmp_lbl]
        else:
            return [self.vl_imgs[step].reshape(self.shape),
                    self.vl_lbls[step].reshape(self.shape)]

    def train(self, epochs=30):
        base_dir = self.proj_dir
        sample, _ = self.get_tr_data(step=0)
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
                tr_img, tr_lbl = self.get_tr_data(step)
                vl_img, vl_lbl = self.get_vl_data(step)
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
    layers = ['conv', 'maxpool', 'conv', 'maxpool', 'conv', 'maxpool', 'conv', 'maxpool', \
              'deconv', 'conv', 'deconv', 'conv', 'deconv', 'conv', 'deconv', 'conv']
    neurons = [32, None, 64, None, 128, None, 256, None, None, 128, None, 64, None, 32, None, 3]
    activations = [tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.softmax, None, \
                   None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax]
    loss= "mse"
    model = {"id":0, "lr":1e-4, "layers":layers, "neurons":neurons, "activations":activations, "loss":"mse"}
    seg = Seg(model)
    seg.train()
