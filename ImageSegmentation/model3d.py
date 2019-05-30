"""-------------------------------------------
Author: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
Program: 3D Image Segmentation using Deep Learning
--------------------------------------------"""

from os import listdir, environ
from os.path import join, abspath
environ["PYTHONPATH"] = '/home/rajkumarcm/Documents/TensorFlow-Examples/'
environ["CUDA_VISIBLE_DEVICES"]="1"

from Simple_TF2 import Simple_TF
import numpy as np
import nibabel as nib
import tensorflow as tf
from multiprocessing.pool import ThreadPool
from matplotlib import pyplot as plt
import time


def extract_patches(n_classes, img, lbl):
    """
    Extracts 3d patches from input_X
    :param img:
    :param lbl:
    :return:
    """
    depth, height, width, _ = img.shape
    patch_shape = [128, 128]
    h_si = 0
    h_ei = 128
    imgs = []
    lbls = []
    for i in range(int(height / patch_shape[0])):
        w_si = 0
        w_ei = 128
        for j in range(int(width / patch_shape[1])):
            imgs.append(img[:, h_si:h_ei, w_si:w_ei, 0].reshape([1, depth, patch_shape[0], patch_shape[1], 1]))
            lbls.append(lbl[:, h_si:h_ei, w_si:w_ei, :].reshape([1, depth, patch_shape[0], patch_shape[1],
                                                                 n_classes]))
            w_si = w_ei
            w_ei = w_si + patch_shape[1]
        h_si = h_ei
        h_ei = h_si + patch_shape[0]
    return [imgs, lbls]

def join_patches(predictions):
    patch_shape = [128, 128]
    height = 512
    width = 512
    predicted_lbl = np.zeros([512, 512, 371]).astype(np.uint8)

    h_si = 0
    h_ei = 128
    count = 0
    for i in range(int(height / patch_shape[0])):
        w_si = 0
        w_ei = 128
        for j in range(int(width / patch_shape[1])):
            predicted_lbl[h_si:h_ei, w_si:w_ei, :] = predictions[count]
            w_si = w_ei
            w_ei = w_si + patch_shape[1]
            count += 1
        h_si = h_ei
        h_ei = h_si + patch_shape[0]

    return predicted_lbl

def get_data(args):
    depth = args['depth']
    n_classes = args['n_classes']
    data_path = args['data_path']
    tr_files = args['tr_files']
    vl_files = args['vl_files']
    test_files = args['test_files']
    vl_count = args['vl_count']
    test_count = args['test_count']
    data_type = args['data_type']
    step = args['step']
    dense_rep = args['dense_rep']

    fname = ''
    if data_type == 'training':
        fname = tr_files[step]
        # print(fname)
    elif data_type == 'validation':
        if vl_count >= len(vl_files):
            vl_count = 0
        fname = vl_files[vl_count]
        vl_count += 1
    else:
        if test_count >= len(test_files):
            test_count = 0
        fname = test_files[test_count]
        test_count += 1
    img_path = join(data_path, "aff_imgs")
    img_path = join(img_path, fname)
    lbl_path = join(data_path, "aff_lbls")
    lbl_path = join(lbl_path, fname)

    img = nib.load(img_path).get_data().astype(np.float32)
    for d in range(depth):
        img[:, :, d, 0] /= np.max(img[:, :, d, 0])
    height, width, _, _ = img.shape
    img = np.transpose(img, [2, 0, 1, 3])  # [self.depth, height, width, 1]
    lbl = nib.load(lbl_path).get_data().astype(np.uint8)
    lbl = np.transpose(lbl, [2, 0, 1, 3])  # [self.depth, height, width, 1]
    lbl = lbl[:, :, :, 0]  # [371, 512, 512]

    tmp = np.zeros([depth, height, width, n_classes]).astype(np.uint8)
    tmp_2d_background = np.zeros([depth, height, width]).astype(np.uint8)
    tmp_2d_kidney = np.zeros([depth, height, width]).astype(np.uint8)
    tmp_2d_liver = np.zeros([depth, height, width]).astype(np.uint8)
    if dense_rep:
        indices = lbl == 0
        tmp_2d_background[indices] = 1
        indices = lbl == 3
        tmp_2d_kidney[indices] = 1
        indices = lbl == 4
        tmp_2d_liver[indices] = 1

        tmp[:, :, :, 0] = tmp_2d_background
        tmp[:, :, :, 1] = tmp_2d_kidney
        tmp[:, :, :, 2] = tmp_2d_liver

        del tmp_2d_background
        del tmp_2d_kidney
        del tmp_2d_liver

        return fname, vl_count, test_count, extract_patches(n_classes, img, tmp)
    else:
        return [img, lbl]

def get_data_main(args):
    args['data_type'] = 'training'
    _, vl_count, test_count, tr_data = get_data(args)
    args['vl_count'] = vl_count
    args['test_count'] = test_count
    args['data_type'] = 'validation'
    _, vl_count, test_count, vl_data = get_data(args)
    return vl_count, test_count, tr_data, vl_data

class Seg:

    proj_dir = None
    files = None
    tr_files = None
    vl_files = None
    test_files = None
    tr_count = 0
    vl_count = 0
    tr_data = None
    vl_data = None
    test_count = 0
    model = None
    device = None
    buffered = False
    data_path = "/media/rajkumarcm/Linux Prog/data/segmentation/Medical Data/Data"
    if environ["CUDA_VISIBLE_DEVICES"] == "0,1":
        device = "device1"
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
        # self.height = 112
        # self.width = 112
        #-------------------------------------------------
        self.target_shape = [self.batch_size, self.depth, 128, 128, model["neurons"][::-1][0]]
        self.tr_files = self.files[:partition_size[0]]
        self.steps = int(len(self.tr_files) / self.batch_size)
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

    def update_data(self, args):
        args = args[0]
        self.vl_count = args[0]
        self.test_count = args[1]
        self.tr_data = args[2]
        self.vl_data = args[3]
        self.buffered = True

    def train(self, epochs=10):
        base_dir = self.proj_dir
        args = {}
        args['depth'] = self.depth
        args['n_classes'] = self.n_classes
        args['data_path'] = self.data_path
        args['tr_files'] = self.tr_files
        args['vl_files'] = self.vl_files
        args['test_files'] = self.test_files
        args['vl_count'] = self.vl_count
        args['test_count'] = self.test_count
        args['data_type'] = 'training'
        args['step'] = 0
        args['dense_rep'] = True

        _, _, _, [samples, sample_lbls] = get_data(args)
        sample = samples[0]
        del samples
        seg = Simple_TF(proj_dir=base_dir,
                        sample=sample,
                        lr=self.model["lr"],
                        layers=self.model["layers"],
                        neurons=self.model["neurons"],
                        activations=self.model["activations"],
                        loss=self.model["loss"],
                        epochs=epochs,
                        steps=self.steps,
                        restore=True,
                        device=self.device,
                        batch_size=self.batch_size,
                        output_shape=self.target_shape,
                        checkpoint="3d_model_5.meta")

        plt.figure()
        pool = ThreadPool(1)

        pool.map_async(get_data_main, [args], callback=self.update_data)
        # self.update_data(result.get()[0])
        tr_cost_epoch = []
        vl_cost_epoch = []
        id = model["id"]
        for epoch in range(6, epochs):
            avg_tr_cost = 0
            avg_vl_cost = 0
            for step in range(self.steps):
                global_count = (epoch * self.steps) + step
                while not self.buffered:
                    time.sleep(1)
                tr_imgs, tr_lbls = self.tr_data
                vl_imgs, vl_lbls = self.vl_data
                self.buffered = False
                self.tr_data = None
                self.vl_data = None
                args['vl_count'] = self.vl_count
                args['test_count'] = self.test_count
                args['step'] = step
                pool.map_async(get_data_main, [args], callback=self.update_data)
                tr_cost_img = 0
                vl_cost_img = 0
                n_patches = len(tr_imgs)
                for tr_img, tr_lbl, vl_img, vl_lbl in zip(tr_imgs, tr_lbls, vl_imgs, vl_lbls):
                    tmp_tr_cost, tmp_vl_cost = seg.optimize(X=tr_img,
                                                            y=tr_lbl,
                                                            vl_input=vl_img,
                                                            vl_y=vl_lbl,
                                                            lr=None,
                                                            global_count=global_count)
                    tr_cost_img += tmp_tr_cost
                    vl_cost_img += tmp_vl_cost
                avg_tr_cost += (tr_cost_img/n_patches)
                avg_vl_cost += (vl_cost_img/n_patches)
                print("count: %d, epoch %d step: %d : tr_cost: %.3f, vl_cost: %.3f" %
                      (global_count, epoch, step, tr_cost_img/n_patches, vl_cost_img/n_patches))
                # self.update_data(result.get()[0])
            avg_tr_cost /= self.steps
            avg_vl_cost /= self.steps
            tr_cost_epoch.append(avg_tr_cost)
            vl_cost_epoch.append(avg_vl_cost)
            plt.scatter(epoch, avg_tr_cost, c='b', marker='.', label="Training Error")
            plt.scatter(epoch, avg_vl_cost, c='r', marker='^', label="Validation Error")
            plt.pause(1e-5)
            if epoch == 0:
                plt.legend()
            if (epoch % 5 == 0) and (epoch != 0):
                seg.save_model(filename="3d_model_%d" % epoch)
                """Testing-------------------------------------------"""
                args['step'] = 10
                args['data_type'] = 'testing'
                predictions = []
                fname, _, _, [tmp_imgs, tmp_lbls] = get_data(args)

                for tmp_img, tmp_lbl, i in zip(tmp_imgs, tmp_lbls, range(len(tmp_imgs))):
                    prediction = seg.predict(X=tmp_img, feed_dict=None)
                    #prediction shape: [N, D, H, W, C=3]
                    prediction = np.transpose(prediction, [0, 2, 3, 1, 4]) # [N, H, W, D, Classes]
                    prediction = np.argmax(prediction[0], axis=3).astype(np.uint8)
                    indices = prediction == 1
                    prediction[indices] = 3
                    indices = prediction == 2
                    prediction[indices] = 4
                    prediction = prediction.reshape([128, 128, self.depth])
                    predictions.append(prediction)

                del tmp_imgs
                del tmp_lbls
                del prediction
                predicted_lbl = join_patches(predictions)
                img_path = join(self.data_path, "aff_imgs")
                img_path = join(img_path, fname)
                lbl_path = join(self.data_path, "aff_lbls")
                lbl_path = join(lbl_path, fname)
                tmp_img = nib.load(img_path).get_data()
                tmp_lbl = nib.load(lbl_path).get_data()
                tmp_img = tmp_img[:, :, :, 0]
                tmp_lbl = tmp_lbl[:, :, :, 0]

                """---------------------------------------------------"""
                # np.save('predicted_lbl.npy', predicted_lbl)
                np.save("3d figures/Model %d/3d_prediction_%d.npy" % (id, epoch), predicted_lbl)
                # fig, axes = plt.subplots(1, 3)
                # axes[0].imshow(tmp_img[:, :, 100], cmap="bone")
                # axes[1].imshow(tmp_lbl[:, :, 100], cmap="bone")
                # axes[2].imshow(predicted_lbl[:, :, 100], cmap="bone")
                # plt.show()
        np.save("3d_model_%d.npy" % epochs, [tr_cost_epoch, vl_cost_epoch])
        response = input('Save the model?\n')
        if response == "y":
            seg.save_model()
        else:
            print("Program terminating...")
            exit(0)


if __name__ == '__main__':

    layers = ['conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', \
              'deconv3d', 'conv3d', 'deconv3d', 'conv3d', 'deconv3d', 'conv3d']
    neurons = [8, None, 16, None, 32, None, None, 16, None, 8, None, 3]
    activations = [tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.softmax, None, \
                   None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax]
    # layers = ['conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', 'conv3d', 'maxpool3d', \
    #           'deconv3d', 'conv3d', 'deconv3d', 'conv3d', 'deconv3d', 'conv3d', 'deconv3d', 'conv3d']
    # neurons = [32, None, 64, None, 128, None, 256, None, None, 128, None, 64, None, 32, None, 3]
    # activations = [tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.tanh, None, tf.nn.softmax, None, \
    #                None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax, None, tf.nn.softmax]
    loss= "cross_entropy"
    model = {"id":2, "lr":1e-4, "layers":layers, "neurons":neurons, "activations":activations, "loss":loss}
    seg = Seg(model)
    seg.train(epochs=50)
