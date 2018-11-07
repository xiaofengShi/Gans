'''
File: main.py
Project: sketchKeras
File Created: Sunday, 7th October 2018 5:51:22 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Sunday, 7th October 2018 7:09:45 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''

from keras.models import load_model
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from keras.utils import plot_model
import datetime
import cv2
import os
import numpy as np
import pickle
from helper_sketch import *


class Sketch:
    def __init__(self, gpu=0):

        print("start")
        self.root = "./images/"
        self.batchsize = 1
        self.outdir = self.root + "sketch/"
        self.gpu = gpu
        self._dtype = np.float32

        if not os.path.isfile("./sketchKeras/mod.h5"):
            print("/sketchKeras/mod.h5 not found. Please download them from github")

        print("load model")
        if self.gpu >= 0:
            self.gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
            self.model_config = tf.ConfigProto(device_count={"CPU": 7},
                                               gpu_options=self.gpu_option,
                                               intra_op_parallelism_threads=0,
                                               inter_op_parallelism_threads=0)

        else:
            self.model_config = tf.ConfigProto(device_count={"CPU": 2, "GPU": 0},
                                               intra_op_parallelism_threads=0,
                                               inter_op_parallelism_threads=0)

        self.model = load_model('./sketchKeras/mod.h5')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def tosketch(self, id_str):
        path = os.path.join(self.root, 'line', id_str + '.png')
        saved_path = os.path.join(self.outdir, id_str+'.jpg')
        from_mat = cv2.imread(path)
        width = float(from_mat.shape[1])
        height = float(from_mat.shape[0])
        new_width = 0
        new_height = 0
        if (width > height):
            from_mat = cv2.resize(
                from_mat, (512, int(512 / width * height)),
                interpolation=cv2.INTER_AREA)
            new_width = 512
            new_height = int(512 / width * height)
        else:
            from_mat = cv2.resize(from_mat, (int(512 / height * width), 512),
                                  interpolation=cv2.INTER_AREA)
            new_width = int(512 / height * width)
            new_height = 512
        from_mat = from_mat.transpose((2, 0, 1))
        light_map = np.zeros(from_mat.shape, dtype=np.float)
        for channel in range(3):
            light_map[channel] = get_light_map_single(from_mat[channel])
        light_map = normalize_pic(light_map)
        light_map = resize_img_512_3d(light_map)
        line_mat = self.model.predict(light_map, batch_size=self.batchsize)
        line_mat = line_mat.transpose((3, 1, 2, 0))[0]
        line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
        # show_active_img_and_save('sketchKeras_colored', line_mat, saved_path)
        line_mat = np.amax(line_mat, 2)
        # show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, saved_path)
        show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, saved_path)
        # show_active_img_and_save_denoise('sketchKeras', line_mat, saved_path)
        # cv2.waitKey(0)


if __name__ == '__main__':
    for n in range(1):
        s = Sketch()
        s.tosketch(n * s.batchsize)
