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
from helper import *


GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
DEVICE_CONFIG_GPU = tf.ConfigProto(device_count={"CPU": 7},
                                   gpu_options=GPU_OPTIONS,
                                   intra_op_parallelism_threads=0,
                                   inter_op_parallelism_threads=0)
K.set_session(tf.Session(config=DEVICE_CONFIG_GPU))

mod = load_model('./mod.h5')
plot_model(mod, to_file='model.png', show_shapes=True)


def get(path, save_path):
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
    # cv2.imshow('raw', from_mat)
    # cv2.imwrite('raw.jpg', from_mat)
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    # show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    # show_active_img_and_save_denoise_filter2('sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, save_path)
    # show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')
    # cv2.waitKey(0)


# file_list = []

# def SearchFiles(root_dir, text):
#     global file_list
#     assert os.path.exists(root_dir)
#     try:
#         dir_list = os.listdir(root_dir)
#         for files in dir_list:
#             child_dir = os.path.join(root_dir, files)
#             if os.path.isdir(child_dir):
#                 SearchFiles(child_dir, text)
#             elif os.path.isfile(child_dir) and os.path.splitext(child_dir)[1] in text:
#                 file_list.append(child_dir)
#     except Exception as e:
#         print('ERROR：', e)


# def main():
#     global file_list
#     img_path = '/hd2/Share/xiaofeng/uniform_dataset/danbooru2017-sfw512px-torrent/danbooru2017/saved_path'
#     save_dir = '/hd2/Share/xiaofeng/uniform_dataset/danbooru2017-sfw512px-torrent/danbooru2017/sketch_path'
#     file_list_saved = '/hd2/Share/xiaofeng/uniform_dataset/danbooru2017-sfw512px-torrent/danbooru2017/file_names_from_saved.pkl'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if os.path.exists(file_list_saved):
#         with open(file_list_saved, 'rb') as fi:
#             file_list = pickle.load(fi)
#     else:
#         SearchFiles(img_path, ['.jpg', '.png'])
#         # print(file_list[:10])
#         with open(file_list_saved, 'wb') as fi:
#             pickle.dump(file_list, fi)
#     total_num = len(file_list)
#     print('total nums is:', total_num)
#     start = datetime.datetime.now()
#     for i in range(total_num):
#         child_path = file_list[i].replace(img_path+'/', '')
#         saved_file = os.path.join(save_dir, child_path)
#         saved_dir = os.path.dirname(saved_file)
#         if not os.path.exists(saved_dir):
#             os.makedirs(saved_dir)
#         try:
#             get(file_list[i], saved_file)
#             if i % 100 == 0:
#                 mid = datetime.datetime.now()
#                 print('[{:^10d}]{:^5s}[{:^10d}]'.format(i, ':', total_num))
#                 print('\t[{:^10s}]{:^5s}[{:010d}]'.format('Time', ':', (mid-start).seconds))
#         except Exception as e:
#             print(e)


if __name__ == '__main__':
    # main()
    
