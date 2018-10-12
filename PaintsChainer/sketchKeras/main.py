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
from keras.utils import plot_model

import cv2
import numpy as np
from helper import *

mod = load_model('./mod.h5')
plot_model(mod, to_file='model.png')


def get(path):
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
    cv2.imwrite('raw.jpg', from_mat)
    from_mat = from_mat.transpose((2, 0, 1))
    light_map = np.zeros(from_mat.shape, dtype=np.float)
    for channel in range(3):
        light_map[channel] = get_light_map_single(from_mat[channel])
    light_map = normalize_pic(light_map)
    light_map = resize_img_512_3d(light_map)
    line_mat = mod.predict(light_map, batch_size=1)
    line_mat = line_mat.transpose((3, 1, 2, 0))[0]
    line_mat = line_mat[0:int(new_height), 0:int(new_width), :]
    show_active_img_and_save('sketchKeras_colored', line_mat, 'sketchKeras_colored.jpg')
    line_mat = np.amax(line_mat, 2)
    show_active_img_and_save_denoise_filter2(
        'sketchKeras_enhanced', line_mat, 'sketchKeras_enhanced.jpg')
    show_active_img_and_save_denoise_filter('sketchKeras_pured', line_mat, 'sketchKeras_pured.jpg')
    show_active_img_and_save_denoise('sketchKeras', line_mat, 'sketchKeras.jpg')
    cv2.waitKey(0)
    return


get('test.jpg')
