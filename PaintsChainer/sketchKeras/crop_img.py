
'''
File: crop_img.py
Project: sketchKeras
File Created: Thursday, 18th October 2018 7:31:16 pm
Author: xiaofeng (sxf1052566766@163.com)
-----
Last Modified: Friday, 19th October 2018 5:32:45 pm
Modified By: xiaofeng (sxf1052566766@163.com>)
-----
Copyright 2018.06 - 2018 onion Math, onion Math
'''

import cv2
import os
import datetime
import glob
import pickle

file_list = []


def change_size(read_file):
    image_ori = cv2.imread(read_file)
    image = cv2.imread(read_file, 0)  # 读取图片 image_name应该是变量
    ret, thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    # binary_image=thresh[1]               #二值图--具有三通道
    # cv2.imwrite('./crop.jpg', thresh)
    # binary_image=cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
    binary_image = thresh
    # print(binary_image.shape)  # 改为单通道
    originshape = binary_image.shape
    # print('origin shape is:', originshape)
    x = binary_image.shape[0]
    y = binary_image.shape[1]
    edges_x = []
    edges_y = []

    for i in range(x):

        for j in range(y):

            if binary_image[i][j] == 255:
                # print("横坐标",i)
                # print("纵坐标",j)
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right-left  # 宽度

    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top-bottom  # 高度

    pre1_picture = image_ori[left:left+width, bottom:bottom+height, :]  # 图片截取
    cropshape = pre1_picture.shape
    # print('new shape is:', cropshape)
    return pre1_picture  # 返回图片数据


def SearchFiles(root_dir, text):
    global file_list
    assert os.path.exists(root_dir)
    try:
        dir_list = os.listdir(root_dir)
        for files in dir_list:
            child_dir = os.path.join(root_dir, files)
            if os.path.isdir(child_dir):
                SearchFiles(child_dir, text)
            elif os.path.isfile(child_dir) and os.path.splitext(child_dir)[1] in text:
                file_list.append(child_dir)
    except Exception as e:
        print('ERROR：', e)


def main(root_dir=None, child_dirname='512px',saved_dirname='saved_path'):
    global file_list
    if root_dir is None:
         # 获取当前文件路径
        current_path = os.path.abspath(__file__)
        # 获取当前文件的父目录
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        root_dir = father_path
    # 获取图片文件路径
    img_path = os.path.join(root_dir, child_dirname)
    # print('image_path: ',img_path)
    # 保存文件路径
    saved_path = os.path.join(root_dir, saved_dirname)
    file_list_saved = os.path.join(root_dir, 'file_names_from_dataset.pkl')

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    assert os.path.exists(img_path)

    if os.path.exists(file_list_saved):
        with open(file_list_saved, 'rb') as fi:
            file_list = pickle.load(fi)
    else:
        SearchFiles(img_path, ['.jpg', '.png'])
        # print(file_list[:10])
        print('total nus is:', len(file_list))
        with open(file_list_saved, 'wb') as fi:
            pickle.dump(file_list, fi)

    starttime = datetime.datetime.now()
    for i in range(len(file_list)):
        # print('file name is: {:s}:'.format(file_list[i]))
        crop = change_size(file_list[i])  # 得到文件名
        child_path = file_list[i].replace(img_path+'/','')
        saved_file = os.path.join(saved_path, child_path)
        # 待保存文件主目录路径
        saved_dir = os.path.dirname(saved_file)
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)
        cv2.imwrite(saved_file, crop)
        # print("裁剪数量:", i)
        # while(i == 2600):
        #     break
        if i % 100 == 0 and i != 0:
            mid_time = datetime.datetime.now()
            mid_time=(mid_time-starttime).seconds
            print('[{:^10d}] files completed and time cost is {:^010f}'.format(i,mid_time))
    print("裁剪完毕")
    endtime = datetime.datetime.now()  # 记录结束时间
    endtime = (endtime-starttime).seconds
    print("裁剪总用时", endtime)


if __name__ == '__main__':
    main('/hd2/Share/xiaofeng/uniform_dataset/danbooru2017-sfw512px-torrent/danbooru2017')
