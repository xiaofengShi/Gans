import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, SearchFiles, cvt2YUV
from PIL import Image
import cv2
import numpy as np
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class PaintsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.train = opt.isTrain
        self._dtype = np.float32
        self._leak = (0, 20)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, 'sketch')
        self.dir_B = os.path.join(opt.dataroot, opt.phase, 'color')
        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        assert self.A_size == self.B_size, 'Dataset for A and B must be same nums'
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        assert os.path.basename(A_path) == os.path.basename(B_path), 'File name must be same'
        A_img = cv2.imread(A_path, cv2.IMREAD_COLOR)
        B_img = cv2.imread(B_path, cv2.IMREAD_COLOR)

        h_l, w_l, cha_l = A_img.shape
        h_c, w_c, cha_c = B_img.shape

        B_img = cv2.resize(
            B_img, (w_l, h_l), interpolation=cv2.INTER_AREA)

        A_img = np.asarray(A_img, self._dtype)
        B_img = np.asarray(B_img, self._dtype)

        if self._leak[1] > 0:
            if self.train:
                n = np.random.randint(16, self._leak[1])
                r = np.random.rand()
                if r < 0.2:
                    n = 0
                elif r < 0.7:
                    n = np.random.randint(2, 16)
                x = np.random.randint(1, A_img.shape[0] - 1, n)
                y = np.random.randint(1, A_img.shape[1] - 1, n)
                for i in range(n):
                    A_img[x[i], y[i], :] = B_img[x[i], y[i], :]
                    if np.random.rand() > 0.5:
                        A_img[x[i], y[i]-1, :] = A_img[x[i], y[i], :]
                        A_img[x[i], y[i]+1, :] = A_img[x[i], y[i], :]
                    if np.random.rand() > 0.5:
                        A_img[x[i]-1, y[i], :] = A_img[x[i], y[i], :]
                        A_img[x[i]+1, y[i], :] = A_img[x[i], y[i], :]

        A_img = Image.fromarray(np.uint8(A_img))
        B_img = Image.fromarray(np.uint8(B_img))
        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'PaintsDataset'
