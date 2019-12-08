# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn

import os.path as osp
import numpy as np
import random
import cv2
from torch.utils import data


"""
CamVid is a road scene understanding dataset with 367 training images and 233 testing images of day and dusk scenes. 
The challenge is to segment 11 classes such as road, building, cars, pedestrians, signs, poles, side-walk etc. We 
resize images to 360x480 pixels for training and testing.
"""

CAMVID_CLASSES = ['Sky',
                  'Building',
                  'Column-Pole',
                  'Road',
                  'Sidewalk',
                  'Tree',
                  'Sign-Symbol',
                  'Fence',
                  'Car',
                  'Pedestrain',
                  'Bicyclist',
                  'Void']

CAMVID_CLASS_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
    (0, 0, 0),
]


class CamVidDataSet(data.Dataset):
    """
       CamVidDataSet is employed to load train set
       Args:
        root: the CamVid dataset path,
        list_path: camvid_train_list.txt, include partial path

    """
    def __init__(self, root=None, list_path='./dataset/list/CamVid/camvid_train_list.txt',
                 max_iters=None, crop_size=(360, 360),
                 mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, vars=(1,1,1), RGB=False):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.vars = vars
        self.is_mirror = mirror
        self.rgb = RGB
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print("length of train set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label[label==11] = self.ignore_label
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            f_scale = 0.5 + random.randint(0, 15) / 10.0  # random resize between 0.5 and 2
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        if self.rgb:
            image = image[:,:, ::-1]  ## BGR -> RGB
            image /= 255         ## using pytorch pretrained models

        image -= self.mean
        image /= self.vars

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)

        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))  # NHWC -> NCHW

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class CamVidTestDataSet(data.Dataset):
    """
       CamVidValDataSet is employed to load val set
       Args:
        root: the CamVid dataset path,
        list_path: camvid_val_list.txt, include partial path

    """

    def __init__(self, root='/home/DataSet/CamVid', list_path='./dataset/list/CamVid/camvid_val_list.txt',
                 f_scale=1, mean=(128, 128, 128), ignore_label=255, vars=(1,1,1), RGB=False):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.vars = vars
        self.rgb = RGB
        self.f_scale = f_scale
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            label_file = osp.join(self.root, name.split()[1])
            image_name = name.strip().split()[0].strip().split('/', 1)[1].split('.')[0]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of Test Set: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.f_scale != 1:
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_NEAREST)

        label[label == 11] = self.ignore_label

        image = np.asarray(image, np.float32)

        if self.rgb:
            image = image[:, :, ::-1]  ## BGR -> RGB
            image /= 255  ## using pytorch pretrained models

        image -= self.mean
        image /= self.vars

        image = image.transpose((2, 0, 1))  # HWC -> CHW

        # print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name
