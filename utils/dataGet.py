import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as tf

import os
from typing import Callable
from torch.utils.data import Dataset
from os.path import join,exists
from PIL import Image
import torch
import os
import os.path as osp
import numpy as np 
# import seg_transforms as st
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

def get_files(path, ext):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith(ext)]


class TransformOCTBilinear(object):
    def __new__(cls, img_size=(128,128),*args, **kwargs):
        return tf.Compose([
            tf.Resize(img_size)
            ])


def get_data(data_path, img_size, batch_size, val_batch_size=1):
    train_dataset_path = os.path.join(data_path, "train")
    val_dataset_path = os.path.join(data_path, "val")
    test_dataset_path = os.path.join(data_path, "test")

    size_transform = TransformOCTBilinear(img_size=(img_size, img_size))

    img_transform = None

    train_dataset = DatasetOct(train_dataset_path, size_transform=size_transform, normalized=True, image_transform=img_transform)
    val_dataset = DatasetOct(val_dataset_path, size_transform=size_transform, normalized=True)
    test_dataset = DatasetOct(test_dataset_path, size_transform=size_transform, normalized=True)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset


class TransformStandardization(object):
    """
    Standardizaton / z-score: (x-mean)/std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + f": mean {self.mean}, std {self.std}"


class TransformOCTMaskAdjustment(object):
    """
    Adjust OCT 2015 Mask
    from: classes [0,1,2,3,4,5,6,7,8,9], where 9 is fluid, 0 is empty space above and 8 empty space below
    to: class 0: not class, classes 1-7: are retinal layers, class 8: fluid
    """
    def __call__(self, mask):
        mask[mask == 8] = 0
        mask[mask == 9] = 8
        return mask


class DatasetOct(Dataset):
    """
    Map style dataset object for oct 2015 data
    - expects .npy files
    - assumes sliced images, masks - produced by our project: dataloading/preprocessing.py
        (valid dims of images,masks and encoding: pixel label e [0..9], for every pixel)

    Parameters:
        dataset_path: path to the dataset path/{images,masks}
        size_transform: deterministic transformation for resizing applied to image and mask separately
        joint_transform: random transformations applied to image and mask jointly after size_transform
        image_transform: transformation applied only to the image and after joint_transform
    _getitem__(): returns image and corresponding mask
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, size_transform: Callable = None,
                 image_transform: Callable = None, normalized=True) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'images')
        self.output_path = os.path.join(dataset_path, 'masks')
        self.images_list = get_files(self.input_path, ".npy")

        # size transform
        self.size_transform = size_transform

        self.joint_transform = joint_transform

        self.mask_adjust = TransformOCTMaskAdjustment()

        self.image_transform = image_transform

        self.normalized = normalized
        # gray scale oct 2015: calculated with full tensor in memory {'mean': tensor([46.3758]), 'std': tensor([53.9434])}
        # calculated with batched method {'mean': tensor([46.3756]), 'std': tensor([53.9204])}
        self.normalize = TransformStandardization((46.3758),
                                                  (53.9434))  # torchvision.transforms.Normalize((46.3758), (53.9434))

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        img = np.load(os.path.join(self.input_path, image_filename))
        mask = np.load(os.path.join(self.output_path, image_filename))

        # img_size 128 works - general transforms require (N,C,H,W) dims
        img = img.squeeze()
        mask = mask.squeeze()

        img = torch.Tensor(img).reshape(1, 1, *img.shape)
        mask = torch.Tensor(mask).reshape(1, 1, *mask.shape).int()

        # adjust mask classes
        mask = self.mask_adjust(mask)

        # note for some reason some masks differ in size from the actual image (dims)
        if self.size_transform:
            img = self.size_transform(img)
            mask = self.size_transform(mask)

        # normalize after size_transform
        if self.normalized:
            img = self.normalize(img)

        if self.joint_transform:
            img, mask = self.joint_transform([img, mask])

        #img = img.reshape(1, img.shape[2], img.shape[3])
        if self.image_transform:
            img = self.image_transform(img)

        #img = img.reshape(1, *img.shape)

        # set image dim to (C,H,W)
        img = img.squeeze()
        img = img.reshape(1, *img.shape)
        # set mask dim to (H,W)  where value at (h,w) maps to class of corresponding pixel
        mask = mask.squeeze(dim=1).long()

        return img, mask


class segList(Dataset):
    def __init__(self, data_dir, phase):
        self.data_dir = data_dir
        self.phase = phase
        # self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        if self.phase == 'train':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            
            image = cv2.imread(self.image_list[index])  # 读取图像文件，返回NumPy数组，其中包含图像的像素值。
            label = cv2.imread(self.label_list[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) 
            
            image = im_trans(image)     
            if label.max()>1:
                label = label/255        
            label = im_to_tensor(label)            
            return image, label

        
        if self.phase == 'eval' or 'test':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            image_vis = [Image.open(self.image_list[index])]
            image = cv2.imread(self.image_list[index])
            imt = torch.from_numpy(np.array(image_vis[0]))
            # imt = im_trans_imt(imt)
            label = cv2.imread(self.label_list[index])
            imn = self.image_list[index].split('/')[-1]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            image = im_trans(image) 
            if label.max()>1:
                label = label/255
            label = im_to_tensor(label)            
            return (image, label, imt, imn)

        if self.phase == 'predict':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            data = [Image.open(self.image_list[index])]
            imt = torch.from_numpy(np.array(data[0]))
            # data = list(self.transforms(*data))
            image = data[0]
            imn = self.image_list[index].split('/')[-1]
            return (image, imt, imn)

    # def __len__(self):
    #     return len(self.image_list)

    def read_lists(self):    
        self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
        print('Total amount of {} images is : {}'.format(self.phase, len(self.image_list)))
        self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
        print('Total amount of {} labels is : {}'.format(self.phase, len(self.image_list)))

def get_list_dir(phase, type, data_dir):
    data_dir = osp.join(data_dir, phase, type)
    files = os.listdir(data_dir)
    list_dir = []
    for file in files:
        file_dir = osp.join(data_dir, file)
        list_dir.append(file_dir)
    return sorted(list_dir)

def im_to_tensor(im):
    d = np.unique(im)
    # print('d:', d)
    l = [] 
    for idx, n in enumerate(d[0:]): 
        i = np.where(im == n)
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t[i] = 1
        t = transforms.ToPILImage()(t)  
        t = transforms.Resize((496, 496), transforms.InterpolationMode.NEAREST)(t)  
        # t = transforms.CenterCrop((400, 512))(t)
        # t = transforms.Resize((256, 512))(t)
        l.append(t)
        
    if len(d) == 9:
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((496, 496), transforms.InterpolationMode.NEAREST)(t)
        # t = transforms.CenterCrop((400, 512))(t)
        # t = transforms.Resize((256, 512))(t)
        l.append(t)
    st = np.dstack(tuple(l)) 
    st = np.swapaxes(st, 0, 2)
    st = np.swapaxes(st, 1, 2)    
    tensor = torch.from_numpy(st)
    return tensor

im_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((496, 496), transforms.InterpolationMode.BILINEAR)
    # transforms.CenterCrop((400, 512))
    # transforms.Resize((256,512))
])    