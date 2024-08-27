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
import json

class segList(Dataset):
    def __init__(self, data_dir, phase):
        self.data_dir = data_dir
        self.phase = phase
        # self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        # self.info = json.load(open(osp.join(data_dir, 'info.json'), 'r'))

    def __getitem__(self, index):
        if self.phase == 'train':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            
            image = cv2.imread(self.image_list[index])  
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
            image = data[0]
            imn = self.image_list[index].split('/')[-1]
            return (image,imt,imn)

    def __len__(self):
        return len(self.image_list)

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
        t = transforms.Resize((496,496), transforms.InterpolationMode.NEAREST)(t)
        l.append(t)
        
    if len(d) == 8:        
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((496,496), transforms.InterpolationMode.NEAREST)(t)
        l.append(t)
    st = np.dstack(tuple(l))
    st = np.swapaxes(st, 0, 2)
    st = np.swapaxes(st, 1, 2)
    tensor = torch.from_numpy(st)
    return tensor

im_trans = transforms.Compose([
    transforms.ToTensor(),    
    transforms.Normalize([0.5], [0.5]),
    transforms.Resize((496,496), transforms.InterpolationMode.NEAREST)
])    


if __name__ == '__main__':
    mydata = '' # path dir
    test = segList(mydata,'train')
    for image, seg_image in test:
        print('image.shape:', image.shape)
        print('seg_image.shape:', seg_image.shape)