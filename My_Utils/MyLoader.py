#coding=utf-8
import torch
import cv2
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import ntpath
from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torch.utils.data import DataLoader


def im_to_tensor(im):
    d = np.unique(im)      #unique() 寻找唯一元素
    # print('d:', d.shape)
    l = []
    for idx, n in enumerate(d[0:]):       #将d组合成一个索引序列 同时列出数据和数据下标
        i = np.where(im == n)
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)     #im.shape[0]：矩阵的行数, im.shape[1] 矩阵的列数
        t[i] = 1
        t = transforms.ToPILImage()(t)    #将数组转换为图片
        t = transforms.Resize((224,224), transforms.InterpolationMode.NEAREST)(t)     #图片标准化
        # t = transforms.CenterCrop((400, 512))(t)
        # t = transforms.Resize((256, 512))(t)

        l.append(t)
    if len(d) == 9:
        t = np.zeros((im.shape[0], im.shape[1]), np.float32)
        t = transforms.ToPILImage()(t)
        t = transforms.Resize((224,224), transforms.InterpolationMode.NEAREST)(t)
        # t = transforms.CenterCrop((400, 512))(t)
        # t = transforms.Resize((256, 512))(t)
        l.append(t)
    st = np.dstack(tuple(l))     #tuple 元组 dstack：沿深度方向按顺序叠放数组
    st = np.swapaxes(st, 0, 2)
    st = np.swapaxes(st, 1, 2)     #交换轴的位置
    tensor = torch.from_numpy(st)
    return tensor

im_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),      #数据归一化，标准正态分布
    transforms.Resize((224,224), transforms.InterpolationMode.BILINEAR)
    # transforms.CenterCrop((400, 512))
    # transforms.Resize((256,512))
])

class ISBI_Loader():
    def __init__(self, data_path, patient_left=-1):
        # 初始化功能，读取data_path下的所有图片
        self.data_path = data_path
        self.patient_left = patient_left
        discard = glob.glob(os.path.join(data_path, f'img_{patient_left}_*'))
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_*'))
        # print(discard)
        for path in discard:
            # print(path)
            self.imgs_path.remove(path)
            # self.labs_path.remove(path)

    def __getitem__(self, index):
        # print("__getitem__被执行了")
        # 根据索引阅读图片
        image_path = self.imgs_path[index]
        # 把img旧字符串替换成seg新的字符串，label_path里面就存的是标签图片
        label_path = image_path.replace('img', 'seg')
        # 阅读训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
              
        # Convert the data to a single channel picture  将数据转换为单通道图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       #转灰度
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # image = image[20:350,135:634]
        # label = label[20:350,135:634]

        image = im_trans(image)   # 这里将图片转换为张量并resize为(224, 224)
        # 处理标签，将像素值从255改为1
        if label.max()>1:
            label = label/255
        # print("test")
        return image, im_to_tensor(label)

    def __len__(self):
        # Return the training set size  返回训练集大小
        return len(self.imgs_path)

class Test_Loader(ISBI_Loader):
    def __init__(self, data_path, patient_left=0):
        # Initialization function, read all the pictures under data_path   读取图片
        super().__init__(data_path, patient_left)
        self.imgs_path = glob.glob(os.path.join(data_path, f'img_{patient_left}_*'))

    def __getitem__(self, index):
        img, seg = ISBI_Loader.__getitem__(self, index)
        image_path = self.imgs_path[index]
        filename = ntpath.basename(image_path)
        return img, seg, filename

def keep_image_size_open(path, size=(256, 256)):
    """
    这里实现图片等比缩放
    Args:
        path (_type_): _description_
        size (tuple, optional): _description_. Defaults to (256, 256).
    """
    img = Image.open(path)  # 将图片读进来
    temp = max(img.size)    # 取图片的最长边
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0)) # 将图片粘贴上来
    mask = mask.resize(size)
    return mask

transform = transforms.Compose([
    transforms.ToTensor() # 转换为张量数据类型
])

class MyDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.name1 = os.listdir(os.path.join(path, 'image2'))  # os.listdir可以获取该文件夹下的所有路径,所以self.name中保存的是每一张图片的名称
        self.name2 = os.listdir(os.path.join(path, 'mask2'))                                                      
    
    def __len__(self):
        return len(self.name1)
    
    def __getitem__(self, index):
        segment_name1 = self.name1[index]  # 拿到每一张图片的名称xx.png
        segment_name2 = self.name2[index]  
        
        
        segment_path = os.path.join(self.path, 'image2', segment_name1)
        image_path = os.path.join(self.path, 'mask2', segment_name2)
        
        segment_image = keep_image_size_open(image_path)
        image = keep_image_size_open(segment_path)

        return transform(image), transform(segment_image)

if __name__ == "__main__":
    mydata = r'/home/zhouzhilin/data/duke'        # 数据地址
    test = ISBI_Loader(mydata)
    for image, seg_image in test:
        print(image.type)
        print(seg_image.type)
    
    
    
    # train_loader = DataLoader(ISBI_Loader(mydata), batch_size=6, shuffle=False)   #实例化
    # print("一个batch的图片个数:", len(train_loader))
    # for i, (img, label) in enumerate(train_loader):
    #     print("i:", i)
    #     print("img:", img.shape)
    #     print("seg_img:", len(label))
    #     print('--------------------')

    #     unloader = transforms.ToPILImage()
    #     # img = img.transpose(1,2,3,0)
    #     image_show = img.cpu().clone()  # clone the tensor
    #     print(image_show.shape)
    #     image_show = np.argmax(image_show, axis=0)
    #     print(image_show.shape)
    #     image_show = unloader(image_show)
    #     image_show = image_show.transpose(1,2,3,0)
    #     image_show.save('/home/zhouzhilin/Code_Myself/U_Net/test_image')
    #     print("-----------")
    #     print(label.shape)
    #     save_path="D:/data/1"
    #     # image = image.cpu().detach().numpy()
    #     # image = np.argmax(image, axis=1)
    #     # image = Image.fromarray(image[0].squeeze())
    #     image.save(f'{save_path}/.png')