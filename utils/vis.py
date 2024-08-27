import numpy as np
import os
import os.path as osp
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from os.path import exists
from PIL import Image

def vis_result(imn, imt, ant, pred, save_dir, n_class=9, has_fluid=0):  
    imn = imn[0]
    imt = transforms.ToPILImage()(imt)
    imt = transforms.Resize((496,496), transforms.InterpolationMode.NEAREST)(imt) 
    imt = np.array(imt) 
    
    img = gray2rgbimage(imt)
    pred_img = draw_img(imt, pred)
    if(ant is None):
        cv2.imwrite(osp.join(save_dir, imn), np.hstack((img, pred_img)).astype('uint8'))
    else:
        ant_img = draw_img(imt, ant)
        if not exists(osp.join(save_dir, 'stacked')):
            os.makedirs(osp.join(save_dir, 'stacked'))
        cv2.imwrite(osp.join(save_dir, 'stacked/' + imn), np.hstack((img, ant_img, pred_img)).astype('uint8'))
        cv2.imwrite(osp.join(save_dir, 'label/' + imn), ant_img.astype('uint8'))
        cv2.imwrite(osp.join(save_dir, 'pred/' + imn), pred_img.astype('uint8'))

def draw_img(img, seg, title = None):
    mask = img
    label_set = [1, 2, 3, 4, 5, 6, 7, 8]
    color_set = {
                    # 0:(200,200,200), bg
                    1: (50, 100, 250),  # NFL
                    2: (0, 255, 0),  # GCL-IPL
                    3: (0, 0, 255),  # INL
                    4: (0, 255, 255),  # OPL
                    5: (255, 0, 255),  # ONL-ISM
                    6: (255, 255, 0),  # ISE
                    7: (0, 0, 150),  # OS-RPE
                    8: (255, 0, 0),  # Fluid
                    # 9: (255, 0, 0),  
                }

    mask = gray2rgbimage(mask) 
    img = gray2rgbimage(img)  
    if(title is not None):
        mask = cv2.putText(mask, title, (16, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # white title
    for draw_label in label_set:
        mask[:, :, 0][seg[0,:,:] == draw_label] = (color_set[draw_label][0])
        mask[:, :, 1][seg[0,:,:] == draw_label] = (color_set[draw_label][1])
        mask[:, :, 2][seg[0,:,:] == draw_label] = (color_set[draw_label][2])
    
    img_mask = cv2.addWeighted(img, 0.4, mask, 0.6, 0)
    return img_mask


def gray2rgbimage(image):
    a, b = image.shape
    new_img = np.ones((a,b,3))
    new_img[:,:,0] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,1] = image.reshape((a,b)).astype('uint8')
    new_img[:,:,2] = image.reshape((a,b)).astype('uint8')
    return new_img

