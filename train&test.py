##### System library #####
import os
import glob
import os.path as osp
from os.path import exists
import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
##### pytorch library #####
from torchvision import transforms
from torch.utils.data import DataLoader
from d2l import torch as d2l
import torch.backends.cudnn as cudnn
import random
##### My own library #####
from Solver import *
from data.seg_dataset import segList
from utils.PrintLossName import printLossName

# define hyper parameters  
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    # train setting
    parser.add_argument('--device',default=d2l.try_gpu(2), help='set which card to run')
    parser.add_argument('--data_dir',default='/home/image/nvme/ZhouZhiLin/zhouzhilin/Retinal_segmentation/GD-Net/dataset/DUKE', help='Your Data Path Dir')
    parser.add_argument('--name', default='GDNet', dest='name', type=str, help='change model')
    parser.add_argument('--lambCoeff',default=[0, 1, 1, 0], nargs='+', type=int, help='The lambda coefficient of the combined loss function, refer to the following order')
    parser.add_argument('--lossName',default='', type=str, choices=['NLLLoss','DiceLoss','CrossEn','FocalFreq'], help='set loss function')
    parser.add_argument('--optimName',default='Adam', type=str, choices=['Adam','Lion','AdamW','SGD'], help='set optim')
    parser.add_argument('--time',default='', type=str, help='')
    parser.add_argument('--batch_size',default=4, type=int, help='(default: 1)')
    parser.add_argument('--epochs',default=5, type=int, help='(default: 10)')
    parser.add_argument('--lr',default=0.0003, type=float, help='(default: 0.01, 0.00016, 16e-5)')
    parser.add_argument('--step',default=40, type=int, help='The learning rate decays by 10 times per epoch.')
    parser.add_argument('--decay',default=0.2, type=int, help='The multiple of weight decay per step')  
    parser.add_argument('--seed',default=3407, type=int, choices=[3407, 7, 1234]) 
    parser.add_argument('--workers',default=2, type=int) 
    # config
    parser.add_argument('--ImageOrNumpy', default='Image', type=str, choices=['Image','Numpy'])
    parser.add_argument('--lr_mode', default='step', type=str)
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='(default: 1e-4)')
    parser.add_argument('--model_path', default=' ', type=str, help='pretrained model test')
    parser.add_argument('--t',default='', type=str, help='')
    args = parser.parse_known_args()[0]
    return args

def main(learning_rate_decay, learning_rate_step, learning_rate_init, alpha, lambCoeffList, iepo, seedSet=None, model_name=None):
    args = parse_args()
    args.decay = learning_rate_decay
    args.step = learning_rate_step 
    args.lr = learning_rate_init
    args.lambCoeff = lambCoeffList
    
    lambCoeff = args.lambCoeff
    optimName = args.optimName
    args.t = optimName
    device = args.device
    ImageOrNumpy = args.ImageOrNumpy
    seed = args.seed # 1234 3407 7
    
    if model_name is not None:
        args.name = model_name
    
    def set_seed(seed):
        print('Successfully setting seed!')
        os.environ['PYTHONHASHSEED'] = str(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        cudnn.enabled = False
        torch.manual_seed(seed)        
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)        
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)
    if seedSet == True: set_seed(seed)     
    else: 
        print('Not successfully setting seed!')
        cudnn.deterministic = False
        # cudnn.benchmark = True
        args.time = ('time' + str(iepo))

    # 制作保存结果的路径
    Loss_Use = printLossName(lambCoeff)
    datasetting = (args.data_dir.split('/')[-1]).split('_')[-1]
    if args.time:args.time = '_' + args.time
    else:pass
    if args.t:end=f'alpha:{alpha}_' + args.t
    else:end = f'alpha:{alpha}'
    script_path = os.path.dirname(os.path.abspath(__file__))    
    train_result_path = osp.join(script_path, 'result', 'train', args.name + '_LossFun_' + Loss_Use  + '_lr' + str(args.lr) + f'_every{args.step}epoch_decay{args.decay}' + '_batch' + str(args.batch_size) + f'_data{datasetting}'+f'_seed({args.seed})' + '_' + end + args.time)
    print('train_result_path:', train_result_path)
    if not exists(train_result_path):os.makedirs(train_result_path)
    test_result_path = osp.join(script_path, 'result', 'test', args.name + '_LossFun_' + Loss_Use + '_lr' + str(args.lr) + f'_every{args.step}epoch_decay{args.decay}' +'_batch' + str(args.batch_size) + f'_data{datasetting}'+f'_seed({args.seed})' + '_' + end + args.time)
    print('test_result_path:', test_result_path)
    if not exists(test_result_path):os.makedirs(test_result_path)
    
    # Add tensorboard object
    # writer = SummaryWriter("", flush_secs=60)

    # load dataset
    if ImageOrNumpy == 'Image':        
        train_dataset = segList(args.data_dir, 'train')        
        val_dataset = segList(args.data_dir, 'eval')
        test_dataset = segList(args.data_dir, 'test')
    # elif ImageOrNumpy == 'Numpy':
    #     # numpy dataset
    #     _, _, _, train_dataset, val_dataset, test_dataset= get_data('', img_size=224, batch_size=10)
    else: raise NameError("Unknow ImageOrNumpy Name!")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    eval_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    ##### train #####
    # train_seg(args, train_result_path, train_loader, eval_loader, optimName, lambCoeff, device=device, alpha=alpha, writer=None, iepo=iepo)
    # writer.close() # if necessary
    
    ##### test #####
    # model_best_path = osp.join(osp.join(train_result_path,'model'), 'model_best.pth')
    # model_best_path = osp.join(osp.join(train_result_path,'model'), 'model_latest.pth')
    model_best_path = 'GD-Net/checkpoint/model_best.pth' # checkpoint path
    args.model_path = model_best_path
    test_seg(args, test_result_path, test_loader, device=device, alpha=alpha)

if __name__ == '__main__':
    
    main(learning_rate_decay=0.2, 
         learning_rate_step=20, 
         learning_rate_init=0.0006, 
         alpha=0.25, 
         lambCoeffList=[0, 1, 0, 0], 
         iepo=1, 
         seedSet=False)
