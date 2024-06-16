##### pytorch library #####
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import os.path as osp
from os.path import exists
import torch
import logging
import time
from torch.autograd import Variable
import copy
import os
import os.path as osp
import torch.backends.cudnn as cudnn
from os.path import exists
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
##### My own library #####
from My_Utils.optimizer import Lion
from My_Utils.utils import *
from utils.vis import vis_result
from utils.logger import Logger
from model import net_builder
from My_Utils.utils import loss_builder1
# from utils.utils import dice_coeff, adjust_learning_rate
from utils.utils import AverageMeter, save_model


# logger vis
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger_vis = logging.getLogger(__name__)
logger_vis.setLevel(logging.DEBUG)

# training process
def train(args, train_loader, model, loss_fun, optimizer, lambCoeff, device = None):
    # set the AverageMeter，每一轮都清零重新计算, dice里是平均值
    batch_time, losses, dice = AverageMeter(), AverageMeter(), AverageMeter()
    # set每一层的dice，每一轮都清零重新计算
    Dice_1, Dice_2, Dice_3, Dice_4, Dice_5, Dice_6, Dice_7, Dice_8 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # print('input:', input.shape)
        # print('target:', target.shape)
        # variable
        input_var, target_var_seg = input.to(device), target.to(device)
        # forward
        modelName = model.__class__.__name__        
        if modelName == 'FTC':
            input_var = input_var.repeat(1, 3, 1, 1)                
        output_seg = model(input_var)
        output_seg = output_seg.contiguous()
        target_var_loss = torch.argmax(target_var_seg, dim=1)
           
        def chooseLoss(choice):
            if choice == 'NLLLoss': # nn.NLLLoss
                loss = loss_fun[0](output_seg, target_var_loss)
            elif choice == 'DiceLoss': # DiceLoss
                loss = loss_fun[1](output_seg, target_var_loss) 
            elif choice == 'CrossEn': # 交叉熵CrossEn 
                loss = loss_fun[2](output_seg, target_var_loss) 
            elif choice == 'FocalFrequency': # FocalFrequencyLoss
                max_val, idx = torch.max(output_seg, 1)
                pred_oh = torch.nn.functional.one_hot(idx, num_classes=9)
                pred_oh = pred_oh.permute(0, 3, 1, 2)
                
                targetseg = torch.argmax(target_var_seg, dim=1)
                label_oh = torch.nn.functional.one_hot(targetseg, num_classes=9)
                label_oh = label_oh.permute(0, 3, 1, 2)
                loss = loss_fun[3](pred_oh, label_oh)            
            else:
               raise NameError("Unknow Loss Name!")
            return loss
        loss = lambCoeff[0]*chooseLoss('NLLLoss')+lambCoeff[1]*chooseLoss('DiceLoss')+lambCoeff[2]*chooseLoss('CrossEn')+lambCoeff[3]*chooseLoss('FocalFrequency')
        contour_loss = HausdorffDistanceLoss()
        loss_hd = contour_loss(output_seg, target_var_loss)
        loss = loss + loss_hd
        # loss = chooseLoss(lossName)
        # loss = chooseLoss(2)*0.6 + loss_1*0.4   
        # loss = loss_2
        
        losses.update(loss.data, input.size(0))
        
        pred_seg = torch.argmax(output_seg, dim=1)
        pred_seg = pred_seg.data.cpu().numpy()
        target_var_seg = torch.argmax(target_var_seg, dim=1)
        label_seg = target_var_seg.data.cpu().numpy()
        ret_d = compute_dice(label_seg, pred_seg)
        dice_score = compute_single_avg_score(ret_d)  # 求平均
        # 更新dice分数
        dice.update(dice_score) # dice是平均值
        Dice_1.update(ret_d[1])
        Dice_2.update(ret_d[2])
        Dice_3.update(ret_d[3])
        Dice_4.update(ret_d[4])
        Dice_5.update(ret_d[5])
        Dice_6.update(ret_d[6])
        Dice_7.update(ret_d[7])
        Dice_8.update(ret_d[8])
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 测量train一轮的时间
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg,dice.avg,Dice_1.avg,Dice_2.avg,Dice_3.avg,Dice_4.avg,Dice_5.avg,Dice_6.avg,Dice_7.avg,Dice_8.avg
           
# evaluation process
def eval(phase, args, eval_data_loader, model, result_path = None, logger = None, device = None, loss_fun=None):
    batch_time, dice, mpa = AverageMeter(), AverageMeter(), AverageMeter()
    Dice_1, Dice_2, Dice_3, Dice_4, Dice_5, Dice_6, Dice_7, Dice_8 = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    pa_1, pa_2, pa_3, pa_4, pa_5, pa_6, pa_7, pa_8 = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    dice_list, mpa_list = [], []
    ret_dice, ret_pa = [], []
    eval_loss = AverageMeter()
    # switch to eval mode
    model.eval()
    end = time.time()
    for iter, (image, label, imt, imn) in enumerate(eval_data_loader):
        with torch.no_grad():
            image_var = image.to(device)
            # model forward
            modelName = model.__class__.__name__
            if modelName == 'FTC':
                image_var = image_var.repeat(1, 3, 1, 1)              
            ############ 计算模型的推理时间 ############
            # print('计算模型的推理时间')
            # inference_list = []
            # start_time = time.time()
            
            output_seg = model(image_var) # (1, 9, 496, 496)            
            output_seg = output_seg.contiguous()
              
            # end_time = time.time()
            # inference_time = end_time - start_time
            # inference_list.append(inference_time)
            # print(f'推理时间：{inference_time} 秒')
            ############ 计算模型的推理时间 ############           
            
            pred = torch.argmax(output_seg, dim=1) # ([1, 496, 496])                                   
            # 保存可视化结果, 将预测结果的张量转换为数组np
            pred_seg = pred.data.cpu().numpy().astype('uint8')
            if phase == 'eval' or phase == 'test':
                imt = (imt.squeeze().numpy()).astype('uint8')
                label_vis = torch.argmax(label, dim=1)
                ant = label_vis.numpy().astype('uint8')
                save_dir = osp.join(result_path, 'vis_ret')
                if not exists(save_dir): os.makedirs(save_dir)
                if not exists(save_dir + '/label'): os.makedirs(save_dir + '/label')
                if not exists(save_dir + '/pred'): os.makedirs(save_dir + '/pred')
                has_fluid = 0
                vis_result(imn, imt, ant, pred_seg, save_dir, has_fluid=has_fluid)
                # print('Saved visualized results!')
            # 计算分割结果的 dice and pa 分数            
            pred_seg = pred.data.cpu().numpy()
            label = torch.argmax(label, dim=1)
            label_seg = label.data.cpu().numpy()            
            
            ret_d = compute_dice(label_seg, pred_seg)            
            ret_p = compute_pa(label_seg, pred_seg)
            ret_dice.append(ret_d)
            ret_pa.append(ret_p)  
             
            # dice_score = compute_single_avg_score_fluid(ret_d)  # 带着积液计算平均dice 
            dice_score = compute_single_avg_score(ret_d) # 只算7个视网膜层的平均Dice                         
            mpa_score = compute_single_avg_score(ret_p)
            dice_list.append(dice_score)

            # 更新 dice and pa score
            dice.update(dice_score)
            Dice_1.update(ret_d[1])
            Dice_2.update(ret_d[2])
            Dice_3.update(ret_d[3])
            Dice_4.update(ret_d[4])
            Dice_5.update(ret_d[5])
            Dice_6.update(ret_d[6])
            Dice_7.update(ret_d[7])
            Dice_8.update(ret_d[8])         
            mpa_list.append(mpa_score)
            mpa.update(mpa_score)
            pa_1.update(ret_p[1])
            pa_2.update(ret_p[2])
            pa_3.update(ret_p[3])
            pa_4.update(ret_p[4])
            pa_5.update(ret_p[5])
            pa_6.update(ret_p[6])
            pa_7.update(ret_p[7])
            pa_8.update(ret_p[8])
            
            #### 补实验 ####
            if phase == 'eval' or phase == 'train':
                target_var_loss = label
                target_var_loss = target_var_loss.to(device)
                output_seg = output_seg.to(device)
                def chooseLoss(choice):
                    if choice == 'NLLLoss': # nn.NLLLoss
                        loss = loss_fun[0](output_seg, target_var_loss)
                    elif choice == 'DiceLoss': # DiceLoss
                        loss = loss_fun[1](output_seg, target_var_loss) 
                    elif choice == 'CrossEn': # 交叉熵CrossEn 
                        loss = loss_fun[2](output_seg, target_var_loss) 
                    elif choice == 'FocalFrequency': # FocalFrequencyLoss
                        max_val, idx = torch.max(output_seg, 1)
                        pred_oh = torch.nn.functional.one_hot(idx, num_classes=9)
                        pred_oh = pred_oh.permute(0, 3, 1, 2)
                        
                        # targetseg = torch.argmax(target_var_seg, dim=1)
                        targetseg = label
                        label_oh = torch.nn.functional.one_hot(targetseg.to(device), num_classes=9)
                        label_oh = label_oh.permute(0, 3, 1, 2)
                        loss = loss_fun[3](pred_oh, label_oh)
                    else:
                        raise NameError("Unknow Loss Name!")
                    return loss
                loss = 1*chooseLoss('DiceLoss') + 1*chooseLoss('CrossEn') + 0.2*chooseLoss('FocalFrequency')
                contour_loss = HausdorffDistanceLoss()
                loss_hd = contour_loss(output_seg, target_var_loss)
                loss = loss + loss_hd            
                eval_loss.update(loss.data, image.size(0))                                    
        # 测量经过时间
        batch_time.update(time.time() - end)
        end = time.time()
    
    # print('平均推理时间：', np.mean(inference_list))    
    dice_avg = (Dice_1.avg+Dice_2.avg+Dice_3.avg+Dice_4.avg+Dice_5.avg+Dice_6.avg+Dice_7.avg+Dice_8.avg)/8
    final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8 = dice_avg, Dice_1.avg, Dice_2.avg, Dice_3.avg, Dice_4.avg, Dice_5.avg, Dice_6.avg, Dice_7.avg, Dice_8.avg
    final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8 = mpa.avg, pa_1.avg, pa_2.avg, pa_3.avg, pa_4.avg, pa_5.avg, pa_6.avg, pa_7.avg, pa_8.avg
    if phase == 'eval' or phase == 'test':
        print('Final Dice_avg Score:{:.4f}'.format(final_dice_avg))
        print('Final Dice_Healthy Score:{:.4f}'.format(dice.avg))
        print('Final Dice_1 Score:{:.4f}'.format(final_dice_1))
        print('Final Dice_2 Score:{:.4f}'.format(final_dice_2))
        print('Final Dice_3 Score:{:.4f}'.format(final_dice_3))
        print('Final Dice_4 Score:{:.4f}'.format(final_dice_4))
        print('Final Dice_5 Score:{:.4f}'.format(final_dice_5))
        print('Final Dice_6 Score:{:.4f}'.format(final_dice_6))
        print('Final Dice_7 Score:{:.4f}'.format(final_dice_7))
        print('Final Dice_8 Score:{:.4f}'.format(final_dice_8))
        print('Final PA_avg:{:.4f}'.format(final_pa_avg))
        print('Final PA_1 Score:{:.4f}'.format(final_pa_1))
        print('Final PA_2 Score:{:.4f}'.format(final_pa_2))
        print('Final PA_3 Score:{:.4f}'.format(final_pa_3))
        print('Final PA_4 Score:{:.4f}'.format(final_pa_4))
        print('Final PA_5 Score:{:.4f}'.format(final_pa_5))
        print('Final PA_6 Score:{:.4f}'.format(final_pa_6))
        print('Final PA_7 Score:{:.4f}'.format(final_pa_7))
        print('Final PA_8 Score:{:.4f}'.format(final_pa_8))
    if phase == 'eval' or phase == 'test':
        logger.append(
        [final_dice_avg, dice.avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8,
        final_pa_avg, final_pa_1, final_pa_2, final_pa_3, final_pa_4, final_pa_5, final_pa_6, final_pa_7, final_pa_8])
    return final_dice_avg, final_dice_1, final_dice_2, final_dice_3, final_dice_4, final_dice_5, final_dice_6, final_dice_7, final_dice_8, dice_list, eval_loss.avg

###### train ######
def train_seg(args, train_result_path, train_loader, eval_loader, optimName, lambCoeff, device = None, alpha=None, writer=None):
    # logger setting
    logger_train = Logger(fpath=osp.join(train_result_path,'dice_epoch.txt'), title='dice', resume=False)
    logger_train.set_names(['Epoch','Dice_Train','Dice_Val','Dice_1','Dice_11','Dice_2','Dice_22','Dice_3','Dice_33','Dice_4','Dice_44','Dice_5','Dice_55','Dice_6','Dice_66','Dice_7','Dice_77','Dice_8','Dice_88'])
    # print hyperparameters
    for k, v in args.__dict__.items():
        print(k, ':', v)
    # load the network
    net = net_builder(name=args.name, pretrained_model=None, pretrained=False, ratio_in=alpha)
    # model = torch.nn.DataParallel(net).to(device)
    model = net.to(device)
    print('#'*15, args.name, '#'*15)
    # 定义损失函数
    loss_fun = loss_builder1(device)
    
    # set optimizer
    def chooseOptimier(choice):
        if choice == 'Adam':
            return optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        elif choice == 'Lion':
            return Lion(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        elif choice == 'AdamW':
            return optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9,0.99), weight_decay=args.weight_decay)
        elif choice == 'SGD':
            return optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise NameError("Unknow Optimizer Name!")
    optimizer = chooseOptimier(optimName)
    
    # main training
    best_dice = 0
    # 模拟训练和验证过程并记录数据
    train_losses = []
    eval_losses = []
    for epoch in range(args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch, args.decay) # 动态调整学习率
        # logger_vis.info('Epoch: [{0}]\t'.format(epoch))  # 打印轮次
        # train for one epoch
        loss,dice_train,dice_1,dice_2,dice_3,dice_4,dice_5,dice_6,dice_7,dice_8 = train(args, train_loader, model, loss_fun, optimizer, lambCoeff, device=device)        
        # evaluate on validation set
        dice_val,dice_11,dice_22,dice_33,dice_44,dice_55,dice_66,dice_77,dice_88,dice_list,eval_loss = eval('train', args, eval_loader, model, device=device, loss_fun=loss_fun)
        
        # 将损失函数添加到tensorboard中
        ratio = 20
        train_losses.append((loss/ratio).cpu().numpy())
        eval_losses.append((eval_loss/ratio).cpu().numpy())
        writer.add_scalars('Loss', {'Train loss': loss/ratio, 'Validation Loss': eval_loss/ratio}, epoch)
        # writer.add_scalar("train_loss", loss, epoch)
        # writer.add_scalar("eval_loss", eval_loss, epoch) 
        # writer.add_scalar("dice_train", dice_train, epoch)
        # writer.add_scalar("dice_val", dice_val, epoch)
        
        # save the best model
        is_best = dice_val > best_dice
        best_dice = max(dice_val, best_dice)
        model_dir = osp.join(train_result_path, 'model')
        if not exists(model_dir):
            os.makedirs(model_dir)
        save_model({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'dice_epoch':dice_val, 'best_dice': best_dice}, is_best, model_dir)
        # logger 
        logger_train.append([epoch+1,dice_train,dice_val,dice_1,dice_11,dice_2,dice_22,dice_3,dice_33,dice_4,dice_44,dice_5,dice_55,dice_6,dice_66,dice_7,dice_77,dice_8,dice_88])
    
    # 使用Matplotlib绘制图像
    # 使用Savitzky-Golay滤波器平滑数据 (可选)
    train_losses = savgol_filter(train_losses, window_length=3, polyorder=2)
    eval_losses = savgol_filter(eval_losses, window_length=10, polyorder=2)
    # 使用Savitzky-Golay滤波器平滑数据 (可选)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(eval_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Evaluation Loss')
    # 保存图像为PNG文件
    plt.savefig('')
    plt.close()

###### validation ######
def eval_seg(args, eval_result_path, eval_loader, device = None):
    # logger setting
    logger_eval = Logger(osp.join(eval_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_eval.set_names(
          ['Dice', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4', 'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8', 'Dice_9','Dice_10',
         'mpa', 'pa_1', 'pa_2','pa_3', 'pa_4', 'pa_5', 'pa_6', 'pa_7', 'pa_8'])
    # load the model
    print('Loading eval model: {}'.format(args.name))
    net = net_builder(args.name)
    # model = torch.nn.DataParallel(net).to(device)
    model = net.to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded!')
    # evaluate the model on validation set
    eval('train', args, eval_loader, model, result_path = eval_result_path, logger = logger_eval)

###### test ######
def test_seg(args, test_result_path, test_loader, device = None, alpha=None):
    print('===========开始测试===========')
    # logger setting
    logger_test = Logger(osp.join(test_result_path, 'dice_mpa_epoch.txt'), title='dice&mpa', resume=False)
    logger_test.set_names(
          ['DiceAll', 'DiceHealthy', 'Dice_1', 'Dice_2', 'Dice_3', 'Dice_4', 'Dice_5', 'Dice_6', 'Dice_7', 'Dice_8',
           'mpa', 'pa_1', 'pa_2','pa_3', 'pa_4', 'pa_5', 'pa_6', 'pa_7', 'pa_8'])
    # load the model
    print('Loading test model ...')
    net = net_builder(name=args.name, pretrained_model=None, pretrained=False, ratio_in=alpha)
    # model = torch.nn.DataParallel(net).to(device)
    model = net.to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load(args.model_path))
    print('Model loaded!')
    print('最好的网络模型是第{}个epoch保存下来的'.format(checkpoint['epoch']))
    # test the model on testing set
    eval('test', args, test_loader, model, result_path = test_result_path, logger = logger_test, device=device)

