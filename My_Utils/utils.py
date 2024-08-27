from PIL import Image
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from einops import rearrange,reduce,repeat
import traceback
import torch
import os
import os.path as osp
import cv2
import scipy.misc as misc
import shutil
from skimage import measure
import math
import traceback
from sklearn import metrics
import zipfile
from torch.autograd import Variable
from d2l import torch as d2l
from torchvision import transforms
from scipy.spatial.distance import directed_hausdorff
# from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error

def adjust_learning_rate(args, optimizer, epoch, decay):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if args.lr_mode == 'step':
        lr = args.lr * (decay ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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

def loss_builder1(device):
    criterion_0 = nn.NLLLoss(ignore_index=255)
    # criterion_1 = DiceLoss(class_num=8, device=device)     # DiceLoss
    criterion_1 = DiceLoss()
    criterion_2 = nn.CrossEntropyLoss()                    # 交叉熵损失函数
    criterion_3 = FocalFrequencyLoss()
    # criterion_3 = DiceLoss_GPT()                           # ChatGPT给我找的DiceLoss
    criterion = [criterion_0, criterion_1, criterion_2, criterion_3]
    return criterion

# 师兄的代码
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def make_one_hot(input, shape):
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    师兄的代码
    Args:
         weight: An array of shape [num_classes,]
         ignore_index: class index to ignore
         predict: A tensor of shape [N, C, *]
         target: A tensor of same shape with predict
         other args pass to BinaryDiceLoss
         Return:
         same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
  

    def forward(self, predict, target):
        shape=predict.shape
        # print(shape)
        target = torch.unsqueeze(target, 1)
        target = make_one_hot(target.long(),shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]

class FocalFrequencyLoss(nn.Module):
    """The torch.nn.Module class that implements focal frequency loss - a
    frequency domain loss function for optimizing generative models.
    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>
    Args:
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1, ave_spectrum=False, log_matrix=False, batch_matrix=False):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix: # False
                matrix_tmp = torch.log(matrix_tmp + 1.0)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix: # False
                matrix_tmp = matrix_tmp / matrix_tmp.max()
            else:
                matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
            'The values of spectrum weight matrix should be in the range [0, 1], '
            'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate focal frequency loss.
        Args:
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix) * self.loss_weight

class HausdorffDistanceLoss(nn.Module):
    def __init__(self):
        super(HausdorffDistanceLoss, self).__init__()

    def extract_boundaries(self, image):
        boundaries = np.zeros_like(image)
        for value in np.unique(image):
            if value == 0:
                continue
            mask = (image == value)
            # Detect edges (simple way by finding boundary pixels)
            boundary = np.zeros_like(mask)
            boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
            boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
            boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
            boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
            boundaries[boundary] = value
        return boundaries

    def hausdorff_distance(self, image1, image2):
        boundaries1 = self.extract_boundaries(image1)
        boundaries2 = self.extract_boundaries(image2)
        
        # Get boundary points
        points1 = np.column_stack(np.where(boundaries1 > 0))
        points2 = np.column_stack(np.where(boundaries2 > 0))
        
        # Calculate Hausdorff distance
        forward_distance = directed_hausdorff(points1, points2)[0]
        backward_distance = directed_hausdorff(points2, points1)[0]
        
        return max(forward_distance, backward_distance)

    def forward(self, output, target):
        # Convert PyTorch tensors to numpy arrays
        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Initialize Hausdorff distance
        hausdorff_distances = []

        # Compute Hausdorff distance for each sample in the batch
        for i in range(output_np.shape[0]):
            # Get the prediction and target for the current sample
            pred = np.argmax(output_np[i], axis=0)
            true = target_np[i]

            # Compute Hausdorff distance for the current sample
            distance = self.hausdorff_distance(pred, true)
            hausdorff_distances.append(distance)

        # Convert to tensor
        hausdorff_distances = torch.tensor(hausdorff_distances, dtype=output.dtype, device=output.device)

        # Return the mean Hausdorff distance over the batch
        return hausdorff_distances.mean()

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
        # Sobel kernel for edge detection
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, prediction, target):
        # Apply softmax to prediction to get probabilities
        prediction = F.softmax(prediction, dim=1)
        
        # Get the predicted class (channel with the highest probability)
        pred_classes = torch.argmax(prediction, dim=1, keepdim=True)
        
        # One-hot encode the target
        target_one_hot = F.one_hot(target, num_classes=prediction.size(1)).permute(0, 3, 1, 2).float()
        
        # Detect edges using Sobel operator
        pred_edges = self._detect_edges(pred_classes.float())
        target_edges = self._detect_edges(target_one_hot.float())
        
        # Calculate the L1 loss between the edges
        loss = F.l1_loss(pred_edges, target_edges)
        
        return loss

    def _detect_edges(self, img):
        # Apply Sobel operator to each channel separately using group convolution
        sobel_kernel_x = self.sobel_kernel_x.repeat(img.size(1), 1, 1, 1).to(img.device)
        sobel_kernel_y = self.sobel_kernel_y.repeat(img.size(1), 1, 1, 1).to(img.device)
        
        grad_x = F.conv2d(img, sobel_kernel_x, padding=1, groups=img.size(1))
        grad_y = F.conv2d(img, sobel_kernel_y, padding=1, groups=img.size(1))
        
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return edges

def dice_coeff(pred, target):
    smooth = 1e-8
    # avg_score=0
    num = pred.size(0)
    probs = F.sigmoid(pred)
    m1 = probs.contiguous().view(num, -1)  # Flatten
    m2 = target.contiguous().view(num, -1)  # Flatten
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    # avg_score = score + (score / num)
    return score

def dice_coeff_old(pred, target):
    smooth = 1.
    num = pred.size(0)
    probs = F.sigmoid(pred)
    m1 = probs.contiguous().view(num, -1)  # Flatten
    m2 = target.contiguous().view(num, -1)  # Flatten
    intersection = (m1 * m2)

    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num
    return score

def compute_dice(ground_truth, prediction): 
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  
        for i in range(9): 
            mask1 = (ground_truth == i)
            mask2 = (prediction == i)
            if mask1.sum() != 0:
                ret[i] = float(2 * ((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum() + mask2.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR in computing dice:", e)
        return None
    return ret

def cal_dice(seg, gt, classes=11, background_id=-1):
    channel_dice = []
    for x in range(224):
        for y in range(224):
            seg[x,y] += 1
            gt[x,y] += 1
    for i in range(1,classes):
        if i == background_id:
            continue
        cond = i ** 2
        # 计算相交部分
        inter = len(np.where(seg * gt == cond)[0])
        total_pix = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0])
        if total_pix == 0:
            dice = 0
        else:
            dice = (2 * inter) / total_pix
        # print(dice)
        channel_dice.append(dice)
    # res = np.array(channel_dice).mean()
    return channel_dice

def cal_mIoU(seg, gt, classes=11, background_id=-1):
    channel_iou = []
    for i in range(classes):
        if i == background_id:
            continue
        cond = i ** 2
        # 计算相交部分
        inter = len(np.where(seg * gt == cond)[0])
        union = len(np.where(seg == i)[0]) + len(np.where(gt == i)[0]) - inter
        if union == 0:
            iou = 0
        else:
            iou = inter / union
        channel_iou.append(iou)
    res = np.array(channel_iou).mean()
    return res

def save_model(state, is_best, model_path):
    model_latest_path = osp.join(model_path,'model_latest.pth.tar')   
    torch.save(state, model_latest_path)
    if is_best:
        model_best_path = osp.join(model_path,'model_best.pth.tar')
        shutil.copyfile(model_latest_path, model_best_path)

def save_dice_single(is_best, filename='dice_single.txt'):
    if is_best:
        shutil.copyfile(filename, 'dice_best.txt')

def compute_pa(ground_truth, prediction): 
    ground_truth = ground_truth.flatten()
    prediction = prediction.flatten()
    try:
        ret = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  
        for i in range(10): 
            mask1 = (ground_truth == i)
            if mask1.sum() != 0:
                ret[i] = float(((mask1 * (ground_truth == prediction)).sum()) / (mask1.sum()))
            else:
                ret[i] = float('nan')
    except Exception as e:
        traceback.print_exc()
        print("ERROR msg:", e)
        return None
    return ret

def compute_avg_score(ret_seg):
    BG, NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg, Choroid_seg, Disc_seg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000001
    num = np.array(ret_seg).shape[0]
    for i in range(num):
        if not math.isnan(ret_seg[i][0]):
            BG += ret_seg[i][0]
            n0 += 1
        if not math.isnan(ret_seg[i][1]):
            NFL_seg += ret_seg[i][1]
            n1 += 1
        if not math.isnan(ret_seg[i][2]):
            GCL_seg += ret_seg[i][2]
            n2 += 1
        if not math.isnan(ret_seg[i][3]):
            IPL_seg += ret_seg[i][3]
            n3 += 1
        if not math.isnan(ret_seg[i][4]):
            INL_seg += ret_seg[i][4]
            n4 += 1
        if not math.isnan(ret_seg[i][5]):
            OPL_seg += ret_seg[i][5]
            n5 += 1
        if not math.isnan(ret_seg[i][6]):
            ONL_seg += ret_seg[i][6]
            n6 += 1
        if not math.isnan(ret_seg[i][7]):
            IS_OS_seg += ret_seg[i][7]
            n7 += 1
        if not math.isnan(ret_seg[i][8]):
            RPE_seg += ret_seg[i][8]
            n8 += 1
        if not math.isnan(ret_seg[i][9]):
            Choroid_seg += ret_seg[i][9]
            n9 += 1
        if not math.isnan(ret_seg[i][10]):
            Disc_seg += ret_seg[i][10]
            n10 += 1
    BG /= n0
    NFL_seg /= n1
    GCL_seg /= n2
    IPL_seg /= n3
    INL_seg /= n4
    OPL_seg /= n5
    ONL_seg /= n6
    IS_OS_seg /= n7
    RPE_seg /= n8
    Choroid_seg /= n9
    Disc_seg /= n10
    avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg + RPE_seg + Choroid_seg + Disc_seg) / 10
    return avg_seg, NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg, Choroid_seg, Disc_seg

def compute_single_avg_score_fluid(ret_seg):
    NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg, RPE_seg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not math.isnan(ret_seg[1]):
        NFL_seg = ret_seg[1]
    if not math.isnan(ret_seg[2]):
        GCL_seg = ret_seg[2]
    if not math.isnan(ret_seg[3]):
        IPL_seg = ret_seg[3]
    if not math.isnan(ret_seg[4]):
        INL_seg = ret_seg[4]
    if not math.isnan(ret_seg[5]):
        OPL_seg = ret_seg[5]
    if not math.isnan(ret_seg[6]):
        ONL_seg = ret_seg[6]
    if not math.isnan(ret_seg[7]):
        IS_OS_seg = ret_seg[7]
    if not math.isnan(ret_seg[8]):
        RPE_seg = ret_seg[8]  
              
    if not math.isnan(ret_seg[8]):
        # print('有积液')
        avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg + RPE_seg) / 8
    else:
        # print('无积液')
        avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg) / 8
        
    return avg_seg

def compute_single_avg_score(ret_seg):
    NFL_seg, GCL_seg, IPL_seg, INL_seg, OPL_seg, ONL_seg, IS_OS_seg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not math.isnan(ret_seg[1]):
        NFL_seg = ret_seg[1]
    if not math.isnan(ret_seg[2]):
        GCL_seg = ret_seg[2]
    if not math.isnan(ret_seg[3]):
        IPL_seg = ret_seg[3]
    if not math.isnan(ret_seg[4]):
        INL_seg = ret_seg[4]
    if not math.isnan(ret_seg[5]):
        OPL_seg = ret_seg[5]
    if not math.isnan(ret_seg[6]):
        ONL_seg = ret_seg[6]
    if not math.isnan(ret_seg[7]):
        IS_OS_seg = ret_seg[7]
        
    avg_seg = (NFL_seg + GCL_seg + IPL_seg + INL_seg + OPL_seg + ONL_seg + IS_OS_seg) / 7
    return avg_seg

def per_class_dice(y_pred, y_true, num_class):
    avg_dice = 0
    y_pred = y_pred.data.squeeze() #.cpu().numpy()
    y_true = y_true.data.squeeze() #.cpu().numpy()
    print('y_pred:', y_pred.shape)
    print('y_true:', y_true.shape)
    # dice_all = np.zeros(num_class)
    dice_all = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    for i in range(num_class):
        GT = y_true[:,:,i].view(-1)
        Pred = y_pred[:,:,i].view(-1)
        #print(GT.shape, Pred.shape)
        inter = (GT * Pred).sum() + 0.0001
        union = GT.sum()  + Pred.sum()  + 0.0001
        t = 2 * inter / union
        avg_dice = avg_dice + (t / num_class)
        dice_all[i] = t
    return avg_dice, dice_all

# def mae(y_pred,y_true):
#     y_pred,y_true=y_pred.cpu().detach().numpy(),y_true.cpu().detach().numpy()
#     y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
#     y_true=rearrange(y_true,'b c h w ->(b c) h w')
#     # print(y_pred.shape,y_true.shape)
#     mae=[0.0 for i in range(y_pred.shape[1])]
#     for i in range(y_pred.shape[0]):
#         for j in range(y_pred.shape[1]):
#             mae[j]+=mean_absolute_error(y_pred[i,j],y_true[i,j])
#     return [x/y_pred.shape[0] for x in mae]

# def rmse(y_pred,y_true):
#     y_pred,y_true=y_pred.cpu().detach().numpy(),y_true.cpu().detach().numpy()
#     y_pred=rearrange(y_pred,'b c h w ->(b c) h w')
#     rmse=[0.0 for i in range(y_pred.shape[1])]
#     for i in range(y_pred.shape[0]):
#         for j in range(y_pred.shape[1]):
#             rmse[j]+=np.sqrt(mean_squared_error(y_pred[i,j],y_true[i,j]))
#     return [x/y_pred.shape[0] for x in rmse]
