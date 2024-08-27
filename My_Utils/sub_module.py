# List of APIs
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from collections import OrderedDict

class MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)

####↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ ReLayNet ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓#
class BasicBlock(nn.Module):
    def __init__(self, params):
        super(BasicBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(params['kernel_h'], params['kernel_w']),padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
        self.prelu = nn.PReLU()

    def forward(self, input):
        out_conv = self.conv(input)
        out_bn = self.batchnorm(out_conv)
        out_prelu = self.prelu(out_bn)
        return out_prelu

class EncoderBlock(BasicBlock):
    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input):
        out_block = super(EncoderBlock, self).forward(input)
        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices

class DecoderBlock(BasicBlock):
    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block, indices):
        unpool = self.unpool(input, indices)
        concat = torch.cat((out_block, unpool), dim=1)
        out_block = super(DecoderBlock, self).forward(concat)

        return out_block

class ClassifierBlock(nn.Module):
    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])
        self.softmax = nn.Softmax2d()

    def forward(self, input):
        out_conv = self.conv(input)
        #out_logit = self.softmax(out_conv)
        return out_conv

#↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ ReLayNet ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑#

class DoubleConv(nn.Module): 
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):   
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
        # return self.layer(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    #上采样，输出为输入的2倍

            # self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #                         nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
            #                         nn.ReLU(inplace=True))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)


        self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)      
        return self.conv(x)

class OutConv(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def img2df(img, mask):
        img[mask == 0] = 0
        img[mask == 2] = 0
        return img

def feature_fusion(out1, out2):
        output2 = F.log_softmax(out2, dim=1)
        out1_bg = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
        out1_disc = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
        out2_layer = torch.zeros([out2.shape[0], 9, out2.shape[2], out2.shape[3]]).cuda()
        out1_bg[:, 0, :, :] = out1[:, 0, :, :]
        out1_disc[:, 0, :, :] = out1[:, 2, :, :]
        out2_layer[:, :, :, :] = out2[:, 1:, :, :]
        out = torch.cat([out1_bg, out2_layer, out1_disc], 1)
        return output2, out

class MyBasicBlock(nn.Module):    
    def __init__(self, in_channels, features):
        super().__init__()
        self.myBasicBlock = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True)
        )        
            
    def forward(self, x):
        outMyBasicBlock = self.myBasicBlock(x)
        return outMyBasicBlock

class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)        
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            if g2l_gate == 0:
                out_xl = self.convl2l(x_l) 
            else:
                out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        #if use_se:
        #    if se_kwargs is None:
        #        se_kwargs = {}
        #    self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        clamp = False
        remove = False
        if clamp:
            ffted = torch.clamp(ffted, min=-10, max=10)
        if remove:
            fftedmin10 = torch.clamp(ffted, min=10)
            fftedmax10 = torch.clamp(ffted, max=-10)
            ffted = torch.where(ffted > 0, fftedmax10, fftedmin10)

        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        #if self.use_se:
        #    ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class GloRe_Unit(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = Basconv(num_in, self.num_s, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # generate projection and inverse projection functions
        self.conv_proj = Basconv(num_in, self.num_n, is_batchnorm = True, kernel_size=kernel_size, padding=padding)   
        self.conv_reproj = Basconv(num_in, self.num_n, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)   
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)  
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1), 
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in) 

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) 
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)  
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out

class Basconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, is_batchnorm = False, kernel_size = 3, stride = 1, padding=1):
        super(Basconv, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        return x
