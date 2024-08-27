######## Syetem ########
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from functools import partial
from torch import Tensor
import torchvision
import numpy as np
import cv2
import My_Utils.sub_module as sm
from My_Utils.sub_module import *

def net_builder(name, pretrained_model=None, pretrained=False, ratio_in=0.25):
    if name == 'ReLayNet':
        net = ReLayNet()
    elif name == 'UNet':
        net = UNet(1, 9)
    elif name == 'GDNet':
        net = GDNet(in_channels=1, out_channels=9, ratio_in=ratio_in, ffc=True, skip_ffc=False)
    else:
        raise NameError("Unknow Model Name!")
    return net

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)        
        x4 = self.down3(x3)        
        x5 = self.down4(x4)        
        x = self.up1(x5, x4)        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class ReLayNet(nn.Module):
    def __init__(self):
        super(ReLayNet, self).__init__()
        params = {
                'num_channels': 1,
                'num_filters': 64,
                'kernel_h': 7,
                'kernel_w': 3,
                'kernel_c': 1,
                'stride_conv': 1,
                'pool': 2,
                'stride_pool': 2,
                'num_class': 9
            }
        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        # params['num_channels'] = 64  # This can be used to change the numchannels for each block
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        bn = self.bottleneck.forward(e3)

        d3 = self.decode1.forward(bn, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)

        return prob
         
class GDNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=11, init_features=32, ratio_in=0.25, ffc=True, skip_ffc=False,
                 cat_merge=True):
        super(GDNet, self).__init__()
        self.ffc = ffc
        self.skip_ffc = skip_ffc
        self.ratio_in = ratio_in
        self.cat_merge = cat_merge
        features = init_features
        self.mgb = MGR_Module(features * 8, features * 8)
                
        self.encoder1 = MyBasicBlock(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = MyBasicBlock(features, features * 2)  # was 1,2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = MyBasicBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = MyBasicBlock(features * 4, features * 4)  # was 8
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        if ffc:
            self.encoder1_f = FFC_BN_ACT(in_channels, features, kernel_size=1, ratio_gin=0, ratio_gout=ratio_in)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = FFC_BN_ACT(features, features * 2, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = FFC_BN_ACT(features * 2, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = FFC_BN_ACT(features * 4, features * 4, kernel_size=1, ratio_gin=ratio_in,
                                         ratio_gout=ratio_in)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)
        else:            
            self.encoder1_f = MyBasicBlock(in_channels, features)
            self.pool1_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder2_f = MyBasicBlock(features, features * 2)  # was 1,2
            self.pool2_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder3_f = MyBasicBlock(features * 2, features * 4)  #
            self.pool3_f = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder4_f = MyBasicBlock(features * 4, features * 4)  # was 8
            self.pool4_f = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = MyBasicBlock(features * 8, features * 16)  # 8, 16

        if skip_ffc and not ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = MyBasicBlock((features * 8) * 2, features * 8)  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = MyBasicBlock((features * 6) * 2, features * 4)
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = MyBasicBlock((features * 3) * 2, features * 2)
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = MyBasicBlock(features * 3, features)  # 2,3
        elif skip_ffc and ffc:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = MyBasicBlock((features * 8) * 2, features * 8)  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = MyBasicBlock((features * 6) * 2, features * 4)
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = MyBasicBlock((features * 3) * 2, features * 2)
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = MyBasicBlock(features * 3, features)  # 2,3
        else:
            self.upconv4 = nn.ConvTranspose2d(
                features * 16, features * 8, kernel_size=2, stride=2  # 16
            )
            self.decoder4 = MyBasicBlock((features * 6) * 2, features * 8)  # 8, 12
            self.upconv3 = nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=2, stride=2
            )
            self.decoder3 = MyBasicBlock((features * 4) * 2, features * 4)
            self.upconv2 = nn.ConvTranspose2d(
                features * 4, features * 2, kernel_size=2, stride=2
            )
            self.decoder2 = MyBasicBlock((features * 2) * 2, features * 2)
            self.upconv1 = nn.ConvTranspose2d(
                features * 2, features, kernel_size=2, stride=2
            )
            self.decoder1 = MyBasicBlock(features * 2, features)  # 2,3

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self.softmax = nn.Softmax2d()
        self.catLayer = ConcatTupleLayer()

    def forward(self, x):        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc4_2 = self.pool4(enc4)        
        
        if self.ffc:
            enc1_f = self.encoder1_f(x) # x:([4, 1, 496, 496])
            enc1_l, enc1_g = enc1_f # enc1_l:([4, 24, 496, 496]) enc1_g:([4, 8, 496, 496])
            if self.ratio_in == 0:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), enc1_g))
            elif self.ratio_in == 1:
                enc2_f = self.encoder2_f((enc1_l, self.pool1_f(enc1_g)))
            else:
                enc2_f = self.encoder2_f((self.pool1_f(enc1_l), self.pool1_f(enc1_g)))

            enc2_l, enc2_g = enc2_f # enc2_l:([4, 48, 248, 248]) enc2_g:([4, 16, 248, 248])
            if self.ratio_in == 0:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), enc2_g))
            elif self.ratio_in == 1:
                enc3_f = self.encoder3_f((enc2_l, self.pool2_f(enc2_g)))
            else:
                enc3_f = self.encoder3_f((self.pool2_f(enc2_l), self.pool2_f(enc2_g)))

            enc3_l, enc3_g = enc3_f
            if self.ratio_in == 0: # enc3_l:([4, 96, 124, 124]) enc3_g:([4, 32, 124, 124])
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), enc3_g))
            elif self.ratio_in == 1:
                enc4_f = self.encoder4_f((enc3_l, self.pool3_f(enc3_g)))
            else:
                enc4_f = self.encoder4_f((self.pool3_f(enc3_l), self.pool3_f(enc3_g)))

            enc4_l, enc4_g = enc4_f
            if self.ratio_in == 0:
                enc4_f2 = self.pool1_f(enc4_l)
            elif self.ratio_in == 1:
                enc4_f2 = self.pool1_f(enc4_g)
            else:
                enc4_f2 = self.catLayer((self.pool4_f(enc4_l), self.pool4_f(enc4_g)))
                # print('enc4_f2:', enc4_f2.shape)
        else:
            enc1_f = self.encoder1_f(x)
            enc2_f = self.encoder2_f(self.pool1_f(enc1_f))
            enc3_f = self.encoder3_f(self.pool2_f(enc2_f))
            enc4_f = self.encoder4_f(self.pool3_f(enc3_f))
            enc4_f2 = self.pool4(enc4_f)

        if self.cat_merge:
            a = torch.zeros_like(enc4_2)
            b = torch.zeros_like(enc4_f2)

            enc4_2 = enc4_2.view(torch.numel(enc4_2), 1)
            enc4_f2 = enc4_f2.view(torch.numel(enc4_f2), 1)

            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
            bottleneck = bottleneck.view_as(torch.cat((a, b), 1))
        else:
            bottleneck = torch.cat((enc4_2, enc4_f2), 1)
                    
        bottleneck = self.mgb(bottleneck)        
        
        bottleneck = self.bottleneck(bottleneck)
        dec4 = self.upconv4(bottleneck)
        if self.ffc and self.skip_ffc:
            enc4_in = torch.cat((enc4, self.catLayer((enc4_f[0], enc4_f[1]))), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, self.catLayer((enc3_f[0], enc3_f[1]))), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, self.catLayer((enc2_f[0], enc2_f[1]))), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, self.catLayer((enc1_f[0], enc1_f[1]))), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)
        elif self.skip_ffc:
            enc4_in = torch.cat((enc4, enc4_f), dim=1)

            dec4 = torch.cat((dec4, enc4_in), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)

            enc3_in = torch.cat((enc3, enc3_f), dim=1)
            dec3 = torch.cat((dec3, enc3_in), dim=1)
            dec3 = self.decoder3(dec3)

            dec2 = self.upconv2(dec3)
            enc2_in = torch.cat((enc2, enc2_f), dim=1)
            dec2 = torch.cat((dec2, enc2_in), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            enc1_in = torch.cat((enc1, enc1_f), dim=1)
            dec1 = torch.cat((dec1, enc1_in), dim=1)
        else:
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)

        dec1 = self.decoder1(dec1)

        return self.softmax(self.conv(dec1))


if __name__ == '__main__':  
    x = torch.randn(4, 1, 256, 256)    
    net = net_builder('ReLayNet')
    print(net(x).shape)
    print("Num params: ", sum(p.numel() for p in net.parameters()))
