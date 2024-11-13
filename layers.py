import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolutional_Block, self).__init__()
        
        #on ne fait pas de rétro action pour l' instant!
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size = (3,3), padding = "same")
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size = (3,3), padding = "same")
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        
        conv_1 = self.conv1(x)
        batch_norm_1 = self.batch_norm1(conv_1)
        relu_1 = self.relu1(batch_norm_1)
        
        conv_2 = self.conv2(relu_1)
        batch_norm_2 = self.batch_norm2(conv_2)

        output = self.relu2(batch_norm_2)

        return output
    
class DeConvolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvolutional_Block, self).__init__()
        

        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size = (3,3), padding = "same")
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size = (3,3), padding = "same")
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        
        conv_1 = self.conv1(x)
        batch_norm_1 = self.batch_norm1(conv_1)
        relu_1 = self.relu1(batch_norm_1)
        
        conv_2 = self.conv2(relu_1)
        batch_norm_2 = self.batch_norm2(conv_2)

        output = self.relu2(batch_norm_2)

        return output

class Conv(nn.Module):
    def __init__(self, in_cn, out_cn):
        super().__init__()
        self.conv = nn.Conv2d(in_cn, out_cn, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_cn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class DeConv(nn.Module):
    def __init__(self, in_cn, out_cn):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_cn, out_cn, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_cn)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
