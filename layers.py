import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,with_attn=False):
        super(Self_Attn,self).__init__()
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convolutional_Block, self).__init__()
        
        #on ne fait pas de rétro action pour l' instant!
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size = 3, padding = 1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        ()
        
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size =3, padding = 1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self,x):
        
        conv_1 = self.conv1(x)
        batch_norm_1 = self.batch_norm1(conv_1)
        relu_1 = self.relu1(batch_norm_1)
        
        conv_2 = self.conv_2(relu_1)
        batch_norm_2 = self.batch_norm_2(conv_2)
        relu_2 = self.relu2(batch_norm_2)
        
        output = relu_2

        return output