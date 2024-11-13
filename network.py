import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Convolutional_Block, DeConvolutional_Block


class Modele(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 32):
        super(Modele, self).__init__()
        """
    # ENCODEUR
        # BLOC 1
        self.c1 = Conv(6,16)
        self.c2 = Conv(16,16)

        self.c3 = Conv(16,32)
        self.c4 = Conv(32,32)

        self.c5 = Conv(32,64)
        self.c6 = Conv(64,64)
        self.c7 = Conv(64,64)

        self.c8 = Conv(64,128)
        self.c9 = Conv(128,128)
        self.c10 = Conv(128,128)

        self.c11 = DeConv(128,128)
        self.c12 = DeConv(128,128)
        self.c13 = DeConv(128,64)

        self.c14 = DeConv(64,64)
        self.c16 = DeConv(64,32)

        self.c17 = DeConv(32,32)
        self.c19 = DeConv(32,16)

        self.c20 = DeConv(16,16)
        self.c21 = DeConv(16,2)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        """

        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #output_size # pour le downsampling
        self.conv1 = Convolutional_Block(in_channels, out_channels)
        self.conv2 = Convolutional_Block(out_channels, out_channels * 2)
        self.conv3 = Convolutional_Block(out_channels*2 , out_channels*4)
        self.conv4 = Convolutional_Block(out_channels*4 , out_channels*8)
    
    # DECODEUR
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2) # si pas bon result changer le upsampling
        self.deconv1 = DeConvolutional_Block(out_channels*16 , out_channels *8)
        self.deconv2 = DeConvolutional_Block(out_channels *8 , out_channels *4)
        self.deconv3 = DeConvolutional_Block(out_channels *4 , out_channels*2)
        self.deconv4 = DeConvolutional_Block(out_channels*2 , out_channels)
        self.deconv5 = DeConvolutional_Block(out_channels , 1)

        self.encodeur = nn.Sequential(
            self.conv1,
            self.max_pool,
            self.conv2,
            self.max_pool,
            self.conv3,
            self.max_pool,
            self.conv4,
            self.max_pool
        )

        self.decodeur = nn.Sequential(
            self.deconv1,
            self.upsampling,
            self.deconv2,
            self.upsampling,
            self.deconv3,
            self.upsampling,
            self.deconv4,
            self.upsampling,
            self.deconv5
        )
    # SIGMOID
        #self.sigmoid = nn.Sigmoid()
        
    

    def forward(self, image1, image2, with_attn=False):
        
        # Encode
        encoded_im1 = self.encodeur(image1)
        encoded_im2 = self.encodeur(image2)


        # Combine encoded images
        encode_stack = torch.concat((encoded_im1, encoded_im2), dim=1)  # Concat on channels

        # Decode
        decode = self.decodeur(encode_stack)
        return decode
        #return self.sigmoid(decode)

        """
        x = torch.cat((image1, image2), 1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)
        x = self.c13(x)
        x = self.c14(x)
        x = self.c16(x)
        x = self.c17(x)
        x = self.c19(x)
        x = self.c20(x)
        x = self.c21(x)
        x = self.logsoftmax(x)
        return x
        """