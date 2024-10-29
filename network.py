import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Convolutional_Block, Self_Attn, Conv, DeConv

class Modele(nn.Module):
    def __init__(self, input_shape = (256, 256), in_channels = 3, out_channels = 32):
        super(Modele, self).__init__()
    
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


        self.convolutional_block1 = Convolutional_Block(in_channels, out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((128,128)) #output_size # pour le downsampling
        
        # BLOC 2
        self.convolutional_block2 = Convolutional_Block(out_channels, out_channels * 2)
        self.avg_pool2 = nn.AdaptiveAvgPool2d((64,64))
        
        # BLOC 3
        self.convolutional_block3 = Convolutional_Block(out_channels*2 , out_channels*4)
        self.avg_pool3 = nn.AdaptiveAvgPool2d((32,32))

        # BLOC 4
        self.convolutional_block4 = Convolutional_Block(out_channels*4 , out_channels*8)
        self.avg_pool4 = nn.AdaptiveAvgPool2d((16,16))
        
    # ATTENTION BLOCK
        self.attention_block = Self_Attn(in_dim = out_channels * 16, with_attn=True)
                
    # DECODEUR
        
        self.convolutional_block_decodeur1 = Convolutional_Block(out_channels*16 , out_channels *8)
        self.convolutional_block_decodeur2 = Convolutional_Block(out_channels *8 , out_channels *4)
        
        self.convolutional_block_decodeur3 = Convolutional_Block(out_channels *4 , out_channels*2)
        
        self.convolutional_block_decodeur4 = Convolutional_Block(out_channels*2 , out_channels)
        
        self.convolutional_block_decodeur5 = Convolutional_Block(out_channels , 1)

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2) # si pas bon result changer le upsampling
        #upsampling pour augmenter taille puis pour les channels: (conv ou dedans on divise par 2 le nombre de channels) 
        
    # SIGMOID
        self.sigmoid = nn.Sigmoid()
        
        
    

    def forward(self, image1, image2):
        """
        # Encode image 1
        conv_block_1_im1 = self.convolutional_block1(image1)
        avg_pool_1_im1 = self.avg_pool(conv_block_1_im1)
        conv_block_2_im1 = self.convolutional_block2(avg_pool_1_im1)
        avg_pool_2_im1 = self.avg_pool2(conv_block_2_im1)
        
        conv_block_3_im1 = self.convolutional_block3(avg_pool_2_im1)
        avg_pool_3_im1 = self.avg_pool3(conv_block_3_im1)
        
        conv_block_4_im1 = self.convolutional_block4(avg_pool_3_im1)
        avg_pool_4_im1 = self.avg_pool4(conv_block_4_im1)
        
        # Encode image 2
        conv_block_1_im2 = self.convolutional_block1(image2)
        avg_pool_1_im2 = self.avg_pool(conv_block_1_im2)
        
        conv_block_2_im2 = self.convolutional_block2(avg_pool_1_im2)
        avg_pool_2_im2 = self.avg_pool2(conv_block_2_im2)
        
        conv_block_3_im2 = self.convolutional_block3(avg_pool_2_im2)
        avg_pool_3_im2 = self.avg_pool3(conv_block_3_im2)
        
        conv_block_4_im2 = self.convolutional_block4(avg_pool_3_im2)
        avg_pool_4_im2 = self.avg_pool4(conv_block_4_im2)

        
        output_encoder_combined = torch.concat((avg_pool_4_im1, avg_pool_4_im2), dim=1) #Concat on channels
        
        # Attention : image 1 + image 2
        attention, mask = self.attention_block(output_encoder_combined) # MAYBE WE NEED TO CONCATENATE RATHEN THAN JUST ADDED THE TWO TERMS
        #C'est pas un maybe, il faut le faire comme ca lol


        # DECODEUR
        
        conv_block_1_dec = self.convolutional_block_decodeur1(attention)
        conv_block_2_dec = self.convolutional_block_decodeur2(conv_block_1_dec)
        
        upsampling2 =self.upsampling(conv_block_2_dec)
        conv_block_3_dec=self.convolutional_block_decodeur3(upsampling2)
        
        upsampling3 =self.upsampling(conv_block_3_dec)
        conv_block_4_dec=self.convolutional_block_decodeur4(upsampling3)
        
        upsampling4 =self.upsampling(conv_block_4_dec)
        conv_block_5_dec=self.convolutional_block_decodeur5(upsampling4)
        
        output_cnn = self.sigmoid(conv_block_5_dec)

        return output_cnn, mask"""


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