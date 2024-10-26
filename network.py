import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Convolutional_Block, Self_Attn

class Modele(nn.Module):
    def __init__(self, input_shape = (256, 256), in_channels = 3, out_channels = 32):
        super(Modele, self).__init__()
    
    # ENCODEUR
        # BLOC 1
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
        self.attention_block = Self_Attn(in_dim = out_channels * 16)
                
    # DECODEUR
        
        self.convolutional_block_decodeur1 = Convolutional_Block(out_channels*16 , out_channels *8)
        self.convolutional_block_decodeur2 = Convolutional_Block(out_channels *8 , out_channels *4)
        
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2) # si pas bon result changer le upsampling
        self.convolutional_block_decodeur3 = Convolutional_Block(out_channels *4 , out_channels*2)
        
        self.upsampling3 = nn.UpsamplingNearest2d(scale_factor=2) # si pas bon result changer le upsampling
        self.convolutional_block_decodeur4 = Convolutional_Block(out_channels*2 , out_channels)
        
        self.upsampling4 = nn.UpsamplingNearest2d(scale_factor=2) # si pas bon result changer le upsampling
        self.convolutional_block_decodeur5 = Convolutional_Block(out_channels , 1)

        self.upsampling5 = nn.UpsamplingNearest2d(scale_factor=2) # si pas bon result changer le upsampling
        #upsampling pour augmenter taille puis pour les channels: (conv ou dedans on divise par 2 le nombre de channels) 
        
    # SIGMOID
        self.sigmoid = nn.Hardsigmoid()
        

    def forward(self, image1, image2):
        
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
        attention = self.attention_block(output_encoder_combined) # MAYBE WE NEED TO CONCATENATE RATHEN THAN JUST ADDED THE TWO TERMS
        #C'est pas un maybe, il faut le faire comme ca lol


        # DECODEUR
        
        conv_block_1_dec = self.convolutional_block_decodeur1(attention)
        conv_block_2_dec = self.convolutional_block_decodeur2(conv_block_1_dec)
        
        upsampling2 =self.upsampling2(conv_block_2_dec)
        conv_block_3_dec=self.convolutional_block_decodeur3(upsampling2)
        
        upsampling3 =self.upsampling3(conv_block_3_dec)
        conv_block_4_dec=self.convolutional_block_decodeur4(upsampling3)
        
        upsampling4 =self.upsampling4(conv_block_4_dec)
        conv_block_5_dec=self.convolutional_block_decodeur5(upsampling4)
        
        output_final = self.sigmoid(conv_block_5_dec)

        return output_final