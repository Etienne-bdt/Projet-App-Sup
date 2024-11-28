import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class ChangeDetectUnet(nn.Module):
    def __init__(self, in_chan=9, out_chan=32):
        
        super(ChangeDetectUnet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_chan, out_chan) # 9,32
        self.enc2 = ConvBlock(out_chan, out_chan * 2) # 32,64
        self.enc3 = ConvBlock(out_chan * 2, out_chan * 4) # 64, 128
        self.enc4 = ConvBlock(out_chan * 4, out_chan * 8) # 128, 256

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(out_chan * 8, out_chan * 16) # 128,254

        # Decoder
        self.up4 = nn.ConvTranspose2d(out_chan * 16, out_chan * 8, kernel_size=2, stride=2) # 254,128
        self.dec4 = ConvBlock(out_chan * 16, out_chan * 8)
        self.up3 = nn.ConvTranspose2d(out_chan * 8, out_chan * 4, kernel_size=2, stride=2) # 254,128
        self.dec3 = ConvBlock(out_chan * 8, out_chan * 4)
        self.up2 = nn.ConvTranspose2d(out_chan * 4, out_chan * 2, kernel_size=2, stride=2) # 128,64
        self.dec2 = ConvBlock(out_chan * 4, out_chan * 2)
        self.up1 = nn.ConvTranspose2d(out_chan * 2, out_chan, kernel_size=2, stride=2) # 64,32
        self.dec1 = ConvBlock(out_chan * 2, out_chan)

        self.final = nn.Conv2d(out_chan, 1, kernel_size=1)

    def forward(self, img1, img2):
        
        # différence absolue 
        diff = img1-img2
        diff_norm = torch.clamp((torch.abs(diff)-torch.mean(diff)),0,1)
        # on ajoute cette différence aux canaux d'entrée
        x = torch.cat((img1, img2, diff_norm), dim=1)  # (batch, 9, 128, 128)

        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 =self.pool(enc2)
        
        enc3 = self.enc3(pool2)

        # Bottleneck
        pool3 = self.pool(enc3)

        enc4 = self.enc4(pool3)
        pool4 = self.pool(enc4)

        bottleneck = self.bottleneck(pool4)

        # Decoder
        concat4 = torch.cat((self.up4(bottleneck), enc4), dim=1)
        dec4 = self.dec4(concat4)

        concat3 = torch.cat((self.up3(dec4), enc3), dim=1)
        dec3 = self.dec3(concat3)
        
        concat2=torch.cat((self.up2(dec3), enc2), dim=1)
        dec2 = self.dec2(concat2)
        
        concat1 = torch.cat((self.up1(dec2), enc1), dim=1)
        dec1 = self.dec1(concat1)

        out = torch.sigmoid(self.final(dec1))  # Sigmoid donne map de proba
        return out
