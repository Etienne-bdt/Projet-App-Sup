import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


####################################################################################################
##                                            Test 1                                              ##
####################################################################################################

# Bloc convolutionnel de base
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Encodeur partagé
class SharedEncoder(nn.Module):
    def __init__(self, in_channels, base_channels=32):
        super(SharedEncoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, base_channels)
        self.conv2 = ConvBlock(base_channels, base_channels * 2)
        self.conv3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(self.pool(enc1))
        enc3 = self.conv3(self.pool(enc2))
        return enc1, enc2, enc3

# Décodeur U-Net avec supervision
class Decoder(nn.Module):
    def __init__(self, base_channels=32):
        super(Decoder, self).__init__()
        # Convolutions pour ajuster les canaux
        self.reduce_channels_enc1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.reduce_channels_enc2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.reduce_channels_enc3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)

        # Décodeur
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(base_channels * 4, base_channels * 2)
        self.conv2 = ConvBlock(base_channels * 2, base_channels)
        self.final = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, enc1, enc2, enc3):
        # Réduire les canaux pour correspondre aux attentes
        enc1 = self.reduce_channels_enc1(enc1)
        enc2 = self.reduce_channels_enc2(enc2)
        enc3 = self.reduce_channels_enc3(enc3)

        # Étape de décodeur
        x = self.up1(enc3)  # Upsampling de enc3
        x = torch.cat((x, enc2), dim=1)  # Concaténation avec enc2
        x = self.conv1(x)  # Convolution

        x = self.up2(x)  # Upsampling de x
        x = torch.cat((x, enc1), dim=1)  # Concaténation avec enc1
        x = self.conv2(x)  # Convolution

        return self.final(x)


# Modèle principal
class DeeplySupervisedFusionNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(DeeplySupervisedFusionNet, self).__init__()
        self.encoder = SharedEncoder(in_channels, base_channels)
        self.decoder = Decoder(base_channels)

    def forward(self, img1, img2):
        # Encodeurs pour les deux images
        enc1_img1, enc2_img1, enc3_img1 = self.encoder(img1)
        enc1_img2, enc2_img2, enc3_img2 = self.encoder(img2)
        
        # print(f"enc1_img1:   {np.shape(enc1_img1)}")
        # print(f"enc2_img1:   {np.shape(enc2_img1)}")
        # print(f"enc3_img1:   {np.shape(enc3_img1)}")
        # print(f"enc1_img2:   {np.shape(enc1_img2)}")
        # print(f"enc2_img2:   {np.shape(enc2_img2)}")
        # print(f"enc3_img2:   {np.shape(enc3_img2)}")
        
        # Fusion des caractéristiques (concaténation canal)
        enc1_fused = torch.cat((enc1_img1, enc1_img2), dim=1)
        #print(f"enc1_fusedv:   {np.shape(enc1_fused)}")
        enc2_fused = torch.cat((enc2_img1, enc2_img2), dim=1)
        #print(f"enc2_fused:   {np.shape(enc2_fused)}")
        enc3_fused = torch.cat((enc3_img1, enc3_img2), dim=1)
        #print(f"enc3_fused:   {np.shape(enc3_fused)}")

        # Décodage et supervision profonde
        out = self.decoder(enc1_fused, enc2_fused, enc3_fused)
        return torch.sigmoid(out)  # Probabilités des changements
    
    
   
####################################################################################################
##                                            Test 2                                              ##
####################################################################################################

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2, self).__init__()
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
        self.enc1 = ConvBlock2(in_chan, out_chan) # 9,32
        self.enc2 = ConvBlock2(out_chan, out_chan * 2) # 32,64
        self.enc3 = ConvBlock2(out_chan * 2, out_chan * 4) # 64, 128

        # Pooling
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock2(out_chan * 4, out_chan * 8) # 128,254

        # Decoder
        self.up3 = nn.ConvTranspose2d(out_chan * 8, out_chan * 4, kernel_size=2, stride=2) # 254,128
        self.dec3 = ConvBlock2(out_chan * 8, out_chan * 4)
        self.up2 = nn.ConvTranspose2d(out_chan * 4, out_chan * 2, kernel_size=2, stride=2) # 128,64
        self.dec2 = ConvBlock2(out_chan * 4, out_chan * 2)
        self.up1 = nn.ConvTranspose2d(out_chan * 2, out_chan, kernel_size=2, stride=2) # 64,32
        self.dec1 = ConvBlock2(out_chan * 2, out_chan)

        self.final = nn.Conv2d(out_chan, 1, kernel_size=1)

    def forward(self, img1, img2):
        
        # différence absolue 
        diff = torch.abs(img1 - img2)  # (batch, 3, 128, 128)

        # on ajoute cette différence aux canaux d'entrée
        x = torch.cat((img1, img2, diff), dim=1)  # (batch, 9, 128, 128)

        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 =self.pool(enc2)
        
        enc3 = self.enc3(pool2)

        # Bottleneck
        pool3 = self.pool(enc3)
        bottleneck = self.bottleneck(pool3)

        # Decoder
        concat3 = torch.cat((self.up3(bottleneck), enc3), dim=1)
        dec3 = self.dec3(concat3)
        
        concat2=torch.cat((self.up2(dec3), enc2), dim=1)
        dec2 = self.dec2(concat2)
        
        concat1 = torch.cat((self.up1(dec2), enc1), dim=1)
        dec1 = self.dec1(concat1)

        out = torch.sigmoid(self.final(dec1))  # Sigmoid donne map de proba
        return out
