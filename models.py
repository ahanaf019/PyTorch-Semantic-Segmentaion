import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes, ):
        super().__init__()
    
    
    def foward(self, x):
        pass
    
    
    def conv_bn_relu(self, in_channels, out_channels, kernel_size, ):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )