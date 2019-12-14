"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import torch
import torch.nn as nn

from .CustomLayers import Conv2DMod


class ConvModLayer(nn.Module):
    def __init__(self):
        super(ConvModLayer, self).__init__()
        self.mod_weight = None
        self.mod_bias = None
        self.weight = None
        self.bias = None
        self.noise_strength = None

    def forward(self, x):
        return x


class ToRGB(nn.Module):
    def __init__(self):
        super(ToRGB, self).__init__()

    def forward(self, x, y=None):
        t = x
        return t if y is None else y + t


class InputBlock(nn.Module):
    def __init__(self):
        super(InputBlock, self).__init__()
        self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
        self.conv = ConvModLayer(layer_idx=, fmaps=, kernel=)
        self.to_rgb = ToRGB()

    def forward(self, dlatents_in):
        x = self.const.expand(batch_size, -1, -1, -1)
        x = self.conv(x, dlatents_in)

        return x


class GSynthesisBlock(nn.Module):
    """
    Building blocks for main layers
    """

    def __init__(self, res):
        super(GSynthesisBlock, self).__init__()
        self.conv0_up = ConvModLayer(layer_idx=, fmaps=, kernel=, up=True)
        self.conv1 = ConvModLayer(layer_idx=, fmaps=, kernel=)
        self.to_rgb = ToRGB()

    def forward(self, x, dlatents_in, y):
        x = self.conv0_up(x, dlatents_in)
        x = self.conv1(x, dlatents_in)
        y = self.to_rgb(x, y)
        return x, y
