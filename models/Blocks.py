"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import torch.nn as nn


class Layer(nn.Module):
    """
    Single convolution layer with all the bells and whistles.
    """

    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, x):
        return x


class InputBlock(nn.Module):
    def __init__(self):
        super(InputBlock, self).__init__()

    def forward(self, x):
        pass


class GSynthesisBlock(nn.Module):
    """
    Building blocks for main layers
    """

    def __init__(self):
        super(GSynthesisBlock, self).__init__()

    def forward(self, x, dlatents_in_range):
        return x
