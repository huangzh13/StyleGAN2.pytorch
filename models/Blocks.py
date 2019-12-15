"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn

from .CustomLayers import EqualizedModConv2d, upscale2d


def apply_bias_act(x, b, act='lrelu', gain=None, lrmul=1):
    act_layer, def_gain = {'relu': (torch.relu, np.sqrt(2)),
                           'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[act]

    b = b * lrmul
    # Add bias
    x += b.view(1, -1, 1, 1)
    # Evaluate activation function.
    x = act_layer(x)
    # Scale by gain.
    if gain is None:
        gain = def_gain
    if gain != 1:
        x *= gain

    return x


class ModConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, act, up=False):
        super(ModConvLayer, self).__init__()
        self.act = act

        self.modulated_conv2d_layer = EqualizedModConv2d(input_channels=input_channels,
                                                         output_channels=output_channels,
                                                         kernel=kernel, up=up)
        self.noise_strength = nn.Parameter(torch.zeros(output_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_channels), requires_grad=True)

    def forward(self, x, dlatents_in_range, noise_input=None):
        x = self.modulated_conv2d_layer(x, dlatents_in_range)
        if noise_input is None:
            noise_input = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        x += self.noise_strength.view(1, -1, 1, 1) * noise_input
        return apply_bias_act(x, self.bias, act=self.act)


class InputBlock(nn.Module):
    def __init__(self, number_channels, input_fmaps, output_fmaps, act, fused_modconv=True):
        super(InputBlock, self).__init__()
        self.const = nn.Parameter(torch.randn(1, input_fmaps, 4, 4), requires_grad=True)
        self.conv = ModConvLayer(input_channels=input_fmaps,
                                 output_channels=output_fmaps,
                                 kernel=3, act=act)
        self.to_rgb = EqualizedModConv2d(input_channels=output_fmaps,
                                         output_channels=number_channels, kernel=1,
                                         demodulate=False, fused_modconv=fused_modconv)

    def forward(self, dlatents_in):
        x = self.const.expand(dlatents_in.size(0), -1, -1, -1)
        x = self.conv(x, dlatents_in[:, 0])
        y = self.to_rgb(x, dlatents_in[:, 1])

        return x, y


class GSynthesisBlock(nn.Module):
    """
    Building blocks for main layers
    """

    def __init__(self, num_channels, res, input_fmaps, output_fmaps, act, fused_modconv=True):
        super(GSynthesisBlock, self).__init__()

        self.res = res

        self.conv0_up = ModConvLayer(input_channels=input_fmaps,
                                     output_channels=output_fmaps,
                                     kernel=3, act=act, up=True)
        self.conv1 = ModConvLayer(input_channels=input_fmaps,
                                  output_channels=output_fmaps,
                                  kernel=3, act=act)
        self.to_rgb = EqualizedModConv2d(input_channels=output_fmaps,
                                         output_channels=num_channels, kernel=1,
                                         demodulate=False, fused_modconv=fused_modconv)

    def forward(self, x, dlatents_in, y):
        x = self.conv0_up(x, dlatents_in[:, self.res * 2 - 5])
        x = self.conv1(x, dlatents_in[:, self.res * 2 - 4])

        y = upscale2d(y)
        y = self.to_rgb(x, dlatents_in[:, self.res * 2 - 3]) + y

        return x, y
