"""
-------------------------------------------------
   File Name:    CustomLayers.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn


def apply_bias_act(x, act_funcs='lrelu', lrmul=1, bias_var='bias'):
    act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                 'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[act_funcs]

    # Add bias
    bias = nn.Parameter(torch.zeros(x.shape[1]), requires_grad=True)
    x = x + bias

    # Evaluate activation function
    x = act(x)

    # scale by gain
    if gain != 1:
        x *= gain

    return x


class Conv2DMod(nn.Module):
    def __init__(self, layer_idx, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1,
                 use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight',
                 mod_bias_var='mod_bias'):
        """
        """
        super(Conv2DMod, self).__init__()
        self.mod_weight = None
        self.mod_bias = None
        self.weight = None
        self.bias = None
        self.noise_strength = None

    def forward(self, x, y):
        # Modulate

        # Demodulate

        # Reshape/scale input

        # Convolution with optional up/down sampling

        # Reshape/scale output

        return x
