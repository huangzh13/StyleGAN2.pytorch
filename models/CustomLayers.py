"""
-------------------------------------------------
   File Name:    CustomLayers.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import torch.nn as nn


class Conv2DMod(nn.Module):
    def __init__(self, fmaps, kernel, up=False, down=False, demodulate=True, resample_kernel=None, gain=1,
                 use_wscale=True, lrmul=1, fused_modconv=True, weight_var='weight', mod_weight_var='mod_weight',
                 mod_bias_var='mod_bias'):
        super(Conv2DMod, self).__init__()

    def forward(self, x):
        # Get weight

        # Modulate

        # Demodulate

        # Reshape/scale input

        # Convolution with optional up/down sampling

        # Reshape/scale output

        return x
