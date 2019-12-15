"""
-------------------------------------------------
   File Name:    CustomLayers.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    assert isinstance(factor, int) and factor >= 1

    if gain != 1:
        x = x * gain

    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])

    return x


def downscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x = x * gain

    # No-op => early exit.
    if factor == 1:
        return x

    return F.avg_pool2d(x, factor)


def upsample_conv_2d(x, w):
    return x


def conv_downsample_2d(x, w):
    return x


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=1., use_wscale=True, lrmul=1.):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init

        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std, requires_grad=True)

    def forward(self, x):
        return F.linear(x, self.weight * self.w_mul)


class EqualizedModConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, up=False, down=False, demodulate=True,
                 gain=1, use_wscale=True, lrmul=1, fused_modconv=True, resample_kernel=None):
        """
        """
        super(EqualizedModConv2d, self).__init__()

        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1

        self.gain = gain
        self.up = up
        self.down = down
        self.fused_modconv = fused_modconv
        self.demodulate = demodulate

        self.dense = EqualizedLinear()
        self.bias_act = None

        he_std = self.gain * (input_channels * kernel ** 2) ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel, kernel) * init_std, requires_grad=True)

    def forward(self, x, y):
        # Get weight.
        ww = None  # [BkkIO] Introduce minibatch dimension.

        # Modulate
        s = self.dense(y)  # [BI] Transform incoming W to style.
        s = self.bias_act(s)  # [BI] Add bias (initially 1).
        ww *= s  # [BkkIO] Scale input feature maps.

        # Demodulate
        if self.demodulate:
            d = None  # [BO] Scaling factor.
            ww *= d  # [BkkIO] Scale output feature maps.

        # Reshape/scale input
        if self.fused_modconv:
            x = x  # Fused => reshape minibatch to convolution groups.
            w = ww
        else:
            x *= s  # [BIhw] Not fused => scale input activations.

        # Convolution with optional up/down sampling
        if self.up:
            x = upsample_conv_2d(x, self.weight)
        elif self.down:
            x = conv_downsample_2d(x, self.weight)
        else:
            x = F.conv2d(x, self.weight)

        # Reshape/scale output
        if self.fused_modconv:
            x = x  # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x *= d  # [BOhw] Not fused => scale output activations.

        return x
