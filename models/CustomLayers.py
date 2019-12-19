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
import torch.nn.functional as F


def apply_bias_act(x, b, act='lrelu', gain=None, lrmul=1):
    act_layer, def_gain = {'relu': (torch.relu, np.sqrt(2)),
                           'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[act]

    b = b * lrmul
    # Add bias
    if len(x.shape) == 4:
        x += b.view(1, -1, 1, 1)
    else:
        x += b.view(1, -1)
    # Evaluate activation function.
    x = act_layer(x)
    # Scale by gain.
    if gain is None:
        gain = def_gain
    if gain != 1:
        x *= gain

    return x


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


def upsample_conv_2d(x, w, groups):
    # x = upscale2d(x)
    # w = w.permute(1, 0, 2, 3)
    x = F.conv_transpose2d(x, w, stride=2, padding=w.shape[-1] // 2, output_padding=1, groups=groups)
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
    def __init__(self, dlatent_size, input_channels, output_channels, kernel,
                 up=False, down=False, demodulate=True, fused_modconv=True,
                 gain=1, use_wscale=True, lrmul=1, resample_kernel=None):
        """
        """
        super(EqualizedModConv2d, self).__init__()

        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1

        # self.gain = gain
        self.up = up
        self.down = down
        self.fused_modconv = fused_modconv
        self.demodulate = demodulate
        self.fmaps = output_channels

        self.mod_layer = EqualizedLinear(input_size=dlatent_size, output_size=input_channels)
        self.bias = nn.Parameter(torch.ones(input_channels), requires_grad=True)

        he_std = gain * (input_channels * kernel ** 2) ** (-0.5)  # He init
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
        w = self.weight
        ww = self.weight.repeat(x.shape[0], 1, 1, 1, 1)  # [BOIkk] Introduce minibatch dimension.

        # Modulate
        s = self.mod_layer(y)  # [BI] Transform incoming W to style.
        s = apply_bias_act(s, b=self.bias)  # [BI] Add bias (initially 1).
        ww *= s.view(-1, 1, s.shape[1], 1, 1)  # [BOIkk] Scale input feature maps.

        # Demodulate
        if self.demodulate:
            d = torch.rsqrt(torch.sum(ww ** 2, dim=[2, 3, 4]) + 1e-8)  # [BO] Scaling factor.
            ww *= d.view(-1, d.shape[1], 1, 1, 1)  # [BOIkk] Scale output feature maps.

        # Reshape/scale input
        groups = x.shape[0]
        if self.fused_modconv:
            x = x.view(1, -1, x.shape[2], x.shape[3])  # Fused => reshape minibatch to convolution groups.
            size = ww.shape
            if self.up:
                w = ww.transpose(1, 2).contiguous().view(-1, size[1], size[3], size[4])
            else:
                w = ww.view(-1, size[2], size[3], size[4])
        else:
            # TODO
            x *= s.expand(-1, -1, 1, 1)  # [BIhw] Not fused => scale input activations.

        # Convolution with optional up/down sampling
        if self.up:
            x = upsample_conv_2d(x, w, groups)
        elif self.down:
            x = conv_downsample_2d(x, w)
        else:
            x = F.conv2d(x, w, stride=1, padding=w.shape[-1] // 2, groups=groups)

        # Reshape/scale output
        if self.fused_modconv:
            # Fused => reshape convolution groups back to minibatch.
            x = x.view(-1, self.fmaps, x.shape[2], x.shape[3])
        elif self.demodulate:
            # TODO
            x *= d.expand(-1, -1, 1, 1)  # [BOhw] Not fused => scale output activations.

        return x


if __name__ == '__main__':
    conv_up = EqualizedModConv2d(dlatent_size=512, input_channels=512, output_channels=512, kernel=3, up=True)
    conv = EqualizedModConv2d(dlatent_size=512, input_channels=512, output_channels=512, kernel=3)

    fmaps = torch.randn(4, 512, 8, 8)
    dlatents_in = torch.randn(4, 512)

    print(conv_up(fmaps, dlatents_in).shape)
    print(conv(fmaps, dlatents_in).shape)

    print('Done.')
