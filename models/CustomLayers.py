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

from op import fused_leaky_relu, upfirdn2d


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0., activation=None,
                 gain=1., use_wscale=True, lrmul=1.):
        super(EqualizedLinear, self).__init__()

        # Equalized learning rate and custom learning rate multiplier.
        he_std = gain * in_dim ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(
            out_dim, in_dim) * init_std, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                out_dim).fill_(bias_init), requires_grad=True)
            self.b_mul = lrmul
        else:
            self.bias = None

        self.activation = activation

    def forward(self, x):
        if self.activation == 'lrelu':  # act='lrelu'
            out = F.linear(x, self.weight * self.w_mul,
                           bias=self.bias * self.b_mul)
            out = fused_leaky_relu(out, self.bias * self.b_mul)
        else:
            out = F.linear(x, self.weight * self.w_mul,
                           bias=self.bias * self.b_mul)

        return out


class EqualizedModConv2d(nn.Module):
    def __init__(self, dlatent_size, in_channel, out_channel, kernel,
                 up=False, down=False, demodulate=True, resample_kernel=None,
                 gain=1., use_wscale=True, lrmul=1.):
        """
        """
        super(EqualizedModConv2d, self).__init__()

        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1

        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.kernel = kernel

        if up:
            factor = 2
            p = (len(resample_kernel) - factor) - (kernel - 1)
            self.blur = Blur(resample_kernel, pad=(
                (p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)

        if down:
            factor = 2
            p = (len(resample_kernel) - factor) + (kernel - 1)
            self.blur = Blur(resample_kernel, pad=((p + 1) // 2, p // 2))

        self.mod = EqualizedLinear(
            in_dim=dlatent_size, out_dim=in_channel, bias_init=1.)

        he_std = gain * (in_channel * kernel ** 2) ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel, kernel) * init_std, requires_grad=True)

    def forward(self, x, y):
        batch, in_channel, height, width = x.shape

        # Modulate
        s = self.mod(y).view(batch, 1, in_channel, 1, 1)
        ww = self.w_mul * self.weight * s

        # Demodulate
        if self.demodulate:
            # [BO] Scaling factor.
            d = torch.rsqrt(ww.pow(2).sum([2, 3, 4]) + 1e-8)
            # [BOIkk] Scale output feature maps.
            ww *= d.view(batch, self.out_channel, 1, 1, 1)

        weight = ww.view(batch * self.out_channel,
                         in_channel, self.kernel, self.kernel)

        if self.up:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel,
                                 in_channel, self.kernel, self.kernel)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel,
                                                    self.kernel, self.kernel)
            out = F.conv_transpose2d(
                x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.down:
            x = self.blur(x)
            _, _, height, width = x.shape
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=self.kernel // 2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.up}, downsample={self.down})'
        )


if __name__ == '__main__':
    print('Done.')
