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

import numpy as np

from op import upfirdn2d_native, fused_leaky_relu


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


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
            # out = F.linear(x, self.weight * self.w_mul)
            # out = fused_leaky_relu(out, self.bias * self.b_mul)
            out = F.linear(x, self.weight * self.w_mul,
                           bias=self.bias * self.b_mul)
            out = (2 ** 0.5) * F.leaky_relu(out, negative_slope=0.2)
        else:
            out = F.linear(x, self.weight * self.w_mul,
                           bias=self.bias * self.b_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, x):
        out = upfirdn2d_native(x, kernel=self.kernel,
                               padx0=self.pad[0], padx1=self.pad[1],
                               pady0=self.pad[0], pady1=self.pad[1])

        return out


# class Upsample(nn.Module):
#     def __init__(self, kernel, factor=2):
#         super().__init__()

#         self.factor = factor
#         kernel = make_kernel(kernel) * (factor ** 2)
#         self.register_buffer('kernel', kernel)

#         p = kernel.shape[0] - factor

#         pad0 = (p + 1) // 2 + factor - 1
#         pad1 = p // 2

#         self.pad = (pad0, pad1)

#     def forward(self, input):
#         # out = upfirdn2d(input, self.kernel, up=self.factor,
#         #                 down=1, pad=self.pad)
#         out = upfirdn2d_native(input, kernel=self.kernel,
#                                up_x=self.factor, up_y=self.factor,
#                                down_x=1, down_y=1,
#                                pad_x0=self.pad[0], pad_x1=self.pad[1],
#                                pad_y0=self.pad[0], pad_y1=self.pad[1])

#         return out

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 1:
        k = np.outer(k, k)
    k /= np.sum(k)
    assert k.ndim == 2
    assert k.shape[0] == k.shape[1]
    return k


class Upsample(nn.Module):
    def __init__(self,
                 opts,
                 kernel=[1, 3, 3, 1],
                 factor=2,
                 down=1,
                 gain=1):
        """
            Upsample2d method in G_synthesis_stylegan2.
        :param k: FIR filter of the shape `[firH, firW]` or `[firN]` (separable).
                  The default is `[1] * factor`, which corresponds to average pooling.
        :param factor: Integer downsampling factor (default: 2).
        :param gain:   Scaling factor for signal magnitude (default: 1.0).
            Returns: Tensor of the shape `[N, C, H // factor, W // factor]`
        """
        super().__init__()
        assert isinstance(
            factor, int) and factor >= 1, "factor must be larger than 1! (default: 2)"

        self.gain = gain
        self.factor = factor
        self.opts = opts

        self.k = _setup_kernel(kernel) * (self.gain * (factor ** 2))  # 4 x 4
        self.k = torch.FloatTensor(self.k).unsqueeze(0).unsqueeze(0)
        self.k = torch.flip(self.k, [2, 3])
        # self.k = nn.Parameter(self.k, requires_grad=False)
        self.register_buffer('kernel', self.k)

        self.p = self.k.shape[0] - self.factor

        self.padx0, self.pady0 = (self.p + 1) // 2 + \
            factor - 1, (self.p + 1) // 2 + factor - 1
        self.padx1, self.pady1 = self.p // 2, self.p // 2

        self.kernelH, self.kernelW = self.k.shape[2:]
        self.down = down

    def forward(self, x):
        y = x.clone()
        # N C H W ---> N*C H W 1
        y = y.reshape([-1, x.shape[2], x.shape[3], 1])

        inC, inH, inW = x.shape[1:]
        # step 1: upfirdn2d

        # 1) Upsample
        y = torch.reshape(y, (-1, inH, 1, inW, 1, 1))
        y = F.pad(y, (0, 0, self.factor - 1, 0, 0,
                      0, self.factor - 1, 0, 0, 0, 0, 0))
        y = torch.reshape(y, (-1, 1, inH * self.factor, inW * self.factor))

        # 2) Pad (crop if negative).
        y = F.pad(y, (0, 0,
                      max(self.pady0, 0), max(self.pady1, 0),
                      max(self.padx0, 0), max(self.padx1, 0),
                      0, 0
                      ))
        y = y[:,
              max(-self.pady0, 0): y.shape[1] - max(-self.pady1, 0),
              max(-self.padx0, 0): y.shape[2] - max(-self.padx1, 0),
              :]

        # 3) Convolve with filter.
        y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
        y = y.reshape(-1, 1, inH * self.factor + self.pady0 +
                      self.pady1, inW * self.factor + self.padx0 + self.padx1)
        y = F.conv2d(y, self.kernel)
        y = y.view(-1, 1,
                   inH * self.factor + self.pady0 + self.pady1 - self.kernelH + 1,
                   inW * self.factor + self.padx0 + self.padx1 - self.kernelW + 1)

        # 4) Downsample (throw away pixels).
        if inH * self.factor != y.shape[1]:
            y = F.interpolate(y, size=(inH * self.factor,
                                       inW * self.factor), mode='bilinear')
        y = y.permute(0, 2, 3, 1)
        y = y.reshape(-1, inC, inH * self.factor, inW * self.factor)

        return y


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
        out = upfirdn2d(input, self.kernel, up=1,
                        down=self.factor, pad=self.pad)

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
    # conv_up = EqualizedModConv2d(dlatent_size=512, input_channels=512, output_channels=512, kernel=3, up=True)
    # conv = EqualizedModConv2d(dlatent_size=512, input_channels=512, output_channels=512, kernel=3)
    #
    # fmaps = torch.randn(4, 512, 8, 8)
    # dlatents_in = torch.randn(4, 512)
    #
    # print(conv_up(fmaps, dlatents_in).shape)
    # print(conv(fmaps, dlatents_in).shape)

    print('Done.')
