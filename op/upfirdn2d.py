import os

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load
import torch.nn.functional as F


module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    'upfirdn2d',
    sources=[
        os.path.join(module_path, 'upfirdn2d.cpp'),
        os.path.join(module_path, 'upfirdn2d_kernel.cu'),
    ],
)


class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(
            in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1,
                                                ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = UpFirDn2dBackward.apply(
            grad_output,
            kernel,
            grad_kernel,
            ctx.up,
            ctx.down,
            ctx.pad,
            ctx.g_pad,
            ctx.in_size,
            ctx.out_size,
        )

        return grad_input, None, None, None, None


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = UpFirDn2d.apply(
        input, kernel, (up, up), (down, down), (pad[0], pad[1], pad[0], pad[1])
    )

    return out


def upfirdn2d_native(x, kernel, factor=2, padx0=0, padx1=0, pady0=0, pady1=0):
    kernelH, kernelW = kernel.shape

    y = x.clone()
    # N C H W ---> N*C H W 1
    y = y.reshape([-1, x.shape[2], x.shape[3], 1])

    inC, inH, inW = x.shape[1:]
    # step 1: upfirdn2d

    # 1) Upsample
    y = torch.reshape(y, (-1, inH, 1, inW, 1, 1))
    y = F.pad(y, (0, 0, factor - 1, 0, 0,
                  0, factor - 1, 0, 0, 0, 0, 0))
    y = torch.reshape(y, (-1, 1, inH * factor, inW * factor))

    # 2) Pad (crop if negative).
    y = F.pad(y, (0, 0,
                  max(pady0, 0), max(pady1, 0),
                  max(padx0, 0), max(padx1, 0),
                  0, 0
                  ))
    y = y[:,
          max(-pady0, 0): y.shape[1] - max(-pady1, 0),
          max(-padx0, 0): y.shape[2] - max(-padx1, 0),
          :]

    # 3) Convolve with filter.
    y = y.permute(0, 3, 1, 2)  # N*C H W 1 --> N*C 1 H W
    y = y.reshape(-1, 1, inH * factor + pady0 +
                  pady1, inW * factor + padx0 + padx1)
    y = F.conv2d(y, kernel)
    y = y.view(-1, 1,
               inH * factor + pady0 + pady1 - kernelH + 1,
               inW * factor + padx0 + padx1 - kernelW + 1)

    # 4) Downsample (throw away pixels).
    if inH * factor != y.shape[1]:
        y = F.interpolate(y, size=(inH * factor,
                                   inW * factor), mode='bilinear')
    y = y.permute(0, 2, 3, 1)
    y = y.reshape(-1, inC, inH * factor, inW * factor)

    return y