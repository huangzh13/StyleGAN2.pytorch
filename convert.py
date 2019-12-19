"""
-------------------------------------------------
   File Name:    convert.py.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import numpy as np
import torch

from models.GAN import Generator


def weight_translate(k, w):
    k = key_translate(k)
    if k.endswith('.weight'):
        if w.dim() == 2:
            w = w.t()
        elif w.dim() == 1:
            pass
        else:
            assert w.dim() == 4
            w = w.permute(3, 2, 0, 1)
    return w


def key_translate(k):
    k = k.lower().split('/')
    if k[0] == 'g_synthesis':
        if not k[1].startswith('torgb'):
            if k[1] != '4x4':
                k.insert(1, 'blocks')
                k[2] = str(int(np.log2(int(k[2].split('x')[0])) - 3))
            else:
                k[1] = 'init_block'
        k = '.'.join(k)
        k = (k.replace('const.const', 'const')
             .replace('torgb', 'to_rgb')
             .replace('.weight', '.mod_conv.weight')
             .replace('mod_bias', 'mod_conv.bias')
             .replace('mod_weight', 'mod_conv.mod_layer.weight'))

    elif k[0] == 'g_mapping':
        k.insert(1, 'map')
        k = '.'.join(k)
        k = (k.replace('weight', 'linear.weight'))
    else:
        k = '.'.join(k)

    return k


if __name__ == '__main__':
    input_file = './weights/stylegan2-ffhq-config-f-torch.pt'

    state_G, state_D, state_Gs, dlatent_avg = torch.load(input_file)
    param_dict = {key_translate(k): weight_translate(k, v) for k, v in state_Gs.items()}

    gen = Generator()

    sd_shapes = {k: v.shape for k, v in gen.state_dict().items()}
    param_shapes = {k: v.shape for k, v in param_dict.items()}

    # check for mismatch
    for k in list(sd_shapes) + list(param_shapes):
        pds = param_shapes.get(k)
        sds = sd_shapes.get(k)
        if pds is None:
            print("sd only", k, sds)
        elif sds is None:
            print("pd only", k, pds)
        elif sds != pds:
            print("mismatch!", k, pds, sds)

    gen.load_state_dict(param_dict, strict=False)  # needed for the blur kernels
    torch.save(gen.state_dict(), 'ffhq_gen.pth')
    print('Done.')
