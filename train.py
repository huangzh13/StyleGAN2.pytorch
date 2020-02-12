"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/12/14
   Description:  Modified from:
-------------------------------------------------
"""

import math
import numpy as np

import torch

from models.GAN import Generator


def convert_dense(weights, source_name, target_name):
    weight = weights[source_name + '/weight'].value().eval()
    bias = weights[source_name + '/bias'].value().eval()
    dic = {'weight': weight.transpose((1, 0)), 'bias': bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def convert_torgb(weights, source_name, target_name):
    weight = weights[source_name + '/weight'].value().eval()
    mod_weight = weights[source_name + '/mod_weight'].value().eval()
    mod_bias = weights[source_name + '/mod_bias'].value().eval()
    bias = weights[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.mod.weight': mod_weight.transpose((1, 0)),
        'conv.mod.bias': mod_bias + 1,
        'bias': bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    return dic_torch


def convert_modconv(weights, source_name, target_name, flip=False):
    weight = weights[source_name + '/weight'].value().eval()
    mod_weight = weights[source_name + '/mod_weight'].value().eval()
    mod_bias = weights[source_name + '/mod_bias'].value().eval()
    noise = weights[source_name + '/noise_strength'].value().eval()
    bias = weights[source_name + '/bias'].value().eval()

    dic = {
        'conv.weight': np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        'conv.mod.weight': mod_weight.transpose((1, 0)),
        'conv.mod.bias': mod_bias + 1,
        'noise_strength': np.array([noise]),
        'bias': bias.reshape((-1, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + '.' + k] = torch.from_numpy(v)

    if flip:
        dic_torch[target_name + '.conv.weight'] = torch.flip(
            dic_torch[target_name + '.conv.weight'], [3, 4]
        )

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        if k not in state_dict:
            raise KeyError(k + ' is not found')

        if v.shape != state_dict[k].shape:
            raise ValueError(
                f'Shape mismatch: {v.shape} vs {state_dict[k].shape}')

        state_dict[k] = v


def fill_statedict(state_dict, weights, size):
    log_size = int(math.log(size, 2))

    for i in range(8):
        update(state_dict, convert_dense(
            weights, f'G_mapping/Dense{i}', f'g_mapping.map.dense_{i}'))

    update(
        state_dict,
        {
            'g_synthesis.init_block.const': torch.from_numpy(
                weights['G_synthesis/4x4/Const/const'].value().eval()
            )
        },
    )

    update(state_dict, convert_torgb(
        weights, 'G_synthesis/4x4/ToRGB', 'g_synthesis.init_block.to_rgb'))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(
                weights,
                f'G_synthesis/{reso}x{reso}/ToRGB',
                f'g_synthesis.blocks.{i}.to_rgb'),
        )

    update(state_dict, convert_modconv(
        weights, 'G_synthesis/4x4/Conv', 'g_synthesis.init_block.conv'))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                weights,
                f'G_synthesis/{reso}x{reso}/Conv0_up',
                f'g_synthesis.blocks.{i}.conv0_up',
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                weights,
                f'G_synthesis/{reso}x{reso}/Conv1',
                f'g_synthesis.blocks.{i}.conv1'
            ),
        )

    return state_dict


if __name__ == '__main__':
    import pickle
    from dnnlib import tflib
    from torchvision import utils

    tflib.init_tf()

    with open('./weights/stylegan2-ffhq-config-f.pkl', 'rb') as f:
        _, _, state_Gs = pickle.load(f)

    gen = Generator()
    state_dict = gen.state_dict()
    state_dict = fill_statedict(state_dict, state_Gs.vars, 1024)
    gen.load_state_dict(state_dict)

    torch.save(state_dict, './weights/stylegan2-ffhq-config-f.pth')

    device = 'cuda:3'

    batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = batch_size.get(1024, 25)

    g = gen.to(device)

    x = torch.randn(n_sample, 512).to(device)

    with torch.no_grad():
        img = g(x)

    utils.save_image(
        img, 'case' + '.png', nrow=int(n_sample ** 0.5), normalize=True, range=(-1, 1)
    )

    print('Done.')
