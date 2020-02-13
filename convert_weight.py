import argparse
import os
import sys
import pickle

from torchvision import utils

import math
import numpy as np

import torch

from models.GAN import Generator


# from model import Generator, Discriminator


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
        'bias': bias,
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
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, default='../stylegan2')
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--disc', action='store_true')
    parser.add_argument('--path', default='../StyleGAN2.pytorch/weights/stylegan2-ffhq-config-f.pkl')

    args = parser.parse_args()

    sys.path.append(args.repo)

    import dnnlib
    from dnnlib import tflib

    tflib.init_tf()

    with open(args.path, 'rb') as f:
        generator, discriminator, g_ema = pickle.load(f)

    size = g_ema.output_shape[2]

    g = Generator()
    state_dict = g.state_dict()
    state_dict = fill_statedict(state_dict, g_ema.vars, size)

    g.load_state_dict(state_dict)

    latent_avg = torch.from_numpy(g_ema.vars['dlatent_avg'].value().eval())

    ckpt = {'g_ema': state_dict, 'latent_avg': latent_avg}

    if args.gen:
        g_train = Generator(size, 512, 8)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator.vars, size)
        ckpt['g'] = g_train_state

    # if args.disc:
    #     disc = Discriminator(size)
    #     d_state = disc.state_dict()
    #     d_state = discriminator_fill_statedict(d_state, discriminator.vars, size)
    #     ckpt['d'] = d_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(ckpt, name + '.pt')

    batch_size = {256: 16, 512: 9, 1024: 1}
    n_sample = batch_size.get(size, 4)

    g = g.to(device)

    z = np.random.RandomState(0).randn(n_sample, 512).astype('float32')

    with torch.no_grad():
        # img_pt, _ = g([torch.from_numpy(z).to(device)], truncation=0.5, truncation_latent=latent_avg.to(device))
        img_pt = g(torch.randn(n_sample, 512).to(device))

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    img_tf = g_ema.run(z, None, **Gs_kwargs)
    img_tf = torch.from_numpy(img_tf).to(device)

    img_diff = ((img_pt + 1) / 2).clamp(0.0, 1.0) - ((img_tf.to(device) + 1) / 2).clamp(0.0, 1.0)

    img_concat = torch.cat((img_tf, img_pt, img_diff), dim=0)
    utils.save_image(img_concat, name + '.png', nrow=n_sample, normalize=True, range=(-1, 1))
