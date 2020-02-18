"""
-------------------------------------------------
   File Name:    generate_grid.py
   Author:       Zhonghao Huang
   Date:         2019/10/27
   Description:  Generate a image sample grid from
                 a particular depth of a model
-------------------------------------------------
"""

import os
import argparse
import numpy as np

import torch
from torchvision.utils import save_image

from models.GAN import Generator


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_file", action="store", type=str,
                        default="./weights/stylegan2-ffhq-config-f.pt",
                        help="pretrained weights file for generator")
    parser.add_argument("--n_row", action="store", type=int,
                        default=4, help="number of synchronized grids to be generated")
    parser.add_argument("--n_col", action="store", type=int,
                        default=4, help="number of synchronized grids to be generated")
    parser.add_argument("--output_dir", action="store", type=str,
                        default="output/",
                        help="path to the output directory for the frames")

    args = parser.parse_args()

    return args


def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
    """
    adjust the dynamic colour range of the given input data
    :param data: input image data
    :param drange_in: original range of input
    :param drange_out: required range of output
    :return: img => colour range adjusted images
    """
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return torch.clamp(data, min=0, max=1)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    device = 'cuda'

    print("Creating generator object ...")
    # create the generator object
    gen = Generator()

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    gen.load_state_dict(torch.load(args.generator_file)['g_ema'])
    gen = gen.to(device)

    # path for saving the files:
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    print("Generating scale synchronized images ...")
    # generate the images:
    with torch.no_grad():
        point = torch.randn(args.n_row * args.n_col, 512).to(device)
        # point = (point / point.norm()) * (latent_size ** 0.5)
        ss_image = gen(point)
        # color adjust the generated image:
        ss_image = adjust_dynamic_range(ss_image)

    # save the ss_image in the directory
    save_image(ss_image, os.path.join(save_path, "grid.jpg"), nrow=args.n_row,
               normalize=True, scale_each=True, pad_value=128, padding=1)

    print('Done.')


if __name__ == '__main__':
    main(parse_arguments())
