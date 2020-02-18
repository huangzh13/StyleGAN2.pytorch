"""
-------------------------------------------------
   File Name:    generate_truncation_figure.py
   Author:       Zhonghao Huang
   Date:         2020/2/18
   Description:
-------------------------------------------------
"""

import argparse
import numpy as np
from PIL import Image

import torch

from generate_grid import adjust_dynamic_range
from models.GAN import Generator

device = "cuda"


def draw_truncation_trick_figure(png, gen, dlatent_avg, seeds, psis, ):
    w = h = 1024
    latent_size = gen.g_mapping.latent_size

    with torch.no_grad():
        latents_np = np.stack([np.random.RandomState(seed).randn(latent_size) for seed in seeds])
        latents = torch.from_numpy(latents_np.astype(np.float32)).to(device)
        dlatents = gen.g_mapping(latents).detach().cpu().numpy()  # [seed, layer, component]
        # dlatent_avg = gen.truncation.avg_latent.numpy()  # [component]

        canvas = Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
        for row, dlatent in enumerate(list(dlatents)):
            row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
            row_dlatents = torch.from_numpy(row_dlatents.astype(np.float32)).to(device)
            row_images = gen.g_synthesis(row_dlatents)
            for col, image in enumerate(list(row_images)):
                image = adjust_dynamic_range(image)
                image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                canvas.paste(Image.fromarray(image, 'RGB'), (col * w, row * h))
        canvas.save(png)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    # from config import cfg as opt
    #
    # opt.merge_from_file(args.config)
    # opt.freeze()

    print("Creating generator object ...")
    # create the generator object
    gen = Generator()

    print("Loading the generator weights from:", args.generator_file)
    # load the weights into it
    state_dict = torch.load(args.generator_file)
    gen.load_state_dict(state_dict['g_ema'])
    gen = gen.to(device)

    avg = state_dict['latent_avg'].numpy()
    draw_truncation_trick_figure('figure-truncation-trick.jpg', gen, avg,
                                 seeds=[91, 388], psis=[1, 0.7, 0.5, 0, -0.5, -1])

    print('Done.')


def parse_arguments():
    """
    default command line argument parser
    :return: args => parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='./configs/sample.yaml')
    parser.add_argument("--generator_file", action="store", type=str,
                        help="pretrained weights file for generator",
                        default='./weights/stylegan2-ffhq-config-f.pt')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())
