"""
-------------------------------------------------
   File Name:    train.py
   Author:       Zhonghao Huang
   Date:         2019/12/14
   Description:  
-------------------------------------------------
"""

import torch

from models.GAN import Generator

if __name__ == '__main__':
    # g_synthesis = GSynthesis()
    # test_dlatents_in = torch.randn(4, 18, 512)
    # test_imgs_out = g_synthesis(test_dlatents_in)

    # g_mapping = GMapping()
    # test_latents_in = torch.randn(4, 512)
    # test_dlatents_out = g_mapping(test_latents_in)
    device = 'cuda'

    gen = Generator().to(device)
    test_latents_in = torch.randn(4, 512).to(device)
    test_imgs_out = gen(test_latents_in)

    print('Done.')

