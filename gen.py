import numpy as np

import torch
from torchvision import utils

from models.GAN import Generator

if __name__ == '__main__':
    gen = Generator()
    gen.load_state_dict(torch.load('./weights/stylegan2-ffhq-config-f.pt')['g_ema'])

    device = 'cuda'

    batch_size = {256: 16, 512: 9, 1024: 9}
    n_sample = batch_size.get(1024, 25)

    g = gen.to(device)

    z = np.random.RandomState(1).randn(n_sample, 512).astype('float32')
    # x = torch.randn(n_sample, 512).to(device)

    with torch.no_grad():
        img = g(torch.from_numpy(z).to(device))

    utils.save_image(
        img, 'gen' + '.png', nrow=int(n_sample ** 0.5),
        normalize=True, range=(-1, 1))

    print('Done.')
