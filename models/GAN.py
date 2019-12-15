"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import numpy as np

import torch.nn as nn

from .Blocks import GSynthesisBlock, InputBlock


class GMapping(nn.Module):
    def __init__(self):
        super(GMapping, self).__init__()

    def forward(self, latents_in, labels_in=None):
        pass


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 randomize_noise=True, architecture='skip', nonlinearity='lrelu',
                 resample_kernel=None, fused_modconv=True, **_kwargs):
        """

        Args:
            dlatent_size: Disentangled latent (W) dimensionality.
            num_channels: Number of output color channels.
            resolution: Output resolution.
            fmap_base: Overall multiplier for the number of feature maps.
            fmap_decay: log2 feature map reduction when doubling the resolution.
            fmap_min: Minimum number of feature maps in any layer.
            fmap_max: Maximum number of feature maps in any layer.
            randomize_noise: True = randomize noise inputs every time (non-deterministic),
                             False = read noise inputs from variables.
            architecture: Architecture: 'orig', 'skip', 'resnet'.
            nonlinearity: Activation function: 'relu', 'lrelu', etc.
            dtype: Data type to use for activations and outputs.
            resample_kernel: Low-pass filter to apply when resampling activations. None = no filtering.
            fused_modconv: Implement modulated_conv2d_layer() as a single fused op?
            **_kwargs: Ignore unrecognized keyword args.):
        """

        super(GSynthesis, self).__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert architecture in ['orig', 'skip', 'resnet']
        act, gain = {'relu': (nn.ReLU(), np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

        # Early layers
        self.init_block = InputBlock(fmaps=nf(1))

        # Main layers
        blocks = []
        for res in range(3, resolution_log2 + 1):
            blocks.append(GSynthesisBlock(res=res, fmaps=nf(res - 1)))

    def forward(self, dlatents_in):
        """

        Args:
            dlatents_in: Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].

        Returns:

        """
        pass


class Generator(nn.Module):
    def __init__(self, truncation_psi=0.5, truncation_cutoff=None, truncation_psi_val=None,
                 truncation_cutoff_val=None, dlatent_avg_beta=0.995, style_mixing_prob=0.9, **_kwargs):
        """

        Args:
            truncation_psi: Style strength multiplier for the truncation trick. None = disable.
            truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
            truncation_psi_val: Value for truncation_psi to use during validation.
            truncation_cutoff_val: Value for truncation_cutoff to use during validation.
            dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
            style_mixing_prob: Probability of mixing styles during training. None = disable.
            **_kwargs: Arguments for sub-networks (mapping and synthesis). ):
        """
        super(Generator, self).__init__()

        self.g_mapping = GMapping(latent_size, dlatent_size, dlatent_broadcast=self.num_layers, **_kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **_kwargs)

    def forward(self, latents_in, labels_in=None, return_dlatents=False):
        """

        Args:
            latents_in: First input: Latent vectors (Z) [minibatch, latent_size].
            labels_in: Second input: Conditioning labels [minibatch, label_size].)
            return_dlatents: Return dlatents in addition to the images?

        Returns:

        """
        dlatents_in = self.g_mapping(latents_in)
        # TODO
        images_out = self.g_synthesis(dlatents_in)

        if return_dlatents:
            return images_out, dlatents_in
        return images_out


class Discriminator(nn.Module):
    def __init__(self, num_channels=3, resolution=1024, label_size=0,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 architecture='resnet', nonlinearity='lrelu',
                 mbstd_group_size=4, mbstd_num_features=1,
                 resample_kernel=None, **_kwargs):
        """
        Args:
            num_channels: Number of input color channels. Overridden based on dataset.
            resolution: Input resolution. Overridden based on dataset.
            label_size: Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
            fmap_base: Overall multiplier for the number of feature maps.
            fmap_decay: log2 feature map reduction when doubling the resolution.
            fmap_min: Minimum number of feature maps in any layer.
            fmap_max: Maximum number of feature maps in any layer.
            architecture: Architecture: 'orig', 'skip', 'resnet'.
            nonlinearity: Activation function: 'relu', 'lrelu', etc.
            mbstd_group_size: Group size for the minibatch standard deviation layer, 0 = disable.
            mbstd_num_features: Number of features for the minibatch standard deviation layer.
            resample_kernel: Low-pass filter to apply when resampling activations. None = no filtering.
            **_kwargs: Ignore unrecognized keyword args.):
        """
        super(Discriminator, self).__init__()
        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

    def forward(self, images_in, labels_in=None):
        """

        Args:
            images_in: First input: Images [minibatch, channel, height, width].
            labels_in: Second input: Labels [minibatch, label_size].

        Returns:

        """
        pass


class StyleGAN2:
    def __init__(self):
        pass


if __name__ == '__main__':
    gen = Generator()
