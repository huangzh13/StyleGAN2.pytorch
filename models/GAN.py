"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from models.Blocks import GSynthesisBlock, InputBlock
from models.CustomLayers import EqualizedLinear


class GMapping(nn.Module):

    def __init__(self, latent_size=512, label_size=0, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 normalize_latents=True, **_kwargs):
        """
        Mapping network used in the StyleGAN paper.

        :param latent_size: Latent vector(Z) dimensionality.
        :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param _kwargs: Ignore unrecognized keyword args.
        """

        super(GMapping, self).__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast
        self.normalize_latents = normalize_latents

        # Embed labels and concatenate them with latents.
        if label_size:
            # TODO
            pass

        layers = []
        for layer_idx in range(0, mapping_layers):
            fmaps_in = self.latent_size if layer_idx == 0 else self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps

            layers.append(
                ('dense_%d' % layer_idx, EqualizedLinear(in_dim=fmaps_in, out_dim=fmaps_out,
                                                         lrmul=mapping_lrmul, activation=mapping_nonlinearity)))

        # Output
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # Normalize latents.
        if self.normalize_latents:
            x = x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)

        return x


class GSynthesis(nn.Module):

    def __init__(self, dlatent_size=512, num_channels=3, resolution=1024,
                 fmap_base=16 << 10, fmap_decay=1.0, fmap_min=1, fmap_max=512,
                 randomize_noise=True, architecture='skip', **_kwargs):
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
            # nonlinearity: Activation function: 'relu', 'lrelu', etc.
            # resample_kernel: Low-pass filter to apply when resampling activations. None = no filtering.
            # fused_modconv: Implement modulated_conv2d_layer() as a single fused op?
            **_kwargs: Ignore unrecognized keyword args.):
        """

        super(GSynthesis, self).__init__()

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert architecture in ['orig', 'skip', 'resnet']

        self.architecture = architecture
        self.resolution_log2 = resolution_log2
        self.randomize_noise = randomize_noise

        def nf(stage):
            return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)

        # Early layers
        self.init_block = InputBlock(dlatent_size=dlatent_size, num_channels=num_channels,
                                     in_fmaps=nf(1), out_fmaps=nf(1), use_noise=randomize_noise)
        # Main layers
        blocks = [GSynthesisBlock(dlatent_size=dlatent_size, num_channels=num_channels, res=res,
                                  in_fmaps=nf(res - 2), out_fmaps=nf(res - 1), use_noise=randomize_noise)
                  for res in range(3, resolution_log2 + 1)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, dlatents_in):
        """
        Args:
            dlatents_in: Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        Returns:
        """
        # TODO noise input

        x, y = self.init_block(dlatents_in)

        for block in self.blocks:
            x, y = block(x, dlatents_in, y)

        images_out = y

        return images_out


class Generator(nn.Module):
    def __init__(self, resolution=1024, latent_size=512, dlatent_size=512,
                 truncation_psi=0.5, truncation_cutoff=None, truncation_psi_val=None,
                 truncation_cutoff_val=None, dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9, **_kwargs):
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

        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.truncation_psi_val = truncation_psi_val
        self.truncation_cutoff_val = truncation_cutoff_val
        self.dlatent_avg_beta = dlatent_avg_beta

        self.style_mixing_prob = style_mixing_prob

        # Setup components.
        self.num_layers = (int(np.log2(resolution)) - 1) * 2
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
        # TODO truncation,style_mixing
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
    # g_synthesis = GSynthesis()
    # test_dlatents_in = torch.randn(4, 18, 512)
    # test_imgs_out = g_synthesis(test_dlatents_in)

    # g_mapping = GMapping()
    # test_latents_in = torch.randn(4, 512)
    # test_dlatents_out = g_mapping(test_latents_in)

    gen = Generator()
    test_latents_in = torch.randn(4, 512)
    test_imgs_out = gen(test_latents_in)

    print('Done.')
