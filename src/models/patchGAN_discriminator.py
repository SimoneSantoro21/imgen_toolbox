import torch.nn as nn
from nn.blocks import Downsampling


class PatchGAN_Discriminator(nn.Module):
    """
    PatchGAN discriminator modified for 128x128 images.
    
    The discriminator supports two modes:
      - patch_size=70: A 70×70 PatchGAN variant.
      - patch_size=16: A 16×16 PatchGAN variant.
    """

    def __init__(self, input_nc, ndf = 64, patch_size = 70, dimensionality = 3):
        super().__init__()

        if patch_size == 70:
            self.model = nn.Sequential(
                Downsampling(input_nc, ndf, dimensionality=dimensionality, normalization=None,
                             activation=True, kernel_size = 4, stride = 2, padding = 1),
                Downsampling(ndf, ndf*2, dimensionality=dimensionality, normalization='batch',
                             activation=True, kernel_size = 4, stride = 2, padding = 1),
                Downsampling(ndf*2, ndf*4, dimensionality=dimensionality, normalization='batch',
                             activation=True, kernel_size = 4, stride = 2, padding = 1),
                Downsampling(ndf*4, ndf*8, dimensionality=dimensionality, normalization='batch',
                             activation=True, kernel_size = 4, stride = 1, padding = 1),
                Downsampling(ndf*8, 1, dimensionality=dimensionality, normalization=None,
                             activation=False, kernel_size = 4, stride = 1, padding = 1),
            )
        elif patch_size == 16:
            self.model = nn.Sequential(
                Downsampling(input_nc, ndf, dimensionality=dimensionality, normalization=None,
                             activation=True, kernel_size=4, stride=2, padding=1),
                Downsampling(ndf, ndf*2, dimensionality=dimensionality, normalization='batch',
                             activation=True, kernel_size=4, stride=2, padding=1),
                Downsampling(ndf*2, ndf*4, dimensionality=dimensionality, normalization='batch',
                             activation=True, kernel_size=4, stride=2, padding=1),
                Downsampling(ndf*4, 1, dimensionality=dimensionality, normalization=None,
                             activation=False, kernel_size=4, stride=1, padding=1),
            )
        else:
            raise ValueError("Unsupported patch size. Please choose patch_size=70 or patch_size=16.")
    
    def forward(self, x):
        return self.model(x)