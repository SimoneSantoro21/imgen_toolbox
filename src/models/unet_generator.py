import torch.nn as nn

from nn.embeddings import MetaMLP
from nn.blocks import UnetBlock

class Generator_Unet3D(nn.Module):
    """
    3D U-Net generator (pix2pix-style) with FiLM metadata conditioning.

    Input:
      x:    (B, in_ch, D, H, W)
      meta: (B, 6)  -> [center_dir(3), neighbor_dir(3)]

    Output:
      y: (B, out_ch, D, H, W)
    """

    def __init__(
        self, in_ch, out_ch, ngf=64, dimensionality=3, use_dropout=True,
        meta_dim=6, emb_dim=128, use_film=True
    ):
        super().__init__()
        self.use_film = use_film

        # metadata embedding
        self.meta_mlp = MetaMLP(meta_dim=meta_dim, emb_dim=emb_dim) if use_film else None

        # anisotropic-ish params (as in your original)
        params = [
            dict(kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  # 80->40, 128->64
            dict(kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  # 40->20, 64->32
            dict(kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  # 20->10, 32->16
            dict(kernel_size=(4,4,4), stride=(2,2,2), padding=(1,1,1)),  # 10->5,  16->8
            dict(kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),  # keep D=5, halve H,W: 8->4
            dict(kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),  # keep D=5, 4->2
            dict(kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1)),  # keep D=5, 2->1
        ]

        # build from innermost outward (same structure as before)
        unet_block = UnetBlock(
            ngf*8, ngf*8, input_nc=None, submodule=None,
            is_innermost=True, dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-1]
        )
        unet_block = UnetBlock(
            ngf*8, ngf*8, input_nc=None, submodule=unet_block,
            use_dropout=use_dropout, dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-2]
        )
        unet_block = UnetBlock(
            ngf*8, ngf*8, input_nc=None, submodule=unet_block,
            use_dropout=use_dropout, dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-3]
        )
        unet_block = UnetBlock(
            ngf*4, ngf*8, input_nc=None, submodule=unet_block,
            dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-4]
        )
        unet_block = UnetBlock(
            ngf*2, ngf*4, input_nc=None, submodule=unet_block,
            dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-5]
        )
        unet_block = UnetBlock(
            ngf, ngf*2, input_nc=None, submodule=unet_block,
            dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-6]
        )
        unet_block = UnetBlock(
            out_ch, ngf, input_nc=in_ch, submodule=unet_block,
            is_outermost=True, dimensionality=dimensionality,
            use_film=use_film, emb_dim=emb_dim, **params[-7]
        )

        self.model = unet_block

    def forward(self, x, meta=None):
        if self.use_film:
            if meta is None:
                raise ValueError("meta must be provided when use_film=True")
            emb = self.meta_mlp(meta)
            return self.model(x, emb)
        else:
            return self.model(x)