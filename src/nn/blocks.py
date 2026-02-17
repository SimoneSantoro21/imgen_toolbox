import torch
import torch.nn as nn
from embeddings import FiLM

#-------------------------------- BASIC OPERATIONS -----------------------------------------

def conv(dimensionality, in_channels, out_channels, **kwargs):
    if dimensionality == 1:
        return nn.Conv1d(in_channels, out_channels, **kwargs)
    elif dimensionality == 2:
        return nn.Conv2d(in_channels, out_channels, **kwargs)
    elif dimensionality == 3:
        return nn.Conv3d(in_channels, out_channels, **kwargs)
    else:
        raise ValueError("dimensionality must be 1, 2, or 3")



def transposed_conv(dimensionality, in_channels, out_channels, **kwargs):
    if dimensionality == 1:
        return nn.ConvTranspose1d(in_channels, out_channels, **kwargs)
    elif dimensionality == 2:
        return nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
    elif dimensionality == 3:
        return nn.ConvTranspose3d(in_channels, out_channels, **kwargs)
    else:
        raise ValueError("dimensionality must be 1, 2, or 3")



#-------------------- NORMALIZATION OPTIONS AND ACTIVATION FUNCTIONS ----------------------

class Normalization(nn.Module):
    """
    Class for defining the normalization operation to apply in network layers
    """

    def __init__(self, num_channels, normalization = "batch", dimensionality = 2, **kwargs):
        super().__init__()

        self.normalization = normalization

        if normalization == "batch":
            if dimensionality==1:
                self.out = nn.BatchNorm1d(num_channels, **kwargs)
            if dimensionality==2:
                self.out = nn.BatchNorm2d(num_channels, **kwargs)
            if dimensionality==3:
                self.out = nn.BatchNorm3d(num_channels, **kwargs)

    def forward(self, x):
        return self.out(x)
    

class Activation(nn.Module):
    """
    Class for defining the Activation layer 
    """

    def __init__(self, activation, **kwargs):
        super().__init__()

        if activation == "relu":
            self.out = nn.ReLU(*kwargs)
        if activation == "leaky_relu":
            self.out = nn.LeakyReLU(**kwargs)
    
    def forward(self, x):
        return self.out(x)
    

#------------------------------- UPSAMPLING AND DOWNSAMPLING BLOCKS -------------------------

class Downsampling(nn.Module):
    """
    Convolution based downsampling block. Works for both 2D and 3D images.
    """

    def __init__(self, in_ch, out_ch, dimensionality = 2, normalization = None, 
                 activation = False, **kwargs):
        super().__init__()

        layers = []

        layers.append(conv(dimensionality, in_ch, out_ch, **kwargs))
        if normalization:
            layers.append(Normalization(out_ch, normalization, dimensionality))
        if activation:
            layers.append(nn.LeakyReLU(0.2, False))

        self.downsampling = nn.Sequential(*layers)

    def forward(self, x):
        return self.downsampling(x)
    

class Upsampling(nn.Module):
    """
    Convolution based upsampling block. Works for both 2D and 3D images.
    """

    def __init__(self, in_ch, out_ch, dimensionality = 2, normalization = None, 
                 activation = False, dropout = False, **kwargs):
        super().__init__()

        layers = []

        if activation:
            layers.append(nn.ReLU(inplace=False))

        layers.append(transposed_conv(dimensionality, in_ch, out_ch, **kwargs))
        
        if normalization:
            layers.append(Normalization(out_ch, normalization, dimensionality))

        if dropout:
            layers.append(nn.Dropout(0.5))
        
        self.upsampling = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.upsampling(x)
    

#--------------------------------- UNET SKIPCONNECTION BLOCK -------------------------

class UnetBlock(nn.Module):
    """
    U-Net block with skip connection.
    This version supports optional FiLM conditioning via an embedding vector.

    Forward signatures:
      - unconditional: y = block(x)
      - conditional:   y = block(x, emb)

    Notes:
      - For non-outermost blocks we return cat([x, up(...)], dim=1) as usual.
      - FiLM is applied after the DOWN path and after the UP path (when enabled).
    """

    def __init__(
        self, outer_nc, inner_nc, input_nc=None, submodule=None,
        is_innermost=False, is_outermost=False, norm_layer="batch",
        use_dropout=False, dimensionality=2, kernel_size=4, stride=2, padding=1,
        use_film=False, emb_dim=128
    ):
        super().__init__()
        self.is_outermost = is_outermost
        self.is_innermost = is_innermost
        self.use_dropout = use_dropout
        self.use_film = use_film

        if input_nc is None:
            input_nc = outer_nc

        if dimensionality == 3:
            if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
            if isinstance(stride, int):      stride = (stride, stride, stride)
            if isinstance(padding, int):     padding = (padding, padding, padding)

        # --- layers ---
        if is_outermost:
            self.down = Downsampling(
                input_nc, inner_nc,
                dimensionality=dimensionality,
                normalization=None,
                activation=False,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.submodule = submodule
            self.up = Upsampling(
                inner_nc * 2, outer_nc,
                dimensionality=dimensionality,
                normalization=None,
                activation=True,
                dropout=False,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.out_act = nn.Sigmoid()

        elif is_innermost:
            self.down = Downsampling(
                input_nc, inner_nc,
                dimensionality=dimensionality,
                normalization=None,
                activation=True,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.submodule = None
            self.up = Upsampling(
                inner_nc, outer_nc,
                dimensionality=dimensionality,
                normalization=norm_layer,
                activation=True,
                dropout=False,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.out_act = None

        else:
            self.down = Downsampling(
                input_nc, inner_nc,
                dimensionality=dimensionality,
                normalization=norm_layer,
                activation=True,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.submodule = submodule
            self.up = Upsampling(
                inner_nc * 2, outer_nc,
                dimensionality=dimensionality,
                normalization=norm_layer,
                activation=True,
                dropout=False,
                kernel_size=kernel_size, stride=stride, padding=padding
            )
            self.dropout = nn.Dropout(0.5) if use_dropout else None
            self.out_act = None

        # --- FiLM modules (optional) ---
        if use_film:
            self.film_down = FiLM(emb_dim=emb_dim, n_ch=inner_nc)
            self.film_up   = FiLM(emb_dim=emb_dim, n_ch=outer_nc)
        else:
            self.film_down = None
            self.film_up   = None

    def forward(self, x, emb=None):
        # Down
        h = self.down(x)
        if self.film_down is not None and emb is not None:
            h = self.film_down(h, emb)

        # Submodule
        if self.submodule is not None:
            h = self.submodule(h, emb)

        # Up
        u = self.up(h)
        if self.film_up is not None and emb is not None:
            u = self.film_up(u, emb)

        if self.out_act is not None:
            u = self.out_act(u)

        # Skip connections
        if self.is_outermost:
            return u
        else:
            out = torch.cat([x, u], dim=1)
            if (not self.is_innermost) and (self.dropout is not None):
                out = self.dropout(out)
            return out