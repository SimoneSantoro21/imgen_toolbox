import torch
import torch.nn as nn


#--------------------------------------- FiLM embedding -------------------------------------
class MetaMLP(nn.Module):
    """Encodes metadata vector to an embedding used by FiLM."""
    def __init__(self, meta_dim=6, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, m):  # (B,meta_dim)
        return self.net(m) # (B,emb_dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: produces per-channel gamma/beta."""
    def __init__(self, emb_dim, n_ch):
        super().__init__()
        self.to_gb = nn.Linear(emb_dim, 2 * n_ch)

    def forward(self, x, emb):
        # x: (B,C,D,H,W) or (B,C,H,W), emb: (B,emb_dim)
        gb = self.to_gb(emb)  # (B,2C)
        gamma, beta = gb.chunk(2, dim=1)
        # broadcast
        view_shape = (gamma.shape[0], gamma.shape[1]) + (1,) * (x.ndim - 2)
        gamma = gamma.view(*view_shape)
        beta  = beta.view(*view_shape)
        return x * (1.0 + gamma) + beta
