import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar time values.
    t:      [B, 1]
    return: [B, dim]
    """
    assert dim % 2 == 0
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )  # [half]
    args = t * freqs[None, :]         # [B, half]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1]
        emb = sinusoidal_embedding(t, self.time_dim)  # [B, time_dim]
        return self.mlp(emb)                          # [B, time_dim]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Conv residual block with FiLM time conditioning (scale + shift).
    GroupNorm requires channels to be divisible by 8 — keep channel counts
    as multiples of 8 (32, 64, 128, ...).
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        # FiLM: produces per-channel scale and shift from time embedding
        self.time_proj = nn.Linear(time_dim, 2 * out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]   t_emb: [B, time_dim]
        h = self.conv1(F.gelu(self.norm1(x)))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)   # each [B, out_channels]
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net velocity field for Flow Matching on 28x28 grayscale images.

    Spatial resolution per stage (base_channels=32):
        enc1  : [B, 32,  28, 28]   <- skip1
        enc2  : [B, 64,  14, 14]   <- skip2
        mid   : [B, 128,  7,  7]
        dec1  : [B, 64,  14, 14]   (cat with skip2 -> 128 in)
        dec2  : [B, 32,  28, 28]   (cat with skip1 ->  64 in)
        output: [B,  1,  28, 28]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels  # stored so sample() can read it
        C = base_channels
        self.time_emb = TimeEmbedding(time_dim)

        # Project raw input to C channels so GroupNorm works throughout
        self.proj_in = nn.Conv2d(in_channels, C, 1)

        # Encoder
        self.enc1 = ResBlock(C,     C,     time_dim)   # 28x28
        self.down1 = nn.Conv2d(C,   C * 2, 3, stride=2, padding=1)   # -> 14x14
        self.enc2 = ResBlock(C * 2, C * 2, time_dim)   # 14x14

        # Bottleneck
        self.down2 = nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=1) # -> 7x7
        self.mid   = ResBlock(C * 4, C * 4, time_dim)

        # Decoder
        self.up1  = nn.ConvTranspose2d(C * 4, C * 2, 2, stride=2)    # -> 14x14
        self.dec1 = ResBlock(C * 4, C * 2, time_dim)  # C*4 = 2C (up) + 2C (skip)

        self.up2  = nn.ConvTranspose2d(C * 2, C, 2, stride=2)        # -> 28x28
        self.dec2 = ResBlock(C * 2, C, time_dim)      # C*2 = C (up) + C (skip)

        # Output head
        self.out_norm = nn.GroupNorm(8, C)
        self.out_conv = nn.Conv2d(C, in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 28, 28]  — noisy image at time t
        t: [B, 1]           — flow time in [0, 1]
        returns: [B, 1, 28, 28] — predicted velocity
        """
        t_emb = self.time_emb(t)         # [B, time_dim]

        x = self.proj_in(x)              # [B, C, 28, 28]

        # Encode
        skip1 = self.enc1(x, t_emb)                    # [B, C,   28, 28]
        skip2 = self.enc2(self.down1(skip1), t_emb)    # [B, 2C,  14, 14]

        # Bottleneck
        h = self.mid(self.down2(skip2), t_emb)         # [B, 4C,   7,  7]

        # Decode (with skip connections)
        h = self.dec1(torch.cat([self.up1(h), skip2], dim=1), t_emb)  # [B, 2C, 14, 14]
        h = self.dec2(torch.cat([self.up2(h), skip1], dim=1), t_emb)  # [B,  C, 28, 28]

        return self.out_conv(F.gelu(self.out_norm(h)))  # [B, 1, 28, 28]
