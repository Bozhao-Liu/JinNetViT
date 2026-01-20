import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic Conv Blocks
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed (in case H,W not perfectly divisible)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.block(x)
        return x


# -----------------------------
# MedFormer Transformer Bottleneck
# -----------------------------
class MedFormerBlock(nn.Module):
    """
    Transformer bottleneck over spatial tokens:
    - Flatten (H, W) to sequence L = H*W
    - Apply Multi-Head Self-Attention
    - MLP with residual
    """

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=False)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # (B, C, H*W) -> (H*W, B, C) for nn.MultiheadAttention
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)   # (L, B, C), L = H*W

        # Self-attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x_flat = x_flat + attn_out

        # MLP
        x_norm2 = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm2)
        x_flat = x_flat + mlp_out

        # Back to (B, C, H, W)
        x_out = x_flat.permute(1, 2, 0).view(B, C, H, W)
        return x_out


# -----------------------------
# Full MedFormer Segmentation Network
# -----------------------------
class MedFormer(nn.Module):
    """
    MedFormer-style 2D segmentation backbone.

    Usage:
        model = MedFormer(in_channels=3, num_classes=1)
        y = model(x)  # x: (B, C, H, W) -> y: (B, num_classes, H, W)
    """

    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 base_channels: int = 64,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)          # H,   W
        self.enc2 = DownBlock(base_channels, base_channels * 2)    # H/2, W/2
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4) # H/4, W/4
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8) # H/8, W/8

        # Bottleneck with Transformer
        self.bottleneck = MedFormerBlock(dim=base_channels * 8,
                                         num_heads=num_heads,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout)

        # Decoder
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4) # H/4
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2) # H/2
        self.dec1 = UpBlock(base_channels * 2, base_channels,     base_channels)     # H

        # Segmentation head
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_classes, H, W)
        """
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # transformer bottleneck
        b = self.bottleneck(e4)

        # decoder with skips
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.head(d1)
        return out


# -----------------------------
# Quick sanity test
# -----------------------------
if __name__ == "__main__":
    x = torch.randn(20, 3, 256, 256)
    model = MedFormer(in_channels=3, num_classes=1)
    y = model(x)
    print("Output shape:", y.shape)  # expected: (2, 1, 256, 256)
