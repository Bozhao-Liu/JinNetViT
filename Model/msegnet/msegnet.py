import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic 2D conv block: {BN, ReLU, Conv3} x 2
# -----------------------------
class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# EnBlock2D: repeat ConvBlock2D n times (like Enblock×k in the table)
# -----------------------------
class EnBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks: int):
        super().__init__()
        blocks = []
        ch_in = in_ch
        for _ in range(num_blocks):
            blocks.append(ConvBlock2D(ch_in, out_ch))
            ch_in = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# -----------------------------
# DeBlock2D: similar to EnBlock, used in decoder
# -----------------------------
class DeBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks: int):
        super().__init__()
        blocks = []
        ch_in = in_ch
        for _ in range(num_blocks):
            blocks.append(ConvBlock2D(ch_in, out_ch))
            ch_in = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# -----------------------------
# DownSample: Conv3 with stride 2 (as in table DownSampleX: Conv3 (Stride 2))
# -----------------------------
class DownSample2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# UpSample: Conv3 -> DeConv -> Conv3 (2D version of "Conv3, DeConv, Conv3")
# -----------------------------
class UpSample2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                mid_ch, mid_ch,
                kernel_size=2, stride=2
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Multi-head self-attention in 2D using nn.MultiheadAttention
# -----------------------------
class MHSA2D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Treat (H*W) as sequence length, C as embedding dim.
        """
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, C, N] -> [N, B, C]
        x_flat = x.view(B, C, -1).permute(2, 0, 1)  # [N, B, C]
        x_norm = self.norm(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)  # [N, B, C]
        out = attn_out.permute(1, 2, 0).view(B, C, H, W)
        return out


# -----------------------------
# Transformer-style block for 2D feature maps
# -----------------------------
class TransformerBlock2D(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.attn = MHSA2D(dim, num_heads=num_heads)
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        # Self-attention residual
        x = x + self.attn(x)
        # MLP residual
        x = x + self.mlp(self.norm(x))
        return x


# -----------------------------
# CMA-like block (simplified for 2D single-modality)
# In the real 3D MSegNet, this does cross-modal attention between modalities.
# Here, we approximate it as a strengthened self-attention block.
# -----------------------------
class CMA2D(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.block = TransformerBlock2D(dim, num_heads=num_heads)

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Multi-View Coupled Transformer stack (approximation)
# -----------------------------
class MultiViewTransformer2D(nn.Module):
    def __init__(self, dim, num_layers=4, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock2D(dim, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


# -----------------------------
# 2D MSegNet-style model
# -----------------------------
class MSegNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, base_channels=16,
                 num_heads=4, img_size=256):
        """
        2D analogue of the MSegNet architecture described in the table:
        - Encoder: InitConv + Enblock1..5 with Conv3 (stride 2) downsampling
        - CMA Transformer + Multi-View Transformer stacks
        - Decoder: DeBlock1..5 with Conv3, DeConv, Conv3 upsampling
        """
        super().__init__()
        ch1 = base_channels          # 16
        ch2 = base_channels * 2      # 32
        ch3 = base_channels * 4      # 64
        ch4 = base_channels * 8      # 128
        ch5 = base_channels * 16     # 256

        # --------- Encoder ---------
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(p=0.1)
        )

        # Enblock1 x1 @ 16 channels
        self.en1 = EnBlock2D(ch1, ch1, num_blocks=1)
        self.down1 = DownSample2D(ch1, ch2)  # 16 -> 32

        # Enblock2 x2 @ 32 channels
        self.en2 = EnBlock2D(ch2, ch2, num_blocks=2)
        self.down2 = DownSample2D(ch2, ch3)  # 32 -> 64

        # Enblock3 x3 @ 64 channels
        self.en3 = EnBlock2D(ch3, ch3, num_blocks=3)
        self.down3 = DownSample2D(ch3, ch4)  # 64 -> 128

        # Enblock4 x3 @ 128 channels
        self.en4 = EnBlock2D(ch4, ch4, num_blocks=3)
        self.down4 = DownSample2D(ch4, ch5)  # 128 -> 256

        # Enblock5 x2 @ 256 channels
        self.en5 = EnBlock2D(ch5, ch5, num_blocks=2)

        # --------- CMA + Transformers (bottleneck) ---------
        self.cma = CMA2D(ch5, num_heads=num_heads)                     # Cross-modal attention analogue
        self.mv_trans1 = MultiViewTransformer2D(ch5, num_layers=4, num_heads=num_heads)
        self.mv_trans2 = MultiViewTransformer2D(ch5, num_layers=4, num_heads=num_heads)

        # Feature mapping (reshape/conv3) – here we keep it as a simple Conv3 to match table
        self.feature_map = nn.Conv2d(ch5, ch5, kernel_size=3, padding=1, bias=False)

        # --------- Decoder ---------
        # DeBlock1 x2 @ 256
        self.deblock1 = DeBlock2D(ch5, ch5, num_blocks=2)
        self.up1 = UpSample2D(ch5, ch4)   # 256 -> 128

        # DeBlock2 x1 @ 128
        self.deblock2 = DeBlock2D(ch4, ch4, num_blocks=1)
        self.up2 = UpSample2D(ch4, ch3)   # 128 -> 64

        # DeBlock3 x2 @ 64
        self.deblock3 = DeBlock2D(ch3, ch3, num_blocks=2)
        self.up3 = UpSample2D(ch3, ch2)   # 64 -> 32

        # DeBlock4 x1 @ 32
        self.deblock4 = DeBlock2D(ch2, ch2, num_blocks=1)
        self.up4 = UpSample2D(ch2, ch1)   # 32 -> 16

        # DeBlock5 x1 @ 16
        self.deblock5 = DeBlock2D(ch1, ch1, num_blocks=1)

        # Final upsample (optional) if you want to go one more scale like UpSample5 in table
        self.up5 = UpSample2D(ch1, ch1)   # 16 -> 16 at 2x spatial

        # EndConv: Conv1, then output
        self.end_conv = nn.Conv2d(ch1, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)  # [B, 16, H, W]

        x1 = self.en1(x)       # [B, 16, H, W]
        x = self.down1(x1)     # [B, 32, H/2, W/2]

        x2 = self.en2(x)       # [B, 32, H/2, W/2]
        x = self.down2(x2)     # [B, 64, H/4, W/4]

        x3 = self.en3(x)       # [B, 64, H/4, W/4]
        x = self.down3(x3)     # [B, 128, H/8, W/8]

        x4 = self.en4(x)       # [B, 128, H/8, W/8]
        x = self.down4(x4)     # [B, 256, H/16, W/16]

        x5 = self.en5(x)       # [B, 256, H/16, W/16]

        # CMA + Multi-View Transformers
        x = self.cma(x5)
        x = self.mv_trans1(x)
        x = self.mv_trans2(x)

        # Feature mapping
        x = self.feature_map(x)

        # Decoder
        x = self.deblock1(x)   # [B, 256, H/16, W/16]
        x = self.up1(x)        # [B, 128, H/8, W/8]

        x = self.deblock2(x)   # [B, 128, H/8, W/8]
        x = self.up2(x)        # [B, 64, H/4, W/4]

        x = self.deblock3(x)   # [B, 64, H/4, W/4]
        x = self.up3(x)        # [B, 32, H/2, W/2]

        x = self.deblock4(x)   # [B, 32, H/2, W/2]
        x = self.up4(x)        # [B, 16, H, W]

        x = self.deblock5(x)   # [B, 16, H, W]

        out = self.end_conv(x) # [B, num_classes, H, W]
        return out


if __name__ == "__main__":
    model = MSegNet(in_channels=1, num_classes=1, base_channels=16)
    dummy = torch.randn(2, 1, 256, 256)
    out = model(dummy)
    print("Output shape:", out.shape)
