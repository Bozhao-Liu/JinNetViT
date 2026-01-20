import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# Axial Attention (Local-Global)
# ----------------------
class AxialAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Conv1d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_h = x.permute(0, 2, 1, 3).reshape(B * H, C, W)  # axial along width
        qkv = self.to_qkv(x_h).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B * H, self.heads, -1, W), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B * H, C, W)
        out_h = self.to_out(out).reshape(B, H, C, W).permute(0, 2, 1, 3)

        x_w = x.permute(0, 3, 1, 2).reshape(B * W, C, H)  # axial along height
        qkv = self.to_qkv(x_w).chunk(3, dim=1)
        q, k, v = map(lambda t: t.reshape(B * W, self.heads, -1, H), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B * W, C, H)
        out_w = self.to_out(out).reshape(B, W, C, H).permute(0, 2, 3, 1)

        return out_h + out_w

# ----------------------
# Gated Axial Attention Block
# ----------------------
class GatedAxialBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.axial_attn = AxialAttention(dim, heads)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        attn_out = self.axial_attn(x)
        x = self.gate * attn_out + (1 - self.gate) * x
        x = self.conv(x)
        return self.norm(F.relu(x))

# ----------------------
# MedT Encoder-Decoder
# ----------------------
class MedicalTransformer(nn.Module):
    def __init__(self, in_channels = 3, num_classes=1, base_dim=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.enc2 = nn.Conv2d(base_dim, base_dim * 2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, stride=2, padding=1)

        # Transformer blocks
        self.block1 = GatedAxialBlock(base_dim)
        self.block2 = GatedAxialBlock(base_dim * 2)
        self.block3 = GatedAxialBlock(base_dim * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.dec2 = nn.Conv2d(base_dim * 4, base_dim * 2, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.dec1 = nn.Conv2d(base_dim * 2, base_dim, 3, padding=1)

        # Output
        self.out_conv = nn.Conv2d(base_dim, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = F.relu(self.enc1(x))
        e1b = self.block1(e1)
        e2 = F.relu(self.enc2(e1b))
        e2b = self.block2(e2)
        e3 = F.relu(self.enc3(e2b))
        e3b = self.block3(e3)

        # Decoder with skip connections
        d2 = self.up2(e3b)
        d2 = torch.cat([d2, e2b], dim=1)
        d2 = F.relu(self.dec2(d2))

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1b], dim=1)
        d1 = F.relu(self.dec1(d1))

        out = self.out_conv(d1)
        return out

# ----------------------
# Example Run
# ----------------------
if __name__ == "__main__":
    model = MedicalTransformer(in_ch=3, num_classes=1)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(out.shape)  # Expected [2, num_classes, 256, 256]
