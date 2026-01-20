from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# -------------------------
# Basic convolutional blocks
# -------------------------
class DoubleConv(nn.Module):
	"""(conv => BN => ReLU) * 2"""
	def __init__(self, in_ch, out_ch, mid_ch: Optional[int] = None):
		super().__init__()
		if not mid_ch:
			mid_ch = out_ch
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(mid_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_ch, out_ch),
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv. Handles skip connections correctly."""
	def __init__(self, dec_ch, skip_ch, out_ch, bilinear=True):
		super().__init__()
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(dec_ch + skip_ch, out_ch, mid_ch=(dec_ch + skip_ch) // 2)
		else:
			self.up = nn.ConvTranspose2d(dec_ch, dec_ch, kernel_size=2, stride=2)
			self.conv = DoubleConv(dec_ch + skip_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


# -------------------------
# Transformer building blocks
# -------------------------
class MLP(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, out_dim),
			nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, dim, n_heads=8, attn_dropout=0.0, proj_dropout=0.0):
		super().__init__()
		assert dim % n_heads == 0, "dim must be divisible by n_heads"
		self.n_heads = n_heads
		self.head_dim = dim // n_heads
		self.scale = self.head_dim ** -0.5

		self.qkv = nn.Linear(dim, dim * 3, bias=True)
		self.attn_drop = nn.Dropout(attn_dropout)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_dropout)

	def forward(self, x):
		B, N, C = x.shape
		qkv = self.qkv(x)
		qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv[0], qkv[1], qkv[2]
		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)
		out = (attn @ v)
		out = out.transpose(1, 2).reshape(B, N, C)
		out = self.proj(out)
		out = self.proj_drop(out)
		return out


class TransformerEncoderLayer(nn.Module):
	def __init__(self, dim, n_heads=8, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = MultiHeadSelfAttention(dim, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
		self.norm2 = nn.LayerNorm(dim)
		hidden_dim = int(dim * mlp_ratio)
		self.mlp = MLP(dim, hidden_dim, dim, dropout=dropout)

	def forward(self, x):
		x = x + self.attn(self.norm1(x))
		x = x + self.mlp(self.norm2(x))
		return x

def checkpoint_safe(module, x):
	# only checkpoint if training and requires_grad
	if x.requires_grad and module.training:
		return checkpoint(module, x, use_reentrant=False)
	else:
		return module(x)
		
class TransformerEncoder(nn.Module):
	def __init__(self, dim, depth, n_heads=8, mlp_ratio=4.0, dropout=0.0, attn_dropout=0.0):
		super().__init__()
		self.layers = nn.ModuleList([
			TransformerEncoderLayer(dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
									dropout=dropout, attn_dropout=attn_dropout)
			for _ in range(depth)
		])

	def forward(self, x):
		for layer in self.layers:
			x = checkpoint_safe(layer, x)
		return x


# -------------------------
# Patch embedding helpers
# -------------------------
class PatchEmbed(nn.Module):
	def __init__(self, in_ch, embed_dim):
		super().__init__()
		self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.proj(x)
		x = x.flatten(2).transpose(1, 2)
		return x, (H, W)


class PatchUnembed(nn.Module):
	def __init__(self, embed_dim, out_ch=None):
		super().__init__()
		self.embed_dim = embed_dim
		self.out_ch = out_ch
		if out_ch is not None:
			self.proj = nn.Conv2d(embed_dim, out_ch, kernel_size=1)
		else:
			self.proj = None

	def forward(self, x, hw):
		B, N, C = x.shape
		H, W = hw
		x = x.transpose(1, 2).reshape(B, C, H, W)
		if self.proj is not None:
			x = self.proj(x)
		return x


# -------------------------
# UNet-Transformer hybrid
# -------------------------
class UNetTransformer(nn.Module):
	def __init__(self,
				 in_channels=3,
				 num_classes=1,
				 base_c=32,
				 depth=4,
				 bilinear=True,
				 trans_dim: Optional[int] = None,
				 n_heads=8,
				 mlp_ratio=4.0,
				 trans_depth=4,
				 dropout=0.0,
				 attn_dropout=0.0,
				 ):
		super().__init__()
		self.depth = depth
		self.bilinear = bilinear

		self.inc = DoubleConv(in_channels, base_c)
		enc_chs = [base_c]
		for i in range(1, depth):
			enc_chs.append(base_c * (2 ** i))

		self.downs = nn.ModuleList()
		for i in range(depth - 1):
			self.downs.append(Down(enc_chs[i], enc_chs[i + 1]))

		bottleneck_ch = enc_chs[-1] * 2
		self.bottleneck_conv = DoubleConv(enc_chs[-1], bottleneck_ch)

		if trans_dim is None:
			trans_dim = bottleneck_ch
		self.patch_embed = PatchEmbed(bottleneck_ch, trans_dim)
		self.pos_embed = None
		self.transformer = TransformerEncoder(dim=trans_dim, depth=trans_depth,
											  n_heads=n_heads, mlp_ratio=mlp_ratio,
											  dropout=dropout, attn_dropout=attn_dropout)
		self.patch_unembed = PatchUnembed(trans_dim, out_ch=bottleneck_ch)

		dec_chs = list(reversed(enc_chs))
		self.ups = nn.ModuleList()
		prev_ch = bottleneck_ch
		for i in range(depth):  # use all skips, including the first one
			skip_ch = dec_chs[i]
			out_ch = skip_ch
			self.ups.append(Up(prev_ch, skip_ch, out_ch, bilinear=bilinear))
			prev_ch = out_ch

		self.outc = OutConv(prev_ch, num_classes)
		self.sigmoid = nn.Sigmoid()

	def _build_pos_embed(self, N, dim, device):
		if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
			pe = nn.Parameter(torch.zeros(1, N, dim), requires_grad=True).to(device)
			nn.init.trunc_normal_(pe, std=0.02)
			self.pos_embed = pe

	def forward(self, x):
		skips = []
		x1 = self.inc(x)
		skips.append(x1)
		xi = x1
		for down in self.downs:
			xi = down(xi)
			skips.append(xi)

		bt = self.bottleneck_conv(xi)

		seq, hw = self.patch_embed(bt)
		B, N, D = seq.shape
		self._build_pos_embed(N, D, seq.device)
		seq = seq + self.pos_embed
		seq = self.transformer(seq)
		bt = self.patch_unembed(seq, hw)

		x_up = bt
		for up in self.ups:
			skip = skips.pop()
			x_up = up(x_up, skip)

		output = self.outc(x_up)
		return output


if __name__ == '__main__':
	model = UNetTransformer(in_channels=3, num_classes=1, base_c=16, depth=4,
							trans_dim=None, n_heads=8, mlp_ratio=4.0, trans_depth=2)
	x = torch.randn(2, 3, 256, 256)
	y = model(x)
	print('in', x.shape, 'out', y.shape)