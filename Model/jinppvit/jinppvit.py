import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.checkpoint import checkpoint
from torchvision.models import vit_b_16, ViT_B_16_Weights

# -------------------------
# Basic helpers / blocks
# -------------------------
def conv1x1(in_ch, out_ch):
	return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

class SEBlock(nn.Module):
	def __init__(self, channels, reduction=16):
		super().__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channels, channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channels // reduction, channels, bias=False),
			nn.Sigmoid()
		)
	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y

class ResidualDilatedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, use_se=False):
		super().__init__()
		padding = (kernel_size - 1) // 2 * dilation
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
							   padding=padding, dilation=dilation, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.LeakyReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
							   padding=padding, dilation=dilation, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.residual_conv = nn.Sequential()
		if in_channels != out_channels:
			self.residual_conv = nn.Sequential(
				conv1x1(in_channels, out_channels),
				nn.BatchNorm2d(out_channels)
			)

		self.use_se = use_se
		if use_se:
			self.se = SEBlock(out_channels)

	def forward(self, x):
		res = self.residual_conv(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.use_se:
			out = self.se(out)
		out = out + res
		out = self.relu(out)
		return out

class MultiScaleBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)

		self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False)
		self.conv5 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
		self.conv7 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)
		self.conv9 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)

		self.bn_cat = nn.BatchNorm2d(out_channels * 5)
		self.relu = nn.LeakyReLU(inplace=True)
		self.project = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
		self.bn_project = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x1 = self.bn1(self.conv1(x))
		x2 = self.conv3(x)
		x3 = self.conv5(x)
		x4 = self.conv7(x)
		x5 = self.conv9(x)
		x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
		x_cat = self.bn_cat(x_cat)
		x_cat = self.relu(x_cat)
		out = self.project(x_cat)
		out = self.bn_project(out)
		out = self.relu(out)
		return out

# -------------------------
# JinPP-style Encoder with multi-stage skips (downsampling between stages)
# -------------------------
class JinPP_Encoder(nn.Module):
	"""
	JinPP-style encoder, but with downsampling between stages so we get multi-scale features.
	f1: H x W, channels 64
	f2: H/2 x W/2, channels 128
	f3: H/4 x W/4, channels 128
	f4: H/8 x W/8, channels 64
	"""
	def __init__(self, in_channels=3, use_se=False):
		super().__init__()
		self.use_se = use_se
		# Stage 1 (no downsample)
		self.l1 = nn.Sequential(
			ResidualDilatedBlock(in_channels, 64, kernel_size=3, dilation=1, use_se=use_se),
			ResidualDilatedBlock(64, 64, kernel_size=3, dilation=1, use_se=use_se)
		)
		# Downsample -> Stage 2
		self.down2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True)
		)
		self.l2 = nn.Sequential(
			ResidualDilatedBlock(128, 128, kernel_size=7, dilation=1, use_se=use_se),
			ResidualDilatedBlock(128, 128, kernel_size=7, dilation=1, use_se=use_se)
		)
		# Downsample -> Stage 3
		self.down3 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True)
		)
		self.l3 = nn.Sequential(
			ResidualDilatedBlock(128, 128, kernel_size=15, dilation=1, use_se=use_se),
			ResidualDilatedBlock(128, 128, kernel_size=15, dilation=1, use_se=use_se)
		)
		# Downsample -> Stage 4 (bottleneck)
		self.down4 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True)
		)
		self.l4 = nn.Sequential(
			ResidualDilatedBlock(64, 64, kernel_size=7, dilation=1, use_se=use_se),
			ResidualDilatedBlock(64, 64, kernel_size=7, dilation=1, use_se=use_se)
		)

	def forward(self, x):
		f1 = self.l1(x)				 # H x W, 64
		f2 = self.down2(f1); f2 = self.l2(f2)  # H/2, 128
		f3 = self.down3(f2); f3 = self.l3(f3)  # H/4, 128
		f4 = self.down4(f3); f4 = self.l4(f4)  # H/8, 64
		return f1, f2, f3, f4

# -------------------------
# ViT bottleneck (tokenize f4 grid)
# -------------------------
class PatchEmbed(nn.Module):
	def __init__(self, in_channels=64, embed_dim=256, patch_size=1):
		super().__init__()
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.proj(x)				 # [B, dim, H, W]
		Hn, Wn = x.shape[2], x.shape[3]
		x = x.flatten(2).transpose(1, 2) # [B, N, dim]
		return x, (Hn, Wn)

class ViTBottleneck(nn.Module):
	def __init__(self, in_channels=64, dim=256, depth=6, heads=8, num_patches = 1024):
		super().__init__()
		self.patch_embed = PatchEmbed(in_channels, dim, patch_size=1)
		self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
		enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
		self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
		self.unpatch = nn.Conv2d(dim, in_channels, kernel_size=1)

	def forward(self, x):
		tokens, (H, W) = self.patch_embed(x)   # [B, N, dim]
		B, N, D = tokens.shape
		if (self.pos_embed is None) or (self.pos_embed.shape[1] != N):
			self.pos_embed = nn.Parameter(torch.randn(1, N, D))
		tokens = tokens + self.pos_embed
		tokens = self.transformer(tokens)	  # [B, N, dim]
		feat = tokens.transpose(1, 2).reshape(B, D, H, W)
		feat = self.unpatch(feat)			 # [B, in_channels, H, W]
		return feat


class PretrainedViTBottleneck(nn.Module):
	def __init__(self, in_channels=64, vit_model='vit_base_patch16_224', pretrained=True):
		super().__init__()
		import timm
		self.vit = timm.create_model(vit_model, pretrained=pretrained)
		self.embed_dim = self.vit.embed_dim
		self.proj_in = nn.Conv2d(in_channels, 3, kernel_size=1)
		self.proj_out = nn.Conv2d(self.embed_dim, in_channels, kernel_size=1)

	def forward(self, x):
		B, C, H, W = x.shape
		# Resize f4 to ViT input
		x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
		x = self.proj_in(x)
		# Get ViT tokens
		tokens = self.vit.forward_features(x)  # [B, N+1, D]
		tokens = tokens[:, 1:, :]  # drop cls token
		N = tokens.shape[1]
		size = int(N**0.5)
		feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, size, size)
		feat = self.proj_out(feat)
		return feat
# -------------------------
# Multi-stage JinPP Decoder (progressive upsample + concat + JinPP blocks)
# -------------------------
class JinPP_DecoderMultiStage(nn.Module):
	def __init__(self, num_classes=1, use_se=False):
		super().__init__()
		self.dec3 = ResidualDilatedBlock(64 + 128, 96, kernel_size=3, use_se=use_se)   # b(up) + f3 -> 96
		self.dec2 = ResidualDilatedBlock(96 + 128, 64, kernel_size=3, use_se=use_se)   # d3(up) + f2 -> 64
		self.dec1 = ResidualDilatedBlock(64 + 64, 32, kernel_size=3, use_se=use_se)	# d2(up) + f1 -> 32

		self.out = nn.Sequential(
			MultiScaleBlock(32, 32),
			nn.Conv2d(32, num_classes, kernel_size=1)
		)

	def forward(self, b, f1, f2, f3):
		# b: feature map at f4 spatial grid (H/8)
		d3 = F.interpolate(b, size=f3.shape[2:], mode='bilinear', align_corners=False)  # -> H/4
		d3 = torch.cat([d3, f3], dim=1)
		d3 = self.dec3(d3)

		d2 = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)  # -> H/2
		d2 = torch.cat([d2, f2], dim=1)
		d2 = self.dec2(d2)

		d1 = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)  # -> H
		d1 = torch.cat([d1, f1], dim=1)
		d1 = self.dec1(d1)

		out = self.out(d1)
		return out

# -------------------------
# JinPPViT full model
# -------------------------
def checkpoint_safe(module, x):
	# only checkpoint if training and requires_grad
	if x.requires_grad and module.training:
		return checkpoint(module, x, use_reentrant=False)
	else:
		return module(x)

class resize(nn.Module):
	def __init__(self, size):
		super().__init__()
		self.size = size

	def forward(self, x):
		return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)

class JinPPViT(nn.Module):
	def __init__(self, in_channels=3, num_classes=1,
				 img_size=(256,256), 
				 use_se_encoder=True, use_se_decoder=True, use_checkpoint=True):
		"""
		JinPPViT:
		 - JinPP-style encoder (multi-stage, with downsampling between stages)
		 - ViT bottleneck on f4
		 - Multi-stage JinPP decoder (progressive upsample + concat)
		Options:
		 - use_se_encoder: insert SE into encoder ResidualDilatedBlocks
		 - use_se_decoder: insert SE into decoder ResidualDilatedBlocks
		 - use_checkpoint: apply checkpointing to heavy modules (encoder.s3/s4 and vit)
		"""
		super().__init__()
		self.use_checkpoint = use_checkpoint
		self.encoder = JinPP_Encoder(in_channels=in_channels, use_se=use_se_encoder)
		# -------------------------
		# Run dummy through encoder to infer f4 spatial size
		# -------------------------
		self.vit = PretrainedViTBottleneck()
		self.decoder = JinPP_DecoderMultiStage(num_classes=num_classes, use_se=use_se_decoder)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		orig_size = x.shape[2:]
		if self.use_checkpoint and self.training:
			x.requires_grad_(True)  # ensure graph tracking
			f1 = self.encoder.l1(x)
			f2 = checkpoint_safe(self.encoder.down2, f1)
			f2 = checkpoint_safe(self.encoder.l2, f2)
			f3 = checkpoint_safe(self.encoder.down3, f2)
			f3 = checkpoint_safe(self.encoder.l3, f3)
			f4 = checkpoint_safe(self.encoder.down4, f3)
			f4 = checkpoint_safe(self.encoder.l4, f4)
		else:
			f1, f2, f3, f4 = self.encoder(x)

		if self.use_checkpoint and self.training:
			b = checkpoint_safe(self.vit, f4)
		else:
			b = self.vit(f4)

		out = self.decoder(b, f1, f2, f3)
		out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
		return out


class ViTHeadCls(nn.Module):
	def __init__(self, in_channels=64, num_classes=10,
				 vit_model='vit_base_patch16_224', pretrained=True):
		super().__init__()
		import timm
		# Create ViT without classifier head
		self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)
		self.embed_dim = self.vit.embed_dim

		# Project f4 -> 3 channels for ViT
		self.proj_in = nn.Conv2d(in_channels, 3, kernel_size=1)

		# New classification head
		self.cls_head = nn.Linear(self.embed_dim, num_classes)

	def forward(self, x):
		B, C, H, W = x.shape
		x = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
		x = self.proj_in(x)

		feat = self.vit.forward_features(x)  # [B, N+1, D]
		feat = self.vit.forward_head(feat, pre_logits=True)

		feat = self.cls_head(feat)
		return feat


class JinPPViT_Classifier(nn.Module):
	def __init__(self, in_channels=3, num_classes=10, img_size=(256,256), 
				 vit_dim=256, vit_depth=6, vit_heads=8, patch_size=1):
		super().__init__()
		self.encoder = JinPP_Encoder(in_channels)

		# fuse f1..f4 -> 64 channels
		self.fuse = nn.Sequential(
			nn.Conv2d(64+128+128+64, 128, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.head = ViTHeadCls(in_channels=64, num_classes=num_classes)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		f1, f2, f3, f4 = self.encoder(x)

		# bring all to f4 resolution
		target_hw = f4.shape[2:]
		f1r = F.adaptive_avg_pool2d(f1, target_hw)
		f2r = F.adaptive_avg_pool2d(f2, target_hw)
		f3r = F.adaptive_avg_pool2d(f3, target_hw)

		fused = torch.cat([f1r, f2r, f3r, f4], dim=1)  # [B, 64+128+128+64, H/8, W/8]
		fused = self.fuse(fused)  # [B, 64, H/8, W/8]

		return self.sigmoid(self.head(fused))  # [B, num_classes]

# -------------------------
# Smoke test
# -------------------------
if __name__ == "__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = JinPPViT_Classifier(num_classes=10).to(device)
	x = torch.randn(20, 3, 256, 256).to(device)
	with torch.no_grad():
		y = model(x)
	print("output:", y.shape)  # expect [2, num_classes, 256, 256]
