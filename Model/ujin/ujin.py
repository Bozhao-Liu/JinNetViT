import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
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
# CNN encoder (UNet-like downsampling)
# -------------------------
class CNNEncoder(nn.Module):
	def __init__(self, in_channels=3, pretrained=True):
		super().__init__()
		from torchvision.models import resnet50, ResNet50_Weights
		
		backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
		if in_channels != 3:
			backbone.conv1  = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.layer0 = nn.Sequential(
			backbone.conv1,
			backbone.bn1,
			backbone.relu
		)  # H/2 initially, NOT returned
		self.pool = backbone.maxpool  # down to H/4

		# Feature stages (skip outputs)
		self.layer1 = backbone.layer1  # → C1: 256 ch, H/4
		self.layer2 = backbone.layer2  # → C2: 512 ch, H/8
		self.layer3 = backbone.layer3  # → C3: 1024 ch, H/16

		# Last feature map passed into ViT
		self.out_channels = 1024

	def forward(self, x):
		x = self.layer0(x)
		x = self.pool(x)
		f1 = self.layer1(x)
		f2 = self.layer2(f1)
		f3 = self.layer3(f2)
		return f1, f2, f3

# -------------------------
# Patch embedding + ViT bottleneck (token-based on f4 grid)
# -------------------------
class PatchEmbed(nn.Module):
	def __init__(self, in_channels=64, embed_dim=256, patch_size=1):
		super().__init__()
		self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

	def forward(self, x):
		B, C, H, W = x.shape
		x = self.proj(x)
		Hn, Wn = x.shape[2], x.shape[3]
		x = x.flatten(2).transpose(1, 2)  # [B, N, dim]
		return x, (Hn, Wn)

class ViTBottleneck(nn.Module):
	def __init__(self, in_channels=64, dim=256, depth=6, heads=8, patch_size=1, num_patches = 1024):
		super().__init__()
		# patch embed operates on f4 features (already downsampled by encoder)
		self.patch_embed = PatchEmbed(in_channels, dim, patch_size=patch_size)

		# register a pos_embed with the expected size so state_dict keys exist
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

		enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, batch_first=True)
		self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
		self.unpatch = nn.Conv2d(dim, in_channels, kernel_size=1)

		# init
		nn.init.trunc_normal_(self.pos_embed, std=0.02)

	def forward(self, x):
		# x is f4 features with spatial size (Hf4, Wf4) = (H//downsample_factor, W//downsample_factor)
		tokens, (Hn, Wn) = self.patch_embed(x)   # tokens: [B, N, dim], Hn/Wn are grid dims on f4 map
		B, N, D = tokens.shape

		# if pos_embed shape does not match actual number of tokens, resize (interpolate) it
		if self.pos_embed is None or self.pos_embed.shape[1] != N:
			# create new registered param (replaces old one)
			new_posemb = nn.Parameter(torch.randn(1, N, D, device=x.device) * 0.02)
			# if old exists and is smaller/larger, try to smart-resize via interpolation
			try:
				old = self.pos_embed.detach()  # (1, N_old, D)
				new = resize_pos_embed(old, new_posemb.shape)  # function from earlier messages
				new_posemb = nn.Parameter(new.to(x.device))
			except Exception:
				# fallback to random init (already set)
				pass
			# register onto module (this updates state_dict)
			self.pos_embed = new_posemb

		tokens = tokens + self.pos_embed
		tokens = self.transformer(tokens)	  # [B, N, dim]
		feat = tokens.transpose(1, 2).reshape(B, D, Hn, Wn)
		feat = self.unpatch(feat)			 # [B, in_channels, Hn, Wn]
		return feat
		
class PretrainedViTBottleneck(nn.Module):
	def __init__(self, in_channels=64, out_channels=64, vit_model='vit_base_patch16_224', pretrained=True):
		super().__init__()
		import timm
		self.vit = timm.create_model(vit_model, pretrained=pretrained)
		self.embed_dim = self.vit.embed_dim
		self.proj_in = nn.Conv2d(in_channels, 3, kernel_size=1)
		self.proj_out = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)

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
# Multi-stage JinPP Decoder (progressive upsample + concat + ResidualDilatedBlock)
# -------------------------
class JinPP_DecoderMultiStage(nn.Module):
	def __init__(self, num_classes=1, use_se=False):
		super().__init__()
		# Stage channels: input concat -> output
		self.dec3 = ResidualDilatedBlock(64 + 1024, 512, kernel_size=3, use_se=use_se)   # b(up) + f3 -> 96
		self.dec2 = ResidualDilatedBlock(512 + 512, 256, kernel_size=3, use_se=use_se)   # d3(up) + f2 -> 64
		self.dec1 = ResidualDilatedBlock(256 + 256, 64, kernel_size=3, use_se=use_se)	# d2(up) + f1 -> 32

		self.out = nn.Sequential(
			MultiScaleBlock(64, 32),
			nn.Conv2d(32, num_classes, kernel_size=1)
		)

	def forward(self, b, f1, f2, f3):
		# b: from ViT, spatial size = f4 (H/8)
		# up to f3 (H/4)
		d3 = F.interpolate(b, size=f3.shape[2:], mode='bilinear', align_corners=False)
		d3 = torch.cat([d3, f3], dim=1)
		d3 = self.dec3(d3)

		d2 = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
		d2 = torch.cat([d2, f2], dim=1)
		d2 = self.dec2(d2)

		d1 = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)
		d1 = torch.cat([d1, f1], dim=1)
		d1 = self.dec1(d1)

		out = self.out(d1)
		return out
	
# -------------------------
# Full Hybrid model
# -------------------------
def checkpoint_safe(module, x):
	# only checkpoint if training and requires_grad
	if x.requires_grad and module.training:
		return checkpoint(module, x, use_reentrant=False)
	else:
		return module(x)

class UJintransformer(nn.Module):
	def __init__(self, in_channels=3, num_classes=1, img_size = (256,256),
					use_se_decoder=True, use_se_encoder=True,
					use_checkpoint=True):
		"""
		use_se_decoder: insert SE blocks into decoder ResidualDilatedBlocks
		use_se_encoder: insert SE blocks into encoder ResidualDilatedBlocks
		use_checkpoint: apply torch.utils.checkpoint to heavy modules:
						- vit always optional
						- encoder.s3 and encoder.s4 optionally
		"""
		super().__init__()
		self.use_checkpoint = use_checkpoint
		self.encoder = CNNEncoder(in_channels=in_channels)
		# -------------------------
		# Run dummy through encoder to infer f4 spatial size
		# -------------------------
		self.vit = PretrainedViTBottleneck(in_channels = self.encoder.out_channels)
		self.decoder = JinPP_DecoderMultiStage(num_classes=num_classes, use_se=use_se_decoder)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		orig_size = x.shape[2:]
		# Optionally checkpoint parts of the encoder to save memory
		if self.use_checkpoint:
			# checkpoint stage s3 and s4 forward paths to reduce activation memory
			# We must provide callables that accept tensors only.
			x = checkpoint_safe(self.encoder.layer0, x)
			x = checkpoint_safe(self.encoder.pool, x)
			# s2 and earlier left as normal
			f1 = checkpoint_safe(self.encoder.layer1, x)
			f2 = checkpoint_safe(self.encoder.layer2, f1)
			f3 = checkpoint_safe(self.encoder.layer3, f2)
		else:
			f1, f2, f3 = self.encoder(x)

		# ViT bottleneck (optionally checkpointed)
		if self.use_checkpoint:
			b = checkpoint_safe(self.vit, f3)
		else:
			b = self.vit(f3)

		out = self.decoder(b, f1, f2, f3)
		out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=False)
		return out

# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
	import os
	os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # show exact failing op
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Device: {device}, CUDA: {torch.version.cuda}, cudnn: {torch.backends.cudnn.version()}")

	# --- Instantiate model ---
	model = UJintransformer(in_channels=3, num_classes=4, use_checkpoint=False).to(device)
	model.eval()   # start with eval mode (no BN update)

	# --- Sanity check forward shapes ---
	x = torch.randn(2, 3, 256, 256, device=device, requires_grad=True)
	print("Input:", x.shape)

	with torch.no_grad():
		fwd = model(x)
	print("Forward OK:", fwd.shape)

	# --- Forward/backward gradient check ---
	model.train()
	x.requires_grad_(True)
	out = model(x)
	print("Forward pass done:", out.shape)

	loss = out.mean()
	loss.backward()
	print("Backward OK — gradients computed successfully.")

	# --- Check for NaNs / Infs ---
	for name, param in model.named_parameters():
		if torch.isnan(param).any() or torch.isinf(param).any():
			print("⚠️ NaN or Inf detected in:", name)

	# --- Memory diagnostics ---
	allocated = torch.cuda.memory_allocated(device) / 1024**2
	reserved  = torch.cuda.memory_reserved(device) / 1024**2
	print(f"GPU memory — allocated: {allocated:.1f} MB | reserved: {reserved:.1f} MB")

	# --- Repeat with larger input to test stability ---
	try:
		big = torch.randn(1, 3, 512, 512, device=device)
		with torch.no_grad():
			y_big = model(big)
		print("Large input test OK:", y_big.shape)
	except Exception as e:
		print("❌ Error with large input:", e)