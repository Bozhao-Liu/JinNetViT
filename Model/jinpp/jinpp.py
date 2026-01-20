import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# 1x1 conv
def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

# Residual Dilated Block
class ResidualDilatedBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
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

	def forward(self, x):
		residual = self.residual_conv(x)
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out += residual
		out = self.relu(out)
		return out

# Multi-scale ASPP-like block (deepened)
class MultiScaleBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)

		# Parallel dilated convs
		self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1, bias=False)
		self.conv5 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
		self.conv7 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)
		self.conv9 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)

		self.bn_cat = nn.BatchNorm2d(out_channels*5)
		self.relu = nn.LeakyReLU(inplace=True)
		self.project = nn.Conv2d(out_channels*5, out_channels, 1, bias=False)
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

def checkpoint_safe(module, x):
	# only checkpoint if training and requires_grad
	if x.requires_grad and module.training:
		return checkpoint(module, x, use_reentrant=False)
	else:
		return module(x)

class JinPP(nn.Module):
	def __init__(self, in_channels = 3, num_classes=1):
		super().__init__()
		# Encoder (stacked residual blocks)
		self.layer1 = nn.Sequential(
			ResidualDilatedBlock(in_channels, 64, kernel_size=3),
			ResidualDilatedBlock(64, 64, kernel_size=3)
		)
		self.layer2 = nn.Sequential(
			ResidualDilatedBlock(64, 128, kernel_size=7),
			ResidualDilatedBlock(128, 128, kernel_size=7)
		)
		self.layer3 = nn.Sequential(
			ResidualDilatedBlock(128, 128, kernel_size=15),
			ResidualDilatedBlock(128, 128, kernel_size=15)
		)
		self.layer4 = nn.Sequential(
			ResidualDilatedBlock(128, 64, kernel_size=7),
			ResidualDilatedBlock(64, 64, kernel_size=7)
		)

		# Multi-scale ASPP-like block (deepened)
		self.outlayer = MultiScaleBlock(64, 64)

		# Skip connections
		self.skip1 = conv1x1(64, 64)
		self.skip2 = conv1x1(128, 64)
		self.skip3 = conv1x1(128, 64)

		# Final classifier
		self.classifier = nn.Sequential(
			conv1x1(64*4, 128),  # bottleneck expansion
			nn.BatchNorm2d(128),
			nn.LeakyReLU(inplace=True),
			conv1x1(128, 64),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(inplace=True),
			conv1x1(64, num_classes)
		)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		f1 = self.layer1(x)  # 64
		f2 = checkpoint_safe(self.layer2, f1)
		f3 = checkpoint_safe(self.layer3, f2)
		f4 = checkpoint_safe(self.layer4, f3)
		out = checkpoint_safe(self.outlayer, f4)

		# fuse skip connections
		s1 = self.skip1(f1)
		s2 = self.skip2(f2)
		s3 = self.skip3(f3)
		out = torch.cat([out, s1, s2, s3], dim=1)

		out = self.classifier(out)
		return out

if __name__ == "__main__":
	data = torch.rand((2,3,256,256)).cuda()
	model = JinPP(num_classes=4).cuda()
	out = model(data)
	print(out.shape)  # [2, 4, 256, 256]
