import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import os
import sys
from typing import Any, Callable, List, Optional, Type, Union

def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
	
class JinNet(nn.Module):
	def __init__(self, in_channels = 3, num_classes: int = 1000):
		super(JinNet, self).__init__()
		
		self.layer1 = nn.Sequential(
				nn.Conv2d(in_channels, 64, kernel_size = 3, stride=1, padding = 1, bias=False), # 
				nn.BatchNorm2d(64),
				nn.LeakyReLU(inplace=False)
				)
		self.layer2 = nn.Sequential(
				nn.Conv2d(64, 128, kernel_size = 7, stride=1, padding = 3, bias=False), # 
				nn.BatchNorm2d(128),
				nn.LeakyReLU(inplace=False)
				)
		self.layer3 = nn.Sequential(
				nn.Conv2d(128, 128, kernel_size = 15, stride=1, padding = 7, bias=False), # 
				nn.BatchNorm2d(128),
				nn.LeakyReLU(inplace=False)
				)
		self.layer4 = nn.Sequential(
				nn.Conv2d(128, 64, kernel_size = 7, stride=1, padding = 3, bias=False), # 
				nn.BatchNorm2d(64),
				nn.LeakyReLU(inplace=False)
				)
		self.outlayer = nn.Sequential(
						nn.Conv2d(64, 64, kernel_size = 31, stride=1, padding = 15, bias=False),
						nn.BatchNorm2d(64),
						nn.LeakyReLU(inplace=False),
						conv1x1(64, 64),
						nn.BatchNorm2d(64),
						nn.LeakyReLU(inplace=False),
						conv1x1(64, num_classes)
						)
		self.sigmoid = nn.Sigmoid()
		
		
	def _forward_impl(self, x: Tensor) -> Tensor:

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.outlayer(x)
		
		return x
		
	def forward(self, x: Tensor) -> Tensor:
		x = self._forward_impl(x)
		return x
	
	
if __name__ == '__main__':
	from time import time
	t = time()
	data = torch.rand((10,3,256,256)).cuda()
	model = JinNet(num_classes = 4).cuda()
	print(model(data).size(), time()-t)
