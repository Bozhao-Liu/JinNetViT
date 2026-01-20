import torch
from torch import Tensor
import torch.nn as nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class Segformer(nn.Module):
	def __init__(self, in_channels=3, num_classes: int = 1, image_size=256):
		super(Segformer, self).__init__()
		config = SegformerConfig(
			num_labels=num_classes,       
			num_channels=in_channels,     
			image_size=image_size
		)
		self.model = SegformerForSemanticSegmentation(config)
		self.sigmoid = nn.Sigmoid()
		self.image_size = image_size
		
	def forward(self, x: Tensor) -> Tensor:
		x = self.model(x, return_dict = False)[0]
		x = torch.nn.functional.interpolate(
					x,
					size=(self.image_size , self.image_size ),
					mode="bilinear",
					align_corners=False
				)
		return x


if __name__ == '__main__':
	pixel_values = torch.randn(1, 3, 256, 256)   # (batch, channels, H, W)
	labels = torch.randint(0, 2, (1, 1, 256, 256))  # (batch, H, W)
	model = Segformer()
	# 3. Forward pass (no labels -> no internal loss)
	outputs = model(pixel_values)
	print(outputs.size())

	# 5. Custom loss (binary segmentation)
	loss_fn = nn.BCELoss()
	loss = loss_fn(outputs, labels.float())

	print("Loss:", loss.item())