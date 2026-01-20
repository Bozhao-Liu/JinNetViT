import torch
import sys, os

from .seg_modeling import VisionTransformer as ViT_seg
from .seg_modeling import CONFIGS as CONFIGS_ViT_seg

def TransUNet(in_channels=3, num_classes=1, pretrained=False):
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3
    model = ViT_seg(config_vit, img_size=256, num_classes=config_vit.n_classes).cuda()
    return model

if __name__ == "__main__":
    x = torch.randn(10,3,256,256).cuda()
    model = TransUNet(3,1)
    y = model(x)
    print("Output:", y.shape)
