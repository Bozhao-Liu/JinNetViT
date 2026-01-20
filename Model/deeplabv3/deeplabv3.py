import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        deeplabv3_resnet101,
    )
except ImportError as e:
    raise ImportError(
        "torchvision is required for DeepLabV3. "
        "Install with `pip install torchvision`."
    ) from e


def _adapt_first_conv(conv, in_channels: int):
    """
    Adapt the first conv layer to accept a different number of input channels.
    - If in_channels == 3: keep as is.
    - If in_channels == 1: sum weights over RGB and use as single-channel kernel.
    - Else: repeat / interpolate weights to match in_channels.
    """
    if in_channels == 3:
        return conv  # nothing to do

    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        if in_channels == 1:
            # average over RGB channels
            w = conv.weight.sum(dim=1, keepdim=True) / 3.0  # (out,1,kh,kw)
            new_conv.weight.copy_(w)
        else:
            # For arbitrary in_channels, repeat or truncate
            w = conv.weight
            if in_channels < 3:
                # repeat some channels
                repeat_factor = (3 + in_channels - 1) // in_channels
                w = w[:, :1, ...].repeat(1, repeat_factor, 1, 1)[:, :in_channels, ...]
            else:
                # tile the original 3-channel weights and cut
                repeat_factor = (in_channels + 2) // 3
                w = w.repeat(1, repeat_factor, 1, 1)[:, :in_channels, ...]
            new_conv.weight.copy_(w[:, :in_channels, ...])

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


class DeepLabV3(nn.Module):
    """
    Wrapper around torchvision DeepLabV3 for segmentation.

    Args:
        in_channels: number of input channels (1, 3, or arbitrary).
        num_classes: number of output classes (channels in logits).
        backbone: "resnet50" or "resnet101".
        pretrained_backbone: if True, use ImageNet-pretrained backbone weights.

    Input:
        x: (B, in_channels, H, W)

    Output:
        y: (B, num_classes, H, W)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone: str = "resnet50",
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        if backbone.lower() == "resnet50":
            self.model = deeplabv3_resnet50(
                weights="DEFAULT" if pretrained_backbone else None,
                aux_loss=True,
            )
        elif backbone.lower() == "resnet101":
            self.model = deeplabv3_resnet101(
                weights="DEFAULT" if pretrained_backbone else None,
                aux_loss=True,
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet50' or 'resnet101'.")

        # Adapt first conv if in_channels != 3
        # torchvision structure: model.backbone.body.conv1 for resnet
        old_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = _adapt_first_conv(old_conv, in_channels)

        # Replace classifier head to match num_classes
        # classifier is DeepLabHead: [ASPP, Conv2d(256, num_classes, 1)]
        in_ch_classifier = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_ch_classifier, num_classes, kernel_size=1)

        self.in_channels = in_channels
        self.num_classes = num_classes

    def forward(self, x):
        """
        x: (B, C, H, W) -> logits: (B, num_classes, H, W)
        """
        # torchvision Deeplab returns dict {"out": logits}
        out = self.model(x)["out"]
        return out


if __name__ == "__main__":
    # quick sanity test
    x = torch.randn(2, 3, 256, 256)
    model = DeepLabV3(in_channels=3, num_classes=1, backbone="resnet50", pretrained_backbone=False)
    y = model(x)
    print("Output shape:", y.shape)

    # test 1-channel input
    x1 = torch.randn(2, 1, 256, 256)
    model1 = DeepLabV3(in_channels=1, num_classes=1, backbone="resnet50", pretrained_backbone=False)
    y1 = model1(x1)
    print("Output shape (1-channel input):", y1.shape)
