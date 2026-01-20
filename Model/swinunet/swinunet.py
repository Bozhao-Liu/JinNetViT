import torch
import torch.nn as nn
import sys, os

from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


# --------------------------------------------------------------------
# Clean wrapper reflecting official implementation
# --------------------------------------------------------------------
class SwinUNet(nn.Module):
    """
    Clean version of the official SwinUNet:
    - Direct constructor: SwinUNet(in_channels, num_classes, img_size)
    - No giant config object
    - Fully reflects official architecture
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        img_size=256,              # official default
        patch_size=4,
        embed_dim=96,
        depths=(2, 2, 2, 2),
        depths_decoder=(1, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
    ):
        super().__init__()

        # Official SwinTransformerSys implementation
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels if in_channels != 1 else 3,  # grayscale → convert
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            depths_decoder=depths_decoder,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            final_upsample=final_upsample,
        )

        self.in_channels = in_channels

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, x):
        # Convert 1-channel input → RGB (official code does this)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.swin_unet(x)


# --------------------------------------------------------------------
# Test code (same style as all your models)
# --------------------------------------------------------------------
if __name__ == "__main__":
    model = SwinUNet(in_channels=3, num_classes=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Output:", y.shape)
