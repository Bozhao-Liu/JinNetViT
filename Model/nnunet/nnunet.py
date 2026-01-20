import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Optional


class nnUNet(nn.Module):
    """
    A faithful, single-class PyTorch nn.Module capturing nnU‑Net's architectural choices.
    
    Key features
    ------------
    • 2D or 3D operation (choose with `dim`).
    • Per‑stage **anisotropic kernels** and **strided downsampling** (tuple per stage),
      matching nnU‑Net plans for anisotropic voxel spacings.
    • Double Conv → InstanceNorm → LeakyReLU blocks (no residuals), as in nnU‑Net.
    • Strided‑conv downsampling (preferred by nnU‑Net) and transposed‑conv upsampling.
    • Channel schedule with capping via `max_num_features` (nnU‑Net caps around 320).
    • **Deep supervision** auxiliary heads on decoder stages (except the last).
    • Optional dropout at bottleneck.

    This is **model only**. Auto‑configuration (patch size, spacing, plans), data
    augmentation, sliding‑window inference, and ensemble are outside this class by design.

    Parameters
    ----------
    in_channels : int
        Input channels (1 for MRI/X‑ray, 3 for RGB endoscopy, etc.).
    num_classes : int
        Number of segmentation classes (logits).
    dim : int
        2 for Conv2d/InstanceNorm2d, 3 for Conv3d/InstanceNorm3d.
    base_channels : int
        First stage channels; doubles each downsampling stage until capped.
    num_stages : int
        UNet depth = number of encoder stages (decoder mirrors this). Typical: 5.
    conv_per_stage : int
        Number of Conv→INorm→LReLU repetitions per stage (nnU‑Net uses 2).
    max_num_features : int
        Cap for channels to control memory (default 320 like nnU‑Net).
    conv_kernel_sizes : Sequence[Sequence[int]] or Sequence[Tuple]
        Per‑stage convolution kernels. Length == num_stages. Each element is a tuple of
        length `dim`. Example (3D, 5 stages):
            [(1,3,3),(3,3,3),(3,3,3),(3,3,3),(3,3,3)]
        For 2D, supply 2‑tuples. If None, defaults to isotropic 3s.
    pool_kernel_sizes : Sequence[Tuple]
        Per‑downsample stride (and upsample stride in decoder), length == num_stages-1.
        Example (3D): [(1,2,2),(2,2,2),(2,2,2),(2,2,2)]
    deep_supervision : bool
        If True return (main_logits, aux_logits_list), else return main_logits only.
    dropout : float
        Dropout probability at bottleneck.
    leaky_relu_slope : float
        Negative slope for LeakyReLU (default 0.01).
    bias : bool
        Use bias in convolutions (nnU‑Net commonly uses bias=False with norm).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dim: int = 2,
        base_channels: int = 32,
        num_stages: int = 5,
        conv_per_stage: int = 2,
        max_num_features: int = 320,
        conv_kernel_sizes: Optional[Sequence[Tuple[int, ...]]] = None,
        pool_kernel_sizes: Optional[Sequence[Tuple[int, ...]]] = None,
        deep_supervision: bool = False,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.01,
        bias: bool = False,
    ):
        super().__init__()
        assert dim in (2, 3), "dim must be 2 or 3"
        self.dim = dim
        self.deep_supervision = deep_supervision
        self.leaky_relu_slope = leaky_relu_slope
        self.num_stages = num_stages
        self.conv_per_stage = conv_per_stage

        # ND ops
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        INorm = nn.InstanceNorm3d if dim == 3 else nn.InstanceNorm2d
        ConvT = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        Drop = nn.Dropout3d if dim == 3 else nn.Dropout2d

        # Channel schedule (double each stage, cap at max_num_features)
        enc_channels: List[int] = []
        ch = base_channels
        for _ in range(num_stages):
            enc_channels.append(min(ch, max_num_features))
            ch *= 2

        # Default kernel/stride plans if not provided
        if conv_kernel_sizes is None:
            # isotropic kernels (3,...,3)
            k = (3, 3, 3) if dim == 3 else (3, 3)
            conv_kernel_sizes = [k for _ in range(num_stages)]
        assert len(conv_kernel_sizes) == num_stages

        if pool_kernel_sizes is None:
            s = (2, 2, 2) if dim == 3 else (2, 2)
            pool_kernel_sizes = [s for _ in range(num_stages - 1)]
        assert len(pool_kernel_sizes) == (num_stages - 1)

        # Helper builders
        def same_padding(kernel: Tuple[int, ...]) -> Tuple[int, ...]:
            return tuple(k // 2 for k in kernel)

        def conv_block(cin: int, cout: int, kernel: Tuple[int, ...]) -> nn.Sequential:
            layers: List[nn.Module] = []
            for _ in range(conv_per_stage):
                layers += [
                    Conv(cin, cout, kernel_size=kernel, padding=same_padding(kernel), bias=bias),
                    INorm(cout, eps=1e-5, affine=True),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                ]
                cin = cout
            return nn.Sequential(*layers)

        # Encoder: stage i uses conv_kernel_sizes[i]
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()  # strided convs
        prev_c = in_channels
        for i in range(num_stages):
            c = enc_channels[i]
            self.enc_blocks.append(conv_block(prev_c, c, conv_kernel_sizes[i]))
            if i < num_stages - 1:
                stride = pool_kernel_sizes[i]
                self.downs.append(
                    Conv(c, c, kernel_size=stride, stride=stride, padding=tuple(0 for _ in stride), bias=bias)
                )
            prev_c = c

        # Bottleneck dropout
        self.bottleneck_dropout = Drop(dropout) if dropout > 0 else nn.Identity()

        # Decoder: mirror encoder (except bottom), upsample with transposed conv using reversed pool strides
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_channels: List[int] = list(reversed(enc_channels[:-1]))  # skip bottom
        cur_c = enc_channels[-1]
        for di, skip_c in enumerate(dec_channels):
            stride = pool_kernel_sizes[-(di + 1)]
            self.ups.append(ConvT(cur_c, skip_c, kernel_size=stride, stride=stride))
            # after concat: skip_c (up) + skip_c (skip)
            self.dec_blocks.append(conv_block(skip_c + skip_c, skip_c, conv_kernel_sizes[-(di + 2)]))
            cur_c = skip_c

        # Heads
        self.final_head = Conv(cur_c, num_classes, kernel_size=1, bias=True)
        if deep_supervision:
            # aux heads for ALL decoder stages EXCEPT the last one
            self.aux_heads = nn.ModuleList([
                Conv(c, num_classes, kernel_size=1, bias=True) for c in dec_channels[:-1]
            ])
        else:
            self.aux_heads = None

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=self.leaky_relu_slope, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _center_crop_to(feat: torch.Tensor, spatial: Tuple[int, ...]) -> torch.Tensor:
        """Center‑crop `feat` to `spatial` (handles 2D/3D)."""
        slices = [slice(None), slice(None)]
        for i, t in enumerate(spatial):
            s = feat.shape[2 + i]
            if s == t:
                slices.append(slice(None))
            else:
                st = (s - t) // 2
                slices.append(slice(st, st + t))
        return feat[tuple(slices)]

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        deep_supervision=True  ->  (main_logits, aux_logits_list)
        deep_supervision=False ->  main_logits
        """
        skips: List[torch.Tensor] = []
        
        h = x
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            if i < len(self.downs):
                skips.append(h)
                h = self.downs[i](h)
        h = self.bottleneck_dropout(h)

        aux: List[torch.Tensor] = []
        # decoder mirror
        for di, (up, dec_block) in enumerate(zip(self.ups, self.dec_blocks)):
            h = up(h)
            skip = skips[-(di + 1)]
            # shape alignment for odd sizes / anisotropy
            if skip.shape[2:] != h.shape[2:]:
                skip = self._center_crop_to(skip, h.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = dec_block(h)
            if self.deep_supervision and di < len(self.dec_blocks) - 1:
                aux.append(self.aux_heads[di](h))

        main = self.final_head(h)
        return (main, aux) if self.deep_supervision else main


if __name__ == "__main__":
    # ---- 2D smoke test (endoscopy/X‑ray) ----
    x2d = torch.randn(2,3,256,256)
    
    model2d = FullNNUNet(
        in_channels=x2d.shape[1], num_classes=3, dim=2, num_stages=5, base_channels=32,
        conv_kernel_sizes=[(3,3)]*5,
        pool_kernel_sizes=[(2,2)]*4,
        deep_supervision=True,
    )
    y2d = model2d(x2d)
    print("2D main:", tuple((y2d[0] if isinstance(y2d, tuple) else y2d).shape))

    # ---- 3D smoke test (MRI/CT) with anisotropic first stage ----
    x3d = torch.randn(1,3,64,192,192)
    model3d = FullNNUNet(
        in_channels=x3d.shape[1], num_classes=2, dim=3, num_stages=5, base_channels=32,
        conv_kernel_sizes=[(1,3,3)] + [(3,3,3)]*4,
        pool_kernel_sizes=[(1,2,2), (2,2,2), (2,2,2), (2,2,2)],
        deep_supervision=True,
    )
    
    y3d = model3d(x3d)
    print("3D main:", tuple((y3d[0] if isinstance(y3d, tuple) else y3d).shape))
