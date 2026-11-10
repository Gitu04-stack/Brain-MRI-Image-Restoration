"""Deep Learning Restoration Models for Brain MRI Restoration.

Implements Residual U-Net (ResUNet) optimized for complex multi-degradation
reconstruction (low resolution, linear motion blur, and non-Gaussian Rician noise).

Key Architectural Components:
1. DoubleConv Blocks: (Conv2d 3x3 -> BatchNorm2d -> LeakyReLU(0.2) -> Conv2d 3x3 -> BatchNorm2d -> LeakyReLU(0.2)).
2. Multi-Scale Encoder: 3 Downsampling stages using MaxPool2d (512x512 -> 256x256 -> 128x128 -> 64x64).
3. Bottleneck Bridge: High-capacity feature representation at the lowest spatial dimension.
4. Multi-Scale Decoder: 3 Upsampling stages via ConvTranspose2d with skip concatenation from encoder levels.
5. Global Residual Learning: Learns the additive restoration perturbation:
       I_restored = clamp(I_degraded + Residual, 0.0, 1.0)
   This directly preserves background anatomical structures and accelerates convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with Batch Normalization and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResUNet(nn.Module):
    """Residual U-Net Architecture for Brain MRI Image Restoration."""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Encoder stages
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8)
        )

        # Decoder stages
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 2, base_channels)

        # Final projection to 1 channel residual
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder forward pass with concatenated skip connections
        d3 = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.conv_up2(torch.cat([self.up2(d3), x2], dim=1))
        d1 = self.conv_up3(torch.cat([self.up3(d2), x1], dim=1))

        # Learned residual perturbation
        residual = self.out_conv(d1)

        # Global residual learning with dynamic range clamping to [0.0, 1.0]
        restored = torch.clamp(x + residual, 0.0, 1.0)
        return restored


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
