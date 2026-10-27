"""Object mask head of Spotter.

A small convolutional head on the fused 1/4-resolution feature map of the dense
(depth) head. It predicts, for every pixel, the logit of the pixel belonging to
the object shared by the views of the sample. It is trained in a separate stage
with the rest of the model frozen (see the paper's supplement), so it does not
change the depth or camera predictions.
"""

import torch
import torch.nn as nn

from .dense_head import pixel_shuffle


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ObjectMaskHead(nn.Module):
    def __init__(self, features: int = 256, num_blocks: int = 3, final_shuffle_factor: int = 4) -> None:
        super().__init__()
        self.final_shuffle_factor = final_shuffle_factor
        self.blocks = nn.Sequential(*[_ResidualConvBlock(features) for _ in range(num_blocks)])
        # One logit per pixel, predicted as final_shuffle_factor**2 sub-pixel channels on the
        # 1/4-resolution grid and decoded to full resolution by pixel shuffle.
        self.proj = nn.Conv2d(features, final_shuffle_factor**2, kernel_size=1)

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """fused_features (B, S, C, H/4, W/4) -> object logits (B, S, H, W)."""
        batch_size, num_frames = fused_features.shape[:2]
        x = self.blocks(fused_features.flatten(0, 1).float())
        logits = pixel_shuffle(self.proj(x), self.final_shuffle_factor).squeeze(1)
        return logits.view(batch_size, num_frames, *logits.shape[1:])
