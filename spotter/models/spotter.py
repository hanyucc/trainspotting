# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Spotter: VGGT-Omega fine-tuned on Trainspotting, with an object mask head.

import warnings

import torch
import torch.nn as nn

from spotter.models.aggregator import Aggregator
from spotter.models.heads import CameraHead, DenseHead, ObjectMaskHead
from spotter.utils.pose_enc import encoding_to_camera


class Spotter(nn.Module):
    """Feed-forward camera, depth and object-mask prediction for a set of photos of one object.

    The architecture is VGGT-Omega (DINOv3 backbone with alternating frame / global attention,
    a single-pass camera head and a dense depth head) plus the object mask head, a small
    convolutional head on the dense head's fused features that predicts which pixels belong to the
    object shared by the views.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_object_mask: bool = True,
        frames_chunk_size: int | None = 8,
    ) -> None:
        super().__init__()
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size)
        self.object_mask_head = ObjectMaskHead() if enable_object_mask else None
        # The dense head decodes this many frames at a time; it only bounds memory.
        self.frames_chunk_size = frames_chunk_size

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images: (S, 3, H, W) or (B, S, 3, H, W) in [0, 1], H and W multiples of the patch size.

        Returns, with a leading batch dimension B:
          pose_enc     (B, S, 9)        translation, quaternion, vertical and horizontal FoV
          extrinsics   (B, S, 3, 4)     OpenCV camera-from-world, in the frame of the first view
          intrinsics   (B, S, 3, 3)     pixel intrinsics for the input resolution
          depth        (B, S, H, W, 1)  depth in the model's (scale-free) units
          depth_conf   (B, S, H, W)     aleatoric depth confidence of the dense head (>= 1)
          object_mask  (B, S, H, W)     probability that the pixel belongs to the object
          images       (B, S, 3, H, W)  the inputs, in eval mode only
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)

        device_type = images.device.type
        use_amp = device_type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        if aggregated_tokens_list[-1] is None:
            raise ValueError("Aggregator did not cache the final layer, which the heads need.")

        predictions = {}
        with torch.autocast(device_type=device_type, enabled=False):
            pose_enc = self.camera_head(aggregated_tokens_list, patch_token_start=patch_token_start)
            extrinsics, intrinsics = encoding_to_camera(pose_enc, images.shape[-2:])
            predictions.update(pose_enc=pose_enc, extrinsics=extrinsics, intrinsics=intrinsics)

            depth, depth_conf, fused = self.dense_head(
                aggregated_tokens_list,
                images=images,
                patch_token_start=patch_token_start,
                frames_chunk_size=self.frames_chunk_size,
                return_features=True,
            )
            predictions.update(depth=depth, depth_conf=depth_conf)

            if self.object_mask_head is not None:
                logits = self.object_mask_head(fused)
                predictions["object_mask_logits"] = logits
                predictions["object_mask"] = torch.sigmoid(logits)

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released checkpoints were trained with 'max'.",
                stacklevel=2,
            )
