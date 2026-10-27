"""Spotter adapter."""

import torch


class SpotterModel:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def predict(self, images):
        out = self.model(images)
        return {
            "extrinsics": out["extrinsics"][0],
            "intrinsics": out["intrinsics"][0],
            "depth": out["depth"][0, ..., 0],
        }


def load(device, checkpoint=None):
    from spotter import load_spotter

    return SpotterModel(load_spotter(checkpoint, device=device, frames_chunk_size=32))
