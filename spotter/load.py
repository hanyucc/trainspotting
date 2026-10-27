"""Load Spotter weights from the Hugging Face Hub or a local file."""

import os

import torch

from spotter.models import Spotter

HF_REPO_ID = "hanyucc/spotter"
HF_FILENAME = "spotter.safetensors"


def load_spotter(checkpoint: str | None = None, device: str | torch.device = "cuda", **model_kwargs) -> Spotter:
    """Build the model and load its weights.

    checkpoint: path to a `.safetensors` or `.pt` state dict. None downloads the released weights
    from the Hub (set HF_TOKEN or run `huggingface-cli login` if the repository is gated).
    """
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    state_dict = load_state_dict(checkpoint)
    model = Spotter(**model_kwargs)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    return {key.removeprefix("module."): value for key, value in state_dict.items()}
