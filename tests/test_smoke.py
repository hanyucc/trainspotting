"""Smoke test: the installed package and the weights reproduce stored reference outputs.

    python tests/test_smoke.py [--checkpoint spotter.safetensors]      (or: pytest tests/)

Runs Spotter on four of the Flying Scotsman example photos and compares the predicted cameras,
the median depth of each view and the object-mask coverage with tests/reference/flying_scotsman_4v.npz
(written by tests/make_reference.py). Tolerances allow for bfloat16 differences between GPUs.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "demo"))

REFERENCE = os.path.join(HERE, "reference", "flying_scotsman_4v.npz")
EXAMPLE_IMAGES = sorted(glob.glob(os.path.join(HERE, "examples", "flying_scotsman", "*.jpg")))[:4]


def summarise(predictions):
    depth = predictions["depth"][..., 0]
    return {
        "pose_enc": predictions["pose_enc"].astype(np.float32),
        "depth_median": np.median(depth.reshape(depth.shape[0], -1), axis=1).astype(np.float32),
        "mask_fraction": (predictions["object_mask"] > 0.5).reshape(depth.shape[0], -1).mean(axis=1).astype(np.float32),
        "image_names": np.array([os.path.basename(p) for p in EXAMPLE_IMAGES]),
    }


def run(checkpoint=None):
    import torch
    from inference import run_inference
    from spotter import load_spotter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_spotter(checkpoint, device=device)
    return summarise(run_inference(model, EXAMPLE_IMAGES))


def compare(result, reference):
    problems = []
    if list(result["image_names"]) != list(reference["image_names"]):
        problems.append("example images differ from the reference")
    pose_diff = np.abs(result["pose_enc"] - reference["pose_enc"]).max()
    if pose_diff > 0.05:
        problems.append(f"camera encoding differs by up to {pose_diff:.3f} (limit 0.05)")
    depth_rel = np.abs(result["depth_median"] / reference["depth_median"] - 1).max()
    if depth_rel > 0.05:
        problems.append(f"median depth differs by up to {depth_rel * 100:.1f}% (limit 5%)")
    mask_diff = np.abs(result["mask_fraction"] - reference["mask_fraction"]).max()
    if mask_diff > 0.03:
        problems.append(f"object-mask coverage differs by up to {mask_diff:.3f} (limit 0.03)")
    return problems


def test_smoke(checkpoint=None):
    assert os.path.exists(REFERENCE), f"missing reference {REFERENCE}"
    reference = dict(np.load(REFERENCE))
    problems = compare(run(checkpoint), reference)
    assert not problems, "; ".join(problems)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    reference = dict(np.load(REFERENCE))
    result = run(args.checkpoint)
    problems = compare(result, reference)
    print("pose_enc max |diff|:", float(np.abs(result["pose_enc"] - reference["pose_enc"]).max()))
    print("depth median rel diff:", float(np.abs(result["depth_median"] / reference["depth_median"] - 1).max()))
    print("mask fraction:", result["mask_fraction"], "reference:", reference["mask_fraction"])
    if problems:
        print("FAIL:", "; ".join(problems))
        sys.exit(1)
    print("OK")
