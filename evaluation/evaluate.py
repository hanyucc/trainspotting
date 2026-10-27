#!/usr/bin/env python
"""Evaluate a model on the Trainspotting held-out samples.

    python evaluation/evaluate.py --data-root /path/to/trainspotting/eval \\
        --model spotter --out evaluation/results/spotter.csv

``--data-root`` holds the extracted held-out objects (one ``<instance_id>/`` folder each).
``--model`` is ``spotter`` or the path of an adapter module (see adapters/__init__.py).
The output is one CSV row per sample; summarise it with tables.py.
"""

import argparse
import csv
import importlib.util
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import HeldOutSamples  # noqa: E402
from metrics import sample_metrics  # noqa: E402

COLUMNS = [
    "instance_id", "num_views", "image_ids",
    "depth_scale", "depth_abs_rel", "depth_delta1_03", "depth_delta1_05", "depth_delta1_10",
    "pose_rot_deg", "pose_rra_mean_deg", "pose_rta_mean_deg",
    "pose_rra_at5", "pose_rra_at15", "pose_rra_at30", "pose_rta_at5", "pose_rta_at15", "pose_rta_at30",
    "pose_auc_at5", "pose_auc_at15", "pose_auc_at30", "focal_rel_abs",
    "points_acc", "points_acc_med", "points_comp", "points_comp_med", "points_extent",
]


def load_adapter(spec):
    path = os.path.join(HERE, "adapters", f"{spec}.py") if not spec.endswith(".py") else spec
    module_spec = importlib.util.spec_from_file_location("eval_adapter", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--samples", default=os.path.join(HERE, "samples", "eval_samples.tsv"))
    ap.add_argument("--model", default="spotter")
    ap.add_argument("--checkpoint", default=None, help="local weights; default: download the released Spotter weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=None, help="evaluate only the first N samples")
    ap.add_argument("--workers", type=int, default=4, help="threads that load and preprocess the next samples")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    dataset = HeldOutSamples(args.data_root, args.samples)
    model = load_adapter(args.model).load(args.device, args.checkpoint)
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    start = time.time()
    with open(args.out, "w", newline="") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        pending = deque(pool.submit(dataset.__getitem__, i) for i in range(min(n, 2 * args.workers)))
        next_index = len(pending)
        for index in range(n):
            sample = pending.popleft().result()
            if next_index < n:
                pending.append(pool.submit(dataset.__getitem__, next_index))
                next_index += 1
            prediction = model.predict(sample["images"].to(args.device))
            row = {"instance_id": sample["instance_id"], "num_views": sample["num_views"], "image_ids": ",".join(sample["image_ids"])}
            row.update({k: v for k, v in sample_metrics(prediction, sample).items() if k in COLUMNS})
            writer.writerow(row)
            handle.flush()
            if (index + 1) % 25 == 0 or index + 1 == n:
                print(f"{index + 1}/{n} samples, {time.time() - start:.0f}s", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
