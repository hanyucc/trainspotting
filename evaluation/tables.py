#!/usr/bin/env python
"""Summarise per-sample result CSVs as the paper's tables.

    python evaluation/tables.py evaluation/results/*.csv
    python evaluation/tables.py evaluation/results/*.csv --by views
    python evaluation/tables.py evaluation/results/*.csv --by regime --ci

Every metric is the mean over samples (both the 4-view and the 16-view sample of an object count).
``--ci`` adds a 95% bootstrap interval over objects. Point-map accuracy and completeness are in
percent of the object extent, depth AbsRel and delta < 1.03 after one scale per sample, camera AUC
at 5 and 30 degrees over all view pairs.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = [
    ("points_acc", "Acc ↓", "{:.2f}"), ("points_comp", "Comp ↓", "{:.2f}"),
    ("depth_abs_rel", "AbsRel ↓", "{:.3f}"), ("depth_delta1_03", "δ<1.03 ↑", "{:.1f}"),
    ("pose_auc_at5", "AUC@5 ↑", "{:.1f}"), ("pose_auc_at30", "AUC@30 ↑", "{:.1f}"),
]
PERCENT = {"depth_delta1_03"}


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def value(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    v = float(v)
    return v * 100 if key in PERCENT else v


def summarise(rows, key, ci, resamples=10000, seed=0):
    by_object = defaultdict(list)
    for row in rows:
        v = value(row, key)
        if v is not None:
            by_object[row["instance_id"]].append(v)
    values = [v for vs in by_object.values() for v in vs]
    if not values:
        return None, None
    mean = float(np.mean(values))
    if not ci:
        return mean, None
    objects = list(by_object.values())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(resamples):
        draw = rng.integers(0, len(objects), len(objects))
        means.append(np.mean([v for i in draw for v in objects[i]]))
    return mean, (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def groups(rows, by, regimes):
    if by == "all":
        return {"all": rows}
    if by == "views":
        out = defaultdict(list)
        for row in rows:
            out[f"{row['num_views']} views"].append(row)
        return dict(sorted(out.items(), key=lambda kv: int(kv[0].split()[0])))
    out = defaultdict(list)
    for row in rows:
        out[regimes.get(row["instance_id"], "unknown")].append(row)
    order = ["fixed", "roaming", "instance-pooled"]
    return {k: out[k] for k in order if k in out} | {k: v for k, v in out.items() if k not in order}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--by", choices=["all", "views", "regime"], default="all")
    ap.add_argument("--regimes", default=os.path.join(HERE, "samples", "regimes.csv"))
    ap.add_argument("--ci", action="store_true")
    args = ap.parse_args()

    regimes = {}
    if args.by == "regime":
        with open(args.regimes, newline="") as handle:
            regimes = {r["instance_id"]: r["regime"] for r in csv.DictReader(handle)}

    models = [(os.path.splitext(os.path.basename(p))[0], read_rows(p)) for p in args.csvs]
    for group_name, _ in groups(models[0][1], args.by, regimes).items():
        print(f"\n### {group_name}\n")
        print("| model | n | " + " | ".join(label for _, label, _ in METRICS) + " |")
        print("|---|---|" + "---|" * len(METRICS))
        for name, rows in models:
            rows = groups(rows, args.by, regimes).get(group_name, [])
            cells = []
            for key, _, fmt in METRICS:
                mean, interval = summarise(rows, key, args.ci)
                if mean is None:
                    cells.append("–")
                elif interval is None:
                    cells.append(fmt.format(mean))
                else:
                    cells.append(f"{fmt.format(mean)} [{fmt.format(interval[0])}, {fmt.format(interval[1])}]")
            print(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
