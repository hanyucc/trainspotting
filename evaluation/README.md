# Trainspotting held-out evaluation

The evaluation protocol of the paper (Section 5.2 and the supplement), self-contained: it needs the
extracted held-out objects of the dataset and a model. `evaluate.py` writes one CSV row per sample,
`tables.py` turns those files into the paper's tables.

## Files

| file | purpose |
|---|---|
| `samples/eval_samples.tsv` | the 1,635 samples: `instance_id` and the `image_ids` of each sample in evaluation order (first = reference view) |
| `samples/regimes.csv` | regime of every object (fixed / roaming / instance-pooled) and its split |
| `results/<model>.csv` | per-sample results of the paper's models (Spotter, VGGT-Ω, VGGT, π³, DA3, MapAnything) |
| `dataset.py` | reads the released object folders and applies the protocol's preprocessing and normalisation |
| `preprocess.py` | the crop / resize of each view (adapted from VGGT, see its license note) |
| `metrics.py`, `alignment.py` | the per-sample metrics |
| `adapters/` | model adapters; `spotter.py` is built in |
| `evaluate.py`, `tables.py` | the command-line tools |

## Protocol

Each view is cropped around its principal point and resized to 512×512 (crop, resize to a 4-pixel
margin, crop); depth outside the object mask is set to 0. Ground-truth cameras and points are expressed in
the first view's camera frame and scaled to unit mean distance of the valid points. Predictions are expected in
the model's own first-camera frame; the metrics are invariant to the model's global scale.

- **Depth**: one median scale (ground truth over prediction) per sample over the valid pixels of all views;
  then AbsRel and the inlier ratios δ < 1.03 / 1.05 / 1.10 pooled over the sample's object pixels.
- **Camera pose**: for every pair of views, the relative rotation error and the angle between the relative
  translation directions; RRA / RTA at 5°, 15°, 30°; AUC of max(rotation, translation) error up to each
  threshold with integer-degree bins (as in VGGT / π³); mean absolute rotation error per view.
- **Point map**: the predicted depth unprojected with the predicted cameras at the pixels that have ground
  truth, aligned to the ground-truth points with a similarity transform (Umeyama on the pixel correspondences)
  refined by point-to-point ICP (threshold 10% of the object extent); accuracy (prediction → ground truth
  nearest-neighbour distance) and completeness (ground truth → prediction), mean and median, in percent of
  the object extent (diagonal of the 5–95 percentile bounding box of the ground-truth points). At most 20,000
  pixels per sample, a fixed-seed random subset.

All metrics are computed per sample and averaged over samples (`tables.py`); `--ci` adds a bootstrap
interval over objects.

## Running

```bash
python evaluation/evaluate.py --data-root <held-out objects> --model spotter --out evaluation/results/spotter_rerun.csv
python evaluation/evaluate.py --data-root <held-out objects> --model my_adapter.py --checkpoint my.ckpt --out evaluation/results/mine.csv
python evaluation/tables.py evaluation/results/*.csv [--by views|regime] [--ci]
```

An adapter is a Python file with `load(device, checkpoint=None)` returning an object whose
`predict(images)` maps a `(S, 3, 512, 512)` tensor in [0, 1] to `extrinsics (S, 3, 4)` (OpenCV
camera-from-world, first view as reference), `intrinsics (S, 3, 3)` (or `None`; then the point-map metrics
are skipped) and `depth (S, 512, 512)`. Inputs are resized to each model's native patch multiple inside the
adapter if needed, and outputs resampled back to 512×512.

## Result columns

`instance_id, num_views, image_ids`, then `depth_scale, depth_abs_rel, depth_delta1_03, depth_delta1_05,
depth_delta1_10` (inlier ratios as fractions), `pose_rot_deg, pose_rra_mean_deg, pose_rta_mean_deg,
pose_rra_at{5,15,30}, pose_rta_at{5,15,30}, pose_auc_at{5,15,30}` (percent), `focal_rel_abs`,
`points_acc, points_acc_med, points_comp, points_comp_med` (percent of extent), `points_extent`.
