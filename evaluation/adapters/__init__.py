"""Model adapters for the Trainspotting evaluation.

An adapter is a module with

    load(device, checkpoint=None) -> model object with
    model.predict(images) -> {"extrinsics": (S, 3, 4), "intrinsics": (S, 3, 3) or None, "depth": (S, H, W)}

where ``images`` is a float tensor (S, 3, H, W) in [0, 1] on ``device`` (H = W = 512 for the
protocol), extrinsics are OpenCV camera-from-world matrices expressed in the frame of the first
view and depth is per-view z-depth in the model's own units. Models without intrinsics still get
the depth and camera metrics; the point-map metrics need intrinsics to unproject the depth.

``evaluate.py --model spotter`` uses adapters/spotter.py; pass ``--model path/to/your_adapter.py``
for another model.
"""
