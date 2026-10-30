"""Object gating with SAM3."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .config import Config
from .images import stem


def _load_sam3(device: str):
    from sam3 import build_sam3_image_model
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

    model = build_sam3_image_model(device=device)
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return model, transform, PostProcessImage


def _datapoint(image: Image.Image, prompt: str, query_id: int):
    from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, InferenceMetadata
    from sam3.train.data.sam3_image_dataset import Image as SAMImage

    w, h = image.size
    return Datapoint(
        find_queries=[FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[h, w],
                object_id=0,
                frame_index=0,
            ),
        )],
        images=[SAMImage(data=image, objects=[], size=[h, w])],
    )


def _largest_box_fraction(boxes: np.ndarray, width: int, height: int) -> float:
    if boxes.ndim != 2 or len(boxes) == 0:
        return 0.0
    b = boxes.astype(np.float64).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
    areas = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    return float(areas.max() / (width * height))


def run_gating(image_root: Path, names: list[str], prompt: str, out_dir: Path, cfg: Config, device: str) -> list[str]:
    """Detect the object in every image; keep images whose largest detection covers enough of the image.

    Writes one `<image>.npz` (masks, boxes, scores) per image and `gating.json`.
    """
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api

    out_dir.mkdir(parents=True, exist_ok=True)
    model, transform, PostProcessImage = _load_sam3(device)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=cfg.detection_threshold,
        to_cpu=False,
    )

    report = {}
    for start in tqdm(range(0, len(names), cfg.sam3_batch_size), desc="object gating"):
        batch_names = names[start:start + cfg.sam3_batch_size]
        datapoints, sizes = [], []
        for query_id, name in enumerate(batch_names):
            with Image.open(image_root / name) as image:
                image = image.convert("RGB")
            sizes.append(image.size)
            datapoints.append(transform(_datapoint(image, prompt, query_id)))
        batch = collate_fn_api(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
        with torch.inference_mode():
            outputs = model(batch)
        results = postprocessor.process_results(outputs, batch.find_metadatas)

        for query_id, (name, (w, h)) in enumerate(zip(batch_names, sizes)):
            result = results[query_id]
            masks = result["masks"].detach().cpu().numpy().astype(bool)
            boxes = result["boxes"].detach().cpu().numpy()
            scores = result["scores"].detach().cpu().numpy()
            np.savez_compressed(out_dir / f"{stem(name)}.npz", masks=masks, boxes=boxes, scores=scores)
            fraction = _largest_box_fraction(boxes, w, h)
            report[name] = {"largest_box_fraction": fraction, "kept": fraction >= cfg.min_box_area_fraction}

    del model
    torch.cuda.empty_cache()
    (out_dir / "gating.json").write_text(json.dumps(report, indent=1))
    return [name for name in names if report[name]["kept"]]


def load_masks(gating_dir: Path, name: str) -> np.ndarray:
    """SAM3 masks of an image as an [N, H, W] boolean array in original image coordinates."""
    with np.load(gating_dir / f"{stem(name)}.npz") as data:
        masks = data["masks"]
    if masks.ndim < 3:
        return np.zeros((0, 1, 1), bool)
    return masks.reshape(-1, *masks.shape[-2:])
