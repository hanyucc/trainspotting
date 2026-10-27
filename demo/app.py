# Adapted from the VGGT-Omega Gradio demo (demo_gradio.py), Copyright (c) Meta Platforms, Inc. and
# affiliates, under the license in the repository root.
"""Spotter demo: upload photos of one object, get its point cloud, cameras and object mask.

    python demo/app.py                       # downloads the released weights from the Hub
    python demo/app.py --checkpoint spotter.safetensors
"""

import argparse
import gc
import glob
import os
import shutil
import sys
from datetime import datetime

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import run_inference  # noqa: E402
from visual_util import predictions_to_glb  # noqa: E402

try:  # Hugging Face ZeroGPU: the decorated function gets a GPU for the duration of a call.
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:  # local GPU
    def gpu(fn):
        return fn

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_VIEWS = int(os.environ.get("SPOTTER_MAX_VIEWS", "24"))
EXAMPLES = [sorted(glob.glob(os.path.join(HERE, "examples", "flying_scotsman", "*.jpg")))]

MODEL = None


def get_model(checkpoint=None):
    global MODEL
    if MODEL is None:
        from spotter import load_spotter

        MODEL = load_spotter(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    return MODEL


def file_path(item):
    if isinstance(item, dict):
        return item.get("name") or item.get("path")
    return getattr(item, "name", str(item))


def handle_uploads(input_images):
    """Copy the uploaded photos into a fresh folder; at most MAX_VIEWS of them, evenly spaced."""
    gc.collect()
    target_dir = os.path.join("demo_outputs", datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    image_dir = os.path.join(target_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    paths = []
    for item in input_images or []:
        src = file_path(item)
        dst = os.path.join(image_dir, os.path.basename(src))
        shutil.copy(src, dst)
        paths.append(dst)
    paths = sorted(paths)
    if len(paths) > MAX_VIEWS:
        paths = paths[:: max(1, len(paths) // MAX_VIEWS)][:MAX_VIEWS]
    return target_dir, paths


ALL_PHOTOS = "All photos"


def view_choices(paths):
    return [ALL_PHOTOS] + [f"{i + 1}: {os.path.basename(p)}" for i, p in enumerate(paths)]


def parse_view(choice):
    """Dropdown value -> 0-based photo index, or None for all photos."""
    if not choice or choice == ALL_PHOTOS:
        return None
    return int(str(choice).split(":", 1)[0]) - 1


def update_gallery_on_upload(input_images):
    if not input_images:
        return None, "None", None, "Upload photos of one object.", gr.update(choices=[ALL_PHOTOS], value=ALL_PHOTOS)
    target_dir, paths = handle_uploads(input_images)
    return None, target_dir, paths, f"{len(paths)} photos ready. Click Reconstruct.", gr.update(choices=view_choices(paths), value=ALL_PHOTOS)


@gpu
def reconstruct(target_dir, mask_threshold, show_cameras, max_points_k, view_choice=ALL_PHOTOS, image_resolution=512):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        raise gr.Error("Upload photos first.")
    paths = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if not paths:
        raise gr.Error("No photos found.")
    predictions = run_inference(get_model(), paths, image_resolution=image_resolution)
    np.savez(os.path.join(target_dir, "predictions.npz"), **predictions)
    glb = render(target_dir, predictions, mask_threshold, show_cameras, max_points_k, parse_view(view_choice))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return glb, f"Reconstructed {len(paths)} photos."


def render(target_dir, predictions, mask_threshold, show_cameras, max_points_k, view=None):
    path = os.path.join(target_dir, f"scene_m{mask_threshold}_cam{show_cameras}_{int(max_points_k)}k_v{'all' if view is None else view}.glb")
    if not os.path.exists(path):
        scene = predictions_to_glb(
            predictions, mask_threshold=mask_threshold, show_cameras=show_cameras,
            max_points=int(max_points_k * 1000), view=view,
        )
        scene.export(file_obj=path)
    return path


def update_visualization(target_dir, mask_threshold, show_cameras, max_points_k, view_choice=ALL_PHOTOS):
    predictions_path = os.path.join(str(target_dir), "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, "No reconstruction yet. Click Reconstruct first."
    with np.load(predictions_path) as loaded:
        predictions = {key: np.array(loaded[key]) for key in loaded.files}
    return render(target_dir, predictions, mask_threshold, show_cameras, max_points_k, parse_view(view_choice)), "Visualization updated."


def build_ui():
    with gr.Blocks(theme=gr.themes.Ocean(), title="Spotter") as demo:
        gr.HTML(
            """
            <h1>Spotter: object-centric 3D reconstruction from found photos</h1>
            <p><a href="https://github.com/hanyucc/trainspotting">Code</a> |
            <a href="https://huggingface.co/datasets/hanyucc/trainspotting">Trainspotting dataset</a></p>
            <p>Upload photos of one object, taken by anyone, anywhere, any time. Spotter predicts the camera of every
            photo, a depth map and an object mask, and shows the object's point cloud with the cameras. The object-mask
            threshold hides the backgrounds, which differ from photo to photo.</p>
            """
        )
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        with gr.Row():
            with gr.Column(scale=2):
                input_images = gr.File(file_count="multiple", label="Upload photos", interactive=True)
                image_gallery = gr.Gallery(label="Photos", columns=4, height="300px", object_fit="contain", preview=True)
            with gr.Column(scale=4):
                log_output = gr.Markdown("Upload photos, then click Reconstruct.")
                reconstruction_output = gr.Model3D(height=700, zoom_speed=0.2, pan_speed=0.2)
                with gr.Row():
                    submit_btn = gr.Button("Reconstruct", variant="primary")
                    update_visual_btn = gr.Button("Update visual")
                    gr.ClearButton([input_images, reconstruction_output, log_output, target_dir_output, image_gallery])
                with gr.Row():
                    mask_threshold = gr.Slider(minimum=0.0, maximum=0.95, value=0.5, step=0.05, label="Object mask threshold (0 = show everything)")
                    max_points_k = gr.Slider(minimum=100, maximum=5000, value=1000, step=100, label="Max points (K)")
                with gr.Row():
                    view_select = gr.Dropdown(choices=[ALL_PHOTOS], value=ALL_PHOTOS, label="Points from")
                    show_cameras = gr.Checkbox(label="Show cameras", value=True)

        viz_inputs = [target_dir_output, mask_threshold, show_cameras, max_points_k, view_select]

        def example_pipeline(images):
            target_dir, paths = handle_uploads(images)
            glb, log = reconstruct(target_dir, 0.5, True, 1000)
            return glb, log, target_dir, paths, gr.update(choices=view_choices(paths), value=ALL_PHOTOS)

        gr.Examples(
            examples=EXAMPLES, inputs=[input_images],
            outputs=[reconstruction_output, log_output, target_dir_output, image_gallery, view_select],
            fn=example_pipeline, cache_examples=False, label="Example: Flying Scotsman, 12 photos from Wikimedia Commons",
        )
        input_images.change(
            fn=update_gallery_on_upload, inputs=[input_images],
            outputs=[reconstruction_output, target_dir_output, image_gallery, log_output, view_select],
        )
        submit_btn.click(fn=lambda: None, outputs=[reconstruction_output]).then(
            fn=lambda: "Reconstructing...", outputs=[log_output]
        ).then(fn=reconstruct, inputs=viz_inputs, outputs=[reconstruction_output, log_output])
        update_visual_btn.click(fn=update_visualization, inputs=viz_inputs, outputs=[reconstruction_output, log_output])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("SPOTTER_CHECKPOINT"))
    ap.add_argument("--server-name", default="0.0.0.0")
    ap.add_argument("--server-port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    get_model(args.checkpoint)
    build_ui().queue(max_size=20).launch(server_name=args.server_name, server_port=args.server_port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
else:  # imported by a Hugging Face Space: app.py is executed as a module and `demo` is launched by the host
    demo = build_ui()
