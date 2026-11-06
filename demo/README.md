# Demo

Gradio demo of Spotter. Upload photos of one object; the app shows the object's point cloud with the predicted
cameras, filtered by the predicted object mask.

```bash
pip install -e ".[demo]"
python demo/app.py                              # downloads the weights from the Hub
python demo/app.py --checkpoint spotter.safetensors
```

The hosted version at https://huggingface.co/spaces/hanyucc/spotter-demo runs this folder on ZeroGPU with a
copy of the `spotter` package. `SPOTTER_MAX_VIEWS` (default 24) caps the number of photos per reconstruction.
