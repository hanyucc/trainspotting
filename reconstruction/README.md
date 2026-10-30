# Trainspotting reconstruction pipeline

Reconstructs one object from a set of photos and a text prompt naming the object.
The output is the set of photos in which the object is the main subject, with camera intrinsics and poses, verified depth, and an object mask for each photo.

## Installation

The pipeline needs a CUDA GPU, conda, and git.

```bash
cd reconstruction
bash install.sh            # creates the conda environment "trainspotting"
conda activate trainspotting
```

SAM3 weights are gated on the Hugging Face Hub.
Request access at https://huggingface.co/facebook/sam3 and log in with `hf auth login` before the first run.
All other weights are downloaded on first use.

## Usage

```bash
trainspotting-recon --images path/to/photos --prompt "steam locomotive" --output path/to/output
```

The prompt is a short noun phrase for the object, as used for SAM3.
Stages whose outputs already exist in the output directory are skipped, so an interrupted run can be restarted with the same command.

## Stages

1. **Object gating.** SAM3 detects the prompted object in every photo. A photo is kept if its largest detection covers at least 10% of the image.
2. **Matching.** Each photo is paired with its 20 nearest neighbours under MASt3R-SfM retrieval. RoMa v2 matches every pair; a correspondence is kept if it is certain, cycle-consistent, and consistent with a MAGSAC++ fundamental matrix. Nearby match endpoints are merged into shared keypoints.
3. **Structure from motion.** COLMAP verifies the matches and runs incremental mapping. Reconstructed components that share images are merged, and the largest model is undistorted.
4. **View selection.** A view is kept if enough of its points lie on tracks of length at least three, if it lies in the largest cluster of the covisibility graph after spectral bisection, and if it shares points with another kept view.
5. **Multi-view stereo.** CasDiffMVS estimates depth for every selected view from its six most covisible views; pixels below the confidence threshold of any stage are discarded.
6. **Depth verification.** The confident depth is compared with MoGe-2 using the spherical filter of MoGe-2 and ViGeo and a global depth-ratio test.
7. **Object masks.** The SAM3 mask that covers the largest share of confident depth is selected.
8. **Final check.** A photo is kept if at least 5% of its mask has verified depth, the object is the dominant subject, and the focal length is plausible. An object needs at least two such photos.

## Output

```
output/
  summary.json            photos remaining after each stage
  images/, images.json    input photos linked under index names, and the index-to-file mapping
  gating/                 SAM3 detections per photo, gating.json
  matching/               shared keypoints, matches, matching.json
  sfm/                    COLMAP database, mapper output, merged model
  sfm/dense/              undistorted images and model (sparse/)
  view_selection/         valid_images.txt, view_selection.json
  mvs/                    CasDiffMVS depth and confidence per view
  verified_depth/         verified depth per view, verification.json
  masks/                  selected object mask per view, masks.json
  final/
    frames.json           source file, intrinsics, camera_to_world (OpenCV convention) and file names per photo
    images/  depths/  masks/
    sparse/               COLMAP text model of the final photos
```

`sfm/dense/sparse` with `view_selection/valid_images.txt` is the reconstruction from which multi-view stereo starts.
`final/` holds only the photos that pass every check.
Depth maps are float32 arrays of the image size, zero where no depth was verified, in the units of the reconstruction.

## License

The code in this folder is MIT licensed (`LICENSE`), except `trainspotting_recon/vendor/`, which keeps the Apache 2.0 license of its source.
The components below are installed or downloaded separately and keep their own licenses.

## Third-party code

| Component | Source | License |
|---|---|---|
| SAM3 | https://github.com/facebookresearch/sam3 | SAM License |
| MASt3R | https://github.com/naver/mast3r (cloned by `install.sh`) | CC BY-NC-SA 4.0 |
| RoMa v2 | https://github.com/Parskatt/RoMaV2 | MIT |
| COLMAP | https://github.com/colmap/colmap | see upstream |
| CasDiffMVS | https://github.com/cvg/diffmvs (model code in `trainspotting_recon/vendor/casdiffmvs`) | Apache 2.0 |
| MoGe-2 | https://github.com/microsoft/MoGe | see upstream |
| ViGeo spherical filter | https://github.com/aigc3d/ViGeo (in `trainspotting_recon/vendor/vigeo`) | Apache 2.0 |
