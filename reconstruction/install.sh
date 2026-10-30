#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-trainspotting}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAST3R_COMMIT=f5209afc300cec36239a7ac992263f36847bbba0

ASMK_COMMIT=2a96d9c03a841dffdfddabc699a20512dcd09363

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c conda-forge python=3.11 colmap=4.0.3 "libfaiss=1.10.0=cpu_openblas_*" "setuptools<81"
fi
run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

run pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
run pip install cython "numpy<2" faiss-gpu-cu12==1.12.0

if [ ! -d "$ROOT/third_party/asmk" ]; then
  git clone https://github.com/jenicek/asmk "$ROOT/third_party/asmk"
  git -C "$ROOT/third_party/asmk" checkout "$ASMK_COMMIT"
fi
(cd "$ROOT/third_party/asmk" && run cython -3 cython/hamming.pyx)
run pip install --no-build-isolation "$ROOT/third_party/asmk"
run pip install -r "$ROOT/requirements.txt"

if [ ! -d "$ROOT/third_party/mast3r" ]; then
  git clone https://github.com/naver/mast3r "$ROOT/third_party/mast3r"
  git -C "$ROOT/third_party/mast3r" checkout "$MAST3R_COMMIT"
  git -C "$ROOT/third_party/mast3r" submodule update --init --recursive
fi

run pip install -e "$ROOT"
SITE="$(run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/third_party/mast3r" > "$SITE/trainspotting_mast3r.pth"
