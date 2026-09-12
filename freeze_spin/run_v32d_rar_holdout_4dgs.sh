#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash freeze_spin/run_v32d_rar_holdout_4dgs.sh /path/to/v32d_action_crops [workdir]
#
# The decisive first GPU test trains on Left Above Rim + Broadcast and asks the
# learned dynamic Gaussian scene to reproduce the *real* Right Above Rim camera.
# Relative frame 0 is item index 6 in the synchronized -6..+6 sequence.

DATASET_INPUT="${1:?pass extracted v32d action-crop dataset directory}"
WORKROOT="${2:-$PWD/v32d_gpu_work}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${THIS_DIR}/v32d_4dgs_quality_config.py"
EVAL="${THIS_DIR}/evaluate_v32d_holdout.py"
FOURDGS_COMMIT="843d5ac636c37e4b611242287754f3d4ed150144"

command -v nvidia-smi >/dev/null || { echo 'CUDA GPU required: nvidia-smi not found' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'CUDA toolkit required: nvcc not found' >&2; exit 2; }
nvidia-smi
nvcc --version

mkdir -p "$WORKROOT"
WORKROOT="$(cd "$WORKROOT" && pwd)"
DATASET_INPUT="$(cd "$DATASET_INPUT" && pwd)"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx v32d4dgs; then
    conda create -y -n v32d4dgs python=3.10
  fi
  conda activate v32d4dgs
else
  python3 -m venv "$WORKROOT/venv"
  # shellcheck disable=SC1091
  source "$WORKROOT/venv/bin/activate"
fi

python -m pip install -U pip setuptools wheel ninja
# Match the reference 4DGaussians environment while using a CUDA wheel that is
# compatible with modern NVIDIA drivers.
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' 'torchaudio==0.13.1'
python -m pip install 'mmcv==1.6.0' matplotlib argparse lpips plyfile pytorch_msssim open3d 'imageio[ffmpeg]' scikit-image

if [[ ! -d "$WORKROOT/4DGaussians/.git" ]]; then
  git clone https://github.com/hustvl/4DGaussians.git "$WORKROOT/4DGaussians"
fi
cd "$WORKROOT/4DGaussians"
git fetch origin "$FOURDGS_COMMIT" --depth=1
git checkout --detach "$FOURDGS_COMMIT"
git submodule update --init --recursive
python -m pip install -e submodules/depth-diff-gaussian-rasterization
python -m pip install -e submodules/simple-knn

python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable after PyTorch install')
print(torch.cuda.get_device_name(0))
PY

DATA="$WORKROOT/data/v32d_rar_holdout"
rm -rf "$DATA"
mkdir -p "$DATA"
cp -a "$DATASET_INPUT"/. "$DATA"/
cp "$DATA/train_meta_holdout_right_above_rim.json" "$DATA/train_meta.json"
cp "$DATA/test_meta_holdout_right_above_rim.json" "$DATA/test_meta.json"

# Guard the exact held-out contract: 13 timestamps, two input cameras, one real
# target camera; no resize or upscale.
python - <<PY
import json, pathlib
p=pathlib.Path(r'''$DATA''')
b=json.load(open(p/'v32d_backend.json'))
tr=json.load(open(p/'train_meta.json')); te=json.load(open(p/'test_meta.json'))
assert b['training_crop_resolution']==[384,448]
assert b['native_pixels_preserved'] is True and b['resized'] is False
assert len(tr['fn'])==13 and all(len(x)==2 for x in tr['fn'])
assert len(te['fn'])==13 and all(len(x)==1 for x in te['fn'])
assert all(x==[2] for x in te['cam_id'])
print('RAR_HOLDOUT_DATA_CONTRACT_OK')
PY

rm -rf output/v32d_rar_holdout
python train.py \
  -s "$DATA" \
  --expname v32d_rar_holdout \
  --configs "$CONFIG" \
  --test_iterations 1500 4000 8000 12000 \
  --save_iterations 1500 4000 8000 12000 \
  --checkpoint_iterations 1500 4000 8000 12000 \
  2>&1 | tee "$WORKROOT/v32d_rar_holdout_train.log"

python render.py \
  --model_path output/v32d_rar_holdout \
  --skip_train --skip_video \
  --configs "$CONFIG" \
  2>&1 | tee "$WORKROOT/v32d_rar_holdout_render.log"

LATEST="$(find output/v32d_rar_holdout/test -maxdepth 1 -type d -name 'ours_*' | sort -V | tail -n1)"
test -n "$LATEST"
RENDER="$LATEST/renders/00006.png"
GT="$LATEST/gt/00006.png"
test -s "$RENDER" && test -s "$GT"

QA="$WORKROOT/v32d_rar_holdout_qa"
rm -rf "$QA"; mkdir -p "$QA"
python "$EVAL" --render "$RENDER" --gt "$GT" --out "$QA"
cp "$RENDER" "$QA/v32d_rar_holdout_freeze_render.png"
cp "$GT" "$QA/v32d_rar_holdout_freeze_ground_truth.png"
cp "$WORKROOT/v32d_rar_holdout_train.log" "$QA/"
cp "$WORKROOT/v32d_rar_holdout_render.log" "$QA/"
cp "$DATA/v32d_backend.json" "$QA/"

cd "$WORKROOT"
zip -0 -r v32d_rar_holdout_qa.zip v32d_rar_holdout_qa >/dev/null
printf '\nGPU_HELDOUT_COMPLETE\nQA_DIR=%s\nQA_ZIP=%s\n' "$QA" "$WORKROOT/v32d_rar_holdout_qa.zip"
