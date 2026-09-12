#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash freeze_spin/run_v32f_freeze_static_4dgs.sh holdout /path/to/v32f [workdir]
#   bash freeze_spin/run_v32f_freeze_static_4dgs.sh orbit   /path/to/v32f [workdir]
#
# HOLDOUT is the mandatory gate: train only Left Above Rim + Broadcast and render
# the real synchronized Right Above Rim image.  ORBIT trains all three real t+00
# views and renders the 61 calibrated 0..25 degree virtual cameras.  It should be
# used only after holdout visual QA is acceptable.

MODE="${1:?mode must be holdout or orbit}"
DATASET_INPUT="${2:?pass extracted v32f dataset directory}"
WORKROOT="${3:-$PWD/v32f_gpu_work}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${THIS_DIR}/v32f_freeze_static_4dgs_config.py"
EVAL="${THIS_DIR}/evaluate_v32d_holdout.py"
FOURDGS_COMMIT="843d5ac636c37e4b611242287754f3d4ed150144"

case "$MODE" in holdout|orbit) ;; *) echo 'mode must be holdout or orbit' >&2; exit 2;; esac
command -v nvidia-smi >/dev/null || { echo 'CUDA GPU required: nvidia-smi not found' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'CUDA toolkit required: nvcc not found' >&2; exit 2; }
nvidia-smi
nvcc --version

mkdir -p "$WORKROOT"
WORKROOT="$(cd "$WORKROOT" && pwd)"
DATASET_INPUT="$(cd "$DATASET_INPUT" && pwd)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx v32f4dgs; then conda create -y -n v32f4dgs python=3.10; fi
  conda activate v32f4dgs
else
  python3 -m venv "$WORKROOT/venv"; source "$WORKROOT/venv/bin/activate"
fi
python -m pip install -U pip setuptools wheel ninja
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' 'torchaudio==0.13.1'
python -m pip install 'mmcv==1.6.0' matplotlib argparse lpips plyfile pytorch_msssim open3d 'imageio[ffmpeg]' scikit-image

if [[ ! -d "$WORKROOT/4DGaussians/.git" ]]; then git clone https://github.com/hustvl/4DGaussians.git "$WORKROOT/4DGaussians"; fi
cd "$WORKROOT/4DGaussians"
git fetch origin "$FOURDGS_COMMIT" --depth=1
git checkout --detach "$FOURDGS_COMMIT"
git submodule update --init --recursive
python -m pip install -e submodules/depth-diff-gaussian-rasterization
python -m pip install -e submodules/simple-knn
python - <<'PY'
import torch
print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable after PyTorch install')
print('gpu',torch.cuda.get_device_name(0))
PY

DATA="$WORKROOT/data/v32f_${MODE}"
rm -rf "$DATA"; mkdir -p "$DATA"; cp -a "$DATASET_INPUT"/. "$DATA"/
python - <<PY
import json,pathlib
p=pathlib.Path(r'''$DATA''')
q=json.load(open(p/'v32f_freeze_only_qa.json')); d=json.load(open(p/'v32f_dense_orbit_qa.json'))
assert q['status']=='PASS_V32F_FREEZE_ONLY_DATASET_AND_ORBIT'
assert d['status']=='PASS_V32F_DENSE_61_VIEW_ORBIT'
assert q['real_training_images']==3 and q['time_steps']==1
assert q['generated_rgb'] is False and d['generated_rgb'] is False
print('V32F_INPUT_CONTRACT_OK')
PY

EXP="v32f_${MODE}"
rm -rf "output/$EXP"

if [[ "$MODE" == holdout ]]; then
  cp "$DATA/train_meta_holdout_right_above_rim.json" "$DATA/train_meta.json"
  cp "$DATA/test_meta_holdout_right_above_rim.json" "$DATA/test_meta.json"
  python - <<PY
import json,pathlib
p=pathlib.Path(r'''$DATA'''); tr=json.load(open(p/'train_meta.json')); te=json.load(open(p/'test_meta.json'))
assert len(tr['fn'])==1 and tr['cam_id']==[[0,1]]
assert len(te['fn'])==1 and te['cam_id']==[[2]]
print('V32F_RAR_HOLDOUT_CONTRACT_OK')
PY
  python train.py -s "$DATA" --expname "$EXP" --configs "$CONFIG" \
    --test_iterations 1000 3000 6000 8000 --save_iterations 1000 3000 6000 8000 \
    --checkpoint_iterations 1000 3000 6000 8000 2>&1 | tee "$WORKROOT/v32f_holdout_train.log"
  python render.py --model_path "output/$EXP" --skip_train --skip_video --configs "$CONFIG" \
    2>&1 | tee "$WORKROOT/v32f_holdout_render.log"
  LATEST="$(find "output/$EXP/test" -maxdepth 1 -type d -name 'ours_*' | sort -V | tail -n1)"
  RENDER="$LATEST/renders/00000.png"; GT="$LATEST/gt/00000.png"
  test -s "$RENDER" && test -s "$GT"
  QA="$WORKROOT/v32f_rar_holdout_qa"; rm -rf "$QA"; mkdir -p "$QA"
  python "$EVAL" --render "$RENDER" --gt "$GT" --out "$QA"
  cp "$RENDER" "$QA/v32f_rar_holdout_freeze_render.png"; cp "$GT" "$QA/v32f_rar_holdout_real_ground_truth.png"
  cp "$WORKROOT/v32f_holdout_train.log" "$WORKROOT/v32f_holdout_render.log" "$QA/"
  (cd "$WORKROOT" && zip -0 -r v32f_rar_holdout_qa.zip v32f_rar_holdout_qa >/dev/null)
  printf '\nV32F_HOLDOUT_COMPLETE\nQA=%s\n' "$QA"
else
  # Train with all three real source views. Keep real-source test metadata during
  # optimisation; swap to the dense virtual-camera file only for rendering.
  python - <<PY
import json,pathlib
p=pathlib.Path(r'''$DATA'''); tr=json.load(open(p/'train_meta.json'))
assert len(tr['fn'])==1 and tr['cam_id']==[[0,1,2]]
print('V32F_ALL_THREE_TRAIN_CONTRACT_OK')
PY
  cp "$DATA/train_meta.json" "$DATA/test_meta.json"
  python train.py -s "$DATA" --expname "$EXP" --configs "$CONFIG" \
    --test_iterations 1000 3000 6000 8000 --save_iterations 1000 3000 6000 8000 \
    --checkpoint_iterations 1000 3000 6000 8000 2>&1 | tee "$WORKROOT/v32f_orbit_train.log"
  cp "$DATA/test_meta_orbit_dense_0_25.json" "$DATA/test_meta.json"
  python render.py --model_path "output/$EXP" --skip_train --skip_video --configs "$CONFIG" \
    2>&1 | tee "$WORKROOT/v32f_orbit_render.log"
  LATEST="$(find "output/$EXP/test" -maxdepth 1 -type d -name 'ours_*' | sort -V | tail -n1)"
  test -d "$LATEST/renders"
  test "$(find "$LATEST/renders" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 61
  OUT="$WORKROOT/v32f_dense_orbit"; rm -rf "$OUT"; mkdir -p "$OUT/frames"
  cp "$LATEST/renders"/*.png "$OUT/frames/"
  # 61 genuine rendered viewpoints at 30 fps = 2.03 s. No optical-flow or frame
  # interpolation is used. The 2160p file is deterministic scaling only.
  ffmpeg -y -v error -framerate 30 -i "$OUT/frames/%05d.png" -c:v libx264 -crf 15 -preset medium -pix_fmt yuv420p -movflags +faststart "$OUT/v32f_freeze_swivel_native.mp4"
  ffmpeg -y -v error -i "$OUT/v32f_freeze_swivel_native.mp4" \
    -vf "scale=-2:2160:flags=lanczos,pad=3840:2160:(ow-iw)/2:0:black" \
    -c:v libx264 -crf 15 -preset medium -pix_fmt yuv420p -movflags +faststart "$OUT/v32f_freeze_swivel_UHD2160p.mp4"
  cp "$DATA/v32f_freeze_only_qa.json" "$DATA/v32f_dense_orbit_qa.json" "$OUT/"
  cp "$WORKROOT/v32f_orbit_train.log" "$WORKROOT/v32f_orbit_render.log" "$OUT/"
  (cd "$WORKROOT" && zip -0 -r v32f_dense_orbit.zip v32f_dense_orbit >/dev/null)
  printf '\nV32F_ORBIT_COMPLETE\nOUTPUT=%s\n' "$OUT"
fi
