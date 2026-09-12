#!/usr/bin/env bash
set -euo pipefail

# Known-pose, held-out-RGB gate for v32g.
#
# Usage:
#   bash freeze_spin/run_v32g_rar_rgb_blind_4dgs.sh /path/to/v32g [workdir]
#
# This runner intentionally has no orbit mode.  The 61-view swivel is unlocked
# only after the real Right Above Rim holdout has been rendered and visually
# accepted.  Camera 2 must never enter training in this script.

DATASET_INPUT="${1:?pass extracted v32g dataset directory}"
WORKROOT="${2:-$PWD/v32g_gpu_work}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${THIS_DIR}/v32f_freeze_static_4dgs_config.py"
EVAL="${THIS_DIR}/evaluate_v32d_holdout.py"
FOURDGS_COMMIT="843d5ac636c37e4b611242287754f3d4ed150144"

# Allow execution from a self-contained artifact as well as from the repo.
if [[ ! -f "$CONFIG" ]]; then CONFIG="$DATASET_INPUT/v32f_freeze_static_4dgs_config.py"; fi
if [[ ! -f "$EVAL" ]]; then EVAL="$DATASET_INPUT/evaluate_v32d_holdout.py"; fi

command -v nvidia-smi >/dev/null || { echo 'CUDA GPU required: nvidia-smi not found' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'CUDA toolkit required: nvcc not found' >&2; exit 2; }
nvidia-smi
nvcc --version

mkdir -p "$WORKROOT"
WORKROOT="$(cd "$WORKROOT" && pwd)"
DATASET_INPUT="$(cd "$DATASET_INPUT" && pwd)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx v32g4dgs; then conda create -y -n v32g4dgs python=3.10; fi
  conda activate v32g4dgs
else
  python3 -m venv "$WORKROOT/venv"
  source "$WORKROOT/venv/bin/activate"
fi
python -m pip install -U pip setuptools wheel ninja
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu117 \
  'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' 'torchaudio==0.13.1'
python -m pip install 'mmcv==1.6.0' matplotlib argparse lpips plyfile pytorch_msssim open3d 'imageio[ffmpeg]' scikit-image opencv-python-headless

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
print('gpu', torch.cuda.get_device_name(0))
PY

DATA="$WORKROOT/data/v32g_holdout"
rm -rf "$DATA"
mkdir -p "$DATA"
cp -a "$DATASET_INPUT"/. "$DATA"/

# Hard provenance gate.  In particular, the canonical train_meta.json must
# already contain only cameras 0 and 1; this runner never swaps in an all-three
# metadata file.
python - <<PY
import json, pathlib, numpy as np
p=pathlib.Path(r'''$DATA''')
q=json.load(open(p/'v32g_provenance_qa.json'))
tr=json.load(open(p/'train_meta.json'))
te=json.load(open(p/'test_meta.json'))
assert q['status']=='PASS_V32G_RAR_RGB_BLIND_PACKAGE'
assert q['heldout_rgb_used_for_initializer'] is False
assert q['heldout_rgb_used_for_training'] is False
assert q['heldout_subject_geometry_used_for_initializer'] is False
assert q['heldout_ball_geometry_used_for_initializer'] is False
assert q['heldout_subject_colour_used_for_initializer'] is False
assert q['heldout_camera_pose_used_for_evaluation_only'] is True
assert tr['cam_id']==[[0,1]], tr['cam_id']
assert te['cam_id']==[[2]], te['cam_id']
assert all('Right_Above_Rim' not in fn for fn in tr['fn'][0])
cloud=np.load(p/'init_pt_cld.npz')['data']
assert cloud.ndim==2 and cloud.shape[1]==6 and len(cloud)>=5000
assert np.isfinite(cloud).all()
print('V32G_RAR_RGB_BLIND_INPUT_CONTRACT_OK', cloud.shape)
PY

EXP="v32g_rar_rgb_blind_holdout"
rm -rf "output/$EXP"
python train.py -s "$DATA" --expname "$EXP" --configs "$CONFIG" \
  --test_iterations 1000 3000 6000 8000 --save_iterations 1000 3000 6000 8000 \
  --checkpoint_iterations 1000 3000 6000 8000 2>&1 | tee "$WORKROOT/v32g_holdout_train.log"

python render.py --model_path "output/$EXP" --skip_train --skip_video --configs "$CONFIG" \
  2>&1 | tee "$WORKROOT/v32g_holdout_render.log"

LATEST="$(find "output/$EXP/test" -maxdepth 1 -type d -name 'ours_*' | sort -V | tail -n1)"
RENDER="$LATEST/renders/00000.png"
GT="$LATEST/gt/00000.png"
test -s "$RENDER" && test -s "$GT"

QA="$WORKROOT/v32g_rar_rgb_blind_holdout_qa"
rm -rf "$QA"
mkdir -p "$QA"
python "$EVAL" --render "$RENDER" --gt "$GT" --out "$QA"
cp "$RENDER" "$QA/v32g_rar_holdout_render_native.png"
cp "$GT" "$QA/v32g_rar_holdout_real_gt_native.png"
cp "$DATA/v32g_provenance_qa.json" "$QA/"
cp "$WORKROOT/v32g_holdout_train.log" "$WORKROOT/v32g_holdout_render.log" "$QA/"

# Project convention: user-facing inspection output is deterministic UHD.  This
# is presentation scaling only; native render/GT are preserved beside it.
python - <<PY
from pathlib import Path
import cv2, numpy as np
p=Path(r'''$QA''')
src=cv2.imread(str(p/'v32d_holdout_freeze_compare.png'), cv2.IMREAD_COLOR)
if src is None: raise SystemExit('comparison panel missing')
h,w=src.shape[:2]
scale=min(3840.0/w,2160.0/h)
nw,nh=int(round(w*scale)),int(round(h*scale))
up=cv2.resize(src,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
canvas=np.zeros((2160,3840,3),dtype=np.uint8)
x=(3840-nw)//2; y=(2160-nh)//2
canvas[y:y+nh,x:x+nw]=up
cv2.imwrite(str(p/'v32g_rar_holdout_compare_UHD2160p.png'),canvas)
PY

python - <<PY
import json, pathlib
p=pathlib.Path(r'''$QA''')
q=json.load(open(p/'v32d_holdout_freeze_qa.json'))
status={
 'status':'GPU_HOLDOUT_COMPLETE_VISUAL_QA_REQUIRED',
 'numeric_diagnostics':q,
 'orbit_unlocked':False,
 'reason':'Visual identity/anatomy QA remains authoritative; do not render the 61-view orbit until explicitly accepted.'
}
(p/'v32g_gate_status.json').write_text(json.dumps(status,indent=2))
print(json.dumps(status,indent=2))
PY

(cd "$WORKROOT" && zip -0 -r v32g_rar_rgb_blind_holdout_qa.zip v32g_rar_rgb_blind_holdout_qa >/dev/null)
printf '\nV32G_HOLDOUT_COMPLETE_VISUAL_QA_REQUIRED\nQA=%s\nZIP=%s\n' "$QA" "$WORKROOT/v32g_rar_rgb_blind_holdout_qa.zip"
