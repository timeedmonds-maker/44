from __future__ import annotations

"""v15 clean-static composite experiment for the native three-camera proof.

v14 increased coverage but MoGe crowd surfels visibly sheared the stands. Earlier
versions also showed a long white player-shaped streak on the court; geometric
inspection proved that streak lies outside every reconstructed player-mesh bbox,
so it is source-player contamination of a nominally static floor/background
layer, not an articulated-mesh failure.

v15 therefore keeps the solved v11 state, camera geometry, anatomical meshes,
mask-safe player texturing and three-view ball, but changes static compositing:

- segmentation returns the same near-play instances used for 3-D association,
  while its static-exclusion mask covers *all* confidently detected on-court
  people from the wider v5 court bounds and is dilated for source-edge safety;
- distant arena uses calibrated multi-camera infinity-ray reprojection with hard
  source ownership, but floor/backboard/rim and all on-court people are excluded
  before sampling; no learned crowd depth is used;
- metric floor/backboard/rim remain the only geometry for those rigid surfaces;
- native 960x540 only, no generated texture, no upscale/UHD.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v14 as v14


_ORIGINAL_DETECT_ONCOURT = v12.v5.detect_oncourt


def detect_near_play_with_broad_static_mask(model, image, K, R, C):
    full_dyn, instances, balls = _ORIGINAL_DETECT_ONCOURT(model, image, K, R, C)
    keep = []
    for inst in instances:
        x, y, _ = inst["foot_world_cm"]
        if -250.0 <= float(x) <= 1150.0 and abs(float(y)) <= 520.0:
            keep.append(inst)

    dyn = full_dyn.astype(np.uint8)
    dyn = cv2.dilate(dyn, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    for b in balls[:3]:
        x1, y1, x2, y2 = [int(round(v)) for v in b["box"]]
        x1 = max(0, x1 - 7); y1 = max(0, y1 - 7)
        x2 = min(v12.W - 1, x2 + 7); y2 = min(v12.H - 1, y2 + 7)
        dyn[y1:y2 + 1, x1:x2 + 1] = True
    return dyn, keep, balls


def far_background_multisource_clean(Kt, Rt, cams, images, dynamic_masks):
    if v13._is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return images[v12.A].copy(), np.ones((v12.H, v12.W), bool)

    yy, xx = np.indices((v12.H, v12.W), np.float64)
    hp = np.stack([xx.ravel(), yy.ravel(), np.ones(v12.H * v12.W)], axis=0)
    dcam = np.linalg.inv(Kt) @ hp
    dw = Rt.T @ dcam

    target_axis = v12.RIM - v12.CURRENT_CT
    target_axis /= max(1e-9, float(np.linalg.norm(target_axis)))
    pref = []
    for label in v12.CAMERAS:
        a = v12.RIM - cams[label][0]
        a /= max(1e-9, float(np.linalg.norm(a)))
        pref.append((float(np.dot(a, target_axis)), label))
    pref.sort(reverse=True)

    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    filled = np.zeros(v12.H * v12.W, bool)
    for _, label in pref:
        Cc, Rc, Kc = cams[label]
        ds = Rc @ dw
        q = Kc @ ds
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (q[:2] / q[2:3]).T
        sign = float(v12.v3.forward_sign(Rc, Cc))
        valid = (
            np.isfinite(uv).all(axis=1)
            & (sign * ds[2] > 1e-6)
            & (uv[:, 0] >= 0) & (uv[:, 0] < v12.W - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] < v12.H - 1)
        )
        if not np.any(valid):
            continue
        cols = v12.v8.bilinear_sample(images[label], uv)
        ui = np.clip(np.rint(uv[:, 0]).astype(np.int32), 0, v12.W - 1)
        vi = np.clip(np.rint(uv[:, 1]).astype(np.int32), 0, v12.H - 1)
        excluded = v13._static_exclusion(label, cams)
        ids = np.where(valid)[0]
        if len(ids):
            valid[ids] &= ~dynamic_masks[label][vi[ids], ui[ids]]
            valid[ids] &= ~excluded[vi[ids], ui[ids]]
        take = valid & (~filled)
        out[take] = cols[take]
        filled[take] = True
    return out.reshape(v12.H, v12.W, 3), filled.reshape(v12.H, v12.W)


def main():
    v12.v6.detect_near_play = detect_near_play_with_broad_static_mask
    v14.far_background_static_surfel = far_background_multisource_clean

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v14.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v15_renderer"] = {
            "resolution": [960, 540],
            "anchor_0deg": "exact Left Above Rim source frame",
            "dynamic_exclusion": "all Mask R-CNN on-court instances under broad v5 court bounds, dilated 9px; near-play subset retained for 3-D identity association",
            "background": "calibrated multi-camera infinity-ray source reprojection with hard ownership; metric floor/backboard/rim and on-court people excluded; no MoGe crowd surfels",
            "players": "v14 mask-safe perspective-correct real-pixel anatomical meshes",
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V15_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
