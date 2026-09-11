from __future__ import annotations

"""v25: source-grounded temporal clean plates for static disocclusions.

v24 combined the corrected v23 focal player with the persistent v21 metric court
atlas, but visual QA still shows large black holes. A substantial class of these
holes comes from the correct decision to exclude on-court people from the static
source layers: when the exact-state frame is occluded, the renderer has no static
RGB behind that person.

For static arena/court appearance only, v25 uses nearby frames from the *same
official physical camera clip*. Each candidate frame is registered back to the
exact-state image with a small RANSAC homography. For pixels hidden by the exact
state's broad human mask, the clean plate selects an actual RGB pixel from the
nearest registered temporal candidates by robust medoid selection; it never uses
the temporal median itself as output. Thus every filled RGB value descends from
one real official source frame. The exact 0-degree output remains the untouched
exact-state Left-Above-Rim frame.

The clean plate is used only by static background and metric-floor appearance.
Player meshes, focal #12 geometry/texture, ball geometry, exact-state timing and
camera solves remain v23/v24. Native 960x540 only; no UHD, upscale, inpainting or
generated texture.
"""

import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v21 as v21
from freeze_spin import build_three_camera_mesh_v24 as v24


SOURCE_RE = re.compile(r"_489_(.+)_SOURCE\.mp4$")
TEMPORAL_OFFSETS = (-75, -50, -25, 25, 50, 75)
_CLIPS_DIR: Path | None = None
_CLEAN = {}
_CLEAN_QA = {}
_ORIG_V15_BG = v15.far_background_multisource_clean
_ORIG_V21_SOURCE_ATLAS = v21._source_atlas


def _label_for_image(image):
    for label, src in v12.IMAGES.items():
        if image is src:
            return label
    return None


def _clip_map(root: Path):
    out = {}
    for p in sorted(root.glob("*_489_*_SOURCE.mp4")):
        m = SOURCE_RE.search(p.name)
        if not m:
            continue
        label = m.group(1).replace("_", " ")
        if label in v12.CAMERAS:
            out[label] = p
    missing = [x for x in v12.CAMERAS if x not in out]
    if missing:
        raise RuntimeError(f"v25 missing temporal source clips: {missing}; found={list(out)}")
    return out


def _decode_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"v25 cannot open {path}")
    n = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if idx < 0 or (n > 0 and idx >= n):
        cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, im = cap.read()
    actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
    cap.release()
    if not ok or actual != int(idx) or im is None or im.shape[:2] != (v12.H, v12.W):
        return None
    return im


def _small_homography(candidate, anchor, anchor_dynamic):
    ga = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    mask = (~cv2.dilate(anchor_dynamic.astype(np.uint8), np.ones((13, 13), np.uint8), iterations=1).astype(bool)).astype(np.uint8) * 255
    sift = cv2.SIFT_create(nfeatures=5500, contrastThreshold=0.018, edgeThreshold=12, sigma=1.4)
    kc, dc = sift.detectAndCompute(gc, mask)
    ka, da = sift.detectAndCompute(ga, mask)
    qa = {"candidate_keypoints": int(len(kc)), "anchor_keypoints": int(len(ka)), "method": "identity_fallback"}
    if dc is None or da is None or len(kc) < 16 or len(ka) < 16:
        qa["reason"] = "insufficient_features"
        return np.eye(3, dtype=np.float64), qa
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for row in bf.knnMatch(dc, da, k=2):
        if len(row) < 2:
            continue
        a, b = row
        if a.distance < 0.74 * b.distance:
            good.append(a)
    qa["ratio_matches"] = int(len(good))
    if len(good) < 14:
        qa["reason"] = "insufficient_ratio_matches"
        return np.eye(3, dtype=np.float64), qa
    src = np.asarray([kc[m.queryIdx].pt for m in good], np.float32).reshape(-1, 1, 2)
    dst = np.asarray([ka[m.trainIdx].pt for m in good], np.float32).reshape(-1, 1, 2)
    H, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=7000, confidence=0.997)
    if H is None or not np.isfinite(H).all():
        qa["reason"] = "homography_failed"
        return np.eye(3, dtype=np.float64), qa
    H = H / H[2, 2]
    inlier = inlier.reshape(-1).astype(bool) if inlier is not None else np.zeros(len(good), bool)
    control = np.asarray([[0,0],[v12.W-1,0],[0,v12.H-1],[v12.W-1,v12.H-1],[(v12.W-1)/2,(v12.H-1)/2]], np.float32).reshape(-1,1,2)
    mapped = cv2.perspectiveTransform(control, H).reshape(-1,2)
    disp = np.linalg.norm(mapped-control.reshape(-1,2),axis=1)
    qa.update({
        "ransac_inliers": int(inlier.sum()),
        "ransac_inlier_fraction": float(inlier.mean()) if len(inlier) else 0.0,
        "control_median_displacement_px": float(np.median(disp)),
        "control_max_displacement_px": float(np.max(disp)),
    })
    plausible = int(inlier.sum()) >= 12 and float(inlier.mean()) >= 0.25 and float(np.median(disp)) <= 28.0 and float(np.max(disp)) <= 75.0
    if not plausible:
        qa["reason"] = "homography_rejected"
        return np.eye(3, dtype=np.float64), qa
    qa["method"] = "sift_ransac_temporal_registration"; qa["reason"] = "accepted"; qa["homography"] = H.tolist()
    return H, qa


def _clean_plate(label, cams, images, dynamic_masks):
    if label in _CLEAN:
        return _CLEAN[label]
    if _CLIPS_DIR is None:
        raise RuntimeError("v25 clips directory was not configured")
    clip = _clip_map(_CLIPS_DIR)[label]
    sync_qa = json.loads(Path(_SYNC_QA).read_text())
    center = int(sync_qa["selected"]["frames"][label])
    anchor = images[label]
    dynamic = dynamic_masks[label].astype(bool)
    rows = []
    warped = []
    valids = []
    used_offsets = []
    for off in TEMPORAL_OFFSETS:
        idx = center + int(off)
        cand = _decode_frame(clip, idx)
        if cand is None:
            rows.append({"offset": int(off), "frame": int(idx), "status": "decode_failed"})
            continue
        H, hq = _small_homography(cand, anchor, dynamic)
        wi = cv2.warpPerspective(cand, H, (v12.W, v12.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        vm = cv2.warpPerspective(np.ones((v12.H,v12.W),np.uint8)*255, H, (v12.W,v12.H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        warped.append(wi); valids.append(vm); used_offsets.append(int(off))
        rows.append({"offset": int(off), "frame": int(idx), "status": "used", "registration": hq, "valid_fraction": float(vm.mean())})
    if len(warped) < 3:
        raise RuntimeError(f"v25 only {len(warped)} temporal frames decoded for {label}")

    stack = np.stack(warped, axis=0)
    vstack = np.stack(valids, axis=0)
    hole = cv2.dilate(dynamic.astype(np.uint8), np.ones((9,9),np.uint8), iterations=1) > 0
    count = vstack.sum(axis=0)
    fillable = hole & (count >= 3)
    out = anchor.copy()

    # Robust center is used only to choose which *actual source frame pixel* wins.
    # Invalid candidates receive NaN and cannot be selected.
    arr = stack.astype(np.float32)
    arr[~vstack[...,None].repeat(3,axis=3)] = np.nan
    med = np.nanmedian(arr, axis=0)
    dist = np.sum(np.square(arr - med[None,...]), axis=3)
    dist[~vstack] = np.inf
    winner = np.argmin(dist, axis=0)
    yy, xx = np.where(fillable)
    out[yy,xx] = stack[winner[yy,xx],yy,xx]

    clean_valid = fillable.copy()
    _CLEAN[label] = (out, clean_valid)
    _CLEAN_QA[label] = {
        "center_frame": center,
        "source_clip": clip.name,
        "temporal_offsets_requested": list(TEMPORAL_OFFSETS),
        "temporal_offsets_used": used_offsets,
        "candidate_frames": rows,
        "selected_dynamic_pixels": int(dynamic.sum()),
        "dilated_dynamic_pixels": int(hole.sum()),
        "clean_plate_filled_pixels": int(clean_valid.sum()),
        "clean_plate_fill_fraction_of_dilated_dynamic": float(clean_valid.sum()/max(1,int(hole.sum()))),
        "output_pixel_policy": "each filled RGB pixel is copied from one registered real temporal source frame; temporal median is selection-only",
    }
    try:
        oi=sys.argv.index("--out"); od=Path(sys.argv[oi+1]); od.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(od/f"v25_clean_plate_{label.replace(' ','_')}.png"),out)
        cv2.imwrite(str(od/f"v25_clean_valid_{label.replace(' ','_')}.png"),clean_valid.astype(np.uint8)*255)
    except Exception:
        pass
    return _CLEAN[label]


def far_background_temporal_clean(Kt, Rt, cams, images, dynamic_masks):
    if v13._is_anchor_pose(cams, Rt, v12.CURRENT_CT):
        return images[v12.A].copy(), np.ones((v12.H, v12.W), bool)
    clean_images = {}; remaining = {}
    for label in v12.CAMERAS:
        plate, valid = _clean_plate(label, cams, images, dynamic_masks)
        clean_images[label] = plate
        remaining[label] = dynamic_masks[label].astype(bool) & (~valid)
    return _ORIG_V15_BG(Kt, Rt, cams, clean_images, remaining)


def source_atlas_temporal_clean(src):
    label = _label_for_image(src["image"])
    if label is None:
        return _ORIG_V21_SOURCE_ATLAS(src)
    cams = {k:(None,None,None) for k in v12.CAMERAS}
    # Reconstruct the camera tuple dictionary from the source rows already built
    # by v12; only this label is needed by _clean_plate, but the helper accepts the
    # common dictionary shape.
    for k in v12.CAMERAS:
        if k == label:
            cams[k] = (src["C"],src["R"],src["K"])
    plate, clean_valid = _CLEAN.get(label, (None,None))
    if plate is None:
        # By normal render order the background creates all plates first. Keep a
        # defensive fallback to the exact source if atlas construction is called first.
        return _ORIG_V21_SOURCE_ATLAS(src)

    s = dict(src)
    s["image"] = plate
    fv = src["floor_vis"].astype(bool).copy()
    _t, Pw = v12.v4.ray_plane_map(src["K"], src["R"], src["C"], "floor")
    court = (
        np.isfinite(Pw).all(axis=2)
        & (Pw[:,:,0] >= float(v12.v4.COURT_X0)) & (Pw[:,:,0] <= float(v12.v4.COURT_X1))
        & (Pw[:,:,1] >= float(v12.v4.COURT_Y0)) & (Pw[:,:,1] <= float(v12.v4.COURT_Y1))
    )
    augmented = clean_valid & src["dynamic"].astype(bool) & court
    fv |= augmented
    s["floor_vis"] = fv
    if label in _CLEAN_QA:
        _CLEAN_QA[label]["floor_visibility_augmented_pixels"] = int(augmented.sum())
        _CLEAN_QA[label]["floor_visible_pixels_after_augmentation"] = int(fv.sum())
    return _ORIG_V21_SOURCE_ATLAS(s)


def _frame_metrics(out: Path):
    rows=[]
    for ang in (0,5,10,15,20,25):
        im=cv2.imread(str(out/f"v12_{ang:02d}deg.png"),cv2.IMREAD_COLOR)
        if im is None: continue
        black=np.all(im==0,axis=2)
        rows.append({"angle_deg":ang,"black_fraction":float(black.mean()),"nonblack_fraction":float((~black).mean())})
    return rows


def _pop_arg(name: str) -> str:
    i=sys.argv.index(name); value=sys.argv[i+1]; del sys.argv[i:i+2]; return value


_SYNC_QA = ""


def main():
    global _CLIPS_DIR, _SYNC_QA
    _CLEAN.clear(); _CLEAN_QA.clear()
    _CLIPS_DIR=Path(_pop_arg("--clips-dir"))
    si=sys.argv.index("--sync-qa"); _SYNC_QA=sys.argv[si+1]

    v15.far_background_multisource_clean = far_background_temporal_clean
    v21._source_atlas = source_atlas_temporal_clean

    assert (int(v12.W),int(v12.H))==(960,540)
    v24.main()

    oi=sys.argv.index("--out"); out=Path(sys.argv[oi+1])
    qp=out/"three_camera_mesh_v12_qa.json"; q=json.loads(qp.read_text())
    q["v25_temporal_clean_plate"]={
        "resolution":[960,540],
        "clean_plate_qa":_CLEAN_QA,
        "frame_black_metrics":_frame_metrics(out),
        "anchor_0deg":"untouched exact-state Left Above Rim source frame",
        "static_fill":"registered nearby frames from the same official physical camera; actual source RGB medoid selection only",
        "foreground_geometry_change":False,
        "ball_geometry_change":False,
        "generated_texture":False,
        "inpainting":False,
        "upscale":False,
        "uhd":False,
    }
    qp.write_text(json.dumps(q,indent=2))
    print(json.dumps(q["v25_temporal_clean_plate"],indent=2),flush=True)


if __name__=="__main__":
    main()
