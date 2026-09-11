from __future__ import annotations

"""v17: cross-camera photometric calibration for metric floor reprojection.

v16 removed additional human pixels from static sampling, but visual QA showed the
long bright court streak largely persisted. The streak follows disocclusion regions
where Left Above Rim cannot supply the floor and Broadcast/Right Above Rim fill the
same metric floor with noticeably different exposure/white balance. That is a
photometric seam, not a geometry or player-mesh failure.

v17 keeps the solved three-camera geometry, v16 semantic exclusion, anatomical
meshes and three-view ball unchanged. Before any virtual floor is rendered it fits
a robust per-channel affine colour transform from each secondary camera into the
Left Above Rim colour space using thousands of mutually visible metric floor
correspondences. Secondary-camera floor pixels are then colour-normalized before
hard ownership compositing.

Native 960x540 only. No generated texture, inpainting, optical-flow fill, upscale
or UHD. Every rendered floor sample still comes from a real official source pixel.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v16 as v16


_ORIGINAL_SAMPLE = v12.v4.sample_plane_from_sources
_PHOTO_CACHE = {}
_PHOTO_QA = {}


def _fit_channel_affine(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]; y = y[keep]
    if len(x) < 100:
        return 1.0, 0.0, {"count": int(len(x)), "status": "INSUFFICIENT"}
    A = np.column_stack([x, np.ones(len(x))])
    p, *_ = np.linalg.lstsq(A, y, rcond=None)
    for _ in range(5):
        pred = A @ p
        e = np.abs(pred - y)
        cap = min(45.0, float(np.percentile(e, 72)))
        k = e <= max(8.0, cap)
        if int(k.sum()) < 100:
            break
        p, *_ = np.linalg.lstsq(A[k], y[k], rcond=None)
    a = float(np.clip(p[0], 0.55, 1.65))
    b = float(np.clip(p[1], -95.0, 95.0))
    before = np.abs(x - y)
    after = np.abs(a * x + b - y)
    return a, b, {
        "count": int(len(x)),
        "slope": a,
        "offset": b,
        "median_abs_before": float(np.median(before)),
        "median_abs_after": float(np.median(after)),
        "p90_abs_before": float(np.percentile(before, 90)),
        "p90_abs_after": float(np.percentile(after, 90)),
    }


def _fit_photo_transforms(sources):
    key = tuple((label, id(sources[label]["image"])) for label in v12.CAMERAS)
    if key in _PHOTO_CACHE:
        return _PHOTO_CACHE[key]

    anchor = v12.A
    sa = sources[anchor]
    Ca, Ra, Ka = sa["C"], sa["R"], sa["K"]
    _t, Pw = v12.v4.ray_plane_map(Ka, Ra, Ca, "floor")
    av = sa["floor_vis"].astype(bool)

    # Avoid high-gradient court markings and player-edge leakage while learning
    # colour response. The transform is then applied to all real floor samples.
    gray_a = cv2.cvtColor(sa["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.magnitude(cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3))
    yy, xx = np.indices((v12.H, v12.W))
    base = av & ((xx % 3) == 0) & ((yy % 3) == 0) & (ga < 70.0)
    ids0 = np.flatnonzero(base.ravel())
    P0 = Pw.reshape(-1, 3)[ids0]
    ay = (ids0 // v12.W).astype(np.int32)
    ax = (ids0 % v12.W).astype(np.int32)
    anchor_cols0 = sa["image"][ay, ax].astype(np.float64)

    transforms = {anchor: {"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}}
    qa = {anchor: {"status": "IDENTITY", "pair_count": int(len(ids0))}}

    for label in v12.CAMERAS:
        if label == anchor:
            continue
        s = sources[label]
        uv, _, valid = v12.v8.project_metric((s["C"], s["R"], s["K"]), P0)
        ui = np.rint(uv[:, 0]).astype(np.int32)
        vi = np.rint(uv[:, 1]).astype(np.int32)
        good = (
            valid & np.isfinite(uv).all(axis=1)
            & (ui >= 1) & (ui < v12.W - 1)
            & (vi >= 1) & (vi < v12.H - 1)
        )
        gi = np.where(good)[0]
        if len(gi):
            gi = gi[s["floor_vis"][vi[gi], ui[gi]].astype(bool)]
        if len(gi):
            gray_s = cv2.cvtColor(s["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gs = cv2.magnitude(cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3))
            gi = gi[gs[vi[gi], ui[gi]] < 70.0]
        if len(gi) > 18000:
            gi = gi[np.linspace(0, len(gi) - 1, 18000).astype(np.int32)]

        src_cols = s["image"][vi[gi], ui[gi]].astype(np.float64) if len(gi) else np.empty((0, 3), np.float64)
        anc_cols = anchor_cols0[gi] if len(gi) else np.empty((0, 3), np.float64)
        aa, bb, channels = [], [], []
        for ch in range(3):
            a, b, cq = _fit_channel_affine(src_cols[:, ch] if len(src_cols) else [], anc_cols[:, ch] if len(anc_cols) else [])
            aa.append(a); bb.append(b); channels.append(cq)
        transforms[label] = {"a": aa, "b": bb}
        if len(src_cols):
            pred = src_cols * np.asarray(aa)[None, :] + np.asarray(bb)[None, :]
            before = np.linalg.norm(src_cols - anc_cols, axis=1)
            after = np.linalg.norm(pred - anc_cols, axis=1)
            qa[label] = {
                "status": "FITTED",
                "pair_count": int(len(src_cols)),
                "channels_bgr": channels,
                "rgb_vector_median_before": float(np.median(before)),
                "rgb_vector_median_after": float(np.median(after)),
                "rgb_vector_p90_before": float(np.percentile(before, 90)),
                "rgb_vector_p90_after": float(np.percentile(after, 90)),
            }
        else:
            qa[label] = {"status": "NO_PAIRS", "pair_count": 0, "channels_bgr": channels}

    _PHOTO_CACHE[key] = transforms
    _PHOTO_QA.clear(); _PHOTO_QA.update(qa)
    return transforms


def _apply_photo(cols, transform):
    a = np.asarray(transform["a"], np.float64).reshape(1, 3)
    b = np.asarray(transform["b"], np.float64).reshape(1, 3)
    return np.clip(cols.astype(np.float64) * a + b, 0, 255).astype(np.uint8)


def sample_plane_photometric(P, support, sources, plane):
    if plane != "floor":
        return _ORIGINAL_SAMPLE(P, support, sources, plane)

    transforms = _fit_photo_transforms(sources)
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    ids = np.where(support)[0]
    if not len(ids):
        return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)
    Pw = P[ids]

    for label in v12.CAMERAS:
        src = sources[label]
        C, R, K = src["C"], src["R"], src["K"]
        sgn = v12.v3.forward_sign(R, C)
        Xc = (R @ (Pw - C).T).T
        q = (K @ Xc.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = q[:, :2] / q[:, 2:3]
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        ok = (
            np.isfinite(uv).all(axis=1) & (sgn * Xc[:, 2] > 20)
            & (u >= 0) & (u < v12.W) & (v >= 0) & (v < v12.H)
        )
        loc = np.where(ok)[0]
        if not len(loc):
            continue
        loc = loc[src["floor_vis"][v[loc], u[loc]].astype(bool)]
        if not len(loc):
            continue
        tgt = ids[loc]
        free = ~owned[tgt]
        loc = loc[free]; tgt = tgt[free]
        if not len(loc):
            continue
        cols = src["image"][v[loc], u[loc]]
        if label != v12.A:
            cols = _apply_photo(cols, transforms[label])
        out[tgt] = cols
        owned[tgt] = True
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    _PHOTO_CACHE.clear(); _PHOTO_QA.clear()
    v12.v4.sample_plane_from_sources = sample_plane_photometric

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v16.main()

    try:
        oi = sys.argv.index("--out")
        out = Path(sys.argv[oi + 1])
        qp = out / "three_camera_mesh_v12_qa.json"
        q = json.loads(qp.read_text())
        q["v17_renderer"] = {
            "resolution": [960, 540],
            "floor_photometric_reference": "Left Above Rim exact-state frame",
            "floor_secondary_camera_transform": "robust per-channel affine fit from mutually visible low-gradient metric floor correspondences",
            "floor_photometric_qa": _PHOTO_QA,
            "source_pixel_policy": "all floor RGB originates in official source pixels; secondary-source values receive deterministic colour calibration only",
            "generated_texture": False,
            "upscale": False,
            "uhd": False,
        }
        qp.write_text(json.dumps(q, indent=2))
    except Exception as exc:
        print("V17_QA_APPEND_WARNING", repr(exc), flush=True)


if __name__ == "__main__":
    main()
