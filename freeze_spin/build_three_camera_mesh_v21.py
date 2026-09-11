from __future__ import annotations

"""v21: registered single metric court atlas from the three solved cameras.

v19 provenance localized the residual coloured floor seam to secondary-camera
metric-floor ownership islands. v20 then removed Broadcast from floor appearance:
the coloured seam disappeared and became an explicit black floor hole. That
establishes the remaining problem as floor texture registration/coverage, not
player geometry, ball geometry or exact-state synchronization.

v21 keeps the validated three-camera player/ball reconstruction unchanged and
replaces view-dependent floor ownership with one persistent basket-local court
texture atlas:

1. Project real floor-visible source pixels from each solved camera into a common
   metric z=0 court atlas.
2. Register Right Above Rim and Broadcast atlas images to the Left Above Rim atlas
   with a small source-grounded planar homography estimated from SIFT matches in
   mutually visible floor texture/markings. Reject implausible corrections.
3. Fit robust per-channel photometric transforms in the *registered* overlap.
4. Build one hard-ownership atlas: LAR first, then registered RAR, then registered
   Broadcast. Unsupported texels remain black.
5. Every virtual floor view samples that same atlas, so source boundaries cannot
   move with virtual-camera angle.

No generated texture, inpainting, optical flow, camera cross-fade, upscale or UHD.
All non-black floor samples descend from official native source pixels through a
planar registration/resampling only. Output remains native 960x540.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v17 as v17


ATLAS_CM = 2.5
X0 = float(v12.v4.COURT_X0)
X1 = float(v12.v4.COURT_X1)
Y0 = float(v12.v4.COURT_Y0)
Y1 = float(v12.v4.COURT_Y1)
AW = int(round((X1 - X0) / ATLAS_CM)) + 1
AH = int(round((Y1 - Y0) / ATLAS_CM)) + 1

_CACHE_KEY = None
_ATLAS = None
_ATLAS_MASK = None
_ATLAS_QA = {}
_OUT = None


def _atlas_world_points():
    yy, xx = np.indices((AH, AW), np.float64)
    xw = X0 + xx.ravel() * ATLAS_CM
    yw = Y0 + yy.ravel() * ATLAS_CM
    return np.column_stack([xw, yw, np.zeros_like(xw)])


def _source_atlas(src):
    P = _atlas_world_points()
    C, R, K = src["C"], src["R"], src["K"]
    sgn = float(v12.v3.forward_sign(R, C))
    Xc = (R @ (P - C).T).T
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    valid = (
        np.isfinite(uv).all(axis=1)
        & (sgn * Xc[:, 2] > 20.0)
        & (uv[:, 0] >= 0.0) & (uv[:, 0] < v12.W - 1.0)
        & (uv[:, 1] >= 0.0) & (uv[:, 1] < v12.H - 1.0)
        & (u >= 0) & (u < v12.W) & (v >= 0) & (v < v12.H)
    )
    ids = np.where(valid)[0]
    if len(ids):
        ids = ids[src["floor_vis"][v[ids], u[ids]].astype(bool)]
    out = np.zeros((AH * AW, 3), np.uint8)
    mask = np.zeros(AH * AW, bool)
    if len(ids):
        out[ids] = v12.v8.bilinear_sample(src["image"], uv[ids])
        mask[ids] = True
    return out.reshape(AH, AW, 3), mask.reshape(AH, AW)


def _feature_image(img, mask):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 8))
    g = clahe.apply(g)
    # Suppress invalid black texels before feature extraction.
    g = g.copy(); g[~mask] = 0
    return g


def _register_to_anchor(anchor_img, anchor_mask, moving_img, moving_mask, label):
    qa = {"label": label, "method": "identity_fallback"}
    if int(anchor_mask.sum()) < 1000 or int(moving_mask.sum()) < 1000:
        qa["reason"] = "insufficient_atlas_support"
        return moving_img, moving_mask, np.eye(3, dtype=np.float64), qa

    ref = _feature_image(anchor_img, anchor_mask)
    mov = _feature_image(moving_img, moving_mask)
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.018, edgeThreshold=12, sigma=1.4)
    kp_m, des_m = sift.detectAndCompute(mov, moving_mask.astype(np.uint8) * 255)
    kp_r, des_r = sift.detectAndCompute(ref, anchor_mask.astype(np.uint8) * 255)
    qa["moving_keypoints"] = int(len(kp_m))
    qa["anchor_keypoints"] = int(len(kp_r))
    if des_m is None or des_r is None or len(kp_m) < 12 or len(kp_r) < 12:
        qa["reason"] = "insufficient_sift_features"
        return moving_img, moving_mask, np.eye(3, dtype=np.float64), qa

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_m, des_r, k=2)
    good = []
    for row in knn:
        if len(row) < 2:
            continue
        a, b = row[0], row[1]
        if a.distance >= 0.74 * b.distance:
            continue
        pm = np.asarray(kp_m[a.queryIdx].pt, np.float64)
        pr = np.asarray(kp_r[a.trainIdx].pt, np.float64)
        # Nominal metric cameras already map to one world plane. A valid residual
        # correction therefore must be near identity, not an arbitrary re-warp.
        if float(np.linalg.norm(pm - pr)) > 140.0:
            continue
        good.append((a, pm, pr))
    qa["ratio_and_prior_matches"] = int(len(good))
    if len(good) < 12:
        qa["reason"] = "insufficient_near_identity_matches"
        return moving_img, moving_mask, np.eye(3, dtype=np.float64), qa

    src_pts = np.asarray([x[1] for x in good], np.float32).reshape(-1, 1, 2)
    dst_pts = np.asarray([x[2] for x in good], np.float32).reshape(-1, 1, 2)
    H, inlier = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.2, maxIters=8000, confidence=0.997)
    if H is None or not np.isfinite(H).all():
        qa["reason"] = "homography_failed"
        return moving_img, moving_mask, np.eye(3, dtype=np.float64), qa
    H = H / H[2, 2]
    inlier = inlier.reshape(-1).astype(bool) if inlier is not None else np.zeros(len(good), bool)
    qa["ransac_inliers"] = int(inlier.sum())
    qa["ransac_inlier_fraction"] = float(inlier.mean()) if len(inlier) else 0.0

    test = np.asarray([
        [0, 0], [AW - 1, 0], [0, AH - 1], [AW - 1, AH - 1],
        [0.5 * (AW - 1), 0.5 * (AH - 1)],
        [0.25 * (AW - 1), 0.5 * (AH - 1)], [0.75 * (AW - 1), 0.5 * (AH - 1)],
        [0.5 * (AW - 1), 0.25 * (AH - 1)], [0.5 * (AW - 1), 0.75 * (AH - 1)],
    ], np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(test, H).reshape(-1, 2)
    disp = np.linalg.norm(mapped - test.reshape(-1, 2), axis=1)
    qa["control_displacement_median_px"] = float(np.median(disp))
    qa["control_displacement_max_px"] = float(np.max(disp))
    qa["homography"] = H.tolist()

    plausible = (
        int(inlier.sum()) >= 10
        and float(inlier.mean()) >= 0.24
        and float(np.median(disp)) <= 45.0
        and float(np.max(disp)) <= 120.0
        and abs(float(np.linalg.det(H[:2, :2]))) > 0.20
        and abs(float(np.linalg.det(H[:2, :2]))) < 4.0
    )
    if not plausible:
        qa["reason"] = "homography_rejected_by_near_identity_gate"
        return moving_img, moving_mask, np.eye(3, dtype=np.float64), qa

    wi = cv2.warpPerspective(moving_img, H, (AW, AH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wm = cv2.warpPerspective(moving_mask.astype(np.uint8) * 255, H, (AW, AH), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    # One-pixel erosion avoids bilinear edge contamination after registration.
    wm = cv2.erode(wm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    qa["method"] = "sift_ransac_metric_atlas_homography"
    qa["reason"] = "accepted"
    qa["registered_support_texels"] = int(wm.sum())
    return wi, wm, H, qa


def _fit_registered_photo(anchor_img, anchor_mask, moving_img, moving_mask):
    overlap = anchor_mask & moving_mask
    if int(overlap.sum()) < 300:
        return moving_img, {"status": "INSUFFICIENT_OVERLAP", "overlap_texels": int(overlap.sum())}
    # Learn colour response away from line/edge discontinuities.
    ga = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gm = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grada = cv2.magnitude(cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3))
    gradm = cv2.magnitude(cv2.Sobel(gm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gm, cv2.CV_32F, 0, 1, ksize=3))
    use = overlap & (grada < 65.0) & (gradm < 65.0)
    ys, xs = np.where(use)
    if len(xs) > 70000:
        take = np.linspace(0, len(xs) - 1, 70000).astype(np.int32)
        ys, xs = ys[take], xs[take]
    src = moving_img[ys, xs].astype(np.float64)
    dst = anchor_img[ys, xs].astype(np.float64)
    aa, bb, cq = [], [], []
    for ch in range(3):
        a, b, q = v17._fit_channel_affine(src[:, ch], dst[:, ch])
        aa.append(a); bb.append(b); cq.append(q)
    corrected = v17._apply_photo(moving_img.reshape(-1, 3), {"a": aa, "b": bb}).reshape(AH, AW, 3)
    before = np.linalg.norm(src - dst, axis=1) if len(src) else np.asarray([])
    pred = src * np.asarray(aa)[None, :] + np.asarray(bb)[None, :]
    after = np.linalg.norm(pred - dst, axis=1) if len(src) else np.asarray([])
    qa = {
        "status": "FITTED" if len(src) >= 100 else "LOW_SUPPORT",
        "overlap_texels": int(overlap.sum()),
        "fit_texels": int(len(src)),
        "channels_bgr": cq,
        "rgb_median_before": float(np.median(before)) if len(before) else None,
        "rgb_median_after": float(np.median(after)) if len(after) else None,
        "rgb_p90_before": float(np.percentile(before, 90)) if len(before) else None,
        "rgb_p90_after": float(np.percentile(after, 90)) if len(after) else None,
    }
    return corrected, qa


def _build_atlas(sources):
    global _CACHE_KEY, _ATLAS, _ATLAS_MASK, _ATLAS_QA
    key = tuple((label, id(sources[label]["image"])) for label in v12.CAMERAS)
    if key == _CACHE_KEY and _ATLAS is not None:
        return _ATLAS, _ATLAS_MASK

    raw = {}; masks = {}
    for label in v12.CAMERAS:
        raw[label], masks[label] = _source_atlas(sources[label])
        if _OUT is not None:
            cv2.imwrite(str(_OUT / f"v21_atlas_raw_{label.replace(' ', '_')}.png"), raw[label])
            cv2.imwrite(str(_OUT / f"v21_atlas_raw_mask_{label.replace(' ', '_')}.png"), masks[label].astype(np.uint8) * 255)

    anchor_img, anchor_mask = raw[v12.A], masks[v12.A]
    registered = {v12.A: anchor_img}
    regmask = {v12.A: anchor_mask}
    regqa = {v12.A: {"method": "anchor_identity", "support_texels": int(anchor_mask.sum())}}
    photoqa = {v12.A: {"status": "IDENTITY"}}

    for label in (v12.C, v12.B):
        wi, wm, H, rq = _register_to_anchor(anchor_img, anchor_mask, raw[label], masks[label], label)
        wi2, pq = _fit_registered_photo(anchor_img, anchor_mask, wi, wm)
        registered[label], regmask[label] = wi2, wm
        regqa[label] = rq; photoqa[label] = pq
        if _OUT is not None:
            cv2.imwrite(str(_OUT / f"v21_atlas_registered_{label.replace(' ', '_')}.png"), wi2)
            cv2.imwrite(str(_OUT / f"v21_atlas_registered_mask_{label.replace(' ', '_')}.png"), wm.astype(np.uint8) * 255)

    atlas = np.zeros((AH, AW, 3), np.uint8)
    owned = np.zeros((AH, AW), bool)
    provenance = np.zeros((AH, AW), np.uint8)
    counts = {label: 0 for label in v12.CAMERAS}
    code = {v12.A: 1, v12.C: 2, v12.B: 3}
    for label in (v12.A, v12.C, v12.B):
        take = regmask[label] & (~owned)
        atlas[take] = registered[label][take]
        owned[take] = True
        provenance[take] = code[label]
        counts[label] = int(take.sum())

    if _OUT is not None:
        cv2.imwrite(str(_OUT / "v21_metric_court_atlas.png"), atlas)
        cv2.imwrite(str(_OUT / "v21_metric_court_atlas_mask.png"), owned.astype(np.uint8) * 255)
        pv = np.zeros_like(atlas)
        pv[provenance == 1] = (0, 200, 0)
        pv[provenance == 2] = (220, 220, 0)
        pv[provenance == 3] = (0, 0, 230)
        cv2.imwrite(str(_OUT / "v21_metric_court_atlas_provenance.png"), pv)

    _CACHE_KEY = key; _ATLAS = atlas; _ATLAS_MASK = owned
    _ATLAS_QA = {
        "atlas_cm_per_texel": ATLAS_CM,
        "atlas_size_px": [int(AW), int(AH)],
        "world_bounds_cm": [X0, Y0, X1, Y1],
        "raw_support_texels": {label: int(masks[label].sum()) for label in v12.CAMERAS},
        "registration": regqa,
        "photometric": photoqa,
        "hard_ownership_texels": counts,
        "resolved_fraction": float(owned.mean()),
    }
    return atlas, owned


def _bilinear_atlas(img, uv):
    uv = np.asarray(uv, np.float64)
    x = uv[:, 0]; y = uv[:, 1]
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, AW - 1); y1 = np.clip(y0 + 1, 0, AH - 1)
    x0 = np.clip(x0, 0, AW - 1); y0 = np.clip(y0, 0, AH - 1)
    wx = (x - x0)[:, None]; wy = (y - y0)[:, None]
    a = img[y0, x0].astype(np.float64) * (1.0 - wx) + img[y0, x1].astype(np.float64) * wx
    b = img[y1, x0].astype(np.float64) * (1.0 - wx) + img[y1, x1].astype(np.float64) * wx
    return np.clip(a * (1.0 - wy) + b * wy, 0, 255).astype(np.uint8)


def sample_plane_registered_atlas(P, support, sources, plane):
    if plane != "floor":
        return v17._ORIGINAL_SAMPLE(P, support, sources, plane)
    atlas, mask = _build_atlas(sources)
    out = np.zeros((v12.H * v12.W, 3), np.uint8)
    owned = np.zeros(v12.H * v12.W, bool)
    ids = np.where(support)[0]
    if not len(ids):
        return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)
    Pw = P[ids]
    uv = np.column_stack([(Pw[:, 0] - X0) / ATLAS_CM, (Pw[:, 1] - Y0) / ATLAS_CM])
    valid = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0) & (uv[:, 0] < AW - 1.0)
        & (uv[:, 1] >= 0.0) & (uv[:, 1] < AH - 1.0)
    )
    loc = np.where(valid)[0]
    if len(loc):
        # Require all four bilinear neighbours to be source-supported. This keeps
        # black unsupported atlas holes explicit and avoids interpolating across them.
        x0 = np.floor(uv[loc, 0]).astype(np.int32); y0 = np.floor(uv[loc, 1]).astype(np.int32)
        x1 = x0 + 1; y1 = y0 + 1
        ok = mask[y0, x0] & mask[y0, x1] & mask[y1, x0] & mask[y1, x1]
        loc = loc[ok]
    if len(loc):
        tgt = ids[loc]
        out[tgt] = _bilinear_atlas(atlas, uv[loc])
        owned[tgt] = True
    return out.reshape(v12.H, v12.W, 3), owned.reshape(v12.H, v12.W)


def main():
    global _CACHE_KEY, _ATLAS, _ATLAS_MASK, _ATLAS_QA, _OUT
    _CACHE_KEY = None; _ATLAS = None; _ATLAS_MASK = None; _ATLAS_QA = {}
    oi = sys.argv.index("--out"); _OUT = Path(sys.argv[oi + 1]); _OUT.mkdir(parents=True, exist_ok=True)
    v17.sample_plane_photometric = sample_plane_registered_atlas
    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v17.main()

    qp = _OUT / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    q["v21_renderer"] = {
        "resolution": [960, 540],
        "floor_representation": "single persistent basket-local registered source-pixel court atlas",
        "atlas_qa": _ATLAS_QA,
        "secondary_registration": "SIFT + RANSAC near-identity planar homography in metric atlas coordinates",
        "ownership": [v12.A, v12.C, v12.B],
        "unsupported_floor_policy": "black; no inpainting or generated fill",
        "generated_texture": False,
        "crossfade": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
