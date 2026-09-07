from __future__ import annotations

"""v3 three-camera diagnostic calibration bridge.

Accepted camera solves are preserved exactly.  Their projective pose lineages do
not all use the same sign for camera-space forward Z: Left Above Rim sees the
known rim at +Z, while Right Above Rim sees it at -Z.  Infer that harmless
projective sign from the regulation rim centre for each accepted camera, then
use a consistent positive metric depth for MoGe alignment and world-cloud
unprojection.
"""

import numpy as np
from scipy.optimize import least_squares

from freeze_spin import build_three_camera_diagnostic_v1 as base


def forward_sign(R, C):
    rim_cam = R @ (base.RIM - C)
    if not np.isfinite(rim_cam).all() or abs(float(rim_cam[2])) < 1e-6:
        raise RuntimeError(f'cannot infer camera forward sign from rim depth {rim_cam}')
    return 1.0 if float(rim_cam[2]) > 0.0 else -1.0


def robust_depth_align(depth, valid, K, R, C):
    s = forward_sign(R, C)
    xs = np.arange(4, base.W - 4, 6, dtype=np.int32)
    ys = np.arange(4, base.H - 4, 6, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    x = xx.ravel(); y = yy.ravel()

    xn = (x.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (y.astype(np.float64) - K[1, 2]) / K[1, 1]
    dc = np.column_stack([xn, yn, np.ones_like(xn)])
    # R maps world -> camera.  s converts each certified pose's projective
    # forward sign into one common positive-depth convention.
    dw = (s * dc) @ R
    denom = dw[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        zmetric = -float(C[2]) / denom
    P = C.reshape(1, 3) + zmetric[:, None] * dw

    court = (
        np.isfinite(zmetric)
        & (zmetric > 30.0)
        & (zmetric < 12000.0)
        & np.isfinite(P).all(axis=1)
        & (P[:, 0] >= -180.0)
        & (P[:, 0] <= 3000.0)
        & (P[:, 1] >= -850.0)
        & (P[:, 1] <= 850.0)
    )
    good = court & valid[y, x] & np.isfinite(depth[y, x]) & (depth[y, x] > 1e-4)
    ids = np.where(good)[0]
    if len(ids) < 100:
        raise RuntimeError(
            f'only {len(ids)} sign-normalized ray-floor anchors; '
            f'court_candidates={int(np.sum(court))}; forward_sign={int(s)}'
        )

    d = depth[y[ids], x[ids]].astype(np.float64)
    z = zmetric[ids].astype(np.float64)
    scale = float(np.median(z / np.maximum(d, 1e-6)))
    p = np.asarray([scale, 0.0], dtype=np.float64)
    start_n = len(d)
    for _ in range(4):
        fit = least_squares(
            lambda q: q[0] * d + q[1] - z,
            p, loss='soft_l1', f_scale=35.0, max_nfev=5000,
        )
        p = fit.x
        e = np.abs(p[0] * d + p[1] - z)
        cap = max(float(np.percentile(e, 68.0)), 20.0)
        keep = e <= cap
        d, z = d[keep], z[keep]
        if len(d) < 100:
            raise RuntimeError(f'sign-normalized floor fit collapsed to {len(d)} anchors')

    pred = p[0] * d + p[1]
    e = np.abs(pred - z)
    return p, {
        'anchor_method': 'camera_ray_z0_intersection_sign_normalized_by_visible_rim',
        'forward_sign': int(s),
        'candidate_anchors': int(start_n),
        'anchors': int(len(d)),
        'scale': float(p[0]),
        'offset_cm': float(p[1]),
        'median_cm': float(np.median(e)),
        'p95_cm': float(np.percentile(e, 95)),
    }


def metric_cloud(image, depth, valid, K, R, C, align, stride=2):
    s = forward_sign(R, C)
    yy, xx = np.indices((base.H, base.W))
    pick = ((xx % stride) == 0) & ((yy % stride) == 0)
    z = align[0] * depth.astype(np.float64) + align[1]
    ok = valid & pick & np.isfinite(z) & (z > 20.0) & (z < 12000.0)
    ys, xs = np.where(ok)
    zz = z[ys, xs]
    xn = (xs - K[0, 2]) / K[0, 0]
    yn = (ys - K[1, 2]) / K[1, 1]
    Xc = s * np.column_stack([xn * zz, yn * zz, zz])
    Xw = (R.T @ Xc.T).T + C
    return Xw.astype(np.float32), image[ys, xs].copy()


base.robust_depth_align = robust_depth_align
base.metric_cloud = metric_cloud

if __name__ == '__main__':
    base.main()
