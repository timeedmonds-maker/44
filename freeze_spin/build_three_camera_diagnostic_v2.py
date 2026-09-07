from __future__ import annotations

"""v2 calibration bridge for the three-camera diagnostic.

The accepted camera models are world->camera poses.  Instead of projecting a
sparse synthetic world grid and hoping it lands on valid MoGe pixels, derive
metric floor depth directly for sampled image rays: back-project each pixel
through K and R, intersect the ray with the regulation z=0 floor plane, and
fit MoGe depth shape to the resulting camera-space Z values robustly.

This preserves the metric-camera gate, uses no generated pixels, and works for
all accepted camera viewpoints so long as the solved ray actually intersects
the court in front of the camera.
"""

import numpy as np
from scipy.optimize import least_squares

from freeze_spin import build_three_camera_diagnostic_v1 as base


def robust_depth_align(depth, valid, K, R, C):
    # Sample image rays on a regular grid.  In the project convention R maps
    # world->camera, so a camera ray d_c maps to world direction R.T @ d_c.
    xs = np.arange(4, base.W - 4, 6, dtype=np.int32)
    ys = np.arange(4, base.H - 4, 6, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    x = xx.ravel()
    y = yy.ravel()

    xn = (x.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (y.astype(np.float64) - K[1, 2]) / K[1, 1]
    dc = np.column_stack([xn, yn, np.ones_like(xn)])
    dw = dc @ R  # row-vector form of R.T @ d_c

    denom = dw[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        zcam = -float(C[2]) / denom
    P = C.reshape(1, 3) + zcam[:, None] * dw

    # Regulation-court/basket-local support, with modest apron tolerance.
    court = (
        np.isfinite(zcam)
        & (zcam > 30.0)
        & (zcam < 12000.0)
        & np.isfinite(P).all(axis=1)
        & (P[:, 0] >= -180.0)
        & (P[:, 0] <= 3000.0)
        & (P[:, 1] >= -850.0)
        & (P[:, 1] <= 850.0)
    )
    good = (
        court
        & valid[y, x]
        & np.isfinite(depth[y, x])
        & (depth[y, x] > 1e-4)
    )
    ids = np.where(good)[0]
    if len(ids) < 100:
        raise RuntimeError(
            f'only {len(ids)} ray-floor depth anchors; '
            f'positive_floor_rays={int(np.sum(court))}'
        )

    d = depth[y[ids], x[ids]].astype(np.float64)
    z = zcam[ids].astype(np.float64)

    scale = float(np.median(z / np.maximum(d, 1e-6)))
    p = np.asarray([scale, 0.0], dtype=np.float64)
    start_n = len(d)
    for _ in range(4):
        fit = least_squares(
            lambda q: q[0] * d + q[1] - z,
            p,
            loss='soft_l1',
            f_scale=35.0,
            max_nfev=5000,
        )
        p = fit.x
        e = np.abs(p[0] * d + p[1] - z)
        # Keep the strongest floor-consistent majority; foreground occluders
        # and crowd pixels are intentionally treated as outliers.
        cap = max(float(np.percentile(e, 68.0)), 20.0)
        keep = e <= cap
        d, z = d[keep], z[keep]
        if len(d) < 100:
            raise RuntimeError(f'floor-anchor robust fit collapsed to {len(d)} samples')

    pred = p[0] * d + p[1]
    e = np.abs(pred - z)
    return p, {
        'anchor_method': 'camera_ray_z0_intersection',
        'candidate_anchors': int(start_n),
        'anchors': int(len(d)),
        'scale': float(p[0]),
        'offset_cm': float(p[1]),
        'median_cm': float(np.median(e)),
        'p95_cm': float(np.percentile(e, 95)),
    }


base.robust_depth_align = robust_depth_align

if __name__ == '__main__':
    base.main()
