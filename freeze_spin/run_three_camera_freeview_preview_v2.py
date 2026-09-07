from __future__ import annotations

"""Sign-convention compatibility driver for the three-camera diagnostic preview.

Right Above Rim v73 is an accepted projective camera whose optimizer permits a
consistent negative camera-Z solution. That convention is valid for projection
but the v1 depth mapper assumed positive camera Z. This driver preserves every
accepted v73 intrinsic/extrinsic parameter and only carries the common depth
sign through monocular-depth metric alignment and pixel unprojection.
"""

import numpy as np
from scipy.optimize import least_squares

from freeze_spin import build_three_camera_freeview_preview_v1 as base


def robust_depth_mapping_signed(depth, valid, cam):
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(cam["center_cm"], dtype=np.float64)
    P = base.regulation_floor_points()
    uv, z_signed = base.project_points(K, R, C, P)
    x = np.rint(uv[:, 0]).astype(int)
    y = np.rint(uv[:, 1]).astype(int)
    good = (
        np.isfinite(uv).all(axis=1)
        & (np.abs(z_signed) > 20.0)
        & (x >= 2) & (x < base.W - 2) & (y >= 2) & (y < base.H - 2)
    )
    ids = np.where(good)[0]
    x, y, z_signed = x[good], y[good], z_signed[good]
    d = depth[y, x].astype(np.float64)
    ok = valid[y, x] & np.isfinite(d) & (d > 0.02)
    ids, d, z_signed = ids[ok], d[ok], z_signed[ok]
    if len(d) < 35:
        raise RuntimeError(f"{cam['label']} insufficient metric floor/depth anchors: {len(d)}")

    # A projective camera may represent all visible scene points with either
    # positive or negative camera Z. Require one coherent sign; do not mix roots.
    med_sign = 1.0 if float(np.median(z_signed)) >= 0.0 else -1.0
    sign_consistency = float(np.mean(np.sign(z_signed) == med_sign))
    if sign_consistency < 0.98:
        raise RuntimeError(
            f"{cam['label']} camera-depth sign is not coherent: {sign_consistency:.4f}"
        )
    z = np.abs(z_signed)

    hold = (ids % 7) == 0
    if int((~hold).sum()) < 25 or int(hold.sum()) < 5:
        hold = (np.arange(len(d)) % 7) == 0
    train = ~hold
    scale0 = float(np.median(z[train] / np.maximum(d[train], 1e-6)))
    fit = least_squares(
        lambda p: p[0] * d[train] + p[1] - z[train],
        [scale0, 0.0], loss="soft_l1", f_scale=45.0, max_nfev=6000
    )
    p = np.asarray(fit.x, dtype=np.float64)
    residual = np.abs(p[0] * d + p[1] - z)
    med = float(np.median(residual[train]))
    mad = float(np.median(np.abs(residual[train] - med)))
    thresh = max(75.0, med + 4.0 * 1.4826 * max(mad, 1.0))
    support = train & (residual <= thresh)
    if int(support.sum()) >= 25:
        fit2 = least_squares(
            lambda q: q[0] * d[support] + q[1] - z[support],
            p, loss="soft_l1", f_scale=35.0, max_nfev=6000
        )
        p = np.asarray(fit2.x, dtype=np.float64)
    pred = p[0] * d + p[1]
    err = np.abs(pred - z)
    held = err[hold]
    mapping = np.asarray([p[0], p[1], med_sign], dtype=np.float64)
    return mapping, {
        "candidate_anchor_count": int(len(d)),
        "training_support_count": int(support.sum()),
        "heldout_count": int(hold.sum()),
        "scale": float(p[0]),
        "offset_cm": float(p[1]),
        "projective_camera_z_sign": int(med_sign),
        "camera_z_sign_consistency": sign_consistency,
        "heldout_median_abs_cm": float(np.median(held)) if len(held) else None,
        "heldout_p95_abs_cm": float(np.percentile(held, 95)) if len(held) else None,
        "all_median_abs_cm": float(np.median(err)),
        "all_p95_abs_cm": float(np.percentile(err, 95)),
    }


def build_cloud_signed(image, depth, valid, mapping, cam, stride=2):
    scale, offset, z_sign = [float(x) for x in mapping]
    z_abs = scale * depth.astype(np.float64) + offset
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R_world_to_camera"], dtype=np.float64)
    C = np.asarray(cam["center_cm"], dtype=np.float64)
    yy, xx = np.indices((base.H, base.W))
    sample = ((xx % stride) == 0) & ((yy % stride) == 0)
    ok = sample & valid & np.isfinite(z_abs) & (z_abs > 20.0) & (z_abs < 12000.0)
    ys, xs = np.where(ok)
    za = z_abs[ys, xs]
    z = z_sign * za
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = np.column_stack([xn * z, yn * z, z])
    Xw = (R.T @ Xc.T).T + C
    colours = image[ys, xs].copy()
    src_uv = np.column_stack([xs, ys]).astype(np.int32)
    return Xw.astype(np.float32), colours, src_uv


base.robust_depth_mapping = robust_depth_mapping_signed
base.build_cloud = build_cloud_signed

if __name__ == "__main__":
    base.main()
