from __future__ import annotations

"""v109: computationally equivalent fast residual wrapper for Right Slash v107.

The v107 geometry/gates are unchanged.  This wrapper removes repeated 3x3 floor-
homography inversions from every implicit court-family evaluation: H^-1 is computed
once per camera state/residual call and reused for that state's regulation curves.
This is an execution optimization only; thresholds, observations, robust loss,
multistart/support/perturbation counts, and fail-closed permissions remain v107.
"""

import numpy as np

from freeze_spin import prove_right_slash_shared_center_v107 as v107


def _implicit_from_inverse(inv_h: np.ndarray, key: str, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    q = (inv_h @ np.column_stack([pixels, np.ones(len(pixels))]).T).T
    world = q[:, :2] / q[:, 2:3]
    v44 = v107.v44
    if key == "three_point_arc":
        return np.hypot(world[:, 0] - v44.RIM_X_CM, world[:, 1]) - v44.THREE_R_CM
    if key == "free_throw_front_semicircle":
        return np.hypot(world[:, 0] - v44.FT_X_CM, world[:, 1]) - v44.FT_R_CM
    if key == "free_throw_line":
        return world[:, 0] - v44.FT_X_CM
    if key == "lane_negative_y":
        return world[:, 1] + v44.PAINT_HALF_CM
    if key == "lane_positive_y":
        return world[:, 1] - v44.PAINT_HALF_CM
    raise KeyError(key)


def _signed_pixel_residual_from_inverse(inv_h: np.ndarray, key: str, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    eps = 0.25
    f = _implicit_from_inverse(inv_h, key, pixels)
    gx = (
        _implicit_from_inverse(inv_h, key, pixels + np.asarray([eps, 0.0]))
        - _implicit_from_inverse(inv_h, key, pixels - np.asarray([eps, 0.0]))
    ) / (2.0 * eps)
    gy = (
        _implicit_from_inverse(inv_h, key, pixels + np.asarray([0.0, eps]))
        - _implicit_from_inverse(inv_h, key, pixels - np.asarray([0.0, eps]))
    ) / (2.0 * eps)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def fast_state_residual(p, s):
    rows = []
    H = v107.v87.floor_homography(p)
    inv_h = np.linalg.inv(H)
    for key, pts in s.get("floor_train_px", {}).items():
        a = np.asarray(pts, dtype=np.float64)
        if len(a):
            rows.append(_signed_pixel_residual_from_inverse(inv_h, key, a))
    for key in v107.TARGET_KEYS:
        a = np.asarray(s.get("target_line_samples_px", {}).get(key, []), dtype=np.float64)
        if len(a):
            uv, _ = v107.v87.project3(p, v107.v87.world_target_lines()[key])
            rows.append(v107.v87.signed_line_distance(a, uv[0], uv[1]))
    return np.concatenate(rows) if rows else np.asarray([1e6], dtype=np.float64)


def main():
    v107.state_residual = fast_state_residual
    v107.main()


if __name__ == "__main__":
    main()
