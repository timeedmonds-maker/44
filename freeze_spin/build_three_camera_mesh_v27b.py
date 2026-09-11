from __future__ import annotations

"""v27b runtime correction for v27 arena-shell intersection masks.

The first v27 run reached the new finite-depth renderer and failed before image
formation because NumPy parsed a multiline comparison/bitwise expression with
comparison precedence different from the intended boolean conjunction.  This
wrapper changes only the shell ray/plane mask construction.  All v27 geometry,
source-pixel policy, temporal acceptance gates, foreground, ball and court remain
unchanged.
"""

import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v27 as v27


def arena_shell_points_fixed(K, R, C):
    C0, d = v27._target_rays(K, R, C)
    C0 = C0[0]
    n = len(d)
    best_t = np.full(n, np.inf, np.float64)
    best_P = np.zeros((n, 3), np.float64)
    best_plane = np.full(n, -1, np.int8)

    def offer(t, P, valid, pid):
        take = valid & np.isfinite(t) & (t > 20.0) & (t < best_t)
        if np.any(take):
            best_t[take] = t[take]
            best_P[take] = P[take]
            best_plane[take] = int(pid)

    den = d[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (v27.BASELINE_X - C0[0]) / den
    P = C0[None, :] + t[:, None] * d
    valid = (
        (np.abs(den) > 1e-8)
        & (np.abs(P[:, 1]) <= v27.Y_HALF_BASE)
        & (P[:, 2] >= v27.Z_MIN) & (P[:, 2] <= v27.Z_MAX)
    )
    offer(t, P, valid, 0)

    for pid, yplane in ((1, -v27.SIDE_Y), (2, v27.SIDE_Y)):
        den = d[:, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (yplane - C0[1]) / den
        P = C0[None, :] + t[:, None] * d
        valid = (
            (np.abs(den) > 1e-8)
            & (P[:, 0] >= v27.X_MIN_SIDE) & (P[:, 0] <= v27.X_MAX_SIDE)
            & (P[:, 2] >= v27.Z_MIN) & (P[:, 2] <= v27.Z_MAX)
        )
        offer(t, P, valid, pid)

    den = d[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (v27.FAR_X - C0[0]) / den
    P = C0[None, :] + t[:, None] * d
    valid = (
        (np.abs(den) > 1e-8)
        & (np.abs(P[:, 1]) <= v27.Y_HALF_BASE)
        & (P[:, 2] >= v27.Z_MIN) & (P[:, 2] <= v27.Z_MAX)
    )
    offer(t, P, valid, 3)
    return best_P, np.isfinite(best_t), best_plane, best_t


def main():
    v27._arena_shell_points = arena_shell_points_fixed
    assert (int(v12.W), int(v12.H)) == (960, 540)
    v27.main()


if __name__ == '__main__':
    main()
