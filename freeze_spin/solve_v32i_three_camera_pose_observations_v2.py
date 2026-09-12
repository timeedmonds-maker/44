from __future__ import annotations

"""v32i pose observation solver v2.

The accepted Right-Above-Rim calibration uses the already-known projective depth
sign convention: image projection is correct although camera-space Z is negative.
Projection validity therefore depends on non-zero projective depth, not Z>0.
This wrapper changes only that convention; all source data, camera matrices,
identity anchors, triangulation and fail-closed gates remain in v1.
"""

import numpy as np

from freeze_spin import solve_v32i_three_camera_pose_observations as v1


def project_projective_sign_safe(cam, X):
    xc = cam["R"] @ (np.asarray(X, np.float64) - cam["C"])
    if abs(float(xc[2])) <= 1e-6:
        return None, float(xc[2])
    q = cam["K"] @ xc
    return q[:2] / q[2], float(xc[2])


def main():
    v1.project = project_projective_sign_safe
    v1.main()


if __name__ == "__main__":
    main()
