from __future__ import annotations

"""V32c: tighten the v32b learned-foreground ROI around quality-qualified action.

The v32b world-box projection was intentionally conservative and left too much
peripheral court/crowd in the Broadcast training image.  For the learned dynamic
foreground we want capacity spent on the action cluster only.  This wrapper keeps
v32b's quality-qualified track selection, calibrated cameras, held-out-camera
contracts and source-only pixel policy, but defines the per-camera ROI directly
from selected source detections plus the user-validated focal action seed and rim.
"""

import math
import numpy as np

from freeze_spin import export_v32b_quality_cluster_4dgs as v32b

W, H = v32b.W, v32b.H

FOCAL_SEEDS = {
    "Left Above Rim": [438, 160, 512, 326],
    "Broadcast": [472, 100, 570, 312],
    "Right Above Rim": [514, 184, 662, 340],
}


def tight_projected_roi(cam: dict, selected: list[dict], label: str, world_box: list[float]):
    del world_box
    fs = FOCAL_SEEDS[label]
    pts = [np.asarray([fs[0], fs[1]], np.float64), np.asarray([fs[2], fs[3]], np.float64)]
    for tr in selected:
        for o in tr.get("observations", []):
            if o.get("camera") == label:
                x1, y1, x2, y2 = o["bbox_xyxy"]
                pts += [np.asarray([x1, y1], np.float64), np.asarray([x2, y2], np.float64)]
    # Basket/rim is mandatory context for this action and gives a stable rigid
    # reference even when Adams is occluded in one of the solved views.
    uv, ok = v32b.project_points_cm(np.asarray([[0.0, 0.0, 305.0]], np.float64), cam)
    if bool(ok[0]):
        p = uv[0]
        pts += [p - np.asarray([38.0, 38.0]), p + np.asarray([38.0, 38.0])]
    P = np.asarray(pts, np.float64)
    pad = 50
    xa = int(max(0, math.floor(np.min(P[:, 0]) - pad)))
    xb = int(min(W - 1, math.ceil(np.max(P[:, 0]) + pad)))
    ya = int(max(0, math.floor(np.min(P[:, 1]) - pad)))
    yb = int(min(H - 1, math.ceil(np.max(P[:, 1]) + pad)))
    if xb - xa < 120 or yb - ya < 120:
        raise RuntimeError(f"implausibly small tight action ROI for {label}: {(xa,ya,xb,yb)}")
    return [xa, ya, xb, yb]


def main():
    v32b.projected_roi = tight_projected_roi
    v32b.main()


if __name__ == "__main__":
    main()
