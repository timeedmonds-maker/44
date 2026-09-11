from __future__ import annotations

"""v24: integrate the validated v23 focal-player reconstruction with the v21 court atlas.

v23 corrected the foreground identity failure: the dark focal #12 player is now
reconstructed from Broadcast pose + Right-Above-Rim silhouette geometry and only
uses Left-Above-Rim for source-grounded visible pixels recovered by projection.
v21 independently improved the floor representation by replacing moving per-view
camera ownership with one persistent metric court atlas.

This experiment combines those two strongest validated components without changing
camera calibration, exact-state frames, focal-player geometry, ball geometry, or
source-pixel policy.

Native 960x540 only. No UHD, no upscale, no generated texture, no inpainting.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v20 as v20
from freeze_spin import build_three_camera_mesh_v21 as v21
from freeze_spin import build_three_camera_mesh_v23 as v23


def _arg_path(name: str) -> Path:
    i = sys.argv.index(name)
    return Path(sys.argv[i + 1])


def _frame_black_metrics(out: Path):
    rows = []
    for ang in (0, 5, 10, 15, 20, 25):
        p = out / f"v12_{ang:02d}deg.png"
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            rows.append({"angle_deg": ang, "status": "MISSING"})
            continue
        # Exact black is deliberate unsupported output in this R&D renderer.
        black = np.all(im == 0, axis=2)
        near_black = np.max(im, axis=2) <= 8
        rows.append({
            "angle_deg": int(ang),
            "black_pixels": int(black.sum()),
            "black_fraction": float(black.mean()),
            "near_black_fraction": float(near_black.mean()),
            "nonblack_fraction": float((~black).mean()),
        })
    return rows


def main():
    # v23 ultimately calls v22 -> v20.main(). Redirect that single call to the
    # registered persistent court-atlas renderer from v21. All v23 foreground
    # monkeypatches are installed before the renderer is entered and therefore
    # remain active through the v21 -> v17 -> ... -> v12 execution chain.
    v20.main = v21.main

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v23.main()

    out = _arg_path("--out")
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    q["v24_integration"] = {
        "resolution": [960, 540],
        "foreground": "v23 dark focal #12 Broadcast-pose + RAR-silhouette geometry with projected LAR source-pixel recovery",
        "floor": "v21 persistent registered metric source-pixel court atlas",
        "ball": q.get("ball", {}),
        "focal_mesh_count": q.get("v23_focal_subject", {}).get("accepted_focal_mesh_count", 0),
        "focal_mesh_views": q.get("v23_focal_subject", {}).get("accepted_focal_mesh_views", []),
        "frame_black_metrics": _frame_black_metrics(out),
        "generated_texture": False,
        "inpainting": False,
        "crossfade": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v24_integration"], indent=2), flush=True)


if __name__ == "__main__":
    main()
