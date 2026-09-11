from __future__ import annotations

"""v28: keep v26 temporal angular arena coverage but restore the clean v21 floor.

Visual QA established a useful split result:
- v26 materially improved distant arena angular coverage using only accepted
  same-camera temporal registrations, but its temporal floor expansion admitted
  a conspicuous bright court seam;
- v27 restored the clean v21 exact-state court atlas, but its coarse finite shell
  lost too much arena coverage.

v28 therefore combines only the validated parts: v23 focal-player geometry,
three-view ball, clean v21 persistent metric court atlas, and v26 accepted-temporal
infinity-depth arena coverage.  v26's temporal floor-atlas hook is deliberately
NOT installed.  This is an ablation/integration step before changing foreground
or adding new geometry.

Native 960x540 only. No generated texture, inpainting, crossfade, UHD or upscale.
Every rendered RGB sample comes from official native source footage.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_mesh_v12 as v12
from freeze_spin import build_three_camera_mesh_v15 as v15
from freeze_spin import build_three_camera_mesh_v21 as v21
from freeze_spin import build_three_camera_mesh_v24 as v24
from freeze_spin import build_three_camera_mesh_v26 as v26


def _pop_arg(name: str) -> str:
    i = sys.argv.index(name)
    value = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    return value


def _arg_path(name: str) -> Path:
    i = sys.argv.index(name)
    return Path(sys.argv[i + 1])


def _black_metrics(out: Path):
    rows = []
    for ang in (0, 5, 10, 15, 20, 25):
        im = cv2.imread(str(out / f"v12_{ang:02d}deg.png"), cv2.IMREAD_COLOR)
        if im is None:
            rows.append({"angle_deg": ang, "status": "MISSING"})
            continue
        black = np.all(im == 0, axis=2)
        rows.append({
            "angle_deg": int(ang),
            "black_fraction": float(black.mean()),
            "nonblack_fraction": float((~black).mean()),
        })
    return rows


def main():
    # Configure v26's accepted temporal registrations, but do not call v26.main()
    # because that would patch v21._source_atlas to the contaminated temporal
    # floor expansion.  v24 retains v21's clean exact-state registered atlas.
    v26._TEMPORAL.clear(); v26._REG_QA.clear(); v26._ATLAS_QA.clear(); v26._BG_QA.clear()
    v26._CONTEXT = None
    v26._CLIPS_DIR = Path(_pop_arg("--clips-dir"))
    # --sync-qa is also required by the underlying v12 parser, so read it without
    # removing it from sys.argv.
    v26._SYNC_QA_PATH = _arg_path("--sync-qa")

    # v15.main installs its current background function into v14 immediately
    # before the lower renderer runs. Redirect only that symbol.
    v15.far_background_multisource_clean = v26.far_background_temporal_angular

    assert (int(v12.W), int(v12.H)) == (960, 540), (v12.W, v12.H)
    v24.main()

    out = _arg_path("--out")
    qp = out / "three_camera_mesh_v12_qa.json"
    q = json.loads(qp.read_text())
    q["v28_clean_floor_temporal_background"] = {
        "resolution": [960, 540],
        "floor": "unchanged clean v21 exact-state registered metric court atlas",
        "background": "v26 accepted-temporal multi-camera angular coverage; rejected temporal registrations never sampled",
        "temporal_registration": v26._REG_QA,
        "background_render_calls": v26._BG_QA,
        "temporal_floor_atlas_enabled": False,
        "foreground_geometry_change": False,
        "ball_geometry_change": False,
        "black_metrics": _black_metrics(out),
        "generated_texture": False,
        "inpainting": False,
        "crossfade": False,
        "upscale": False,
        "uhd": False,
    }
    qp.write_text(json.dumps(q, indent=2))
    print(json.dumps(q["v28_clean_floor_temporal_background"], indent=2), flush=True)


if __name__ == "__main__":
    main()
