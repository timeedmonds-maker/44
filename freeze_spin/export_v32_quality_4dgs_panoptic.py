from __future__ import annotations

"""Quality-focused wrapper around the v32 4DGaussians exporter.

The full source frames remain available to the backend, but the dynamic person
initialization is restricted to the quality-qualified action cluster selected by
select_v32_quality_cluster.py.  This allocates Gaussian capacity to Adams and
near-action players we can defend visually, while peripheral people are not part
of the final quality contract.
"""

import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import export_v32_4dgs_panoptic as base


def _person_seed(xy_m, col):
    pts, cols = [], []
    for z in np.arange(0.10, 2.21, 0.14):
        # Denser than the previous whole-scene seed because this cluster is now
        # intentionally small and receives the dynamic capacity budget.
        r = 0.10 if z < 0.25 else (0.12 if z > 1.85 else 0.23)
        n = 10 if r >= 0.2 else 6
        for a in np.linspace(0, 2*np.pi, n, endpoint=False):
            pts.append([xy_m[0] + r*np.cos(a), xy_m[1] + r*np.sin(a), z])
            cols.append(col)
    return pts, cols


def _obs_color(stage_a: Path, obs):
    # Works for both ordinary Stage-A observations and focal observation dicts.
    p = obs.get("rgba_crop")
    if not p:
        return None
    return base.crop_color(stage_a, {"rgba_crop": p})


def quality_player_initial_points(manifest: dict, stage_a: Path):
    qp = stage_a / "v32_quality_cluster.json"
    if not qp.is_file():
        raise RuntimeError("v32_quality_cluster.json missing: refuse whole-scene player seeding")
    q = json.loads(qp.read_text())
    pts, cols = [], []

    focal = q["focal_player"]
    fxy = np.asarray(focal["centroid_xy_cm"], np.float64) / 100.0
    fcols = []
    for o in focal.get("observations", {}).values():
        c = _obs_color(stage_a, o)
        if c is not None: fcols.append(c)
    fcol = np.median(np.stack(fcols), axis=0) if fcols else np.asarray([.15,.15,.15],np.float32)
    p, c = _person_seed(fxy, fcol); pts.extend(p); cols.extend(c)

    for tr in q.get("supporting_players", []):
        xy = np.asarray(tr["centroid_xy_cm"], np.float64) / 100.0
        cs = []
        for o in tr.get("observations", []):
            col = _obs_color(stage_a, o)
            if col is not None: cs.append(col)
        col = np.median(np.stack(cs), axis=0) if cs else np.asarray([.45,.45,.45],np.float32)
        p, c = _person_seed(xy, col); pts.extend(p); cols.extend(c)

    return np.asarray(pts,np.float32), np.asarray(cols,np.float32)


def main():
    # Restrict only the player initialization; preserve the accepted camera,
    # frame, court, ball, and held-out dataset logic from the base exporter.
    base.player_initial_points = quality_player_initial_points
    base.main()

    # Enrich backend manifest with the quality-cluster contract.
    argv = sys.argv
    out = Path(argv[argv.index("--out") + 1])
    stage = Path(argv[argv.index("--stage-a") + 1])
    bpath = out / "v32_4dgs_backend.json"
    b = json.loads(bpath.read_text())
    q = json.loads((stage / "v32_quality_cluster.json").read_text())
    b["person_quality_policy"] = "quality-qualified action cluster; peripheral people excluded from required final composition"
    b["quality_required_person_count"] = q["quality_required_person_count"]
    b["supporting_player_count"] = q["supporting_player_count"]
    b["excluded_person_track_count"] = q["excluded_person_track_count"]
    b["quality_cluster_manifest"] = "../stage_a/v32_quality_cluster.json"
    b["final_camera_contract"] = q["final_camera_contract"]
    b["evaluation_order"] = [
        "held-out-camera validation on Adams plus selected supporting action players",
        "exclude any supporting player that fails visual quality instead of degrading the scene",
        "fit final camera path/framing so excluded peripheral people are not conspicuously visible",
        "only after held-out pass, train all-three-camera model and render 0/5/10/15/20/25 degree static arc",
    ]
    bpath.write_text(json.dumps(b, indent=2))
    print(json.dumps({
        "quality_required_person_count": q["quality_required_person_count"],
        "supporting_track_ids": [x["track_id"] for x in q["supporting_players"]],
        "excluded_person_track_count": q["excluded_person_track_count"],
        "player_seed_points": b["initial_point_cloud"]["player_seed_points"],
    }, indent=2))


if __name__ == "__main__":
    main()
