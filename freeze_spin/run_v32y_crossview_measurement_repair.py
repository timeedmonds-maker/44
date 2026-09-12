from __future__ import annotations

"""v32y: repair the two remaining measurement failures without changing QA gates.

Evidence from v32x:
- coherent LAR t+03 / detection 3 is the best whole-body state;
- one LAR<->RAR correspondence (right_wrist) exceeds the already-established
  32 px raw-state consistency tolerance;
- the held-out Broadcast selector visibly latched onto a Utah defender instead
  of Steven Adams.

v32y changes measurement acceptance/identity only:
1) keep the v32x rank-1 whole-body LAR state;
2) mark any LAR joint with calibrated LAR<->RAR epipolar error >32 px as
   unsupported (drop rather than invent/mix a replacement joint);
3) choose the Broadcast t+00 person detection by calibrated Broadcast<->RAR
   whole-body epipolar agreement, with dark-uniform appearance only as a
   secondary tie-breaker;
4) fit the unchanged v32v MHR objective to LAR+RAR only; Broadcast remains
   strictly held out;
5) preserve the unchanged v32v numerical gates.

No camera refinement, no temporal joint mixing, no generated RGB, no upscale.
"""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import run_v32x_lar_candidate_sweep as v32x
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF
MAX_ACCEPTED_PAIR_EPI_PX = 32.0
CTX = {"lar_rejected": [], "broadcast_rank": []}


def choose_rank1_with_crossview_prune(model, stage, qual, scene, rar_xy, rar_valid):
    candidates, frames = v32x.all_coherent_candidates(model, stage, scene, rar_xy, rar_valid)
    best = copy.deepcopy(candidates[0])
    rejected = []
    for row in best["epipolar"].get("per_joint", []):
        j = int(row["joint"])
        e = float(row["epi_px"])
        if e > MAX_ACCEPTED_PAIR_EPI_PX:
            best["valid0"][j] = False
            best["anchor_conf"][j] = 0.0
            rejected.append({"joint": j, "name": row.get("name", v32j.NAMES[j]), "epi_px": e})
    CTX.update({
        "scene": scene,
        "rar_xy": np.asarray(rar_xy, float),
        "rar_valid": np.asarray(rar_valid, bool),
        "lar_rejected": rejected,
        "lar_selected_pre_prune": v32x.serial_candidate(1, candidates[0]),
    })
    serial = [v32x.serial_candidate(i + 1, c) for i, c in enumerate(candidates)]
    return best, serial, frames


def broadcast_geometry_rank(img, detections, _bbox):
    """Direct t+00 held-out identity selector; never uses the fitted MHR body."""
    scene = CTX["scene"]
    rar_xy = CTX["rar_xy"]
    rar_valid = CTX["rar_valid"]
    rows = []
    for i, d in enumerate(detections):
        valid = np.asarray(d["conf"], float) >= MIN_CONF
        epi = v32v.epipolar_stats(scene, BCAST, RAR, np.asarray(d["xy"], float), valid, rar_xy, rar_valid)
        dark = float(v32j.dark_fraction(img, d["box"]))
        torso = float(v32w.appearance(d, img)[1])
        confident = int(sum(bool(valid[j]) for j in BODY))
        shortage = max(0, 7 - int(epi["joint_count"]))
        # Geometry dominates. Appearance cannot rescue a geometrically bad body.
        appearance_penalty = 3.0 * max(0.0, 0.35 - torso) + 1.5 * max(0.0, 0.25 - dark)
        score = float(epi["median_px"] + .30 * epi["p90_px"] + 14.0 * shortage + appearance_penalty)
        rows.append({
            "index": int(i), "score": score,
            "box_xyxy": np.asarray(d["box"], float).tolist(),
            "dark_fraction": dark, "torso_dark_score": torso,
            "confident_body_joints": confident, "epipolar": epi,
        })
    rows.sort(key=lambda r: (r["score"], r["epipolar"]["median_px"], -r["torso_dark_score"]))
    CTX["broadcast_rank"] = rows
    if not rows:
        return None, rows
    return int(rows[0]["index"]), rows


def repro_stats_with_joints(project_fn, Xw, camera_label, observations, cams, jslot):
    vals = []
    rows = []
    for j, row in observations.items():
        uv = project_fn(cams[camera_label], Xw[jslot[j]].reshape(1, 3))[0]
        if not np.all(np.isfinite(uv)):
            continue
        measured = np.asarray(row["xy"], float)
        err = float(np.linalg.norm(uv - measured))
        vals.append(err)
        rows.append({
            "joint": int(j), "name": v32j.NAMES[int(j)], "residual_px": err,
            "measured_xy": measured.tolist(), "projected_xy": np.asarray(uv, float).tolist(),
        })
    rows.sort(key=lambda r: r["residual_px"], reverse=True)
    if not vals:
        return {"joint_count": 0, "median_px": 999.0, "p75_px": 999.0,
                "p90_px": 999.0, "max_px": 999.0, "per_joint": rows}
    a = np.asarray(vals, float)
    return {
        "joint_count": int(len(a)), "median_px": float(np.median(a)),
        "p75_px": float(np.percentile(a, 75)), "p90_px": float(np.percentile(a, 90)),
        "max_px": float(np.max(a)), "per_joint": rows,
    }


def main():
    original_selector = v32v.choose_global_lar_state
    original_rank = v32v.focal_rank
    original_repro = v32v.repro_stats
    v32v.choose_global_lar_state = choose_rank1_with_crossview_prune
    v32v.focal_rank = broadcast_geometry_rank
    v32v.repro_stats = repro_stats_with_joints
    rc = 0
    try:
        v32v.main()
    except SystemExit as exc:
        rc = int(exc.code or 0)
    finally:
        v32v.choose_global_lar_state = original_selector
        v32v.focal_rank = original_rank
        v32v.repro_stats = original_repro

    # Recover --out path from the v32v CLI arguments passed through unchanged.
    out = None
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    if out is not None:
        qpath = out / "v32v_qa.json"
        qa = json.loads(qpath.read_text()) if qpath.exists() else None
        policy = {
            "version": "v32y_crossview_measurement_repair",
            "lar_policy": f"v32x rank1 whole-body state; drop only correspondences with LAR-RAR epipolar > {MAX_ACCEPTED_PAIR_EPI_PX:.1f}px",
            "lar_rejected": CTX.get("lar_rejected", []),
            "broadcast_policy": "direct t00 held-out identity selected by whole-body Broadcast-RAR epipolar geometry; not used in MHR optimization",
            "broadcast_rank": CTX.get("broadcast_rank", []),
            "qa": qa,
            "solver_exit_code": rc,
        }
        (out / "v32y_policy_and_residuals.json").write_text(json.dumps(policy, indent=2))
    if rc:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
