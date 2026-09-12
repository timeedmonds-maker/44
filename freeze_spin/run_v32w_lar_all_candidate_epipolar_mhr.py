from __future__ import annotations

"""v32w: geometry-select the coherent LAR Adams state, then run v32v MHR.

v32v proved that a manually/manifest-selected LAR detector person can be the
wrong Houston player.  v32w removes that brittle identity anchor completely.
For each real LAR candidate state (direct t+00 and whole-pose tracks from real
+2..+6 anchors), EVERY RF-DETR person detection is evaluated as one coherent
body against the trusted RAR t+00 pose using the accepted calibrated epipolar
geometry.  No joints are mixed across people or times.

The winning complete LAR candidate is then passed into the unchanged v32v
LAR+RAR articulated MHR solve.  Broadcast remains strictly held out.
"""

import json
import math

import cv2
import numpy as np

from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR, RAR = v32v.LAR, v32v.RAR
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF


def appearance(d, img):
    dark = float(v32j.dark_fraction(img, d["box"]))
    torso = float(v32k.torso_dark_score(img, d))
    confident = int(sum(float(d["conf"][j]) >= MIN_CONF for j in BODY))
    return dark, torso, confident


def coherent_candidate(scene, source, rel, det_index, d, img, xy0, valid0, trace, rar_xy, rar_valid):
    valid0 = np.asarray(valid0, bool) & (np.asarray(d["conf"], float) >= MIN_CONF)
    epi = v32v.epipolar_stats(scene, LAR, RAR, np.asarray(xy0, float), valid0, rar_xy, rar_valid)
    dark, torso, confident = appearance(d, img)
    shortage = max(0, 8 - int(epi["joint_count"]))
    # Epipolar agreement is primary. Appearance only breaks plausible geometric
    # ties toward Houston's dark uniform; it cannot rescue a bad geometric fit.
    appearance_penalty = 4.0 * max(0.0, 0.45 - torso) + 2.0 * max(0.0, 0.35 - dark)
    temporal_penalty = 0.0 if rel == 0 else 0.35
    score = float(epi["median_px"] + .30 * epi["p90_px"] + 12.0 * shortage + appearance_penalty + temporal_penalty)
    return {
        "source": source,
        "anchor_rel": int(rel),
        "selected_index": int(det_index),
        "xy0": np.asarray(xy0, float),
        "valid0": valid0,
        "anchor_conf": np.asarray(d["conf"], float),
        "trace": trace,
        "epipolar": epi,
        "selection_score": score,
        "dark_fraction": dark,
        "torso_dark_score": torso,
        "confident_body_joints": confident,
        "box_xyxy": np.asarray(d["box"], float).tolist(),
        "det_conf": float(d.get("det_conf", 0.0)),
    }


def choose_global_lar_state_all_candidates(model, stage, qual, scene, rar_xy, rar_valid):
    frames = v32v.load_burst(stage, LAR, range(0, 7))
    candidates = []

    for rel in (0, 2, 3, 4, 5, 6):
        dets = v32j.infer(model, frames[rel])
        for i, d in enumerate(dets):
            dark, torso, confident = appearance(d, frames[rel])
            # Keep broad recall. Only discard detections with too little body
            # support to form a meaningful coherent cross-view constraint.
            if confident < 6:
                continue
            if rel == 0:
                xy0 = np.asarray(d["xy"], float)
                valid0 = np.asarray(d["conf"], float) >= MIN_CONF
                trace = None
                source = "direct_t00"
            else:
                xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
                source = f"temporal_t+{rel}"
            c = coherent_candidate(scene, source, rel, i, d, frames[rel], xy0, valid0, trace, rar_xy, rar_valid)
            # Require at least six actually comparable body joints after flow.
            if int(c["epipolar"]["joint_count"]) >= 6:
                candidates.append(c)

    if not candidates:
        raise RuntimeError("v32w: no coherent LAR candidate has >=6 RAR-comparable joints")

    candidates.sort(key=lambda c: (c["selection_score"], c["epipolar"]["median_px"], -c["torso_dark_score"]))
    best = candidates[0]

    serial = []
    for rank, c in enumerate(candidates, start=1):
        serial.append({
            "rank": rank,
            "source": c["source"],
            "anchor_rel": c["anchor_rel"],
            "selected_index": c["selected_index"],
            "box_xyxy": c["box_xyxy"],
            "selection_score": c["selection_score"],
            "dark_fraction": c["dark_fraction"],
            "torso_dark_score": c["torso_dark_score"],
            "confident_body_joints": c["confident_body_joints"],
            "epipolar": c["epipolar"],
        })

    # Extra audit independent of v32v's standard QA.
    audit = {
        "version": "v32w_lar_all_candidate_epipolar_selection",
        "policy": "select one whole-body LAR person/time candidate by geometry; never mix joints across people or times",
        "candidate_count": len(serial),
        "winner": serial[0],
        "top_candidates": serial[:20],
    }
    (stage.parent / "v32w_lar_selection_preview.json").write_text(json.dumps(audit, indent=2))
    return best, serial, frames


def main():
    # Monkey-patch only the LAR identity/state selector. All camera models,
    # MHR fitting, held-out Broadcast validation and fail-closed gates remain
    # the v32v implementation.
    v32v.choose_global_lar_state = choose_global_lar_state_all_candidates
    v32v.main()


if __name__ == "__main__":
    main()
