from __future__ import annotations

"""v32t: per-joint Broadcast source selection using calibrated RAR epipolar evidence.

v32q showed the exact-freeze Broadcast detector is contaminated by a defender.
v32s showed that replacing the whole Broadcast pose with one dense-flow track is
too coarse: identity improved, but some limb tracks produced large tail errors.

v32t therefore treats each body joint independently. Candidate observations are:
  - direct real Broadcast t+00 RF-DETR;
  - real Broadcast t+02..t+06 RF-DETR joints dense-tracked back to t+00.
The trusted RAR t+04 -> t+00 temporal observation supplies cross-view evidence.
For every body joint, the Broadcast candidate with the lowest symmetric epipolar
error to RAR is selected. Direct t+00 wins ties to avoid unnecessary flow error.
Candidates with weak identity, unstable flow, or poor epipolar consistency are
rejected rather than invented.

The selected 2-D Broadcast observation is then injected into the unchanged v32q
image-space MHR fit. LAR remains fully withheld until validation. No 3-D joint
positions are used as fit targets and no pixels are generated.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview

from freeze_spin import fit_v32q_mhr_direct_multiview_2d as base
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

BODY = tuple(range(5, 17))
ANCHORS = (2, 3, 4, 5, 6)
MIN_CONF = 0.20
MAX_ANCHOR_CENTER_DIST = 75.0
MAX_EPI_PX = 14.0
DIRECT_TIE_MARGIN_PX = 1.5


def direct_broadcast(model, img, bbox):
    dets = v32j.infer(model, img)
    ranked = []
    for i, d in enumerate(dets):
        score = 4.0 * v32j.iou(d["box"], bbox) + 0.5 * v32j.dark_fraction(img, d["box"])
        ranked.append((float(score), i))
    ranked.sort(reverse=True)
    if not ranked:
        raise RuntimeError("v32t: no Broadcast t+00 detections")
    return dets, ranked[0][1]


def temporal_rar_t0(model, stage: Path, qual: dict):
    frames = {rel: cv2.imread(str(v32k.burst_path(stage, base.RAR, rel))) for rel in range(0, 5)}
    if not all(x is not None for x in frames.values()):
        raise RuntimeError("v32t: missing RAR frames 0..+4")
    rb = np.asarray(qual["focal_player"]["observations"][base.RAR]["bbox_xyxy"], float)
    target_center = np.array([(rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5], float)
    dets = v32j.infer(model, frames[4])
    idx, ranked = v32k.select_adams_detection(frames[4], dets, target_center)
    if idx is None:
        raise RuntimeError("v32t: no RAR t+04 Adams candidate")
    d = dets[idx]
    xy0, valid0, trace = v32l.dense_track_back(frames, 4, d["xy"], d["conf"])
    return {
        "xy0": np.asarray(xy0, float),
        "valid0": np.asarray(valid0, bool),
        "anchor_conf": np.asarray(d["conf"], float),
        "selected_index": int(idx),
        "ranked": ranked,
        "trace": trace,
    }


def broadcast_candidates(model, stage: Path, qual: dict):
    frames = {rel: cv2.imread(str(v32k.burst_path(stage, base.BCAST, rel))) for rel in range(0, 7)}
    if not all(x is not None for x in frames.values()):
        raise RuntimeError("v32t: missing Broadcast frames 0..+6")
    bb = np.asarray(qual["focal_player"]["observations"][base.BCAST]["bbox_xyxy"], float)
    target_center = np.array([(bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5], float)
    direct_dets, direct_idx = direct_broadcast(model, frames[0], bb)
    direct = direct_dets[direct_idx]

    rows = []
    for rel in ANCHORS:
        dets = v32j.infer(model, frames[rel])
        idx, ranked = v32k.select_adams_detection(frames[rel], dets, target_center)
        if idx is None:
            rows.append({"rel": rel, "usable": False, "reason": "no_candidate"})
            continue
        d = dets[idx]
        center = np.array([(d["box"][0] + d["box"][2]) * 0.5, (d["box"][1] + d["box"][3]) * 0.5], float)
        center_dist = float(np.linalg.norm(center - target_center))
        dark = float(v32j.dark_fraction(frames[rel], d["box"]))
        xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
        stable = int(sum(bool(valid0[j]) and float(d["conf"][j]) >= MIN_CONF for j in BODY))
        usable = bool(center_dist <= MAX_ANCHOR_CENTER_DIST and dark >= 0.50 and stable >= 7)
        rows.append({
            "rel": rel, "usable": usable, "idx": int(idx), "center_dist": center_dist,
            "dark": dark, "stable_body": stable, "xy0": np.asarray(xy0, float),
            "valid0": np.asarray(valid0, bool), "anchor_conf": np.asarray(d["conf"], float),
            "ranked": ranked, "trace": trace,
        })
    return frames[0], direct, int(direct_idx), rows


def choose_joint_sources(scene: dict, direct: dict, temporal_rows, rar: dict):
    cams = {c: v32j.cam(scene, c) for c in base.CAMS}
    F = v32j.fundamental(cams[base.BCAST], cams[base.RAR])

    xy = np.asarray(direct["xy"], float).copy()
    conf = np.zeros_like(np.asarray(direct["conf"], float))
    audit = {}

    for j in BODY:
        r_ok = bool(rar["valid0"][j]) and float(rar["anchor_conf"][j]) >= MIN_CONF
        if not r_ok:
            audit[str(j)] = {"name": v32j.NAMES[j], "accepted": False, "reason": "no_trusted_rar_joint", "candidates": []}
            continue
        ruv = np.asarray(rar["xy0"][j], float)
        candidates = []
        if float(direct["conf"][j]) >= MIN_CONF:
            uv = np.asarray(direct["xy"][j], float)
            candidates.append({
                "source": "direct_t00", "rel": 0, "uv": uv, "confidence": float(direct["conf"][j]),
                "epi_px": float(v32j.epi(F, uv, ruv)),
            })
        for row in temporal_rows:
            if not row.get("usable"):
                continue
            if not bool(row["valid0"][j]) or float(row["anchor_conf"][j]) < MIN_CONF:
                continue
            uv = np.asarray(row["xy0"][j], float)
            candidates.append({
                "source": f"temporal_t+{row['rel']}", "rel": int(row["rel"]), "uv": uv,
                "confidence": float(row["anchor_conf"][j]), "epi_px": float(v32j.epi(F, uv, ruv)),
            })
        if not candidates:
            audit[str(j)] = {"name": v32j.NAMES[j], "accepted": False, "reason": "no_broadcast_candidate", "candidates": []}
            continue
        candidates.sort(key=lambda x: (x["epi_px"], -x["confidence"], abs(x["rel"])))
        best = candidates[0]
        direct_row = next((c for c in candidates if c["source"] == "direct_t00"), None)
        if direct_row is not None and direct_row["epi_px"] <= best["epi_px"] + DIRECT_TIE_MARGIN_PX:
            best = direct_row
        accepted = bool(best["epi_px"] <= MAX_EPI_PX)
        if accepted:
            xy[j] = best["uv"]
            conf[j] = min(0.99, max(MIN_CONF, best["confidence"]))
        audit[str(j)] = {
            "name": v32j.NAMES[j], "accepted": accepted,
            "selected_source": best["source"] if accepted else None,
            "selected_epi_px": float(best["epi_px"]),
            "rar_xy": ruv.tolist(),
            "candidates": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in candidates],
        }

    # Non-body points are not used by the MHR body loss. Retain only direct face points for display.
    for j in range(5):
        if float(direct["conf"][j]) >= MIN_CONF:
            xy[j] = np.asarray(direct["xy"][j], float)
            conf[j] = float(direct["conf"][j])

    retained = [j for j in BODY if conf[j] >= MIN_CONF]
    if len(retained) < 8:
        raise RuntimeError(f"v32t: epipolar selection retained only {len(retained)} Broadcast body joints")
    pseudo = dict(direct)
    pseudo["xy"] = xy
    pseudo["conf"] = conf
    pseudo["box"] = np.asarray(direct["box"], float)
    return pseudo, audit, retained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=320)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())
    model = RFDETRKeypointPreview()
    b0, direct, direct_idx, temporal_rows = broadcast_candidates(model, stage, qual)
    rar = temporal_rar_t0(model, stage, qual)
    pseudo, joint_audit, retained = choose_joint_sources(scene, direct, temporal_rows, rar)

    serial_temporal = []
    for r in temporal_rows:
        serial_temporal.append({k: v for k, v in r.items() if k not in ("xy0", "valid0", "anchor_conf", "ranked", "trace")})
    selected_counts = {}
    for row in joint_audit.values():
        src = row.get("selected_source")
        if src:
            selected_counts[src] = selected_counts.get(src, 0) + 1
    audit = {
        "version": "v32t_epipolar_broadcast_source_selection",
        "direct_detection_index": direct_idx,
        "rar_t04_detection_index": rar["selected_index"],
        "max_epipolar_error_px": MAX_EPI_PX,
        "direct_tie_margin_px": DIRECT_TIE_MARGIN_PX,
        "retained_body_joint_count": len(retained),
        "retained_body_joints": [v32j.NAMES[j] for j in retained],
        "selected_source_counts": selected_counts,
        "broadcast_temporal_anchor_summary": serial_temporal,
        "joint_audit": joint_audit,
        "policy": "each Broadcast joint independently selected by symmetric epipolar agreement with trusted temporal RAR; no 3-D target enters MHR loss",
    }
    (a.out / "v32t_epipolar_selection_audit.json").write_text(json.dumps(audit, indent=2))

    original_infer = v32j.infer
    injected = {"n": 0}
    def patched_infer(model_obj, img):
        if img.shape == b0.shape and np.array_equal(img, b0):
            injected["n"] += 1
            return [pseudo]
        return original_infer(model_obj, img)
    v32j.infer = patched_infer
    base.v32j.infer = patched_infer

    old_argv = sys.argv[:]
    exit_code = 0
    try:
        sys.argv = [old_argv[0], "--b32-root", str(a.b32_root), "--work", str(a.work), "--out", str(a.out), "--max-nfev", str(a.max_nfev)]
        try:
            base.main()
        except SystemExit as e:
            exit_code = int(e.code or 0)
    finally:
        sys.argv = old_argv
        v32j.infer = original_infer
        base.v32j.infer = original_infer

    if injected["n"] != 1:
        raise RuntimeError(f"v32t: expected one Broadcast injection, got {injected['n']}")
    qpath = a.out / "v32q_qa.json"
    if qpath.exists():
        q = json.loads(qpath.read_text())
        q["version"] = "v32t_epipolar_selected_broadcast_mhr"
        q["broadcast_observation_selection"] = audit
        q["v32t_base_exit_code"] = exit_code
        q["status"] = ("PASS_V32T_EPIPOLAR_SELECTED_MULTIVIEW_ANATOMICAL_POSITION" if q.get("status", "").startswith("PASS_")
                       else "FAIL_CLOSED_V32T_EPIPOLAR_SELECTED_MULTIVIEW_ANATOMICAL_POSITION")
        (a.out / "v32t_qa.json").write_text(json.dumps(q, indent=2))

    print(json.dumps({"retained_body_joints": len(retained), "selected_source_counts": selected_counts, "base_exit_code": exit_code}, indent=2), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
