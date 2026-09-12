from __future__ import annotations

"""v32s: decontaminate Broadcast identity temporally, then run the v32q MHR solve.

The v32q direct multiview solve showed a strong RAR fit and a passing withheld LAR
validation, but Broadcast remained ~19.6 px median. Visual inspection showed the
exact-freeze Broadcast pose is contaminated by the overlapping Utah defender.

v32s changes observations, not geometry or acceptance thresholds:
  * search real Broadcast t+02..t+06 for the cleanest Adams RF-DETR pose;
  * dense-track that one articulated pose back through the real Broadcast burst;
  * keep direct t+00 joints only when they agree with the temporal identity track;
  * replace disagreeing t+00 joints with the temporal source measurement;
  * pass that fail-closed pseudo-detection into the unchanged v32q direct MHR fit.

RAR remains the existing real t+04 -> t+00 temporal identity solve. LAR remains
withheld from optimization. No generated RGB, fourth camera, interpolation,
upscale, capsule body, or Gaussian player anatomy is introduced.
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
DIRECT_AGREE_PX = 11.0
TEMPORAL_MIN_CONF = 0.20


def _select_direct_broadcast(model, bimg, bbox):
    dets = v32j.infer(model, bimg)
    rows = []
    for i, d in enumerate(dets):
        score = 4.0 * v32j.iou(d["box"], bbox) + 0.5 * v32j.dark_fraction(bimg, d["box"])
        rows.append((float(score), i))
    rows.sort(reverse=True)
    if not rows:
        raise RuntimeError("v32s: no direct Broadcast t+00 detections")
    return dets, rows[0][1]


def _anchor_quality(img, dets, idx, tracked_valid, anchor_conf, target_center):
    d = dets[idx]
    stable = [j for j in BODY if bool(tracked_valid[j]) and float(anchor_conf[j]) >= TEMPORAL_MIN_CONF]
    conf_med = float(np.median([anchor_conf[j] for j in stable])) if stable else 0.0
    centre = np.array([(d["box"][0] + d["box"][2]) * 0.5, (d["box"][1] + d["box"][3]) * 0.5], float)
    centre_dist = float(np.linalg.norm(centre - target_center))
    dark = float(v32j.dark_fraction(img, d["box"]))
    overlap = 0.0
    for k, other in enumerate(dets):
        if k != idx:
            overlap = max(overlap, float(v32j.iou(d["box"], other["box"])))
    score = 5.0 * len(stable) + 3.0 * conf_med + 2.0 * dark - 4.0 * overlap - centre_dist / 70.0
    return score, {
        "stable_body_joint_count": len(stable),
        "median_anchor_confidence": conf_med,
        "centre_distance_px": centre_dist,
        "dark_fraction": dark,
        "max_other_box_iou": overlap,
        "score": float(score),
    }


def build_temporal_broadcast(stage: Path, qual: dict):
    model = RFDETRKeypointPreview()
    frames = {rel: cv2.imread(str(v32k.burst_path(stage, base.BCAST, rel))) for rel in range(0, 7)}
    if not all(x is not None for x in frames.values()):
        raise RuntimeError("v32s: missing Broadcast burst frames 0..+6")

    bbox = np.asarray(qual["focal_player"]["observations"][base.BCAST]["bbox_xyxy"], float)
    target_center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], float)
    direct_dets, direct_idx = _select_direct_broadcast(model, frames[0], bbox)
    direct = direct_dets[direct_idx]

    candidates = []
    for rel in ANCHORS:
        dets = v32j.infer(model, frames[rel])
        idx, ranked = v32k.select_adams_detection(frames[rel], dets, target_center)
        if idx is None:
            candidates.append({"rel": rel, "usable": False, "reason": "no Adams candidate"})
            continue
        d = dets[idx]
        xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
        score, detail = _anchor_quality(frames[rel], dets, idx, valid0, d["conf"], target_center)
        candidates.append({
            "rel": rel,
            "usable": True,
            "idx": int(idx),
            "ranked": ranked,
            "xy0": xy0,
            "valid0": valid0,
            "anchor_conf": np.asarray(d["conf"], float),
            "trace": trace,
            "detail": detail,
            "score": float(score),
        })

    usable = [r for r in candidates if r.get("usable") and r["detail"]["stable_body_joint_count"] >= 7]
    if not usable:
        raise RuntimeError("v32s: no Broadcast temporal anchor retained >=7 body joints")
    usable.sort(key=lambda r: (r["score"], r["rel"]), reverse=True)
    best = usable[0]

    fused_xy = np.asarray(direct["xy"], float).copy()
    fused_conf = np.zeros_like(np.asarray(direct["conf"], float))
    joint_audit = {}
    temporal_count = 0
    agreeing_count = 0
    replaced_count = 0

    for j in range(len(fused_conf)):
        t_ok = bool(best["valid0"][j]) and float(best["anchor_conf"][j]) >= TEMPORAL_MIN_CONF
        d_ok = float(direct["conf"][j]) >= TEMPORAL_MIN_CONF
        if t_ok:
            txy = np.asarray(best["xy0"][j], float)
            temporal_count += 1
            if d_ok:
                delta = float(np.linalg.norm(np.asarray(direct["xy"][j], float) - txy))
                if delta <= DIRECT_AGREE_PX:
                    # Agreement is independent evidence. Stay closest to the temporal identity
                    # track while allowing the exact-state direct observation to sharpen it.
                    fused_xy[j] = 0.72 * txy + 0.28 * np.asarray(direct["xy"][j], float)
                    fused_conf[j] = float(min(1.0, max(best["anchor_conf"][j], direct["conf"][j])))
                    agreeing_count += 1
                    source = "temporal+direct_agree"
                else:
                    fused_xy[j] = txy
                    fused_conf[j] = float(min(0.95, best["anchor_conf"][j]))
                    replaced_count += 1
                    source = "temporal_replaces_disagreeing_direct"
            else:
                delta = None
                fused_xy[j] = txy
                fused_conf[j] = float(min(0.90, best["anchor_conf"][j]))
                source = "temporal_only"
        else:
            delta = None
            # Fail closed for body joints: a direct-only limb at this overlap is precisely the
            # contamination mode v32s is designed to remove. Non-body face points are irrelevant
            # to v32q's MHR body loss and may remain for diagnostic display only.
            if j not in BODY and d_ok:
                fused_xy[j] = np.asarray(direct["xy"][j], float)
                fused_conf[j] = float(direct["conf"][j])
                source = "direct_nonbody_only"
            else:
                fused_conf[j] = 0.0
                source = "rejected_no_temporal_support"
        joint_audit[str(j)] = {
            "name": v32j.NAMES[j] if j < len(v32j.NAMES) else str(j),
            "source": source,
            "direct_conf": float(direct["conf"][j]),
            "anchor_conf": float(best["anchor_conf"][j]),
            "direct_temporal_delta_px": delta,
            "fused_conf": float(fused_conf[j]),
        }

    retained_body = [j for j in BODY if fused_conf[j] >= TEMPORAL_MIN_CONF]
    if len(retained_body) < 7:
        raise RuntimeError(f"v32s: temporal Broadcast retained only {len(retained_body)} body joints")

    pseudo = dict(direct)
    pseudo["xy"] = fused_xy
    pseudo["conf"] = fused_conf
    # Keep the exact t+00 identity box only so v32q's existing B32 identity-selector selects
    # this observation. The body keypoints themselves come from the real temporal track above.
    pseudo["box"] = np.asarray(direct["box"], float)

    serial_candidates = []
    for r in candidates:
        rr = {k: v for k, v in r.items() if k not in ("xy0", "valid0", "anchor_conf", "trace", "ranked")}
        if r.get("usable"):
            rr["selected_detection_index"] = int(r["idx"])
        serial_candidates.append(rr)

    audit = {
        "version": "v32s_broadcast_temporal_identity",
        "anchor_candidates_rel": list(ANCHORS),
        "selected_anchor_rel": int(best["rel"]),
        "selected_anchor_detection_index": int(best["idx"]),
        "selected_anchor_quality": best["detail"],
        "candidate_summary": serial_candidates,
        "direct_t00_detection_index": int(direct_idx),
        "direct_agreement_threshold_px": DIRECT_AGREE_PX,
        "temporal_joint_count_all": int(temporal_count),
        "retained_body_joint_count": int(len(retained_body)),
        "direct_temporal_agree_count": int(agreeing_count),
        "disagreeing_direct_joints_replaced": int(replaced_count),
        "joint_audit": joint_audit,
        "policy": "temporal Adams identity is authoritative for overlapping Broadcast body joints; direct t+00 is used only when it agrees",
    }
    return frames[0], pseudo, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=300)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())
    b0, pseudo, audit = build_temporal_broadcast(stage, qual)
    (a.out / "v32s_broadcast_temporal_audit.json").write_text(json.dumps(audit, indent=2))

    original_infer = v32j.infer
    injected = {"n": 0}

    def patched_infer(model, img):
        if img.shape == b0.shape and np.array_equal(img, b0):
            injected["n"] += 1
            return [pseudo]
        return original_infer(model, img)

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
        raise RuntimeError(f"v32s: expected exactly one Broadcast t+00 injection, got {injected['n']}")

    qa_path = a.out / "v32q_qa.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        qa["version"] = "v32s_broadcast_temporal_identity_mhr"
        qa["broadcast_observation"] = audit
        qa["v32s_base_exit_code"] = int(exit_code)
        if qa.get("status", "").startswith("PASS_"):
            qa["status"] = "PASS_V32S_TEMPORAL_BROADCAST_MULTIVIEW_ANATOMICAL_POSITION"
        else:
            qa["status"] = "FAIL_CLOSED_V32S_TEMPORAL_BROADCAST_MULTIVIEW_ANATOMICAL_POSITION"
        (a.out / "v32s_qa.json").write_text(json.dumps(qa, indent=2))

    print(json.dumps({
        "broadcast_temporal_anchor_rel": audit["selected_anchor_rel"],
        "retained_body_joints": audit["retained_body_joint_count"],
        "direct_temporal_agreements": audit["direct_temporal_agree_count"],
        "direct_joints_replaced": audit["disagreeing_direct_joints_replaced"],
        "base_exit_code": exit_code,
    }, indent=2), flush=True)

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
