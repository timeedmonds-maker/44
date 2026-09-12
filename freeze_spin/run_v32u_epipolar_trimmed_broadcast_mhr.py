from __future__ import annotations

"""v32u: adaptive/top-k epipolar trimming for Broadcast observations.

v32t proved calibrated epipolar ranking is useful but its absolute 14 px cutoff
was too strict under the accepted three-camera calibration/timing residuals,
retaining only three Broadcast body joints. v32u keeps the ranking and removes
the brittle absolute cutoff.

For every RAR-supported body joint:
  * compare direct real Broadcast t+00 against each valid real +2..+6 pose
    dense-tracked back to t+00;
  * choose the Broadcast candidate with minimum symmetric epipolar error to the
    trusted real RAR t+04 -> t+00 temporal joint, with a small direct tie bias;
  * reject only absurd (>60 px) cross-view mismatches;
  * rank the surviving joints by epipolar error and retain the best nine
    (or the best eight if only eight survive).

This is deterministic outlier trimming, not threshold relaxation: the unchanged
v32q direct image-space MHR fit still has to pass Broadcast, RAR and withheld LAR
reprojection gates. No 3-D joint target, generated RGB, fourth camera or upscale.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import fit_v32q_mhr_direct_multiview_2d as base
from freeze_spin import run_v32t_epipolar_broadcast_mhr as v32t
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

BODY = v32t.BODY
MIN_CONF = v32t.MIN_CONF
ABSURD_EPI_PX = 60.0
TARGET_KEEP = 9
MIN_KEEP = 8
DIRECT_TIE_MARGIN_PX = 1.5


def choose_joint_sources_trimmed(scene: dict, direct: dict, temporal_rows, rar: dict):
    cams = {c: v32j.cam(scene, c) for c in base.CAMS}
    F = v32j.fundamental(cams[base.BCAST], cams[base.RAR])

    xy = np.asarray(direct["xy"], float).copy()
    conf = np.zeros_like(np.asarray(direct["conf"], float))
    ranked_rows = []
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
                "source": "direct_t00", "rel": 0, "uv": uv,
                "confidence": float(direct["conf"][j]),
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
                "confidence": float(row["anchor_conf"][j]),
                "epi_px": float(v32j.epi(F, uv, ruv)),
            })
        if not candidates:
            audit[str(j)] = {"name": v32j.NAMES[j], "accepted": False, "reason": "no_broadcast_candidate", "candidates": []}
            continue
        candidates.sort(key=lambda x: (x["epi_px"], -x["confidence"], abs(x["rel"])))
        best = candidates[0]
        direct_row = next((c for c in candidates if c["source"] == "direct_t00"), None)
        if direct_row is not None and direct_row["epi_px"] <= best["epi_px"] + DIRECT_TIE_MARGIN_PX:
            best = direct_row
        row = {
            "joint": j,
            "name": v32j.NAMES[j],
            "best": best,
            "rar_xy": ruv,
            "candidates": candidates,
        }
        if best["epi_px"] <= ABSURD_EPI_PX:
            ranked_rows.append(row)
        audit[str(j)] = {
            "name": v32j.NAMES[j], "accepted": False,
            "selected_source": None,
            "selected_epi_px": float(best["epi_px"]),
            "rar_xy": ruv.tolist(),
            "candidates": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in candidates],
        }

    ranked_rows.sort(key=lambda r: (r["best"]["epi_px"], -r["best"]["confidence"], abs(r["best"]["rel"])))
    if len(ranked_rows) < MIN_KEEP:
        raise RuntimeError(f"v32u: only {len(ranked_rows)} Broadcast joints survive absurd-error guard")

    # Keep the strongest nine cross-view constraints. This intentionally drops
    # the worst outlying limbs rather than letting them drag the articulated body.
    keep_n = min(TARGET_KEEP, len(ranked_rows))
    retained_rows = ranked_rows[:keep_n]
    retained = []
    for rank, row in enumerate(retained_rows, start=1):
        j = row["joint"]; best = row["best"]
        xy[j] = best["uv"]
        conf[j] = float(min(0.99, max(MIN_CONF, best["confidence"])))
        retained.append(j)
        audit[str(j)].update({
            "accepted": True,
            "selection_rank": rank,
            "selected_source": best["source"],
            "selected_epi_px": float(best["epi_px"]),
        })

    # Face points are diagnostic only; v32q maps body joints 5..16 into MHR.
    for j in range(5):
        if float(direct["conf"][j]) >= MIN_CONF:
            xy[j] = np.asarray(direct["xy"][j], float)
            conf[j] = float(direct["conf"][j])

    pseudo = dict(direct)
    pseudo["xy"] = xy
    pseudo["conf"] = conf
    pseudo["box"] = np.asarray(direct["box"], float)

    selected_errors = [float(r["best"]["epi_px"]) for r in retained_rows]
    all_best_errors = [float(r["best"]["epi_px"]) for r in ranked_rows]
    meta = {
        "retained_body_joint_count": len(retained),
        "retained_body_joints": [v32j.NAMES[j] for j in retained],
        "retained_epi_px": selected_errors,
        "retained_epi_median_px": float(np.median(selected_errors)),
        "retained_epi_max_px": float(np.max(selected_errors)),
        "all_surviving_best_epi_px": all_best_errors,
        "absurd_epi_guard_px": ABSURD_EPI_PX,
        "target_keep": TARGET_KEEP,
    }
    return pseudo, audit, retained, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=340)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())

    # Reuse v32t's exact real-frame observation generation.
    from rfdetr import RFDETRKeypointPreview
    model = RFDETRKeypointPreview()
    b0, direct, direct_idx, temporal_rows = v32t.broadcast_candidates(model, stage, qual)
    rar = v32t.temporal_rar_t0(model, stage, qual)
    pseudo, joint_audit, retained, meta = choose_joint_sources_trimmed(scene, direct, temporal_rows, rar)

    serial_temporal = []
    for r in temporal_rows:
        serial_temporal.append({k: v for k, v in r.items() if k not in ("xy0", "valid0", "anchor_conf", "ranked", "trace")})
    selected_counts = {}
    for row in joint_audit.values():
        src = row.get("selected_source")
        if src:
            selected_counts[src] = selected_counts.get(src, 0) + 1
    audit = {
        "version": "v32u_epipolar_trimmed_broadcast_source_selection",
        "direct_detection_index": direct_idx,
        "rar_t04_detection_index": rar["selected_index"],
        "selected_source_counts": selected_counts,
        "broadcast_temporal_anchor_summary": serial_temporal,
        "joint_audit": joint_audit,
        "policy": "rank per-joint Broadcast candidates by symmetric epipolar consistency with trusted temporal RAR, drop worst outliers, fit only best nine",
        **meta,
    }
    (a.out / "v32u_epipolar_trimmed_audit.json").write_text(json.dumps(audit, indent=2))

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
        raise RuntimeError(f"v32u: expected one Broadcast injection, got {injected['n']}")

    qpath = a.out / "v32q_qa.json"
    if qpath.exists():
        q = json.loads(qpath.read_text())
        q["version"] = "v32u_epipolar_trimmed_broadcast_mhr"
        q["broadcast_observation_selection"] = audit
        q["v32u_base_exit_code"] = exit_code
        q["status"] = ("PASS_V32U_EPIPOLAR_TRIMMED_MULTIVIEW_ANATOMICAL_POSITION" if q.get("status", "").startswith("PASS_")
                       else "FAIL_CLOSED_V32U_EPIPOLAR_TRIMMED_MULTIVIEW_ANATOMICAL_POSITION")
        (a.out / "v32u_qa.json").write_text(json.dumps(q, indent=2))

    print(json.dumps({
        "retained_body_joints": len(retained),
        "retained_names": [v32j.NAMES[j] for j in retained],
        "retained_epi_median_px": meta["retained_epi_median_px"],
        "retained_epi_max_px": meta["retained_epi_max_px"],
        "selected_source_counts": selected_counts,
        "base_exit_code": exit_code,
    }, indent=2), flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
