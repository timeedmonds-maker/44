from __future__ import annotations

"""v86: compare v82 lane evidence with a native-pixel lane-centre audit.

Discovery only. No threshold is relaxed and no permission can be granted here.
The v83 NBA 2-inch painted-line centre convention is used for both evidence sets.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_spec(spec: dict, perturb_trials: int = 64) -> dict:
    h0 = v85.h_seed_from_spec(spec)
    train, held = v44.split_groups(spec["observations_px"], spec["held_out_indices"])
    dense = v44.dense_features()
    original_indices = v85.train_original_indices(spec)

    z, roots = v44.solve_multistart(h0, train, return_roots=True)
    H = v44.H_from_z(z, h0)
    held_max, held_metrics = v85.max_heldout(H, held, dense)

    best_score = min(r["median_abs_pixel_residual"] for r in roots)
    competitive = [r for r in roots if r["median_abs_pixel_residual"] <= best_score + 0.25]
    pairwise_max = 0.0
    for i in range(len(competitive)):
        for j in range(i + 1, len(competitive)):
            sh = v44.curve_shift(v44.H_from_z(competitive[i]["z"], h0), v44.H_from_z(competitive[j]["z"], h0), dense)
            pairwise_max = max(pairwise_max, v85.max_shift(sh))

    loo = []
    for key in v44.GROUPS:
        for ti in range(len(train[key])):
            g = v85.copied(train)
            removed = g[key][ti].copy()
            g[key] = np.delete(g[key], ti, axis=0)
            row = v85.solve_variant(h0, z, g, held, dense)
            row.update({
                "feature": key,
                "train_index": ti,
                "original_observation_index": original_indices[key][ti],
                "removed_pixel": removed.tolist(),
            })
            loo.append(row)
    loo.sort(key=lambda r: r["max_curve_p95_shift_px"], reverse=True)

    rng = np.random.default_rng(441903)
    perturb = []
    for trial in range(perturb_trials):
        noise = {k: rng.uniform(-0.5, 0.5, size=v.shape) for k, v in train.items()}
        g = {k: train[k] + noise[k] for k in v44.GROUPS}
        zp = v44.solve_warm(h0, g, z)
        Hp = v44.H_from_z(zp, h0)
        sh = v44.curve_shift(H, Hp, dense)
        mx = v85.max_shift(sh)
        perturb.append({"trial": trial, "max_curve_p95_shift_px": mx, "curve_shift": sh})
    perturb.sort(key=lambda r: r["max_curve_p95_shift_px"], reverse=True)

    return {
        "nominal_max_heldout_feature_p95_px": held_max,
        "heldout_pixel_error": held_metrics,
        "competitive_root_count": len(competitive),
        "competitive_pairwise_max_p95_shift_px": pairwise_max,
        "single_observation_leave_one_out": {
            "trial_count": len(loo),
            "max_curve_p95_shift_px": float(loo[0]["max_curve_p95_shift_px"]),
            "count_over_2px": int(sum(r["max_curve_p95_shift_px"] > 2.0 for r in loo)),
            "worst_10": loo[:10],
        },
        "half_pixel_perturbation": {
            "trial_count": len(perturb),
            "max_curve_p95_shift_px": float(perturb[0]["max_curve_p95_shift_px"]),
            "count_over_2px": int(sum(r["max_curve_p95_shift_px"] > 2.0 for r in perturb)),
            "p50_curve_p95_shift_px": float(np.percentile([r["max_curve_p95_shift_px"] for r in perturb], 50)),
            "p95_curve_p95_shift_px": float(np.percentile([r["max_curve_p95_shift_px"] for r in perturb], 95)),
            "worst_10": perturb[:10],
        },
        "homography_world_to_image": H.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual = sha256(args.frame)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Immutable Broadcast frame changed: {actual}")

    v85.patch_line_aware_geometry()
    baseline = json.loads(args.baseline.read_text())
    audited = json.loads(args.observations.read_text())
    for spec in (baseline, audited):
        if spec.get("camera_label") != "Broadcast" or spec.get("event_id") != 489:
            raise RuntimeError("Broadcast evidence provenance changed")

    a = run_spec(baseline)
    b = run_spec(audited)
    report = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_BROADCAST_LANE_EVIDENCE_V86",
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Broadcast",
        "immutable_frame_sha256": actual,
        "geometry": "NBA 2-inch painted-line centre convention from v83/v84",
        "baseline_v82": a,
        "audited_v86": b,
        "delta": {
            "nominal_heldout_max_p95_px": b["nominal_max_heldout_feature_p95_px"] - a["nominal_max_heldout_feature_p95_px"],
            "loo_max_curve_p95_shift_px": b["single_observation_leave_one_out"]["max_curve_p95_shift_px"] - a["single_observation_leave_one_out"]["max_curve_p95_shift_px"],
            "perturbation_max_curve_p95_shift_px": b["half_pixel_perturbation"]["max_curve_p95_shift_px"] - a["half_pixel_perturbation"]["max_curve_p95_shift_px"],
        },
        "interpretation": (
            "LANE_EVIDENCE_BIAS_MATERIALLY_REDUCES_SUPPORT_INSTABILITY"
            if b["single_observation_leave_one_out"]["max_curve_p95_shift_px"] < a["single_observation_leave_one_out"]["max_curve_p95_shift_px"]
            else "LANE_EVIDENCE_AUDIT_DID_NOT_REDUCE_SUPPORT_INSTABILITY"
        ),
        "permissions": {
            "broadcast_floor_homography_allowed": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "guardrail": "This is an evidence audit, not a promotion gate. Preserve the <=2 px robustness thresholds for any subsequent strict validation.",
    }
    p = args.out / "broadcast_lane_evidence_v86.json"
    p.write_text(json.dumps(v44.json_safe(report), indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "baseline_nominal": a["nominal_max_heldout_feature_p95_px"],
        "v86_nominal": b["nominal_max_heldout_feature_p95_px"],
        "baseline_loo_max": a["single_observation_leave_one_out"]["max_curve_p95_shift_px"],
        "v86_loo_max": b["single_observation_leave_one_out"]["max_curve_p95_shift_px"],
        "baseline_perturb_max": a["half_pixel_perturbation"]["max_curve_p95_shift_px"],
        "v86_perturb_max": b["half_pixel_perturbation"]["max_curve_p95_shift_px"],
        "v86_perturb_over_2": b["half_pixel_perturbation"]["count_over_2px"],
        "interpretation": report["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
