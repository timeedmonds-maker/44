from __future__ import annotations

"""v88: independent orange-rim validation of the v87 Broadcast pinhole camera.

v87 fits only regulation floor paint and three white backboard-target line families.
The orange rim is deliberately absent from that solve. v88 freezes that model,
checks the held-out regulation rim in image space, then repeats the exact v87
64-trial half-pixel perturbation schedule while keeping the rim observations
unperturbed.

A rim pass can validate the camera's same-frame 3D projection. It cannot waive
v87's physical-centre stability gate. This stage is discovery-only and grants no
replay, metric-camera or physical-centre permission.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

FT = 30.48
IN = 2.54
EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"
RIM_CENTER_X_CM = 15.0 * IN
RIM_RADIUS_CM = 9.0 * IN
RIM_Z_CM = 10.0 * FT
CENTER_STABILITY_LIMIT_CM = 75.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rim_circle(samples: int = 4000) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return np.column_stack([
        RIM_CENTER_X_CM + RIM_RADIUS_CM * np.cos(theta),
        RIM_RADIUS_CM * np.sin(theta),
        np.full_like(theta, RIM_Z_CM),
    ]).astype(np.float64)


def rim_metrics(p: np.ndarray, observed: np.ndarray) -> dict:
    projected, depth = v87.project3(p, rim_circle())
    if float(np.min(depth[:, 2])) <= 20.0:
        return {"median_px": float("inf"), "p95_px": float("inf"), "max_px": float("inf")}
    d = np.sqrt(np.sum((observed[:, None, :] - projected[None, :, :]) ** 2, axis=2)).min(axis=1)
    return {
        "count": int(len(d)),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
        "per_point_px": d.tolist(),
    }


def nominal_v87(floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray], target_spec: dict) -> tuple[np.ndarray, list[dict]]:
    roots = []
    for start in v87.pnp_starts(target_spec):
        try:
            p = v87.solve_warm(start, floor_train, target_obs, max_nfev=7000)
        except Exception:
            continue
        r = v87.data_residual(p, floor_train, target_obs)
        score = float(np.median(np.abs(r[:-8]))) if len(r) > 8 else float(np.median(np.abs(r)))
        C = v87.camera_center(p)
        _, q = v87.project3(p, np.vstack(list(v87.world_target_lines().values())))
        if np.all(np.isfinite(p)) and np.all(q[:, 2] > 20.0) and np.isfinite(C).all():
            roots.append({"p": p, "score": score, "center_cm": C})
    if not roots:
        raise RuntimeError("No physically forward Broadcast v88 roots")
    roots.sort(key=lambda row: row["score"])
    return np.asarray(roots[0]["p"], dtype=np.float64), roots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--target-lines", type=Path, required=True)
    ap.add_argument("--rim", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual = sha256(args.frame)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Immutable Broadcast Frame C SHA changed: {actual}")
    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (540, 960):
        raise RuntimeError("Expected immutable native 960x540 Broadcast Frame C")

    v85.patch_line_aware_geometry()
    floor_spec = json.loads(args.floor.read_text(encoding="utf-8"))
    target_spec = json.loads(args.target_lines.read_text(encoding="utf-8"))
    rim_spec = json.loads(args.rim.read_text(encoding="utf-8"))
    if any(spec.get("camera_label") != "Broadcast" for spec in (floor_spec, target_spec, rim_spec)):
        raise RuntimeError("Broadcast provenance changed")
    if any(int(spec.get("event_id", -1)) != 489 for spec in (floor_spec, target_spec, rim_spec)):
        raise RuntimeError("Broadcast event provenance changed")
    if rim_spec.get("independence", {}).get("used_by_v87_fit") is not False:
        raise RuntimeError("v88 rim evidence is not declared held out from v87")

    floor_train, floor_held = v44.split_groups(floor_spec["observations_px"], floor_spec["held_out_indices"])
    target_obs = {
        key: np.asarray(target_spec["observed_line_samples_px"][key], dtype=np.float64)
        for key in v87.TARGET_KEYS
    }
    rim_obs = np.asarray(rim_spec["rim_contour_samples_px"], dtype=np.float64)
    if rim_obs.shape != (15, 2):
        raise RuntimeError("v88 rim evidence schema changed")

    pb, roots = nominal_v87(floor_train, target_obs, target_spec)
    Cb = v87.camera_center(pb)
    H = v87.floor_homography(pb)
    dense = v44.dense_features()
    floor_metrics = v44.pixel_metrics(H, floor_held, dense)
    floor_max = float(max(row["p95_px"] for row in floor_metrics.values()))
    target_metrics = v87.target_metrics(pb, target_obs)
    target_max = float(max(row["p95_px"] for row in target_metrics.values()))
    rim_nominal = rim_metrics(pb, rim_obs)

    best_score = float(roots[0]["score"])
    competitive = []
    for row in roots:
        if float(row["score"]) > best_score + 0.25:
            continue
        shift = v87.projection_shift(pb, row["p"])
        competitive.append({
            "score": float(row["score"]),
            "max_projection_p95_shift_px": v87.max_p95(shift),
            "center_shift_cm": float(np.linalg.norm(row["center_cm"] - Cb)),
        })
    competitive_projection_max = float(max(row["max_projection_p95_shift_px"] for row in competitive))

    rng = np.random.default_rng(870903)
    perturb = []
    for trial in range(args.perturbation_trials):
        fg = {
            key: floor_train[key] + rng.uniform(-0.5, 0.5, size=floor_train[key].shape)
            for key in v44.GROUPS
        }
        tg = {
            key: target_obs[key] + rng.uniform(-0.5, 0.5, size=target_obs[key].shape)
            for key in v87.TARGET_KEYS
        }
        try:
            p = v87.solve_warm(pb, fg, tg, max_nfev=4500)
        except Exception:
            perturb.append({"trial": trial, "failed": True})
            continue
        C = v87.camera_center(p)
        shift = v87.projection_shift(pb, p)
        held_metrics = v44.pixel_metrics(v87.floor_homography(p), floor_held, dense)
        held_max = float(max(row["p95_px"] for row in held_metrics.values()))
        tmetrics = v87.target_metrics(p, target_obs)
        target_trial_max = float(max(row["p95_px"] for row in tmetrics.values()))
        rmetrics = rim_metrics(p, rim_obs)
        perturb.append({
            "trial": trial,
            "failed": False,
            "max_projection_p95_shift_px": v87.max_p95(shift),
            "center_shift_cm": float(np.linalg.norm(C - Cb)),
            "heldout_floor_max_p95_px": held_max,
            "target_line_max_p95_px": target_trial_max,
            "heldout_rim_p95_px": float(rmetrics["p95_px"]),
            "heldout_rim_max_px": float(rmetrics["max_px"]),
            "focal_fraction_shift": float(abs(np.exp(p[6]) - np.exp(pb[6])) / np.exp(pb[6])),
            "principal_point_shift_px": float(np.linalg.norm(p[7:9] - pb[7:9])),
        })

    good = [row for row in perturb if not row.get("failed")]
    failures = len(perturb) - len(good)
    max_projection = float(max((row["max_projection_p95_shift_px"] for row in good), default=float("inf")))
    max_center = float(max((row["center_shift_cm"] for row in good), default=float("inf")))
    max_floor = float(max((row["heldout_floor_max_p95_px"] for row in good), default=float("inf")))
    max_target = float(max((row["target_line_max_p95_px"] for row in good), default=float("inf")))
    max_rim_p95 = float(max((row["heldout_rim_p95_px"] for row in good), default=float("inf")))
    max_rim_max = float(max((row["heldout_rim_max_px"] for row in good), default=float("inf")))

    gates = {
        "nominal_heldout_floor_p95_at_most_2px": floor_max <= 2.0,
        "nominal_target_line_p95_at_most_1_5px": target_max <= 1.5,
        "nominal_independent_rim_p95_at_most_1_5px": float(rim_nominal["p95_px"]) <= 1.5,
        "at_least_3_competitive_roots": len(competitive) >= 3,
        "competitive_projection_p95_at_most_0_5px": competitive_projection_max <= 0.5,
        "all_64_half_pixel_trials_converged": failures == 0 and len(good) == args.perturbation_trials,
        "half_pixel_projection_p95_at_most_2px": max_projection <= 2.0,
        "half_pixel_heldout_floor_p95_at_most_2_5px": max_floor <= 2.5,
        "half_pixel_target_line_p95_at_most_2px": max_target <= 2.0,
        "half_pixel_independent_rim_p95_at_most_1_5px": max_rim_p95 <= 1.5,
        "half_pixel_center_shift_at_most_75cm": max_center <= CENTER_STABILITY_LIMIT_CM,
        "pinhole_only_no_distortion_parameters": True,
    }
    center_gate = bool(gates["half_pixel_center_shift_at_most_75cm"])
    projection_gate_names = [key for key in gates if key != "half_pixel_center_shift_at_most_75cm"]
    projection_only_pass = bool(all(gates[key] for key in projection_gate_names))
    all_pass = bool(all(gates.values()))

    report = {
        "schema_version": 1,
        "status": "DISCOVERY_ONLY_BROADCAST_INDEPENDENT_RIM_V88",
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Broadcast",
        "immutable_frame_sha256": actual,
        "rim_policy": "orange rim is independent held-out 3D evidence and is never included in the v87 fit or perturbation fit",
        "camera_candidate": {
            "focal_px": float(np.exp(pb[6])),
            "principal_point_px": pb[7:9].tolist(),
            "center_cm": Cb.tolist(),
            "center_ft": (Cb / FT).tolist(),
        },
        "nominal": {
            "heldout_floor": floor_metrics,
            "heldout_floor_max_p95_px": floor_max,
            "target_lines": target_metrics,
            "target_line_max_p95_px": target_max,
            "independent_rim": rim_nominal,
            "competitive_root_count": len(competitive),
            "competitive_projection_max_p95_shift_px": competitive_projection_max,
        },
        "half_pixel_perturbation_64": {
            "trial_count": len(perturb),
            "converged_count": len(good),
            "failed_count": failures,
            "max_projection_p95_shift_px": max_projection,
            "max_center_shift_cm": max_center,
            "max_heldout_floor_p95_px": max_floor,
            "max_target_line_p95_px": max_target,
            "max_independent_rim_p95_px": max_rim_p95,
            "max_independent_rim_max_px": max_rim_max,
            "worst_center_trials": sorted(good, key=lambda row: row["center_shift_cm"], reverse=True)[:10],
            "worst_rim_trials": sorted(good, key=lambda row: row["heldout_rim_p95_px"], reverse=True)[:10],
            "all_trials": perturb,
        },
        "gates": gates,
        "candidate_passes_projection_only_gates": projection_only_pass,
        "candidate_passes_all_gates": all_pass,
        "classification": {
            "same_frame_projection_geometry_independently_validated": projection_only_pass,
            "physical_center_identifiability_resolved": center_gate,
            "diagnosis": (
                "INDEPENDENT_RIM_CONFIRMS_IMAGE_SPACE_3D_PROJECTION_BUT_CAMERA_CENTER_REMAINS_UNDERIDENTIFIED"
                if projection_only_pass and not center_gate
                else "V88_RESULT_REQUIRES_REVIEW"
            ),
        },
        "permissions": {
            "broadcast_floor_homography_allowed": False,
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "broadcast_freeview_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "guardrail": "Do not relax the inherited 75 cm centre-stability gate. If rim projection passes while centre stability fails, move to independent rig/mount evidence or a physically constrained multi-state pivot model; do not promote this single-state centre.",
    }

    (args.out / "broadcast_independent_rim_v88.json").write_text(
        json.dumps(v44.json_safe(report), indent=2) + "\n", encoding="utf-8"
    )

    overlay = image.copy()
    pred, _ = v87.project3(pb, rim_circle(720))
    qq = np.round(pred).astype(int)
    ok = (qq[:, 0] >= 0) & (qq[:, 0] < 960) & (qq[:, 1] >= 0) & (qq[:, 1] < 540)
    if int(np.sum(ok)) > 1:
        cv2.polylines(overlay, [qq[ok].reshape(-1, 1, 2)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for point in np.round(rim_obs).astype(int):
        cv2.circle(overlay, tuple(point), 2, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "broadcast_independent_rim_v88_overlay.png"), overlay)

    print("status", report["status"])
    print("projection_only_pass", projection_only_pass)
    print("all_gates_pass", all_pass)
    print("nominal_rim_p95_px", rim_nominal["p95_px"])
    print("perturb_max_rim_p95_px", max_rim_p95)
    print("perturb_max_center_shift_cm", max_center)
    print("diagnosis", report["classification"]["diagnosis"])


if __name__ == "__main__":
    main()
