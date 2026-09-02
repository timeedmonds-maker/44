from __future__ import annotations

"""Exact Frame C metric camera with independently self-calibrated exact-frame intrinsics.

The physical camera centre comes from independent same-game metric evidence. Exact
Frame C principal point and focal length come from static-scene rotational
self-calibration in the target clip and do not use NBA metric landmarks. This stage
therefore solves only world-to-camera rotation from regulation NBA geometry.

Passing promotes only this exact metric event camera. Replay rendering remains
forbidden pending a second trustworthy event camera / physical baseline and foreground
reconstruction.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v27 import H, W, draw_overlay, rotation_angle_deg
from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v28 import dense_validation_geometry
from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks


def project(rvec: np.ndarray, obj: np.ndarray, center: np.ndarray, pp: np.ndarray, focal: float):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    cam = (R @ (obj - center).T).T
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + pp[0],
        focal * cam[:, 1] / cam[:, 2] + pp[1],
    ])
    return uv, cam, R


def solve_rotation(obj: np.ndarray, obs: np.ndarray, center: np.ndarray, pp: np.ndarray, focal: float, seed: np.ndarray) -> np.ndarray:
    def residual(rv: np.ndarray) -> np.ndarray:
        uv, cam, _ = project(rv, obj, center, pp, focal)
        return np.concatenate([
            (uv - obs).ravel(),
            np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0,
        ])
    opt = least_squares(
        residual, np.asarray(seed, dtype=np.float64),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=30000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def fit_metrics(rv: np.ndarray, obj: np.ndarray, obs: np.ndarray, center: np.ndarray, pp: np.ndarray, focal: float) -> dict:
    uv, cam, R = project(rv, obj, center, pp, focal)
    err = np.linalg.norm(uv - obs, axis=1)
    return {
        "uv": uv, "cam": cam, "R": R,
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "per_point_px": [float(x) for x in err],
    }


def dense_displacement(base_rv: np.ndarray, test_rv: np.ndarray, center: np.ndarray,
                       base_pp: np.ndarray, base_f: float, test_pp: np.ndarray, test_f: float,
                       dense: np.ndarray) -> dict:
    buv, bcam, _ = project(base_rv, dense, center, base_pp, base_f)
    tuv, tcam, _ = project(test_rv, dense, center, test_pp, test_f)
    valid = (
        (bcam[:, 2] > 20.0) & (tcam[:, 2] > 20.0)
        & (buv[:, 0] >= 0.0) & (buv[:, 0] < W)
        & (buv[:, 1] >= 0.0) & (buv[:, 1] < H)
    )
    if int(valid.sum()) < 100:
        raise RuntimeError("Insufficient dense validation support")
    d = np.linalg.norm(tuv[valid] - buv[valid], axis=1)
    return {
        "visible_point_count": int(valid.sum()),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--rotational-intrinsics", type=Path, required=True)
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-fit-rmse-px", type=float, default=1.0)
    ap.add_argument("--max-fit-p95-px", type=float, default=1.5)
    ap.add_argument("--max-loo-median-px", type=float, default=2.0)
    ap.add_argument("--max-loo-max-px", type=float, default=4.0)
    ap.add_argument("--max-direct-center-disagreement-cm", type=float, default=10.0)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    ap.add_argument("--max-projective-p95-px", type=float, default=1.5)
    ap.add_argument("--max-projective-max-px", type=float, default=3.0)
    ap.add_argument("--max-rotation-delta-deg", type=float, default=0.75)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Frame C image")

    spec = json.loads(args.landmarks.read_text(encoding="utf-8"))
    freeze = spec.get("freeze_lock", {})
    if freeze.get("authority_camera") != "Right Slash" or freeze.get("chooser_option") != "C":
        raise RuntimeError("Immutable Frame C authority changed")
    if abs(float(freeze.get("right_slash_local_time", -1.0)) - 8.275733) > 5e-7 or int(freeze.get("decoded_frame_index", -1)) != 248:
        raise RuntimeError("Immutable Frame C timing changed")
    view = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if view is None:
        raise RuntimeError("Missing target metric view")
    if any(t in json.dumps(view).lower() for t in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Dynamic metric anchors forbidden")

    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    camreg = reg["cameras"][args.camera_label]
    if not camreg["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Independent game physical-centre prior not accepted")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    intr = json.loads(args.rotational_intrinsics.read_text(encoding="utf-8"))
    if intr.get("status") != "PASS_ROTATIONAL_INTRINSICS_PRIOR":
        raise RuntimeError("Rotational exact-frame intrinsics prior not accepted")
    if not intr.get("principal_point_prior_allowed") or not intr.get("target_focal_prior_allowed"):
        raise RuntimeError("Rotational intrinsics permissions incomplete")
    if intr.get("metric_event_camera_allowed") or intr.get("replay_render_allowed"):
        raise RuntimeError("Intrinsics stage exceeded permission scope")
    if intr.get("camera_label") != args.camera_label or intr.get("game_id") != reg["game_id"] or int(intr.get("event_id", -1)) != 489:
        raise RuntimeError("Rotational intrinsics identity mismatch")
    pp = np.asarray(intr["target_principal_point_px"], dtype=np.float64)
    focal = float(intr["target_focal_px"])

    world = world_landmarks()
    names = list(view["landmarks"])
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    obs = np.asarray([view["landmarks"][n] for n in names], dtype=np.float64)
    if len(names) < 8:
        raise RuntimeError("Expected at least eight regulation NBA landmarks")

    # The unconstrained metric solve is retained only as an independent sanity check
    # on the already-promoted physical camera centre.
    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError("Direct metric sanity solve implausible")
    dp = np.asarray(direct[2], dtype=np.float64)
    direct_center = np.asarray(direct[4], dtype=np.float64)
    seed_rv = dp[:3]

    rv = solve_rotation(obj, obs, center, pp, focal, seed_rv)
    fit = fit_metrics(rv, obj, obs, center, pp, focal)
    draw_overlay(image, names, obs, fit["uv"], args.out / "frame_c_left_above_rim_metric_overlay_v32.png")

    loo = []
    for hold in range(len(names)):
        keep = np.asarray([i for i in range(len(names)) if i != hold], dtype=int)
        rh = solve_rotation(obj[keep], obs[keep], center, pp, focal, rv)
        pred = project(rh, obj[[hold]], center, pp, focal)[0][0]
        loo.append({"held_out": names[hold], "error_px": float(np.linalg.norm(pred - obs[hold]))})
    le = np.asarray([x["error_px"] for x in loo], dtype=np.float64)

    dense = dense_validation_geometry()
    buv, bcam, _ = project(rv, dense, center, pp, focal)
    visible = (
        (bcam[:, 2] > 20.0)
        & (buv[:, 0] >= 0.0) & (buv[:, 0] < W)
        & (buv[:, 1] >= 0.0) & (buv[:, 1] < H)
    )

    intr_trials = intr.get("half_pixel_static_correspondence_perturbation") or []
    if len(intr_trials) < args.perturbation_trials:
        raise RuntimeError("Intrinsics prior lacks enough independent perturbation trials")
    rng = np.random.default_rng(321902)
    perturb = []
    for trial in range(args.perturbation_trials):
        it = intr_trials[trial]
        ppi = np.asarray(it["principal_point_px"], dtype=np.float64)
        fi = float(it["target_focal_px"])
        po = obs + rng.uniform(-0.5, 0.5, size=obs.shape)
        rt = solve_rotation(obj, po, center, ppi, fi, rv)
        mt = fit_metrics(rt, obj, po, center, ppi, fi)
        ds = dense_displacement(rv, rt, center, pp, focal, ppi, fi, dense)
        perturb.append({
            "trial": trial,
            "intrinsics_principal_point_px": [float(x) for x in ppi],
            "intrinsics_focal_px": fi,
            "rotation_delta_deg": rotation_angle_deg(mt["R"], fit["R"]),
            "fit_rmse_px": mt["rmse_px"],
            "dense_projective_displacement": ds,
        })

    max_p95 = max(x["dense_projective_displacement"]["p95_px"] for x in perturb)
    max_max = max(x["dense_projective_displacement"]["max_px"] for x in perturb)
    max_rot = max(x["rotation_delta_deg"] for x in perturb)
    direct_disagreement = float(np.linalg.norm(direct_center - center))

    R = fit["R"]
    t = -R @ center
    K = np.asarray([[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]])
    P = K @ np.column_stack([R, t])

    gates = {
        "immutable_frame_c_lock": True,
        "fixed_regulation_geometry_only": True,
        "independent_game_physical_center_prior_accepted": True,
        "independent_static_scene_principal_point_prior_accepted": True,
        "independent_static_scene_target_focal_prior_accepted": True,
        "frame_c_metric_stage_solves_rotation_only": True,
        "fit_rmse_at_most_threshold": fit["rmse_px"] <= args.max_fit_rmse_px,
        "fit_p95_at_most_threshold": fit["p95_px"] <= args.max_fit_p95_px,
        "leave_one_out_median_at_most_threshold": float(np.median(le)) <= args.max_loo_median_px,
        "leave_one_out_max_at_most_threshold": float(np.max(le)) <= args.max_loo_max_px,
        "direct_unconstrained_center_agrees_with_prior": direct_disagreement <= args.max_direct_center_disagreement_cm,
        "combined_intrinsics_and_landmark_half_pixel_dense_p95_stability": max_p95 <= args.max_projective_p95_px,
        "combined_intrinsics_and_landmark_half_pixel_dense_max_stability": max_max <= args.max_projective_max_px,
        "combined_intrinsics_and_landmark_half_pixel_rotation_stability": max_rot <= args.max_rotation_delta_deg,
        "positive_depth_all_landmarks": float(np.min(fit["cam"][:, 2])) > 20.0,
        "dense_validation_support_at_least_100_points": int(visible.sum()) >= 100,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_METRIC_EVENT_CAMERA" if passed else "FAIL_METRIC_EVENT_CAMERA",
        "version": "v32_fixed_rotational_intrinsics",
        "game_id": reg["game_id"], "event_id": 489, "date": "2025-11-30", "camera_label": args.camera_label,
        "frame_c_authority": {
            "camera": "Right Slash", "chooser_option": "C", "right_slash_local_time": 8.275733,
            "decoded_frame_index": 248, "left_above_rim_synchronized_time": 8.653093,
            "left_above_rim_decoded_frame_index": 259,
        },
        "method": "independent game physical centre + exact-frame rotational self-calibrated principal point/focal + regulation NBA geometry rotation-only solve",
        "guardrail": "Passing promotes only this exact metric event camera. Foreground geometry, second event camera and replay rendering remain unproven.",
        "camera": {
            "camera_center_world_cm": [float(x) for x in center],
            "principal_point_px": [float(x) for x in pp],
            "focal_px": focal,
            "R_world_to_camera": R.tolist(), "t_world_to_camera_cm": t.tolist(),
            "projection_matrix_KRt": P.tolist(),
        },
        "direct_unconstrained_center_cm": [float(x) for x in direct_center],
        "direct_center_disagreement_cm": direct_disagreement,
        "fit": {
            "rmse_px": fit["rmse_px"], "median_px": fit["median_px"], "p95_px": fit["p95_px"],
            "max_px": fit["max_px"], "per_point_px": {n: float(e) for n, e in zip(names, fit["per_point_px"])},
        },
        "leave_one_out": {
            "results": loo, "median_px": float(np.median(le)),
            "p95_px": float(np.percentile(le, 95)), "max_px": float(np.max(le)),
        },
        "combined_half_pixel_perturbation": {
            "trials": perturb,
            "max_dense_projective_p95_px": max_p95,
            "max_dense_projective_max_px": max_max,
            "max_rotation_delta_deg": max_rot,
        },
        "dense_validation_visible_points": int(visible.sum()),
        "gates": gates,
        "metric_event_camera_allowed": passed,
        "replay_render_allowed": False,
    }
    (args.out / "frame_c_left_above_rim_metric_camera_v32.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "principal_point_px": payload["camera"]["principal_point_px"],
        "focal_px": focal, "fit_rmse_px": fit["rmse_px"], "fit_p95_px": fit["p95_px"],
        "loo_max_px": float(np.max(le)), "direct_center_disagreement_cm": direct_disagreement,
        "max_dense_projective_p95_px": max_p95, "max_dense_projective_max_px": max_max,
        "max_rotation_delta_deg": max_rot, "metric_event_camera_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
