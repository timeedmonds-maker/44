from __future__ import annotations

"""Exact Frame C Left Above Rim metric camera using game-level centre + principal point.

The physical camera centre and optical principal point are learned independently from
same-game frames.  This exact target pose then solves only rotation and focal length
against fixed regulation NBA landmarks.  No player or ball point is used.

Passing may promote only this exact Frame C metric camera. Replay rendering remains
forbidden until foreground geometry and a validated orbit baseline are separately
proved.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v27 import H, W, draw_overlay, rotation_angle_deg
from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v28 import dense_validation_geometry
from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks


def project_pose(p: np.ndarray, obj: np.ndarray, center: np.ndarray, pp: np.ndarray):
    R, _ = cv2.Rodrigues(np.asarray(p[:3], dtype=np.float64))
    focal = float(np.exp(p[3]))
    cam = (R @ (obj - center).T).T
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + pp[0],
        focal * cam[:, 1] / cam[:, 2] + pp[1],
    ])
    return uv, cam, focal, R


def solve_pose(obj: np.ndarray, obs: np.ndarray, center: np.ndarray, pp: np.ndarray, view: dict, p0: np.ndarray) -> np.ndarray:
    focal_prior = float(view.get("focal_prior_px", 900.0))
    focal_sigma_log = float(view.get("focal_prior_sigma_log", 1.8))

    def residual(p: np.ndarray) -> np.ndarray:
        uv, cam, focal, _ = project_pose(p, obj, center, pp)
        return np.concatenate([
            (uv - obs).ravel(),
            np.asarray([(math.log(focal) - math.log(focal_prior)) / focal_sigma_log]),
            np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0,
        ])

    lower = np.asarray([-np.inf, -np.inf, -np.inf, math.log(150.0)])
    upper = np.asarray([np.inf, np.inf, np.inf, math.log(4000.0)])
    opt = least_squares(
        residual,
        np.asarray(p0, dtype=np.float64),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=30000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def metrics(p: np.ndarray, obj: np.ndarray, obs: np.ndarray, center: np.ndarray, pp: np.ndarray) -> dict:
    uv, cam, focal, R = project_pose(p, obj, center, pp)
    err = np.linalg.norm(uv - obs, axis=1)
    return {
        "uv": uv,
        "cam": cam,
        "focal": float(focal),
        "R": R,
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "per_point_px": [float(x) for x in err],
    }


def displacement_stats(base_p: np.ndarray, test_p: np.ndarray, center: np.ndarray, pp: np.ndarray, dense: np.ndarray) -> dict:
    buv, bcam, _, _ = project_pose(base_p, dense, center, pp)
    tuv, tcam, _, _ = project_pose(test_p, dense, center, pp)
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
    ap.add_argument("--game-intrinsics", type=Path, required=True)
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
    ap.add_argument("--max-perturb-focal-fraction", type=float, default=0.05)
    ap.add_argument("--max-perturb-rotation-deg", type=float, default=0.75)
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
        raise RuntimeError("Dynamic camera anchors forbidden")

    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    camreg = reg["cameras"][args.camera_label]
    if not camreg["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Physical center prior not accepted")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    intr = json.loads(args.game_intrinsics.read_text(encoding="utf-8"))
    if intr.get("status") != "PASS_GAME_INTRINSICS_PRIOR" or not intr.get("principal_point_prior_allowed"):
        raise RuntimeError("Game-level principal-point prior not accepted")
    if intr.get("camera_label") != args.camera_label or intr.get("game_id") != reg["game_id"]:
        raise RuntimeError("Game intrinsics identity mismatch")
    pp = np.asarray(intr["shared_principal_point_px"], dtype=np.float64)

    world = world_landmarks()
    names = list(view["landmarks"])
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    obs = np.asarray([view["landmarks"][n] for n in names], dtype=np.float64)
    if len(names) < 8:
        raise RuntimeError("Expected at least eight regulation landmarks")

    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError("Direct metric sanity solve implausible")
    dp = np.asarray(direct[2], dtype=np.float64)
    direct_center = np.asarray(direct[4], dtype=np.float64)
    p0 = np.r_[dp[:3], dp[6]]
    p = solve_pose(obj, obs, center, pp, view, p0)
    fit = metrics(p, obj, obs, center, pp)
    draw_overlay(image, names, obs, fit["uv"], args.out / "frame_c_left_above_rim_metric_overlay_v29.png")

    loo = []
    for hold in range(len(names)):
        keep = np.asarray([i for i in range(len(names)) if i != hold], dtype=int)
        ph = solve_pose(obj[keep], obs[keep], center, pp, view, p)
        pred = project_pose(ph, obj[[hold]], center, pp)[0][0]
        loo.append({"held_out": names[hold], "error_px": float(np.linalg.norm(pred - obs[hold]))})
    le = np.asarray([x["error_px"] for x in loo], dtype=np.float64)

    dense = dense_validation_geometry()
    base_uv, base_cam, _, _ = project_pose(p, dense, center, pp)
    visible = (
        (base_cam[:, 2] > 20.0)
        & (base_uv[:, 0] >= 0.0) & (base_uv[:, 0] < W)
        & (base_uv[:, 1] >= 0.0) & (base_uv[:, 1] < H)
    )

    rng = np.random.default_rng(291902)
    perturb = []
    for trial in range(args.perturbation_trials):
        po = obs + rng.uniform(-0.5, 0.5, size=obs.shape)
        pt = solve_pose(obj, po, center, pp, view, p)
        mt = metrics(pt, obj, po, center, pp)
        ds = displacement_stats(p, pt, center, pp, dense)
        perturb.append({
            "trial": trial,
            "rotation_delta_deg": rotation_angle_deg(mt["R"], fit["R"]),
            "focal_fraction_delta": abs(mt["focal"] - fit["focal"]) / fit["focal"],
            "fit_rmse_px": mt["rmse_px"],
            "dense_projective_displacement": ds,
        })

    max_p95 = max(x["dense_projective_displacement"]["p95_px"] for x in perturb)
    max_max = max(x["dense_projective_displacement"]["max_px"] for x in perturb)
    max_rot = max(x["rotation_delta_deg"] for x in perturb)
    max_ff = max(x["focal_fraction_delta"] for x in perturb)
    dc = float(np.linalg.norm(direct_center - center))

    focal = fit["focal"]
    R = fit["R"]
    t = -R @ center
    K = np.asarray([[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]])
    P = K @ np.column_stack([R, t])

    gates = {
        "immutable_frame_c_lock": True,
        "fixed_regulation_geometry_only": True,
        "independent_center_prior_accepted": True,
        "independent_game_principal_point_prior_accepted": True,
        "fit_rmse_at_most_threshold": fit["rmse_px"] <= args.max_fit_rmse_px,
        "fit_p95_at_most_threshold": fit["p95_px"] <= args.max_fit_p95_px,
        "leave_one_out_median_at_most_threshold": float(np.median(le)) <= args.max_loo_median_px,
        "leave_one_out_max_at_most_threshold": float(np.max(le)) <= args.max_loo_max_px,
        "direct_unconstrained_center_agrees_with_prior": dc <= args.max_direct_center_disagreement_cm,
        "dense_projective_p95_stability": max_p95 <= args.max_projective_p95_px,
        "dense_projective_max_stability": max_max <= args.max_projective_max_px,
        "half_pixel_focal_stability": max_ff <= args.max_perturb_focal_fraction,
        "half_pixel_rotation_stability": max_rot <= args.max_perturb_rotation_deg,
        "positive_depth_all_landmarks": float(np.min(fit["cam"][:, 2])) > 20.0,
        "dense_validation_support_at_least_100_points": int(visible.sum()) >= 100,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_METRIC_EVENT_CAMERA" if passed else "FAIL_METRIC_EVENT_CAMERA",
        "game_id": reg["game_id"],
        "event_id": 489,
        "date": "2025-11-30",
        "camera_label": args.camera_label,
        "frame_c_authority": {
            "camera": "Right Slash",
            "chooser_option": "C",
            "right_slash_local_time": 8.275733,
            "decoded_frame_index": 248,
            "left_above_rim_synchronized_time": 8.653093,
            "left_above_rim_decoded_frame_index": 259,
        },
        "method": "independent game-level fixed camera centre + independent game-level shared principal point + exact Frame C regulation geometry solve for rotation/focal only",
        "guardrail": "Passing promotes only this exact metric event camera. Player/ball depth and replay rendering remain unproven.",
        "camera": {
            "camera_center_world_cm": [float(x) for x in center],
            "principal_point_px": [float(x) for x in pp],
            "focal_px": float(focal),
            "R_world_to_camera": R.tolist(),
            "t_world_to_camera_cm": t.tolist(),
            "projection_matrix_KRt": P.tolist(),
        },
        "direct_unconstrained_center_cm": [float(x) for x in direct_center],
        "direct_center_disagreement_cm": dc,
        "fit": {
            "rmse_px": fit["rmse_px"],
            "median_px": fit["median_px"],
            "p95_px": fit["p95_px"],
            "max_px": fit["max_px"],
            "per_point_px": {n: float(e) for n, e in zip(names, fit["per_point_px"])},
        },
        "leave_one_out": {
            "results": loo,
            "median_px": float(np.median(le)),
            "p95_px": float(np.percentile(le, 95)),
            "max_px": float(np.max(le)),
        },
        "half_pixel_perturbation": {
            "trials": perturb,
            "max_dense_projective_p95_px": max_p95,
            "max_dense_projective_max_px": max_max,
            "max_rotation_delta_deg": max_rot,
            "max_focal_fraction_delta": max_ff,
        },
        "dense_validation_visible_points": int(visible.sum()),
        "thresholds": {
            "max_fit_rmse_px": args.max_fit_rmse_px,
            "max_fit_p95_px": args.max_fit_p95_px,
            "max_loo_median_px": args.max_loo_median_px,
            "max_loo_max_px": args.max_loo_max_px,
            "max_direct_center_disagreement_cm": args.max_direct_center_disagreement_cm,
            "max_dense_projective_p95_px": args.max_projective_p95_px,
            "max_dense_projective_max_px": args.max_projective_max_px,
            "max_perturb_focal_fraction": args.max_perturb_focal_fraction,
            "max_perturb_rotation_deg": args.max_perturb_rotation_deg,
        },
        "gates": gates,
        "metric_event_camera_allowed": passed,
        "replay_render_allowed": False,
    }
    (args.out / "frame_c_left_above_rim_metric_camera_v29.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "principal_point_px": payload["camera"]["principal_point_px"],
        "focal_px": focal,
        "fit_rmse_px": fit["rmse_px"],
        "fit_p95_px": fit["p95_px"],
        "loo_max_px": float(np.max(le)),
        "max_dense_projective_p95_px": max_p95,
        "max_dense_projective_max_px": max_max,
        "max_rotation_delta_deg": max_rot,
        "max_focal_fraction_delta": max_ff,
        "metric_event_camera_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
