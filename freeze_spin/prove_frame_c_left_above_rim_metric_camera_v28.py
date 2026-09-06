from __future__ import annotations

"""v28: exact Frame C Left Above Rim metric camera with projective-stability QA.

v27 showed that with the independently proved physical centre fixed, sub-pixel
annotation perturbations can trade camera rotation against principal point while
leaving the actual image projection nearly unchanged.  Those raw parameters are
therefore retained as diagnostics, but promotion is governed by the quantity that
matters for reconstruction: displacement of dense fixed regulation geometry in
source pixels.

No player or ball point is used.  This may promote only this exact Frame C metric
camera.  It never authorizes foreground depth or replay rendering by itself.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v27 import (
    H,
    W,
    draw_overlay,
    metrics,
    project_fixed,
    rotation_angle_deg,
    solve_fixed,
)
from freeze_spin.solve_nba_geometry_proof_v3 import (
    FOOT_CM,
    PAINT_W_CM,
    RIM_RADIUS_CM,
    RIM_X_CM,
    RIM_Z_CM,
    TARGET_INNER_H_CM,
    TARGET_INNER_W_CM,
    solve_camera,
    world_landmarks,
)


def edge_samples(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, endpoint=True)[:, None]
    return (1.0 - t) * a[None, :] + t * b[None, :]


def dense_validation_geometry() -> np.ndarray:
    """Dense, fixed NBA geometry spanning the basket and visible half-court region.

    This deliberately includes many points that were not observations in the fit:
    target-edge interpolation, the full rim circle, a dense lane-floor grid, a
    restricted-area circle and a wider half-court floor grid.  Points outside the
    actual base image are discarded later.
    """
    pts: list[np.ndarray] = []

    # Regulation target opening (fit used only its four corners).
    hw = TARGET_INNER_W_CM / 2.0
    z0 = RIM_Z_CM + 2.0 * 2.54
    z1 = RIM_Z_CM + 18.0 * 2.54 - 2.0 * 2.54
    tl = np.array([0.0, -hw, z1])
    tr = np.array([0.0, +hw, z1])
    br = np.array([0.0, +hw, z0])
    bl = np.array([0.0, -hw, z0])
    for a, b in ((tl, tr), (tr, br), (br, bl), (bl, tl)):
        pts.append(edge_samples(a, b, 31))

    # Full regulation rim circle: not used by the Left Above Rim fit.
    theta = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
    pts.append(np.column_stack([
        RIM_X_CM + RIM_RADIUS_CM * np.cos(theta),
        RIM_RADIUS_CM * np.sin(theta),
        np.full_like(theta, RIM_Z_CM),
    ]))

    # Dense lane-floor grid from baseline to FT line. Fit used only four corners.
    lane_half = PAINT_W_CM / 2.0
    xs = np.linspace(-4.0 * FOOT_CM, 15.0 * FOOT_CM, 17)
    ys = np.linspace(-lane_half, lane_half, 15)
    pts.append(np.asarray([[x, y, 0.0] for x in xs for y in ys], dtype=np.float64))

    # Restricted-area floor circle, independent of the eight fit landmarks.
    rr = 4.0 * FOOT_CM
    theta2 = np.linspace(-np.pi, np.pi, 120, endpoint=False)
    pts.append(np.column_stack([
        RIM_X_CM + rr * np.cos(theta2),
        rr * np.sin(theta2),
        np.zeros_like(theta2),
    ]))

    # Wider half-court floor test field to expose projection instability away from
    # the fitted lane corners. It is a mathematical validation grid, not an image
    # feature detector.
    xs2 = np.linspace(-4.0 * FOOT_CM, 43.0 * FOOT_CM, 13)
    ys2 = np.linspace(-24.0 * FOOT_CM, 24.0 * FOOT_CM, 13)
    pts.append(np.asarray([[x, y, 0.0] for x in xs2 for y in ys2], dtype=np.float64))

    return np.vstack(pts).astype(np.float64)


def displacement_stats(base_p: np.ndarray, test_p: np.ndarray, center: np.ndarray, dense: np.ndarray) -> dict:
    base_uv, base_cam, _, _ = project_fixed(base_p, dense, center)
    test_uv, test_cam, _, _ = project_fixed(test_p, dense, center)
    valid = (
        (base_cam[:, 2] > 20.0)
        & (test_cam[:, 2] > 20.0)
        & (base_uv[:, 0] >= 0.0) & (base_uv[:, 0] < W)
        & (base_uv[:, 1] >= 0.0) & (base_uv[:, 1] < H)
    )
    if int(valid.sum()) < 100:
        raise RuntimeError(f"Only {int(valid.sum())} dense validation points visible")
    d = np.linalg.norm(test_uv[valid] - base_uv[valid], axis=1)
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
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Frame C image")

    spec = json.loads(args.landmarks.read_text(encoding="utf-8"))
    freeze = spec.get("freeze_lock", {})
    if freeze.get("authority_camera") != "Right Slash" or freeze.get("chooser_option") != "C":
        raise RuntimeError("Landmark spec is not bound to immutable Right Slash chooser C")
    if abs(float(freeze.get("right_slash_local_time", -1.0)) - 8.275733) > 5e-7 or int(freeze.get("decoded_frame_index", -1)) != 248:
        raise RuntimeError("Immutable Frame C timing lock changed")

    view = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if view is None:
        raise RuntimeError(f"No {args.camera_label!r} landmark view")
    lower_view = json.dumps(view).lower()
    if any(token in lower_view for token in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Dynamic anchors are forbidden for metric camera promotion")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    camreg = registry["cameras"][args.camera_label]
    if camreg.get("status") != "FIXED_PHYSICAL_CENTRE_PRIOR_ACCEPTED" or not camreg["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Camera centre prior has not been independently accepted")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    world = world_landmarks()
    names = list(view["landmarks"])
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    obs = np.asarray([view["landmarks"][n] for n in names], dtype=np.float64)
    if len(names) < 8:
        raise RuntimeError("Expected at least eight fixed regulation landmarks")

    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError("Direct metric initialization is implausible")
    direct_params = np.asarray(direct[2], dtype=np.float64)
    direct_center = np.asarray(direct[4], dtype=np.float64)
    p0 = np.asarray([
        direct_params[0], direct_params[1], direct_params[2],
        direct_params[6], direct_params[7], direct_params[8],
    ], dtype=np.float64)

    p = solve_fixed(obj, obs, center, view, p0)
    fit = metrics(p, obj, obs, center)
    draw_overlay(image, names, obs, fit["uv"], args.out / "frame_c_left_above_rim_metric_overlay_v28.png")

    loo = []
    for hold in range(len(names)):
        keep = np.asarray([i for i in range(len(names)) if i != hold], dtype=int)
        ph = solve_fixed(obj[keep], obs[keep], center, view, p)
        held_pred = project_fixed(ph, obj[[hold]], center)[0][0]
        loo.append({"held_out": names[hold], "error_px": float(np.linalg.norm(held_pred - obs[hold]))})
    loo_errors = np.asarray([x["error_px"] for x in loo], dtype=np.float64)

    dense = dense_validation_geometry()
    base_dense_uv, base_dense_cam, _, _ = project_fixed(p, dense, center)
    base_visible = (
        (base_dense_cam[:, 2] > 20.0)
        & (base_dense_uv[:, 0] >= 0.0) & (base_dense_uv[:, 0] < W)
        & (base_dense_uv[:, 1] >= 0.0) & (base_dense_uv[:, 1] < H)
    )

    rng = np.random.default_rng(280902)
    perturb = []
    for trial in range(args.perturbation_trials):
        po = obs + rng.uniform(-0.5, 0.5, size=obs.shape)
        pt = solve_fixed(obj, po, center, view, p)
        mt = metrics(pt, obj, po, center)
        proj = displacement_stats(p, pt, center, dense)
        perturb.append({
            "trial": trial,
            "rotation_delta_deg": rotation_angle_deg(mt["R"], fit["R"]),
            "focal_fraction_delta": abs(mt["focal"] - fit["focal"]) / fit["focal"],
            "principal_point_shift_px": float(np.linalg.norm(pt[4:6] - p[4:6])),
            "fit_rmse_px": mt["rmse_px"],
            "dense_projective_displacement": proj,
        })

    max_proj_p95 = max(x["dense_projective_displacement"]["p95_px"] for x in perturb)
    max_proj_max = max(x["dense_projective_displacement"]["max_px"] for x in perturb)
    max_rot = max(x["rotation_delta_deg"] for x in perturb)
    max_focal_frac = max(x["focal_fraction_delta"] for x in perturb)
    max_pp = max(x["principal_point_shift_px"] for x in perturb)
    direct_center_disagreement = float(np.linalg.norm(direct_center - center))

    focal = fit["focal"]
    cx, cy = float(p[4]), float(p[5])
    R = fit["R"]
    t = -R @ center
    K = np.asarray([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    P = K @ np.column_stack([R, t])

    gates = {
        "immutable_frame_c_lock": True,
        "fixed_regulation_geometry_only": True,
        "independent_center_prior_accepted": True,
        "fit_rmse_at_most_threshold": fit["rmse_px"] <= args.max_fit_rmse_px,
        "fit_p95_at_most_threshold": fit["p95_px"] <= args.max_fit_p95_px,
        "leave_one_out_median_at_most_threshold": float(np.median(loo_errors)) <= args.max_loo_median_px,
        "leave_one_out_max_at_most_threshold": float(np.max(loo_errors)) <= args.max_loo_max_px,
        "direct_unconstrained_center_agrees_with_prior": direct_center_disagreement <= args.max_direct_center_disagreement_cm,
        "dense_projective_p95_stability": max_proj_p95 <= args.max_projective_p95_px,
        "dense_projective_max_stability": max_proj_max <= args.max_projective_max_px,
        "half_pixel_focal_stability": max_focal_frac <= args.max_perturb_focal_fraction,
        "positive_depth_all_landmarks": float(np.min(fit["cam"][:, 2])) > 20.0,
        "dense_validation_support_at_least_100_points": int(base_visible.sum()) >= 100,
    }
    passed = bool(all(gates.values()))

    payload = {
        "status": "PASS_METRIC_EVENT_CAMERA" if passed else "FAIL_METRIC_EVENT_CAMERA",
        "game_id": "0022500301",
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
        "coordinate_system": registry["coordinate_system"],
        "method": "independently proved fixed physical camera centre + exact Frame C regulation target/lane solve + dense projective perturbation stability",
        "guardrail": "This result can promote only this exact Left Above Rim Frame C camera. It does not validate player/ball depth or authorize replay rendering.",
        "center_prior_cm": [float(x) for x in center],
        "direct_unconstrained_center_cm": [float(x) for x in direct_center],
        "direct_unconstrained_center_disagreement_cm": direct_center_disagreement,
        "landmarks": names,
        "fit": {
            "rmse_px": fit["rmse_px"],
            "median_px": fit["median_px"],
            "p95_px": fit["p95_px"],
            "max_px": fit["max_px"],
            "per_point_px": {n: float(e) for n, e in zip(names, fit["per_point_px"])},
        },
        "leave_one_out": {
            "results": loo,
            "median_px": float(np.median(loo_errors)),
            "p95_px": float(np.percentile(loo_errors, 95)),
            "max_px": float(np.max(loo_errors)),
        },
        "dense_validation_geometry": {
            "point_count_total": int(len(dense)),
            "base_visible_point_count": int(base_visible.sum()),
            "contents": ["target opening edge samples", "full rim circle", "dense lane floor grid", "restricted-area circle", "wider half-court floor validation grid"],
            "note": "Dense validation points assess camera projection stability; they are not additional fitted image observations."
        },
        "half_pixel_perturbation": {
            "trials": perturb,
            "max_dense_projective_p95_px": max_proj_p95,
            "max_dense_projective_max_px": max_proj_max,
            "raw_parameter_diagnostics_not_promotion_gates": {
                "max_rotation_delta_deg": max_rot,
                "max_principal_point_shift_px": max_pp,
                "reason": "With optical centre fixed, rotation and principal point covary; promotion is gated on their resulting dense source-pixel projection instead."
            },
            "max_focal_fraction_delta": max_focal_frac,
        },
        "camera": {
            "focal_px": focal,
            "principal_point_px": [cx, cy],
            "K": K.tolist(),
            "R_world_to_camera": R.tolist(),
            "t_world_to_camera_cm": t.tolist(),
            "camera_center_world_cm": [float(x) for x in center],
            "projection_matrix_KRt": P.tolist(),
        },
        "thresholds": {
            "max_fit_rmse_px": args.max_fit_rmse_px,
            "max_fit_p95_px": args.max_fit_p95_px,
            "max_loo_median_px": args.max_loo_median_px,
            "max_loo_max_px": args.max_loo_max_px,
            "max_direct_center_disagreement_cm": args.max_direct_center_disagreement_cm,
            "max_dense_projective_p95_px": args.max_projective_p95_px,
            "max_dense_projective_max_px": args.max_projective_max_px,
            "max_perturb_focal_fraction": args.max_perturb_focal_fraction,
        },
        "gates": gates,
        "metric_event_camera_allowed": passed,
        "replay_render_allowed": False,
    }
    (args.out / "frame_c_left_above_rim_metric_camera_v28.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "fit_rmse_px": fit["rmse_px"],
        "fit_p95_px": fit["p95_px"],
        "loo_median_px": payload["leave_one_out"]["median_px"],
        "loo_max_px": payload["leave_one_out"]["max_px"],
        "direct_center_disagreement_cm": direct_center_disagreement,
        "dense_visible_points": int(base_visible.sum()),
        "max_dense_projective_p95_px": max_proj_p95,
        "max_dense_projective_max_px": max_proj_max,
        "raw_max_rotation_delta_deg": max_rot,
        "raw_max_principal_point_shift_px": max_pp,
        "max_focal_fraction_delta": max_focal_frac,
        "metric_event_camera_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
