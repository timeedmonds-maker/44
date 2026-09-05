from __future__ import annotations

"""Promote the immutable Frame C Left Above Rim view to a metric event camera.

The physical optical centre is NOT estimated here.  It is locked to the independently
validated same-game centre prior in adams_jazz_game_camera_registry_v1.json.  The
exact synchronized Frame C image then solves only orientation, focal length and
principal point against fixed regulation NBA target/lane landmarks.

No player or ball points are used.  Passing this script may promote this one exact
Frame C event camera.  It still does not authorize a replay render or foreground 3D.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks

W, H = 960, 540


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    d = Ra @ Rb.T
    c = float(np.clip((np.trace(d) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def project_fixed(p: np.ndarray, obj: np.ndarray, center: np.ndarray):
    R, _ = cv2.Rodrigues(np.asarray(p[:3], dtype=np.float64))
    focal = float(np.exp(p[3]))
    cx, cy = float(p[4]), float(p[5])
    cam = (R @ (obj - center).T).T
    uv = np.column_stack([
        focal * cam[:, 0] / cam[:, 2] + cx,
        focal * cam[:, 1] / cam[:, 2] + cy,
    ])
    return uv, cam, focal, R


def solve_fixed(
    obj: np.ndarray,
    obs: np.ndarray,
    center: np.ndarray,
    view: dict,
    p0: np.ndarray,
) -> np.ndarray:
    pp_sigma = float(view.get("principal_point_prior_sigma_px", 160.0))
    pp_bound = float(view.get("principal_point_bound_px", 350.0))
    focal_prior = float(view.get("focal_prior_px", 900.0))
    focal_sigma_log = float(view.get("focal_prior_sigma_log", 1.8))

    def residual(p: np.ndarray) -> np.ndarray:
        uv, cam, focal, _ = project_fixed(p, obj, center)
        out = [(uv - obs).ravel()]
        out.append(np.asarray([
            (p[4] - W / 2.0) / pp_sigma,
            (p[5] - H / 2.0) / pp_sigma,
            (math.log(focal) - math.log(focal_prior)) / focal_sigma_log,
        ], dtype=np.float64))
        out.append(np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0)
        return np.concatenate(out)

    lower = np.asarray([-np.inf, -np.inf, -np.inf, math.log(150.0), W / 2.0 - pp_bound, H / 2.0 - pp_bound])
    upper = np.asarray([+np.inf, +np.inf, +np.inf, math.log(4000.0), W / 2.0 + pp_bound, H / 2.0 + pp_bound])
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


def metrics(p: np.ndarray, obj: np.ndarray, obs: np.ndarray, center: np.ndarray) -> dict:
    uv, cam, focal, R = project_fixed(p, obj, center)
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


def draw_overlay(image: np.ndarray, names: list[str], obs: np.ndarray, pred: np.ndarray, out: Path) -> None:
    im = image.copy()
    idx = {n: i for i, n in enumerate(names)}
    rect_names = ["target_inner_top_left", "target_inner_top_right", "target_inner_bottom_right", "target_inner_bottom_left"]
    lane_names = ["baseline_left_lane", "baseline_right_lane", "ft_right_lane", "ft_left_lane"]
    if all(n in idx for n in rect_names):
        pts = np.round(np.asarray([pred[idx[n]] for n in rect_names])).astype(np.int32)
        cv2.polylines(im, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    if all(n in idx for n in lane_names):
        pts = np.round(np.asarray([pred[idx[n]] for n in lane_names])).astype(np.int32)
        cv2.polylines(im, [pts], True, (255, 180, 0), 2, cv2.LINE_AA)
    for i, (o, p) in enumerate(zip(obs, pred)):
        oo = tuple(np.round(o).astype(int))
        pp = tuple(np.round(p).astype(int))
        cv2.circle(im, oo, 4, (0, 215, 255), -1, cv2.LINE_AA)
        cv2.circle(im, pp, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.line(im, oo, pp, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, str(i + 1), (oo[0] + 5, oo[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), im)


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
    ap.add_argument("--max-perturb-rotation-deg", type=float, default=0.75)
    ap.add_argument("--max-perturb-focal-fraction", type=float, default=0.05)
    ap.add_argument("--max-perturb-principal-point-px", type=float, default=10.0)
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
    if camreg["permissions"].get("metric_event_camera_allowed") or camreg["permissions"].get("replay_render_allowed"):
        raise RuntimeError("Registry input must still be centre-prior-only before this proof")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    world = world_landmarks()
    names = list(view["landmarks"])
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    obs = np.asarray([view["landmarks"][n] for n in names], dtype=np.float64)
    if len(names) < 8:
        raise RuntimeError("Expected at least eight fixed regulation landmarks")

    # Unconstrained solve is initialization and an independent sanity comparison only.
    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError("Existing direct metric initialization is implausible")
    direct_params = np.asarray(direct[2], dtype=np.float64)
    direct_center = np.asarray(direct[4], dtype=np.float64)
    R0, _ = cv2.Rodrigues(direct_params[:3])
    r0, _ = cv2.Rodrigues(R0)
    p0 = np.asarray([r0[0, 0], r0[1, 0], r0[2, 0], direct_params[6], direct_params[7], direct_params[8]], dtype=np.float64)

    p = solve_fixed(obj, obs, center, view, p0)
    fit = metrics(p, obj, obs, center)
    pred = fit["uv"]
    draw_overlay(image, names, obs, pred, args.out / "frame_c_left_above_rim_metric_overlay_v27.png")

    loo = []
    for hold in range(len(names)):
        keep = np.asarray([i for i in range(len(names)) if i != hold], dtype=int)
        ph = solve_fixed(obj[keep], obs[keep], center, view, p)
        held_pred = project_fixed(ph, obj[[hold]], center)[0][0]
        held_err = float(np.linalg.norm(held_pred - obs[hold]))
        loo.append({"held_out": names[hold], "error_px": held_err})
    loo_errors = np.asarray([x["error_px"] for x in loo], dtype=np.float64)

    rng = np.random.default_rng(270902)
    perturb = []
    for trial in range(args.perturbation_trials):
        po = obs + rng.uniform(-0.5, 0.5, size=obs.shape)
        pt = solve_fixed(obj, po, center, view, p)
        mt = metrics(pt, obj, po, center)
        perturb.append({
            "trial": trial,
            "rotation_delta_deg": rotation_angle_deg(mt["R"], fit["R"]),
            "focal_fraction_delta": abs(mt["focal"] - fit["focal"]) / fit["focal"],
            "principal_point_shift_px": float(np.linalg.norm(pt[4:6] - p[4:6])),
            "fit_rmse_px": mt["rmse_px"],
        })

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
        "half_pixel_rotation_stability": max_rot <= args.max_perturb_rotation_deg,
        "half_pixel_focal_stability": max_focal_frac <= args.max_perturb_focal_fraction,
        "half_pixel_principal_point_stability": max_pp <= args.max_perturb_principal_point_px,
        "positive_depth_all_landmarks": float(np.min(fit["cam"][:, 2])) > 20.0,
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
        "method": "independently proved fixed physical camera centre + exact Frame C regulation target/lane solve",
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
        "half_pixel_perturbation": {
            "trials": perturb,
            "max_rotation_delta_deg": max_rot,
            "max_focal_fraction_delta": max_focal_frac,
            "max_principal_point_shift_px": max_pp,
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
            "max_perturb_rotation_deg": args.max_perturb_rotation_deg,
            "max_perturb_focal_fraction": args.max_perturb_focal_fraction,
            "max_perturb_principal_point_px": args.max_perturb_principal_point_px,
        },
        "gates": gates,
        "metric_event_camera_allowed": passed,
        "replay_render_allowed": False,
    }
    (args.out / "frame_c_left_above_rim_metric_camera_v27.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "fit_rmse_px": fit["rmse_px"],
        "fit_p95_px": fit["p95_px"],
        "loo_median_px": payload["leave_one_out"]["median_px"],
        "loo_max_px": payload["leave_one_out"]["max_px"],
        "direct_center_disagreement_cm": direct_center_disagreement,
        "max_perturb_rotation_deg": max_rot,
        "max_perturb_focal_fraction_delta": max_focal_frac,
        "max_perturb_principal_point_shift_px": max_pp,
        "metric_event_camera_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
