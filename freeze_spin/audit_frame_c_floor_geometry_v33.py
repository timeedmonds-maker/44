from __future__ import annotations

"""v33: audit the old local eight-landmark solve against wider NBA floor geometry.

The v26 Left Above Rim metric fit used four elevated target corners plus four floor
lane intersections.  A low residual on those eight points does not prove the plane
mapping away from the paint.  This audit therefore builds the floor homography from
ONLY the four v26 floor anchors and asks it to predict two regulation curves that
were never used by the fit:

* the 6-foot free-throw front semicircle;
* the 23'9" NBA three-point arc.

The observed curve samples are real source pixels from the immutable Frame C image.
No player/ball/body point is used.  This script is diagnostic and fail-closed: a
failed held-out curve test invalidates the local floor model but does not alter the
previously proved physical-camera-centre evidence.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FOOT_CM = 30.48
RIM_X_CM = 15.0 * 2.54
FT_X_CM = 15.0 * FOOT_CM
FT_RADIUS_CM = 6.0 * FOOT_CM
THREE_RADIUS_CM = 23.75 * FOOT_CM
THREE_CORNER_Y_CM = 22.0 * FOOT_CM
PAINT_HALF_CM = 8.0 * FOOT_CM
BASELINE_X_CM = -4.0 * FOOT_CM


def project_h(Hm: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.column_stack([np.asarray(xy, dtype=np.float64), np.ones(len(xy))])
    q = (Hm @ p.T).T
    return q[:, :2] / q[:, 2:3]


def invert_h(Hm: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return project_h(np.linalg.inv(Hm), np.asarray(uv, dtype=np.float64))


def three_arc(n: int = 1201) -> np.ndarray:
    tmax = math.asin(THREE_CORNER_Y_CM / THREE_RADIUS_CM)
    t = np.linspace(-tmax, tmax, n)
    return np.column_stack([
        RIM_X_CM + THREE_RADIUS_CM * np.cos(t),
        THREE_RADIUS_CM * np.sin(t),
    ])


def ft_front(n: int = 801) -> np.ndarray:
    t = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
    return np.column_stack([
        FT_X_CM + FT_RADIUS_CM * np.cos(t),
        FT_RADIUS_CM * np.sin(t),
    ])


def nearest_metrics(observed: np.ndarray, projected_dense: np.ndarray) -> dict:
    # Small arrays only: deterministic brute-force nearest curve distance.
    d2 = np.sum((observed[:, None, :] - projected_dense[None, :, :]) ** 2, axis=2)
    d = np.sqrt(np.min(d2, axis=1))
    return {
        "count": int(len(d)),
        "rmse_px": float(np.sqrt(np.mean(d ** 2))),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
        "per_point_px": [float(x) for x in d],
    }


def radial_metrics(floor_xy: np.ndarray, cx: float, cy: float, expected_radius_cm: float) -> dict:
    r = np.sqrt((floor_xy[:, 0] - cx) ** 2 + (floor_xy[:, 1] - cy) ** 2)
    e = r - expected_radius_cm
    return {
        "expected_radius_ft": float(expected_radius_cm / FOOT_CM),
        "implied_radius_ft": [float(x / FOOT_CM) for x in r],
        "median_implied_radius_ft": float(np.median(r) / FOOT_CM),
        "min_implied_radius_ft": float(np.min(r) / FOOT_CM),
        "max_implied_radius_ft": float(np.max(r) / FOOT_CM),
        "median_absolute_radius_error_ft": float(np.median(np.abs(e)) / FOOT_CM),
        "max_absolute_radius_error_ft": float(np.max(np.abs(e)) / FOOT_CM),
    }


def quadratic_vertex(points: np.ndarray) -> dict:
    x = points[:, 0]
    y = points[:, 1]
    a, b, c = np.polyfit(x, y, 2)
    xv = float(-b / (2.0 * a)) if abs(a) > 1e-12 else float(np.median(x))
    yv = float(a * xv * xv + b * xv + c)
    return {"x_px": xv, "y_px": yv, "coefficients": [float(a), float(b), float(c)]}


def draw_overlay(image: np.ndarray, lane_uv: np.ndarray, obs_three: np.ndarray, obs_ft: np.ndarray,
                 pred_three: np.ndarray, pred_ft: np.ndarray, path: Path) -> None:
    out = image.copy()
    # Predicted regulation curves from the old four-anchor floor homography.
    for pts, color in ((pred_three, (0, 0, 255)), (pred_ft, (0, 165, 255))):
        p = np.round(pts).astype(int)
        valid = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        ids = np.where(valid)[0]
        for i in ids:
            cv2.circle(out, tuple(p[i]), 1, color, -1, cv2.LINE_AA)
    for p in np.round(obs_three).astype(int):
        cv2.circle(out, tuple(p), 4, (255, 255, 0), 2, cv2.LINE_AA)
    for p in np.round(obs_ft).astype(int):
        cv2.circle(out, tuple(p), 4, (0, 255, 0), 2, cv2.LINE_AA)
    for p in np.round(lane_uv).astype(int):
        cv2.circle(out, tuple(p), 5, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-heldout-p95-px", type=float, default=3.0)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Frame C")

    spec = json.loads(args.landmarks.read_text(encoding="utf-8"))
    freeze = spec["freeze_lock"]
    if freeze["authority_camera"] != "Right Slash" or freeze["chooser_option"] != "C":
        raise RuntimeError("Immutable Frame C authority changed")
    if abs(float(freeze["right_slash_local_time"]) - 8.275733) > 5e-7 or int(freeze["decoded_frame_index"]) != 248:
        raise RuntimeError("Immutable Frame C timing changed")
    view = next(v for v in spec["views"] if v["label"] == "Left Above Rim")

    curves = json.loads(args.curves.read_text(encoding="utf-8"))
    lock = curves["freeze_lock"]
    if abs(float(lock["left_above_rim_synchronized_time"]) - 8.653093) > 5e-7 or int(lock["left_above_rim_decoded_frame_index"]) != 259:
        raise RuntimeError("Held-out curve spec not bound to immutable Left Above Rim Frame C")

    floor_names = ["baseline_left_lane", "baseline_right_lane", "ft_left_lane", "ft_right_lane"]
    lane_world = np.asarray([
        [BASELINE_X_CM, -PAINT_HALF_CM], [BASELINE_X_CM, +PAINT_HALF_CM],
        [FT_X_CM, -PAINT_HALF_CM], [FT_X_CM, +PAINT_HALF_CM],
    ], dtype=np.float64)
    lane_uv = np.asarray([view["landmarks"][n] for n in floor_names], dtype=np.float64)
    H_lane = cv2.getPerspectiveTransform(lane_world.astype(np.float32), lane_uv.astype(np.float32)).astype(np.float64)

    world_three = three_arc()
    world_ft = ft_front()
    pred_three = project_h(H_lane, world_three)
    pred_ft = project_h(H_lane, world_ft)
    obs_three = np.asarray(curves["observed_curves_px"]["three_point_arc"], dtype=np.float64)
    obs_ft = np.asarray(curves["observed_curves_px"]["free_throw_front_semicircle"], dtype=np.float64)

    three_px = nearest_metrics(obs_three, pred_three)
    ft_px = nearest_metrics(obs_ft, pred_ft)
    three_floor = invert_h(H_lane, obs_three)
    ft_floor = invert_h(H_lane, obs_ft)
    three_radius = radial_metrics(three_floor, RIM_X_CM, 0.0, THREE_RADIUS_CM)
    ft_radius = radial_metrics(ft_floor, FT_X_CM, 0.0, FT_RADIUS_CM)

    # Directly compare front-most centreline positions.  The observed y values are
    # estimated from the real sampled curves only; no old camera parameters enter.
    observed_three_vertex = quadratic_vertex(obs_three)
    observed_ft_vertex = quadratic_vertex(obs_ft)
    test_world = np.asarray([
        [BASELINE_X_CM, 0.0],
        [FT_X_CM, 0.0],
        [FT_X_CM + FT_RADIUS_CM, 0.0],
        [RIM_X_CM + THREE_RADIUS_CM, 0.0],
    ], dtype=np.float64)
    test_uv = project_h(H_lane, test_world)
    centreline = {
        "baseline_center_predicted_px": [float(x) for x in test_uv[0]],
        "free_throw_line_center_predicted_px": [float(x) for x in test_uv[1]],
        "free_throw_circle_front_predicted_px": [float(x) for x in test_uv[2]],
        "three_point_arc_front_predicted_px": [float(x) for x in test_uv[3]],
        "free_throw_circle_observed_curve_vertex_px": observed_ft_vertex,
        "three_point_arc_observed_curve_vertex_px": observed_three_vertex,
        "free_throw_front_y_disagreement_px": float(test_uv[2, 1] - observed_ft_vertex["y_px"]),
        "three_point_front_y_disagreement_px": float(test_uv[3, 1] - observed_three_vertex["y_px"]),
    }

    # Half-pixel perturbation of the four old floor anchors exposes extrapolation
    # conditioning.  We compare the SAME regulation world samples point-for-point.
    rng = np.random.default_rng(330903)
    perturb = []
    for trial in range(args.perturbation_trials):
        q = lane_uv + rng.uniform(-0.5, 0.5, size=lane_uv.shape)
        Hp = cv2.getPerspectiveTransform(lane_world.astype(np.float32), q.astype(np.float32)).astype(np.float64)
        p3 = project_h(Hp, world_three)
        pf = project_h(Hp, world_ft)
        d3 = np.linalg.norm(p3 - pred_three, axis=1)
        df = np.linalg.norm(pf - pred_ft, axis=1)
        perturb.append({
            "trial": trial,
            "three_point_p95_shift_px": float(np.percentile(d3, 95)),
            "three_point_max_shift_px": float(np.max(d3)),
            "free_throw_p95_shift_px": float(np.percentile(df, 95)),
            "free_throw_max_shift_px": float(np.max(df)),
        })

    heldout_pass = three_px["p95_px"] <= args.max_heldout_p95_px and ft_px["p95_px"] <= args.max_heldout_p95_px
    status = "PASS_WIDE_FLOOR_GEOMETRY" if heldout_pass else "FAIL_LOCAL_EIGHT_POINT_FLOOR_MODEL"

    draw_overlay(image, lane_uv, obs_three, obs_ft, pred_three, pred_ft,
                 args.out / "frame_c_floor_geometry_overlay_v33.png")

    payload = {
        "status": status,
        "version": "v33_wide_floor_heldout_audit",
        "game_id": "0022500301",
        "event_id": 489,
        "camera_label": "Left Above Rim",
        "method": "four v26 floor anchors -> exact planar homography -> held-out 6ft free-throw semicircle and 23ft9in three-point arc",
        "guardrail": "This audit can invalidate the local floor model. It cannot revoke the independently proved fixed physical camera centre and cannot authorize replay rendering.",
        "floor_anchor_names": floor_names,
        "floor_anchor_pixels": lane_uv.tolist(),
        "floor_anchor_world_cm": lane_world.tolist(),
        "floor_homography_world_to_image": H_lane.tolist(),
        "heldout_curve_pixel_error": {
            "three_point_arc": three_px,
            "free_throw_front_semicircle": ft_px,
        },
        "heldout_curve_implied_world_radius": {
            "three_point_arc": three_radius,
            "free_throw_front_semicircle": ft_radius,
        },
        "centreline_diagnostic": centreline,
        "half_pixel_floor_anchor_perturbation": {
            "trials": perturb,
            "max_three_point_p95_shift_px": max(x["three_point_p95_shift_px"] for x in perturb),
            "max_three_point_max_shift_px": max(x["three_point_max_shift_px"] for x in perturb),
            "max_free_throw_p95_shift_px": max(x["free_throw_p95_shift_px"] for x in perturb),
            "max_free_throw_max_shift_px": max(x["free_throw_max_shift_px"] for x in perturb),
        },
        "thresholds": {"max_heldout_curve_p95_px": args.max_heldout_p95_px},
        "gates": {
            "immutable_frame_c_lock": True,
            "source_pixel_static_court_curves_only": True,
            "three_point_heldout_p95_at_most_threshold": three_px["p95_px"] <= args.max_heldout_p95_px,
            "free_throw_heldout_p95_at_most_threshold": ft_px["p95_px"] <= args.max_heldout_p95_px,
        },
        "floor_model_validated": heldout_pass,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
        "next_gate": "Replace the local four-floor-anchor model with a wider regulation-court solve using multiple visible lines/curves, then re-run held-out geometry before metric camera promotion."
    }
    (args.out / "frame_c_floor_geometry_audit_v33.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": status,
        "three_point_p95_px": three_px["p95_px"],
        "free_throw_p95_px": ft_px["p95_px"],
        "three_point_front_y_disagreement_px": centreline["three_point_front_y_disagreement_px"],
        "free_throw_front_y_disagreement_px": centreline["free_throw_front_y_disagreement_px"],
        "max_half_pixel_three_point_p95_shift_px": payload["half_pixel_floor_anchor_perturbation"]["max_three_point_p95_shift_px"],
        "floor_model_validated": heldout_pass,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
