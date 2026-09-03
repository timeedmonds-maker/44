from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 540
FOOT_CM = 30.48
INCH_CM = 2.54
BASELINE_X_CM = -4.0 * FOOT_CM
FT_X_CM = 15.0 * FOOT_CM
PAINT_HALF_CM = 8.0 * FOOT_CM
RIM_X_CM = 15.0 * INCH_CM
RESTRICTED_R_CM = 4.0 * FOOT_CM
FT_R_CM = 6.0 * FOOT_CM

WORLD_CORNERS = np.asarray([
    [FT_X_CM, -PAINT_HALF_CM],
    [FT_X_CM, +PAINT_HALF_CM],
    [BASELINE_X_CM, -PAINT_HALF_CM],
    [BASELINE_X_CM, +PAINT_HALF_CM],
], dtype=np.float64)


def line(p1, p2):
    return np.cross(
        np.r_[np.asarray(p1, dtype=np.float64), 1.0],
        np.r_[np.asarray(p2, dtype=np.float64), 1.0],
    )


def intersect(a, b):
    p = np.cross(a, b)
    if abs(float(p[2])) < 1e-10:
        raise RuntimeError("Parallel/degenerate line intersection")
    return p[:2] / p[2]


def corners_from_segments(seg):
    lines = {k: line(v[0], v[1]) for k, v in seg.items()}
    return np.asarray([
        intersect(lines["free_throw_line"], lines["far_lane_sideline"]),
        intersect(lines["free_throw_line"], lines["near_lane_sideline"]),
        intersect(lines["baseline"], lines["far_lane_sideline"]),
        intersect(lines["baseline"], lines["near_lane_sideline"]),
    ], dtype=np.float64)


def homography_from_segments(seg):
    uv = corners_from_segments(seg)
    Hm = cv2.getPerspectiveTransform(
        WORLD_CORNERS.astype(np.float32), uv.astype(np.float32)
    ).astype(np.float64)
    if not np.isfinite(Hm).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        raise RuntimeError("Degenerate floor homography")
    return Hm, uv


def project(Hm, xy):
    xy = np.asarray(xy, dtype=np.float64)
    ph = np.column_stack([xy, np.ones(len(xy))])
    q = (Hm @ ph.T).T
    return q[:, :2] / q[:, 2:3]


def circle(cx, radius, n=1601):
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.column_stack([cx + radius * np.cos(t), radius * np.sin(t)])


def nearest_metrics(obs, pred):
    d = np.sqrt(np.sum((obs[:, None, :] - pred[None, :, :]) ** 2, axis=2)).min(axis=1)
    return {
        "count": int(len(d)),
        "rms_px": float(np.sqrt(np.mean(d * d))),
        "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)),
        "per_point_px": [float(x) for x in d],
    }


def draw(image, uv, restricted_pred, ft_pred, restricted_obs, ft_obs, out):
    im = image.copy()
    for pts, color in ((restricted_pred, (0, 255, 0)), (ft_pred, (0, 0, 255))):
        p = np.round(pts).astype(np.int32)
        valid = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
        for i in np.where(valid)[0]:
            cv2.circle(im, tuple(p[i]), 1, color, -1, cv2.LINE_AA)
    for p in np.round(restricted_obs).astype(int):
        cv2.circle(im, tuple(p), 4, (255, 255, 0), 2, cv2.LINE_AA)
    for p in np.round(ft_obs).astype(int):
        cv2.circle(im, tuple(p), 4, (255, 0, 255), 2, cv2.LINE_AA)
    for p in np.round(uv).astype(int):
        cv2.circle(im, tuple(p), 5, (0, 165, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(out), im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 In Arena Frame C")

    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    digest = hashlib.sha256(args.frame.read_bytes()).hexdigest()
    lock = spec["freeze_lock"]
    if (
        lock["camera"] != "In Arena"
        or abs(float(lock["synchronized_local_time"]) - 9.062573) > 5e-7
        or int(lock["decoded_frame_index"]) != 272
    ):
        raise RuntimeError("In Arena immutable Frame C lock changed")
    if lock.get("sha256_png") and digest != lock["sha256_png"]:
        raise RuntimeError(f"Frame hash mismatch: {digest}")

    seg = {
        k: np.asarray(v, dtype=np.float64)
        for k, v in spec["training_line_segments_px"].items()
    }
    Hm, uv = homography_from_segments(seg)

    restricted_world = circle(RIM_X_CM, RESTRICTED_R_CM)
    ft_world = circle(FT_X_CM, FT_R_CM)
    restricted_pred = project(Hm, restricted_world)
    ft_pred = project(Hm, ft_world)
    restricted_obs = np.asarray(
        spec["heldout_curves_px"]["restricted_area_arc"], dtype=np.float64
    )
    ft_obs = np.asarray(
        spec["heldout_curves_px"]["free_throw_circle_dashed_half"], dtype=np.float64
    )
    restricted_metrics = nearest_metrics(restricted_obs, restricted_pred)
    ft_metrics = nearest_metrics(ft_obs, ft_pred)

    rng = np.random.default_rng(520903)
    trials = []
    for trial in range(int(spec["perturbation_trials"])):
        perturbed = {
            k: v + rng.uniform(-0.5, 0.5, size=v.shape)
            for k, v in seg.items()
        }
        Hp, _ = homography_from_segments(perturbed)
        restricted_p = project(Hp, restricted_world)
        ft_p = project(Hp, ft_world)
        restricted_shift = np.linalg.norm(restricted_p - restricted_pred, axis=1)
        ft_shift = np.linalg.norm(ft_p - ft_pred, axis=1)
        restricted_holdout = nearest_metrics(restricted_obs, restricted_p)
        ft_holdout = nearest_metrics(ft_obs, ft_p)
        trials.append({
            "trial": trial,
            "restricted_curve_p95_shift_px": float(np.percentile(restricted_shift, 95)),
            "free_throw_curve_p95_shift_px": float(np.percentile(ft_shift, 95)),
            "restricted_heldout_p95_px": restricted_holdout["p95_px"],
            "free_throw_heldout_p95_px": ft_holdout["p95_px"],
        })

    worst = {
        "restricted_curve_p95_shift_px": max(x["restricted_curve_p95_shift_px"] for x in trials),
        "free_throw_curve_p95_shift_px": max(x["free_throw_curve_p95_shift_px"] for x in trials),
        "restricted_heldout_p95_px": max(x["restricted_heldout_p95_px"] for x in trials),
        "free_throw_heldout_p95_px": max(x["free_throw_heldout_p95_px"] for x in trials),
    }
    thresholds = spec["thresholds"]
    gates = {
        "immutable_frame_lock": True,
        "regulation_lane_line_families_only": True,
        "restricted_arc_nominal_p95": restricted_metrics["p95_px"] <= thresholds["max_restricted_p95_px"],
        "free_throw_arc_nominal_p95": ft_metrics["p95_px"] <= thresholds["max_free_throw_p95_px"],
        "restricted_arc_half_pixel_stability": worst["restricted_heldout_p95_px"] <= thresholds["max_perturbed_restricted_p95_px"],
        "restricted_projection_half_pixel_stability": worst["restricted_curve_p95_shift_px"] <= thresholds["max_restricted_curve_shift_p95_px"],
        "free_throw_projection_half_pixel_stability": worst["free_throw_curve_p95_shift_px"] <= thresholds["max_free_throw_curve_shift_p95_px"],
    }
    passed = bool(all(gates.values()))

    payload = {
        "status": "PASS_IN_ARENA_FLOOR_V52" if passed else "FAIL_IN_ARENA_FLOOR_V52",
        "game_id": "0022500301",
        "event_id": 489,
        "camera": "In Arena",
        "version": "v52",
        "method": "four independently fitted long regulation line families -> lane-corner intersections -> exact NBA lane homography; held-out restricted-area and free-throw curves; 64 half-pixel endpoint perturbations",
        "guardrail": "No player, ball, rim, backboard or arena feature enters the floor fit. Passing validates only the floor homography, not a 3D physical camera.",
        "source_sha256_png": digest,
        "training_line_segments_px": {k: v.tolist() for k, v in seg.items()},
        "derived_lane_corners_px": uv.tolist(),
        "world_lane_corners_cm": WORLD_CORNERS.tolist(),
        "floor_homography_world_to_image": Hm.tolist(),
        "heldout_curve_error": {
            "restricted_area_arc": restricted_metrics,
            "free_throw_circle_dashed_half": ft_metrics,
        },
        "half_pixel_endpoint_perturbation": {
            "trial_count": len(trials),
            "worst": worst,
            "trials": trials,
        },
        "thresholds": thresholds,
        "gates": gates,
        "floor_homography_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
        "next_gate": (
            "Use this floor homography as the planar base, then solve In Arena intrinsics/extrinsics with independent non-coplanar regulation basket geometry and require held-out basket reprojection plus camera-center perturbation stability."
            if passed
            else "Reject these line identities/annotations and do not proceed to 3D camera recovery."
        ),
    }
    (args.out / "in_arena_floor_v52.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    draw(
        image, uv, restricted_pred, ft_pred, restricted_obs, ft_obs,
        args.out / "in_arena_floor_overlay_v52.png"
    )
    print(json.dumps({
        "status": payload["status"],
        "restricted_rms_px": restricted_metrics["rms_px"],
        "restricted_p95_px": restricted_metrics["p95_px"],
        "free_throw_rms_px": ft_metrics["rms_px"],
        "free_throw_p95_px": ft_metrics["p95_px"],
        "worst_perturbed_restricted_p95_px": worst["restricted_heldout_p95_px"],
        "worst_restricted_curve_p95_shift_px": worst["restricted_curve_p95_shift_px"],
        "floor_homography_allowed": payload["floor_homography_allowed"],
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
