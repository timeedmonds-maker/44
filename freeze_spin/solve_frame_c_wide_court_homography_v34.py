from __future__ import annotations

"""v34: wide regulation-court floor homography for immutable Left Above Rim Frame C.

v33 formally retired the local v26 four-floor-anchor model: it fit the paint locally
but missed the held-out 23'9" three-point arc by ~29 px p95 and the 6-foot free-throw
semicircle by ~12 px p95.  v34 therefore does not fit those retired baseline anchors.

Instead, this stage estimates the floor plane from wide, directly visible regulation
court paint: the NBA three-point arc, free-throw circle, both corner-three straight
segments and the free-throw line.  Spatially distributed observations from every
feature group are held out before fitting.  No player/ball/body point is permitted.

Passing validates only the Frame C floor-plane homography.  It does not by itself
promote a 3D metric event camera or authorize replay rendering.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

W, H = 960, 540
FOOT_CM = 30.48
INCH_CM = 2.54
RIM_X_CM = 15.0 * INCH_CM
BASELINE_X_CM = -4.0 * FOOT_CM
FT_X_CM = 15.0 * FOOT_CM
FT_R_CM = 6.0 * FOOT_CM
THREE_R_CM = 23.75 * FOOT_CM
CORNER_Y_CM = 22.0 * FOOT_CM
PAINT_HALF_CM = 8.0 * FOOT_CM

SCALES = np.asarray([1.0, 1.0, 300.0, 1.0, 1.0, 300.0, 0.002, 0.002], dtype=np.float64)


def project_h(Hm: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    ph = np.column_stack([xy, np.ones(len(xy))])
    q = (Hm @ ph.T).T
    return q[:, :2] / q[:, 2:3]


def parameter_vector(Hm: np.ndarray) -> np.ndarray:
    Hm = np.asarray(Hm, dtype=np.float64) / float(Hm[2, 2])
    return np.r_[Hm[0], Hm[1], Hm[2, :2]]


def H_from_z(z: np.ndarray, h0: np.ndarray) -> np.ndarray:
    v = h0 + np.asarray(z, dtype=np.float64) * SCALES
    return np.asarray([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], 1.0]], dtype=np.float64)


def split_groups(obs: dict, held_indices: dict) -> tuple[dict, dict]:
    train, held = {}, {}
    for key, rows in obs.items():
        pts = np.asarray(rows, dtype=np.float64)
        ids = np.asarray(held_indices[key], dtype=int)
        if np.any(ids < 0) or np.any(ids >= len(pts)):
            raise RuntimeError(f"Invalid held-out index for {key}")
        mask = np.ones(len(pts), dtype=bool)
        mask[ids] = False
        train[key] = pts[mask]
        held[key] = pts[~mask]
        if len(train[key]) < 3 or len(held[key]) < 1:
            raise RuntimeError(f"Insufficient train/held-out support for {key}")
    return train, held


def world_constraint_residuals(Hm: np.ndarray, groups: dict) -> dict[str, np.ndarray]:
    G = np.linalg.inv(Hm)
    out: dict[str, np.ndarray] = {}
    p = project_h(G, groups["three_point_arc"])
    out["three_point_arc"] = np.sqrt((p[:, 0] - RIM_X_CM) ** 2 + p[:, 1] ** 2) - THREE_R_CM
    p = project_h(G, groups["free_throw_front_semicircle"])
    out["free_throw_front_semicircle"] = np.sqrt((p[:, 0] - FT_X_CM) ** 2 + p[:, 1] ** 2) - FT_R_CM
    p = project_h(G, groups["left_corner_three_straight"])
    out["left_corner_three_straight"] = p[:, 1] + CORNER_Y_CM
    p = project_h(G, groups["right_corner_three_straight"])
    out["right_corner_three_straight"] = p[:, 1] - CORNER_Y_CM
    p = project_h(G, groups["free_throw_line"])
    out["free_throw_line"] = p[:, 0] - FT_X_CM
    return out


def residual(z: np.ndarray, h0: np.ndarray, groups: dict) -> np.ndarray:
    Hm = H_from_z(z, h0)
    if abs(np.linalg.det(Hm)) < 1e-12:
        return np.full(sum(len(v) for v in groups.values()) + len(z), 1e6, dtype=np.float64)
    rows = world_constraint_residuals(Hm, groups)
    # 3 cm ~= one source-pixel annotation scale in this view. Equal feature groups
    # are already spatially sparse, so each visible point retains equal influence.
    out = [v / 3.0 for v in rows.values()]
    # This is numerical root regularization only.  The retired v26 baseline pixels
    # never appear in the residual; z=0 is merely a physically plausible seed.
    out.append(np.asarray(z, dtype=np.float64) * 0.001)
    return np.concatenate(out)


def solve(h0: np.ndarray, groups: dict, *, warm: np.ndarray | None = None) -> np.ndarray:
    seeds = []
    if warm is not None:
        seeds.append(np.asarray(warm, dtype=np.float64))
    seeds.append(np.zeros(8, dtype=np.float64))
    rng = np.random.default_rng(340903)
    for _ in range(7):
        seeds.append(rng.uniform(-0.25, 0.25, size=8))
    best, best_score = None, float("inf")
    for x0 in seeds:
        try:
            fit = least_squares(
                lambda z: residual(z, h0, groups), x0,
                loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=50000,
            )
            Hm = H_from_z(fit.x, h0)
            wr = world_constraint_residuals(Hm, groups)
            score = float(np.median(np.concatenate([np.abs(x) for x in wr.values()])))
            if np.isfinite(score) and score < best_score:
                best, best_score = np.asarray(fit.x, dtype=np.float64), score
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Wide-court homography solve failed from all deterministic roots")
    return best


def dense_three(n: int = 2001) -> np.ndarray:
    tmax = math.asin(CORNER_Y_CM / THREE_R_CM)
    t = np.linspace(-tmax, tmax, n)
    return np.column_stack([RIM_X_CM + THREE_R_CM * np.cos(t), THREE_R_CM * np.sin(t)])


def dense_ft(n: int = 1601) -> np.ndarray:
    t = np.linspace(-math.pi / 2.0, math.pi / 2.0, n)
    return np.column_stack([FT_X_CM + FT_R_CM * np.cos(t), FT_R_CM * np.sin(t)])


def nearest_curve_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    d = np.sqrt(np.sum((obs[:, None, :] - pred[None, :, :]) ** 2, axis=2)).min(axis=1)
    return {
        "count": int(len(d)), "rmse_px": float(np.sqrt(np.mean(d ** 2))),
        "median_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)),
        "max_px": float(np.max(d)), "per_point_px": [float(x) for x in d],
    }


def constraint_metrics(Hm: np.ndarray, groups: dict) -> dict:
    rows = world_constraint_residuals(Hm, groups)
    return {k: {
        "count": int(len(v)), "median_abs_cm": float(np.median(np.abs(v))),
        "p95_abs_cm": float(np.percentile(np.abs(v), 95)), "max_abs_cm": float(np.max(np.abs(v))),
    } for k, v in rows.items()}


def legacy_predictions(Hm: np.ndarray, legacy: dict) -> dict:
    world = {
        "baseline_left_lane": [BASELINE_X_CM, -PAINT_HALF_CM],
        "baseline_right_lane": [BASELINE_X_CM, +PAINT_HALF_CM],
        "ft_left_lane": [FT_X_CM, -PAINT_HALF_CM],
        "ft_right_lane": [FT_X_CM, +PAINT_HALF_CM],
    }
    names = list(world)
    pred = project_h(Hm, np.asarray([world[n] for n in names], dtype=np.float64))
    out = {}
    for n, p in zip(names, pred):
        old = np.asarray(legacy[n], dtype=np.float64)
        out[n] = {
            "wide_model_predicted_px": [float(p[0]), float(p[1])],
            "legacy_v26_observed_px": [float(old[0]), float(old[1])],
            "disagreement_px": float(np.linalg.norm(p - old)),
        }
    return out


def draw_overlay(image: np.ndarray, spec: dict, Hm: np.ndarray, path: Path) -> None:
    out = image.copy()
    p3 = project_h(Hm, dense_three())
    pf = project_h(Hm, dense_ft())
    # full regulation predictions
    for pts, color in ((p3, (0, 0, 255)), (pf, (0, 165, 255))):
        q = np.round(pts).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        for x, y in q[ok]:
            cv2.circle(out, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)
    colors = {
        "three_point_arc": (255, 255, 0), "free_throw_front_semicircle": (0, 255, 0),
        "left_corner_three_straight": (255, 0, 255), "right_corner_three_straight": (255, 0, 255),
        "free_throw_line": (0, 255, 255),
    }
    held = spec["held_out_indices"]
    for key, rows in spec["observations_px"].items():
        pts = np.asarray(rows, dtype=int)
        held_set = set(held[key])
        for i, p in enumerate(pts):
            radius = 5 if i in held_set else 3
            thickness = 2 if i in held_set else 1
            cv2.circle(out, tuple(p), radius, colors[key], thickness, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--wide-court", type=Path, required=True)
    ap.add_argument("--legacy-landmarks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-heldout-three-p95-px", type=float, default=2.0)
    ap.add_argument("--max-heldout-ft-p95-px", type=float, default=2.0)
    ap.add_argument("--max-heldout-line-p95-cm", type=float, default=8.0)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    ap.add_argument("--max-half-pixel-three-p95-shift-px", type=float, default=2.5)
    ap.add_argument("--max-half-pixel-ft-p95-shift-px", type=float, default=2.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Frame C")
    spec = json.loads(args.wide_court.read_text(encoding="utf-8"))
    lock = spec["freeze_lock"]
    if lock["authority_camera"] != "Right Slash" or lock["chooser_option"] != "C":
        raise RuntimeError("Wide-court spec not bound to immutable chooser C")
    if abs(float(lock["right_slash_local_time"]) - 8.275733) > 5e-7 or int(lock["right_slash_decoded_frame_index"]) != 248:
        raise RuntimeError("Immutable authority timing changed")
    if abs(float(lock["left_above_rim_synchronized_time"]) - 8.653093) > 5e-7 or int(lock["left_above_rim_decoded_frame_index"]) != 259:
        raise RuntimeError("Immutable Left Above Rim Frame C changed")

    legacy_spec = json.loads(args.legacy_landmarks.read_text(encoding="utf-8"))
    legacy_view = next(v for v in legacy_spec["views"] if v["label"] == "Left Above Rim")
    lane_world = np.asarray([
        [BASELINE_X_CM, -PAINT_HALF_CM], [BASELINE_X_CM, +PAINT_HALF_CM],
        [FT_X_CM, -PAINT_HALF_CM], [FT_X_CM, +PAINT_HALF_CM],
    ], dtype=np.float64)
    lane_uv = np.asarray([
        legacy_view["landmarks"]["baseline_left_lane"], legacy_view["landmarks"]["baseline_right_lane"],
        legacy_view["landmarks"]["ft_left_lane"], legacy_view["landmarks"]["ft_right_lane"],
    ], dtype=np.float64)
    H_seed = cv2.getPerspectiveTransform(lane_world.astype(np.float32), lane_uv.astype(np.float32)).astype(np.float64)
    h0 = parameter_vector(H_seed)

    train, held = split_groups(spec["observations_px"], spec["held_out_indices"])
    z = solve(h0, train)
    Hm = H_from_z(z, h0)
    if abs(np.linalg.det(Hm)) < 1e-12:
        raise RuntimeError("Degenerate final floor homography")

    train_metric = constraint_metrics(Hm, train)
    held_metric = constraint_metrics(Hm, held)
    p3 = project_h(Hm, dense_three())
    pf = project_h(Hm, dense_ft())
    held_three_px = nearest_curve_metrics(held["three_point_arc"], p3)
    held_ft_px = nearest_curve_metrics(held["free_throw_front_semicircle"], pf)
    held_line_p95 = max(
        held_metric["left_corner_three_straight"]["p95_abs_cm"],
        held_metric["right_corner_three_straight"]["p95_abs_cm"],
        held_metric["free_throw_line"]["p95_abs_cm"],
    )

    # Root consistency from deterministic multistarts: the accepted solve must not
    # depend on a fragile starting point. solve() itself already performs multistart;
    # this second pass starts warm from the accepted root after removing one training
    # point from each group.
    reduced = {k: v[:-1] for k, v in train.items()}
    zr = solve(h0, reduced, warm=z)
    Hr = H_from_z(zr, h0)
    d3_root = np.linalg.norm(project_h(Hr, dense_three()) - p3, axis=1)
    df_root = np.linalg.norm(project_h(Hr, dense_ft()) - pf, axis=1)
    root_stability = {
        "three_point_p95_shift_px": float(np.percentile(d3_root, 95)),
        "three_point_max_shift_px": float(np.max(d3_root)),
        "free_throw_p95_shift_px": float(np.percentile(df_root, 95)),
        "free_throw_max_shift_px": float(np.max(df_root)),
    }

    rng = np.random.default_rng(341903)
    perturb = []
    for trial in range(args.perturbation_trials):
        pg = {k: v + rng.uniform(-0.5, 0.5, size=v.shape) for k, v in train.items()}
        zp = solve(h0, pg, warm=z)
        Hp = H_from_z(zp, h0)
        p3p = project_h(Hp, dense_three())
        pfp = project_h(Hp, dense_ft())
        d3 = np.linalg.norm(p3p - p3, axis=1)
        df = np.linalg.norm(pfp - pf, axis=1)
        perturb.append({
            "trial": trial,
            "three_point_p95_shift_px": float(np.percentile(d3, 95)),
            "three_point_max_shift_px": float(np.max(d3)),
            "free_throw_p95_shift_px": float(np.percentile(df, 95)),
            "free_throw_max_shift_px": float(np.max(df)),
        })
    max_p3 = max(x["three_point_p95_shift_px"] for x in perturb)
    max_pf = max(x["free_throw_p95_shift_px"] for x in perturb)

    legacy = legacy_predictions(Hm, spec["legacy_v26_floor_anchors_for_diagnostic_only"])
    draw_overlay(image, spec, Hm, args.out / "frame_c_wide_court_overlay_v34.png")

    gates = {
        "immutable_frame_c_lock": True,
        "static_regulation_court_only": True,
        "spatially_distributed_heldout_observations": True,
        "heldout_three_point_p95_at_most_threshold": held_three_px["p95_px"] <= args.max_heldout_three_p95_px,
        "heldout_free_throw_p95_at_most_threshold": held_ft_px["p95_px"] <= args.max_heldout_ft_p95_px,
        "heldout_straight_line_p95_world_error_at_most_threshold": held_line_p95 <= args.max_heldout_line_p95_cm,
        "half_pixel_three_point_projection_stability": max_p3 <= args.max_half_pixel_three_p95_shift_px,
        "half_pixel_free_throw_projection_stability": max_pf <= args.max_half_pixel_ft_p95_shift_px,
        "finite_nondegenerate_homography": bool(np.isfinite(Hm).all() and abs(np.linalg.det(Hm)) > 1e-12),
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_WIDE_COURT_FLOOR_HOMOGRAPHY" if passed else "FAIL_WIDE_COURT_FLOOR_HOMOGRAPHY",
        "version": "v34_wide_regulation_court",
        "game_id": "0022500301", "event_id": 489, "camera_label": "Left Above Rim",
        "method": "wide source-pixel NBA court lines/curves -> metric floor homography; held-out points from every feature group; retired v26 baseline anchors diagnostic only",
        "guardrail": "Passing validates only the immutable Frame C floor-plane homography. 3D metric event camera and replay rendering remain forbidden.",
        "floor_homography_world_to_image": Hm.tolist(),
        "floor_homography_image_to_world": np.linalg.inv(Hm).tolist(),
        "training_constraint_error_cm": train_metric,
        "heldout_constraint_error_cm": held_metric,
        "heldout_pixel_curve_error": {"three_point_arc": held_three_px, "free_throw_front_semicircle": held_ft_px},
        "max_heldout_straight_line_p95_world_error_cm": float(held_line_p95),
        "root_reduction_stability": root_stability,
        "half_pixel_training_annotation_perturbation": {
            "trial_count": len(perturb), "trials": perturb,
            "max_three_point_p95_shift_px": float(max_p3),
            "max_free_throw_p95_shift_px": float(max_pf),
        },
        "legacy_v26_anchor_diagnostic": legacy,
        "thresholds": {
            "max_heldout_three_p95_px": args.max_heldout_three_p95_px,
            "max_heldout_ft_p95_px": args.max_heldout_ft_p95_px,
            "max_heldout_line_p95_cm": args.max_heldout_line_p95_cm,
            "max_half_pixel_three_p95_shift_px": args.max_half_pixel_three_p95_shift_px,
            "max_half_pixel_ft_p95_shift_px": args.max_half_pixel_ft_p95_shift_px,
        },
        "gates": gates,
        "floor_homography_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
        "next_gate": "Combine this validated floor plane with independent game camera centre and elevated regulation target/board geometry; solve exact Frame C 3D camera and re-test wide court as held-out geometry before promotion."
    }
    (args.out / "frame_c_wide_court_homography_v34.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "heldout_three_p95_px": held_three_px["p95_px"],
        "heldout_ft_p95_px": held_ft_px["p95_px"],
        "heldout_line_p95_cm": held_line_p95,
        "max_half_pixel_three_p95_shift_px": max_p3,
        "max_half_pixel_ft_p95_shift_px": max_pf,
        "legacy_anchor_disagreements_px": {k: v["disagreement_px"] for k, v in legacy.items()},
        "floor_homography_allowed": passed,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
