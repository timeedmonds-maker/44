from __future__ import annotations

"""v45: distortion-aware Broadcast floor proof on immutable Frame C.

v44 falsified the undistorted pinhole-floor hypothesis at the pre-registered
2 px standard.  This experiment preserves the exact same source frame,
regulation geometry, observations, train/held-out split, support-reduction test
and half-pixel annotation perturbation test.  The only added model capacity is
a four-coefficient Brown-Conrady lens distortion model (k1, k2, p1, p2)
centred at the fixed image centre.

A pass requires more than lower training residual.  Held-out geometry must
improve materially over the reproduced v44 baseline, every held-out feature
must be <=2 px p95, support reduction and half-pixel perturbations must remain
<=2 px p95, independent competitive roots must be functionally equivalent,
and the fitted distortion must remain physically mild and non-folding.

Even a pass authorizes only a distortion-aware Broadcast floor model.  It does
not authorize a physical 3D camera centre or replay rendering.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

W, H = v44.W, v44.H
GROUPS = v44.GROUPS
DIST_CENTER = np.asarray([W / 2.0, H / 2.0], dtype=np.float64)
DIST_SCALE = W / 2.0
# Fixed a priori before fitting.  With normalized radius ~1 at the horizontal
# edge these are deliberately broad for a broadcast zoom lens, but still keep
# the model in a physically mild regime.  A separate image-space displacement
# gate below is more interpretable and stricter in practice.
DIST_BOUNDS = np.asarray([0.08, 0.04, 0.01, 0.01], dtype=np.float64)


def json_safe(value):
    return v44.json_safe(value)


def _xy_norm(pixels: np.ndarray) -> np.ndarray:
    return (np.asarray(pixels, dtype=np.float64) - DIST_CENTER[None, :]) / DIST_SCALE


def _xy_pixels(xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy, dtype=np.float64) * DIST_SCALE + DIST_CENTER[None, :]


def distort_pixels(pixels_undistorted: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Forward Brown-Conrady map in fixed image-centred normalized coordinates."""
    xy = _xy_norm(pixels_undistorted)
    x, y = xy[:, 0], xy[:, 1]
    k1, k2, p1, p2 = np.asarray(d, dtype=np.float64)
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return _xy_pixels(np.column_stack([xd, yd]))


def undistort_pixels(pixels_distorted: np.ndarray, d: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Deterministic fixed-point inverse of the bounded Brown-Conrady map."""
    xyd = _xy_norm(pixels_distorted)
    xd, yd = xyd[:, 0], xyd[:, 1]
    x, y = xd.copy(), yd.copy()
    k1, k2, p1, p2 = np.asarray(d, dtype=np.float64)
    for _ in range(iterations):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        radial = np.where(np.abs(radial) < 1e-6, np.sign(radial) * 1e-6 + (radial == 0) * 1e-6, radial)
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    return _xy_pixels(np.column_stack([x, y]))


def project_model(Hm: np.ndarray, d: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    return distort_pixels(v44.project_h(Hm, world_xy), d)


def implicit_world_value(Hm: np.ndarray, d: np.ndarray, key: str, distorted_pixels: np.ndarray) -> np.ndarray:
    return v44.implicit_world_value(Hm, key, undistort_pixels(distorted_pixels, d))


def signed_pixel_residual(Hm: np.ndarray, d: np.ndarray, key: str, pixels: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float64)
    eps = 0.25
    f = implicit_world_value(Hm, d, key, pixels)
    gx = (
        implicit_world_value(Hm, d, key, pixels + np.asarray([eps, 0.0]))
        - implicit_world_value(Hm, d, key, pixels - np.asarray([eps, 0.0]))
    ) / (2.0 * eps)
    gy = (
        implicit_world_value(Hm, d, key, pixels + np.asarray([0.0, eps]))
        - implicit_world_value(Hm, d, key, pixels - np.asarray([0.0, eps]))
    ) / (2.0 * eps)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def unpack(q: np.ndarray, h0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    return v44.H_from_z(q[:8], h0), q[8:12]


def residual(q: np.ndarray, h0: np.ndarray, groups: dict) -> np.ndarray:
    Hm, d = unpack(q, h0)
    if not np.isfinite(Hm).all() or not np.isfinite(d).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        return np.full(sum(len(v) for v in groups.values()) + len(q), 1e6, dtype=np.float64)
    rows = [signed_pixel_residual(Hm, d, key, groups[key]) for key in GROUPS]
    # Tiny zero-centred priors break numerically equivalent roots without
    # meaningfully competing with pixel residuals.
    rows.append(np.asarray(q[:8], dtype=np.float64) * 0.001)
    rows.append((np.asarray(d, dtype=np.float64) / DIST_BOUNDS) * 0.01)
    return np.concatenate(rows)


def bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.r_[-np.ones(8), -DIST_BOUNDS]
    hi = np.r_[ np.ones(8),  DIST_BOUNDS]
    return lo, hi


def solve_multistart(h0: np.ndarray, groups: dict, *, warm: np.ndarray | None = None, return_roots: bool = False):
    seeds: list[np.ndarray] = []
    if warm is not None:
        seeds.append(np.asarray(warm, dtype=np.float64))
    # Reproduce the best v44 homography first, then permit distortion.
    z44 = v44.solve_multistart(h0, groups)
    seeds.append(np.r_[z44, np.zeros(4)])
    rng = np.random.default_rng(450903)
    for _ in range(5):
        seeds.append(np.r_[z44 + rng.uniform(-0.05, 0.05, size=8), rng.uniform(-0.35, 0.35, size=4) * DIST_BOUNDS])

    lo, hi = bounds()
    best, best_score, roots = None, float("inf"), []
    for x0 in seeds:
        x0 = np.minimum(np.maximum(np.asarray(x0, dtype=np.float64), lo + 1e-8), hi - 1e-8)
        try:
            fit = least_squares(
                lambda q: residual(q, h0, groups),
                x0,
                loss="soft_l1",
                f_scale=1.0,
                x_scale="jac",
                max_nfev=5000,
                bounds=(lo, hi),
            )
            q = np.asarray(fit.x, dtype=np.float64)
            Hm, d = unpack(q, h0)
            score = float(np.median(np.abs(np.concatenate([
                signed_pixel_residual(Hm, d, key, groups[key]) for key in GROUPS
            ]))))
            if np.isfinite(score):
                roots.append({"q": q, "median_abs_pixel_residual": score})
                if score < best_score:
                    best, best_score = q, score
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Broadcast v45 Brown solve failed from all deterministic roots")
    if return_roots:
        return best, roots
    return best


def solve_warm(h0: np.ndarray, groups: dict, warm: np.ndarray) -> np.ndarray:
    lo, hi = bounds()
    fit = least_squares(
        lambda q: residual(q, h0, groups),
        np.minimum(np.maximum(np.asarray(warm, dtype=np.float64), lo + 1e-8), hi - 1e-8),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=1800,
        bounds=(lo, hi),
    )
    q = np.asarray(fit.x, dtype=np.float64)
    Hm, d = unpack(q, h0)
    if not np.isfinite(Hm).all() or not np.isfinite(d).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        raise RuntimeError("Broadcast v45 warm solve was degenerate")
    return q


def pixel_metrics(Hm: np.ndarray, d: np.ndarray, groups: dict, dense: dict) -> dict:
    out = {}
    for key in GROUPS:
        pred = project_model(Hm, d, dense[key])
        obs = np.asarray(groups[key], dtype=np.float64)
        distances = np.sqrt(np.sum((obs[:, None, :] - pred[None, :, :]) ** 2, axis=2)).min(axis=1)
        out[key] = {
            "count": int(len(distances)),
            "rms_px": float(np.sqrt(np.mean(distances ** 2))),
            "median_px": float(np.median(distances)),
            "p95_px": float(np.percentile(distances, 95)),
            "max_px": float(np.max(distances)),
            "per_point_px": distances.tolist(),
        }
    return out


def pooled_p95(metrics: dict) -> float:
    vals = np.concatenate([np.asarray(metrics[k]["per_point_px"], dtype=np.float64) for k in GROUPS])
    return float(np.percentile(vals, 95))


def curve_shift(Ha: np.ndarray, da: np.ndarray, Hb: np.ndarray, db: np.ndarray, dense: dict) -> dict:
    out = {}
    for key in GROUPS:
        dpx = np.linalg.norm(project_model(Ha, da, dense[key]) - project_model(Hb, db, dense[key]), axis=1)
        out[key] = {"p95_px": float(np.percentile(dpx, 95)), "max_px": float(np.max(dpx))}
    return out


def distortion_diagnostics(d: np.ndarray) -> dict:
    xs = np.linspace(0.0, W - 1.0, 25)
    ys = np.linspace(0.0, H - 1.0, 15)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    warped = distort_pixels(grid, d)
    disp = np.linalg.norm(warped - grid, axis=1)

    eps = 0.5
    xp = distort_pixels(grid + np.asarray([eps, 0.0]), d)
    xm = distort_pixels(grid - np.asarray([eps, 0.0]), d)
    yp = distort_pixels(grid + np.asarray([0.0, eps]), d)
    ym = distort_pixels(grid - np.asarray([0.0, eps]), d)
    dx = (xp - xm) / (2.0 * eps)
    dy = (yp - ym) / (2.0 * eps)
    det = dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0]

    xy = _xy_norm(grid)
    r2 = np.sum(xy * xy, axis=1)
    k1, k2, p1, p2 = np.asarray(d, dtype=np.float64)
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    return {
        "fixed_center_px": DIST_CENTER.tolist(),
        "normalization_scale_px": float(DIST_SCALE),
        "coefficients": {"k1": float(k1), "k2": float(k2), "p1": float(p1), "p2": float(p2)},
        "coefficient_abs_fraction_of_bound": (np.abs(d) / DIST_BOUNDS).tolist(),
        "grid_max_displacement_px": float(np.max(disp)),
        "grid_p95_displacement_px": float(np.percentile(disp, 95)),
        "min_radial_scale": float(np.min(radial)),
        "max_radial_scale": float(np.max(radial)),
        "min_jacobian_det": float(np.min(det)),
        "max_jacobian_det": float(np.max(det)),
    }


def draw_overlay(image: np.ndarray, spec: dict, Hm: np.ndarray, d: np.ndarray, dense: dict, path: Path) -> None:
    out = image.copy()
    colors = {
        "three_point_arc": (0, 0, 255),
        "free_throw_front_semicircle": (0, 255, 0),
        "free_throw_line": (255, 255, 255),
        "lane_negative_y": (255, 255, 0),
        "lane_positive_y": (255, 255, 0),
    }
    for key in GROUPS:
        q = np.round(project_model(Hm, d, dense[key])).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        qq = q[ok]
        if len(qq) > 1:
            cv2.polylines(out, [qq.reshape(-1, 1, 2)], False, colors[key], 2, cv2.LINE_AA)
        held = set(spec["held_out_indices"][key])
        for i, p in enumerate(np.asarray(spec["observations_px"][key], dtype=int)):
            cv2.circle(out, tuple(p), 5 if i in held else 3, colors[key], 2 if i in held else 1, cv2.LINE_AA)
    cv2.imwrite(str(path), out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-heldout-p95-px", type=float, default=2.0)
    ap.add_argument("--max-root-p95-shift-px", type=float, default=2.0)
    ap.add_argument("--max-half-pixel-p95-shift-px", type=float, default=2.0)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    ap.add_argument("--min-heldout-improvement-px", type=float, default=0.25)
    ap.add_argument("--max-distortion-displacement-px", type=float, default=20.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.frame))
    if image is None or image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable Broadcast Frame C")
    spec = json.loads(args.observations.read_text(encoding="utf-8"))
    lock = spec["freeze_lock"]
    if spec["camera_label"] != "Broadcast" or lock["authority_camera"] != "Right Slash" or lock["chooser_option"] != "C":
        raise RuntimeError("Broadcast v45 observations are not bound to immutable chooser C")
    if abs(float(lock["right_slash_local_time"]) - 8.275733) > 5e-7 or int(lock["right_slash_decoded_frame_index"]) != 248:
        raise RuntimeError("Immutable authority timing changed")
    if abs(float(lock["broadcast_synchronized_time"]) - 9.194613) > 5e-7 or int(lock["broadcast_decoded_frame_index"]) != 276:
        raise RuntimeError("Immutable Broadcast Frame C changed")

    seed = spec["seed_only_correspondences"]
    H_seed, _ = cv2.findHomography(
        np.asarray(seed["world_cm"], dtype=np.float64),
        np.asarray(seed["image_px"], dtype=np.float64),
        method=0,
    )
    if H_seed is None:
        raise RuntimeError("Could not construct seed-only homography")
    h0 = v44.parameter_vector(H_seed)
    train, held = v44.split_groups(spec["observations_px"], spec["held_out_indices"])
    dense = v44.dense_features()

    # Reproduce v44 on this exact input as the comparison baseline.
    z44 = v44.solve_multistart(h0, train)
    H44 = v44.H_from_z(z44, h0)
    v44_train_metrics = v44.pixel_metrics(H44, train, dense)
    v44_held_metrics = v44.pixel_metrics(H44, held, dense)
    v44_max_held = max(row["p95_px"] for row in v44_held_metrics.values())
    v44_pooled_held = pooled_p95(v44_held_metrics)

    q, nominal_roots = solve_multistart(h0, train, warm=np.r_[z44, np.zeros(4)], return_roots=True)
    Hm, d = unpack(q, h0)
    train_metrics = pixel_metrics(Hm, d, train, dense)
    held_metrics = pixel_metrics(Hm, d, held, dense)
    max_held = max(row["p95_px"] for row in held_metrics.values())
    pooled_held = pooled_p95(held_metrics)

    reduced = {key: value[:-1] for key, value in train.items()}
    qr = solve_multistart(h0, reduced, warm=q)
    Hr, dr = unpack(qr, h0)
    root_shift = curve_shift(Hm, d, Hr, dr, dense)
    max_root_p95 = max(row["p95_px"] for row in root_shift.values())

    best_root_score = min(row["median_abs_pixel_residual"] for row in nominal_roots)
    competitive_roots = [row for row in nominal_roots if row["median_abs_pixel_residual"] <= best_root_score + 0.25]
    pairwise_root_rows = []
    for i in range(len(competitive_roots)):
        for j in range(i + 1, len(competitive_roots)):
            Ha, da = unpack(competitive_roots[i]["q"], h0)
            Hb, db = unpack(competitive_roots[j]["q"], h0)
            shifts = curve_shift(Ha, da, Hb, db, dense)
            pairwise_root_rows.append({
                "i": i,
                "j": j,
                "feature_shift": shifts,
                "max_p95_px": max(row["p95_px"] for row in shifts.values()),
            })
    max_pairwise = max((row["max_p95_px"] for row in pairwise_root_rows), default=float("inf"))

    rng = np.random.default_rng(451164)
    perturb_rows = []
    max_half = 0.0
    for trial in range(args.perturbation_trials):
        noisy = {
            key: np.asarray(value, dtype=np.float64) + rng.uniform(-0.5, 0.5, size=np.asarray(value).shape)
            for key, value in train.items()
        }
        try:
            qp = solve_warm(h0, noisy, q)
            Hp, dp = unpack(qp, h0)
            shifts = curve_shift(Hm, d, Hp, dp, dense)
            mx = max(row["p95_px"] for row in shifts.values())
        except Exception as exc:
            shifts = {key: {"p95_px": float("inf"), "max_px": float("inf")} for key in GROUPS}
            mx = float("inf")
        max_half = max(max_half, mx)
        perturb_rows.append({"trial": trial, "feature_shift": shifts, "max_p95_px": mx})

    distortion = distortion_diagnostics(d)
    max_bound_fraction = max(distortion["coefficient_abs_fraction_of_bound"])
    distortion_plausible = (
        max_bound_fraction < 0.95
        and distortion["grid_max_displacement_px"] <= args.max_distortion_displacement_px
        and distortion["min_radial_scale"] >= 0.80
        and distortion["max_radial_scale"] <= 1.20
        and distortion["min_jacobian_det"] >= 0.70
        and distortion["max_jacobian_det"] <= 1.30
    )

    heldout_improvement = v44_max_held - max_held
    pooled_improvement = v44_pooled_held - pooled_held
    gates = {
        "immutable_frame_c_lock": True,
        "native_960x540_source": True,
        "same_v44_observations_and_heldout_split": True,
        "four_coefficient_fixed_centre_brown_model_only": True,
        "at_least_three_competitive_multistart_roots": len(competitive_roots) >= 3,
        "competitive_multistart_roots_functionally_equivalent": max_pairwise <= 0.5,
        "every_heldout_feature_p95_at_most_two_px": max_held <= args.max_heldout_p95_px,
        "heldout_max_p95_improves_over_v44": heldout_improvement >= args.min_heldout_improvement_px,
        "pooled_heldout_p95_improves_over_v44": pooled_improvement > 0.0,
        "support_reduction_root_p95_at_most_threshold": max_root_p95 <= args.max_root_p95_shift_px,
        "half_pixel_annotation_p95_stability": max_half <= args.max_half_pixel_p95_shift_px,
        "distortion_parameters_physically_mild": distortion_plausible,
        "finite_nondegenerate_homography": bool(np.isfinite(Hm).all() and abs(float(np.linalg.det(Hm))) > 1e-12),
    }
    passed = all(gates.values())
    status = "PASS_BROADCAST_BROWN_FLOOR_V45" if passed else "FAIL_BROADCAST_BROWN_FLOOR_V45"

    draw_overlay(image, spec, Hm, d, dense, args.out / "broadcast_frame_c_floor_overlay_v45.png")
    result = {
        "schema_version": 1,
        "status": status,
        "game_id": spec["game_id"],
        "event_id": spec["event_id"],
        "camera_label": "Broadcast",
        "model": "floor homography plus fixed-centre Brown-Conrady k1,k2,p1,p2 distortion",
        "method": "same v44 source pixels, regulation geometry, observations and held-out split; distortion-aware signed pixel fit; support reduction; 64 half-pixel perturbations; explicit v44 held-out comparison",
        "floor_homography_world_to_undistorted_image": Hm,
        "distortion": distortion,
        "training_pixel_error": train_metrics,
        "heldout_pixel_error": held_metrics,
        "max_heldout_feature_p95_px": max_held,
        "pooled_heldout_p95_px": pooled_held,
        "v44_reproduced_baseline": {
            "floor_homography_world_to_image": H44,
            "training_pixel_error": v44_train_metrics,
            "heldout_pixel_error": v44_held_metrics,
            "max_heldout_feature_p95_px": v44_max_held,
            "pooled_heldout_p95_px": v44_pooled_held,
        },
        "heldout_improvement_over_v44": {
            "max_feature_p95_px": heldout_improvement,
            "pooled_p95_px": pooled_improvement,
        },
        "support_reduction_projection_shift": root_shift,
        "max_support_reduction_p95_shift_px": max_root_p95,
        "nominal_multistart": {
            "seed_count": len(nominal_roots),
            "competitive_score_margin_px": 0.25,
            "competitive_root_count": len(competitive_roots),
            "competitive_root_scores_px": [row["median_abs_pixel_residual"] for row in competitive_roots],
            "pairwise_projection_shift": pairwise_root_rows,
            "max_pairwise_p95_shift_px": max_pairwise,
            "max_allowed_pairwise_p95_shift_px": 0.5,
        },
        "half_pixel_training_annotation_perturbation": {
            "trial_count": args.perturbation_trials,
            "max_feature_p95_shift_px": max_half,
            "trials": perturb_rows,
        },
        "thresholds": {
            "max_heldout_p95_px": args.max_heldout_p95_px,
            "max_root_p95_shift_px": args.max_root_p95_shift_px,
            "max_half_pixel_p95_shift_px": args.max_half_pixel_p95_shift_px,
            "min_heldout_improvement_px": args.min_heldout_improvement_px,
            "max_distortion_displacement_px": args.max_distortion_displacement_px,
            "coefficient_bounds": {"k1": 0.08, "k2": 0.04, "p1": 0.01, "p2": 0.01},
            "min_radial_scale": 0.80,
            "max_radial_scale": 1.20,
            "min_jacobian_det": 0.70,
            "max_jacobian_det": 1.30,
        },
        "gates": gates,
        "permissions": {
            "broadcast_distortion_aware_floor_model_allowed": bool(passed),
            "broadcast_physical_camera_center_allowed": False,
            "broadcast_metric_event_camera_allowed": False,
            "replay_render_allowed": False,
        },
        "failure_policy": "Do not relax v44/v45 two-pixel held-out or stability gates. If Brown v45 fails, diagnose observation support/model family or reject Broadcast as the second metric camera rather than granting 3D permission.",
        "independent_baseline_policy": "Broadcast is distinct from Left Above Rim. Mobile Broadcast, Other Broadcast and Play by Play remain excluded as near-duplicate/crop-family feeds.",
    }
    (args.out / "broadcast_frame_c_floor_v45.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(json_safe({
        "status": status,
        "v44_max_heldout_feature_p95_px": v44_max_held,
        "v45_max_heldout_feature_p95_px": max_held,
        "heldout_max_improvement_px": heldout_improvement,
        "v44_pooled_heldout_p95_px": v44_pooled_held,
        "v45_pooled_heldout_p95_px": pooled_held,
        "max_support_reduction_p95_shift_px": max_root_p95,
        "competitive_root_count": len(competitive_roots),
        "max_competitive_pairwise_p95_shift_px": max_pairwise,
        "max_half_pixel_p95_shift_px": max_half,
        "distortion": distortion,
        "gates": gates,
        "permissions": result["permissions"],
    }), indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
