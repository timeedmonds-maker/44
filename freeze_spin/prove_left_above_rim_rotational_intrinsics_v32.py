from __future__ import annotations

"""Exact-clip rotational self-calibration for Left Above Rim.

This stage deliberately does NOT use NBA court/basket landmarks. The already-proved
fixed optical centre means static scene points at every depth are related between
frames by a pure-rotation/zoom homography. We therefore learn the exact Frame C
principal point and focal length directly from robust static-scene correspondences
between non-freeze frames and the immutable Frame C image.

Passing authorizes only an intrinsics prior for the exact target frame. Metric world
pose and replay rendering remain forbidden until separate regulation-geometry gates
pass.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.audit_game_camera_registry_preflight_v1 import sift_points, transfer_err

W, H = 960, 540


def K_of(f: float, pp: np.ndarray) -> np.ndarray:
    return np.asarray([[f, 0.0, pp[0]], [0.0, f, pp[1]], [0.0, 0.0, 1.0]], dtype=np.float64)


def action_core(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return (x > 0.20 * W) & (x < 0.80 * W) & (y > 0.48 * H) & (y < 0.98 * H)


def spatial_subsample(p: np.ndarray, q: np.ndarray, limit: int = 320) -> tuple[np.ndarray, np.ndarray]:
    if len(p) <= limit:
        return p.astype(np.float64), q.astype(np.float64)
    # Deterministic spatial ordering followed by even selection prevents dense texture
    # regions from dominating the camera fit.
    key = p[:, 0] + 17.0 * p[:, 1]
    order = np.argsort(key)
    ids = np.linspace(0, len(order) - 1, limit).round().astype(int)
    pick = order[ids]
    return p[pick].astype(np.float64), q[pick].astype(np.float64)


def build_pair(source: Path, target: Path, meta: dict) -> dict | None:
    a = cv2.imread(str(source))
    b = cv2.imread(str(target))
    if a is None or b is None or a.shape[:2] != (H, W) or b.shape[:2] != (H, W):
        return None
    p, q = sift_points(a, b)
    if len(p) < 30:
        return None
    xa, ya = p[:, 0], p[:, 1]
    xb, yb = q[:, 0], q[:, 1]
    train_geom = (ya < 0.46 * H) | (xa < 0.14 * W) | (xa > 0.86 * W)
    train_geom &= (yb < 0.46 * H) | (xb < 0.14 * W) | (xb > 0.86 * W)
    training = train_geom & ~action_core(p) & ~action_core(q)
    withheld = ~training & ~action_core(p) & ~action_core(q)
    if int(training.sum()) < 24:
        return None
    Hm, mask = cv2.findHomography(p[training], q[training], cv2.RANSAC, 1.5, maxIters=30000, confidence=0.999)
    if Hm is None or mask is None:
        return None
    inlier = mask.ravel().astype(bool)
    pin = p[training][inlier]
    qin = q[training][inlier]
    tr = transfer_err(Hm, pin, qin)
    wh = transfer_err(Hm, p[withheld], q[withheld])
    if len(pin) < 24 or len(wh) < 10:
        return None
    if float(np.percentile(tr, 95)) > 1.5 or float(np.median(wh)) > 2.5 or float(np.percentile(wh, 90)) > 4.0:
        return None
    pin, qin = spatial_subsample(pin, qin)
    return {
        "frame_id": source.stem,
        "source": source,
        "decoded_time_seconds": float(meta["decoded_time_seconds"]),
        "relative_to_freeze_seconds": float(meta["relative_to_freeze_seconds"]),
        "p": pin,
        "q": qin,
        "H": np.asarray(Hm, dtype=np.float64),
        "raw_inliers": int(inlier.sum()),
        "fit_points": int(len(pin)),
        "homography_training_p95_px": float(np.percentile(tr, 95)),
        "homography_withheld_median_px": float(np.median(wh)),
        "homography_withheld_p90_px": float(np.percentile(wh, 90)),
    }


def nearest_rotation(Hm: np.ndarray, pp: np.ndarray, ft: float, fs: float) -> np.ndarray:
    M = np.linalg.inv(K_of(ft, pp)) @ Hm @ K_of(fs, pp)
    det = float(np.linalg.det(M))
    if abs(det) > 1e-12:
        M = M / np.cbrt(abs(det))
        if det < 0:
            M = -M
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    rv, _ = cv2.Rodrigues(R)
    return rv.ravel()


def project_pair(pp: np.ndarray, ft: float, fs: float, rvec: np.ndarray, p: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    Hm = K_of(ft, pp) @ R @ np.linalg.inv(K_of(fs, pp))
    ph = np.column_stack([p, np.ones(len(p))])
    qh = (Hm @ ph.T).T
    return qh[:, :2] / qh[:, 2:3]


def solve(frames: list[dict], seed_pp: np.ndarray, seed_f: float, warm: np.ndarray | None = None) -> np.ndarray:
    # [cx, cy, log(f_target), frame0 log(f_source), rvec(3), ...]
    if warm is None or len(warm) != 3 + 4 * len(frames):
        parts = [np.asarray(seed_pp, dtype=np.float64), [math.log(seed_f)]]
        for fr in frames:
            rv = nearest_rotation(fr["H"], np.asarray(seed_pp, dtype=np.float64), seed_f, seed_f)
            parts.append(np.r_[math.log(seed_f), rv])
        x0 = np.r_[*parts]
    else:
        x0 = np.asarray(warm, dtype=np.float64).copy()

    def residual(x: np.ndarray) -> np.ndarray:
        pp = x[:2]
        ft = float(np.exp(x[2]))
        out = []
        for i, fr in enumerate(frames):
            off = 3 + 4 * i
            fs = float(np.exp(x[off]))
            pred = project_pair(pp, ft, fs, x[off + 1:off + 4], fr["p"])
            # Equalize total influence per frame while retaining subpixel geometry.
            out.append(((pred - fr["q"]) / math.sqrt(max(len(fr["p"]), 1) / 120.0)).ravel())
        # Very broad physical regularization prevents non-physical projective roots;
        # it is intentionally far weaker than the image evidence.
        out.append(np.asarray([(pp[0] - W / 2.0) / 350.0, (pp[1] - H / 2.0) / 350.0]))
        out.append(np.asarray([(math.log(ft) - math.log(500.0)) / 1.6]))
        return np.concatenate(out)

    lower = [0.0, 0.0, math.log(150.0)]
    upper = [float(W), float(H), math.log(4000.0)]
    for _ in frames:
        lower += [math.log(150.0), -np.inf, -np.inf, -np.inf]
        upper += [math.log(4000.0), np.inf, np.inf, np.inf]
    opt = least_squares(
        residual, x0, bounds=(np.asarray(lower), np.asarray(upper)),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=30000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def frame_metrics(x: np.ndarray, frames: list[dict]) -> list[dict]:
    pp = x[:2]
    ft = float(np.exp(x[2]))
    rows = []
    for i, fr in enumerate(frames):
        off = 3 + 4 * i
        fs = float(np.exp(x[off]))
        pred = project_pair(pp, ft, fs, x[off + 1:off + 4], fr["p"])
        err = np.linalg.norm(pred - fr["q"], axis=1)
        rows.append({
            "frame_id": fr["frame_id"],
            "decoded_time_seconds": fr["decoded_time_seconds"],
            "relative_to_freeze_seconds": fr["relative_to_freeze_seconds"],
            "fit_points": fr["fit_points"],
            "source_focal_px": fs,
            "rmse_px": float(np.sqrt(np.mean(err ** 2))),
            "median_px": float(np.median(err)),
            "p95_px": float(np.percentile(err, 95)),
            "max_px": float(np.max(err)),
        })
    return rows


def subset_warm(full_x: np.ndarray, full_frames: list[dict], subset: list[dict]) -> np.ndarray:
    by_id = {f["frame_id"]: i for i, f in enumerate(full_frames)}
    parts = [full_x[:3]]
    for fr in subset:
        i = by_id[fr["frame_id"]]
        off = 3 + 4 * i
        parts.append(full_x[off:off + 4])
    return np.r_[*parts]


def multistart(frames: list[dict]) -> np.ndarray:
    seeds = [
        (np.asarray([480.0, 270.0]), 350.0),
        (np.asarray([480.0, 360.0]), 450.0),
        (np.asarray([480.0, 420.0]), 450.0),
        (np.asarray([440.0, 390.0]), 600.0),
        (np.asarray([520.0, 390.0]), 600.0),
        (np.asarray([480.0, 360.0]), 900.0),
    ]
    best = None
    best_score = float("inf")
    for pp, f in seeds:
        try:
            x = solve(frames, pp, f)
            rows = frame_metrics(x, frames)
            score = float(np.median([r["p95_px"] for r in rows]))
            if np.isfinite(score) and score < best_score:
                best_score, best = score, x
        except Exception:
            continue
    if best is None:
        raise RuntimeError("Rotational self-calibration failed from all multistart roots")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-accepted-frames", type=int, default=6)
    ap.add_argument("--max-frame-rmse-px", type=float, default=1.5)
    ap.add_argument("--max-frame-p95-px", type=float, default=2.5)
    ap.add_argument("--max-loo-pp-shift-px", type=float, default=5.0)
    ap.add_argument("--max-block-pp-shift-px", type=float, default=8.0)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    ap.add_argument("--max-half-pixel-pp-shift-px", type=float, default=5.0)
    ap.add_argument("--max-half-pixel-target-focal-fraction", type=float, default=0.05)
    ap.add_argument("--min-source-focal-range-fraction", type=float, default=0.04)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    target = cv2.imread(str(args.target_frame))
    if target is None or target.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable target frame")
    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    cam = reg["cameras"][args.camera_label]
    if not cam["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Fixed optical-centre evidence is required before rotational self-calibration")

    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_resolution") != [W, H]:
        raise RuntimeError("Sample manifest is not native 960x540")
    freeze_time = float(manifest["immutable_freeze_time_seconds"])
    if abs(freeze_time - 8.653093) > 5e-7:
        raise RuntimeError("Immutable Left Above Rim Frame C time changed")
    meta = {r["file"]: r for r in manifest["samples"]}

    frames = []
    audit = []
    for src in sorted(args.samples.glob("Left_Above_Rim_target_event__*.png")):
        sm = meta.get(src.name)
        if sm is None:
            continue
        if abs(float(sm["decoded_time_seconds"]) - freeze_time) < float(manifest["freeze_exclusion_radius_seconds"]) - 1e-6:
            raise RuntimeError("Calibration sample violates immutable freeze exclusion")
        fr = build_pair(src, args.target_frame, sm)
        if fr is None:
            audit.append({"frame_id": src.stem, "accepted": False})
            continue
        frames.append(fr)
        audit.append({
            "frame_id": fr["frame_id"], "accepted": True, "raw_inliers": fr["raw_inliers"],
            "fit_points": fr["fit_points"], "relative_to_freeze_seconds": fr["relative_to_freeze_seconds"],
            "homography_training_p95_px": fr["homography_training_p95_px"],
            "homography_withheld_median_px": fr["homography_withheld_median_px"],
            "homography_withheld_p90_px": fr["homography_withheld_p90_px"],
        })

    if len(frames) < args.min_accepted_frames:
        raise RuntimeError(f"Only {len(frames)} static-scene frames passed correspondence gates")
    before = sum(f["relative_to_freeze_seconds"] < 0 for f in frames)
    after = sum(f["relative_to_freeze_seconds"] > 0 for f in frames)
    if before < 2 or after < 2:
        raise RuntimeError(f"Insufficient temporal support: before={before}, after={after}")

    full = multistart(frames)
    pp = full[:2]
    ft = float(np.exp(full[2]))
    rows = frame_metrics(full, frames)
    max_rmse = max(r["rmse_px"] for r in rows)
    max_p95 = max(r["p95_px"] for r in rows)
    sf = np.asarray([r["source_focal_px"] for r in rows], dtype=np.float64)
    source_focal_range_fraction = float((sf.max() - sf.min()) / np.median(sf))

    loo = []
    for hold in range(len(frames)):
        subset = [f for i, f in enumerate(frames) if i != hold]
        x = solve(subset, pp, ft, warm=subset_warm(full, frames, subset))
        loo.append({
            "held_out_frame": frames[hold]["frame_id"],
            "principal_point_px": [float(x[0]), float(x[1])],
            "target_focal_px": float(np.exp(x[2])),
            "pp_shift_px": float(np.linalg.norm(x[:2] - pp)),
            "target_focal_fraction_delta": abs(float(np.exp(x[2])) - ft) / ft,
        })

    ordered = sorted(frames, key=lambda f: f["decoded_time_seconds"])
    blocks = np.array_split(np.arange(len(ordered)), 3)
    block_rows = []
    for bi, ids in enumerate(blocks):
        held = {ordered[int(i)]["frame_id"] for i in ids}
        subset = [f for f in frames if f["frame_id"] not in held]
        if len(subset) < 3:
            continue
        x = solve(subset, pp, ft, warm=subset_warm(full, frames, subset))
        block_rows.append({
            "block": bi, "held_out_frames": sorted(held),
            "principal_point_px": [float(x[0]), float(x[1])],
            "target_focal_px": float(np.exp(x[2])),
            "pp_shift_px": float(np.linalg.norm(x[:2] - pp)),
            "target_focal_fraction_delta": abs(float(np.exp(x[2])) - ft) / ft,
        })

    rng = np.random.default_rng(320902)
    perturb = []
    for trial in range(args.perturbation_trials):
        pframes = []
        for fr in frames:
            pf = dict(fr)
            pf["p"] = fr["p"] + rng.uniform(-0.5, 0.5, size=fr["p"].shape)
            pf["q"] = fr["q"] + rng.uniform(-0.5, 0.5, size=fr["q"].shape)
            pframes.append(pf)
        x = solve(pframes, pp, ft, warm=full)
        perturb.append({
            "trial": trial,
            "principal_point_px": [float(x[0]), float(x[1])],
            "target_focal_px": float(np.exp(x[2])),
            "pp_shift_px": float(np.linalg.norm(x[:2] - pp)),
            "target_focal_fraction_delta": abs(float(np.exp(x[2])) - ft) / ft,
        })

    max_loo_pp = max(r["pp_shift_px"] for r in loo)
    max_block_pp = max(r["pp_shift_px"] for r in block_rows) if block_rows else float("inf")
    max_pert_pp = max(r["pp_shift_px"] for r in perturb)
    max_pert_f = max(r["target_focal_fraction_delta"] for r in perturb)
    gates = {
        "fixed_optical_center_prior_accepted": True,
        "nba_metric_landmarks_not_used_for_intrinsics": True,
        "immutable_frame_c_used_only_as_static_image_target": True,
        "accepted_frame_count_at_least_minimum": len(frames) >= args.min_accepted_frames,
        "at_least_two_frames_before_and_after_freeze": before >= 2 and after >= 2,
        "all_rotational_model_frame_rmse_at_most_threshold": max_rmse <= args.max_frame_rmse_px,
        "all_rotational_model_frame_p95_at_most_threshold": max_p95 <= args.max_frame_p95_px,
        "source_focal_diversity_at_least_minimum": source_focal_range_fraction >= args.min_source_focal_range_fraction,
        "leave_one_frame_out_principal_point_stability": max_loo_pp <= args.max_loo_pp_shift_px,
        "leave_temporal_block_out_principal_point_stability": max_block_pp <= args.max_block_pp_shift_px,
        "half_pixel_static_correspondence_principal_point_stability": max_pert_pp <= args.max_half_pixel_pp_shift_px,
        "half_pixel_static_correspondence_target_focal_stability": max_pert_f <= args.max_half_pixel_target_focal_fraction,
        "target_focal_physical_range": 150.0 < ft < 4000.0,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_ROTATIONAL_INTRINSICS_PRIOR" if passed else "FAIL_ROTATIONAL_INTRINSICS_PRIOR",
        "version": "v32_rotational_self_calibration",
        "game_id": reg["game_id"], "event_id": 489, "camera_label": args.camera_label,
        "method": "fixed-centre pure-rotation/zoom self-calibration from real static-scene SIFT correspondences in exact target clip; no NBA metric landmark input",
        "guardrail": "Passing authorizes only exact Frame C principal point and focal priors. Metric world pose and replay rendering remain unproven.",
        "target_principal_point_px": [float(pp[0]), float(pp[1])],
        "target_focal_px": ft,
        "accepted_frame_count": len(frames), "accepted_before_freeze": before, "accepted_after_freeze": after,
        "frame_results": rows, "correspondence_audit": audit,
        "source_focal_range_fraction": source_focal_range_fraction,
        "leave_one_frame_out": loo, "leave_temporal_block_out": block_rows,
        "half_pixel_static_correspondence_perturbation": perturb,
        "max_frame_rmse_px": max_rmse, "max_frame_p95_px": max_p95,
        "max_leave_one_frame_out_pp_shift_px": max_loo_pp,
        "max_leave_temporal_block_out_pp_shift_px": max_block_pp,
        "max_half_pixel_pp_shift_px": max_pert_pp,
        "max_half_pixel_target_focal_fraction_delta": max_pert_f,
        "gates": gates,
        "principal_point_prior_allowed": passed,
        "target_focal_prior_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
    }
    out = args.out / "left_above_rim_rotational_intrinsics_v32.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "frames": len(frames),
        "target_principal_point_px": payload["target_principal_point_px"],
        "target_focal_px": ft, "max_frame_rmse_px": max_rmse, "max_frame_p95_px": max_p95,
        "source_focal_range_fraction": source_focal_range_fraction,
        "max_loo_pp_shift_px": max_loo_pp, "max_block_pp_shift_px": max_block_pp,
        "max_half_pixel_pp_shift_px": max_pert_pp,
        "max_half_pixel_target_focal_fraction_delta": max_pert_f,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
