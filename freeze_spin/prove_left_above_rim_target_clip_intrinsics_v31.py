from __future__ import annotations

"""Learn the exact-event principal point from many non-freeze frames in the same clip.

The physical camera centre is fixed by the independently proved game registry.  NBA
broadcast processing can change crop/stabilization state between event clips, so v31
stops forcing one principal point across the entire game.  Instead, it estimates one
principal point for the exact Left Above Rim event clip while allowing rotation and
focal length to vary per frame.

The immutable Frame C itself is excluded from this prior.  Regulation landmarks are
transported from the trusted Frame C annotation to each non-freeze source frame only
when a held-out-validated static-scene homography passes.  Each transported frame
must also be individually compatible with the independently fixed physical centre
before entering the joint intrinsics fit.

Passing authorizes only a target-clip principal-point prior; not the Frame C pose and
not a replay render.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.audit_game_camera_registry_preflight_v1 import audit_pair
from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v27 import project_fixed, solve_fixed
from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks

W, H = 960, 540


def transform_array(points: np.ndarray, H_target_to_source: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    return cv2.perspectiveTransform(pts[:, None, :], H_target_to_source)[:, 0].astype(np.float64)


def project_metrics(p: np.ndarray, obj: np.ndarray, obs: np.ndarray, center: np.ndarray) -> dict:
    uv, cam, focal, R = project_fixed(p, obj, center)
    err = np.linalg.norm(uv - obs, axis=1)
    return {
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "focal_px": float(focal),
        "principal_point_px": [float(p[4]), float(p[5])],
        "positive_depth": bool(float(np.min(cam[:, 2])) > 20.0),
        "R": R,
    }


def initialize_frame(frame: dict, center: np.ndarray, world: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    view = frame["view"]
    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError(f"Direct metric initialization implausible for {frame['frame_id']}")
    d = np.asarray(direct[2], dtype=np.float64)
    seed = np.r_[d[:3], d[6], d[7], d[8]]
    p = solve_fixed(frame["obj"], frame["obs"], center, view, seed)
    return p, project_metrics(p, frame["obj"], frame["obs"], center)


def joint_solve(frames: list[dict], center: np.ndarray, pp_seed: np.ndarray, *, warm: np.ndarray | None = None) -> np.ndarray:
    # [shared cx,cy, frame0 rvec(3),logf, ...]
    if warm is None or len(warm) != 2 + 4 * len(frames):
        chunks = [np.r_[f["individual_params"][:3], f["individual_params"][3]] for f in frames]
        x0 = np.r_[pp_seed, *chunks]
    else:
        x0 = np.asarray(warm, dtype=np.float64).copy()

    def residual(x: np.ndarray) -> np.ndarray:
        pp = x[:2]
        out = []
        for i, fr in enumerate(frames):
            off = 2 + 4 * i
            p = np.r_[x[off:off + 3], x[off + 3], pp]
            uv, cam, focal, _ = project_fixed(p, fr["obj"], center)
            out.append((uv - fr["obs"]).ravel())
            out.append(np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0)
            fp = float(fr["view"].get("focal_prior_px", 900.0))
            fs = float(fr["view"].get("focal_prior_sigma_log", 1.8))
            out.append(np.asarray([(math.log(float(focal)) - math.log(fp)) / fs]))
        out.append(np.asarray([(pp[0] - W / 2.0) / 160.0, (pp[1] - H / 2.0) / 160.0]))
        return np.concatenate(out)

    lower = [W / 2.0 - 250.0, H / 2.0 - 250.0]
    upper = [W / 2.0 + 250.0, H / 2.0 + 250.0]
    for _ in frames:
        lower += [-np.inf, -np.inf, -np.inf, math.log(150.0)]
        upper += [np.inf, np.inf, np.inf, math.log(4000.0)]
    opt = least_squares(
        residual, x0,
        bounds=(np.asarray(lower), np.asarray(upper)),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=25000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def joint_frame_metrics(x: np.ndarray, idx: int, fr: dict, center: np.ndarray) -> dict:
    pp = x[:2]
    off = 2 + 4 * idx
    p = np.r_[x[off:off + 3], x[off + 3], pp]
    m = project_metrics(p, fr["obj"], fr["obs"], center)
    return {
        "frame_id": fr["frame_id"],
        "decoded_time_seconds": float(fr["decoded_time_seconds"]),
        "relative_to_freeze_seconds": float(fr["relative_to_freeze_seconds"]),
        "rmse_px": m["rmse_px"], "p95_px": m["p95_px"], "max_px": m["max_px"],
        "focal_px": m["focal_px"], "positive_depth": m["positive_depth"],
    }


def subset_warm(full_x: np.ndarray, full_frames: list[dict], subset: list[dict]) -> np.ndarray:
    by_id = {f["frame_id"]: i for i, f in enumerate(full_frames)}
    parts = [full_x[:2]]
    for fr in subset:
        i = by_id[fr["frame_id"]]
        off = 2 + 4 * i
        parts.append(full_x[off:off + 4])
    return np.r_[*parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--target-landmarks", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-accepted-frames", type=int, default=6)
    ap.add_argument("--max-individual-rmse-px", type=float, default=1.5)
    ap.add_argument("--max-individual-p95-px", type=float, default=2.5)
    ap.add_argument("--max-joint-frame-rmse-px", type=float, default=1.5)
    ap.add_argument("--max-loo-frame-pp-shift-px", type=float, default=5.0)
    ap.add_argument("--max-temporal-block-pp-shift-px", type=float, default=8.0)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    ap.add_argument("--max-half-pixel-pp-shift-px", type=float, default=5.0)
    ap.add_argument("--min-focal-range-fraction", type=float, default=0.04)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    target_image = cv2.imread(str(args.target_frame))
    if target_image is None or target_image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 immutable target frame")

    spec = json.loads(args.target_landmarks.read_text(encoding="utf-8"))
    freeze = spec.get("freeze_lock", {})
    if freeze.get("authority_camera") != "Right Slash" or freeze.get("chooser_option") != "C":
        raise RuntimeError("Landmark spec not bound to immutable Frame C")
    view0 = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if view0 is None:
        raise RuntimeError("Missing target camera metric view")
    if any(t in json.dumps(view0).lower() for t in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Dynamic metric anchors forbidden")

    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    camreg = reg["cameras"][args.camera_label]
    if not camreg["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Physical camera centre prior not accepted")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_resolution") != [W, H]:
        raise RuntimeError("Target-clip sample manifest is not native 960x540")
    freeze_time = float(manifest["immutable_freeze_time_seconds"])
    if abs(freeze_time - 8.653093) > 5e-7:
        raise RuntimeError("Left Above Rim immutable synchronized freeze time changed")
    meta = {r["file"]: r for r in manifest["samples"]}

    world = world_landmarks()
    names = list(view0["landmarks"])
    target_obs = np.asarray([view0["landmarks"][n] for n in names], dtype=np.float64)
    obj = np.asarray([world[n] for n in names], dtype=np.float64)

    candidates = []
    transfer_rows = []
    for src in sorted(args.samples.glob("Left_Above_Rim_target_event__*.png")):
        sm = meta.get(src.name)
        if sm is None:
            continue
        if abs(float(sm["decoded_time_seconds"]) - freeze_time) < float(manifest["freeze_exclusion_radius_seconds"]) - 1e-6:
            raise RuntimeError("A calibration sample violates immutable freeze exclusion")
        tr = audit_pair(src, args.target_frame)
        row = {
            "source": str(src), "decoded_time_seconds": sm["decoded_time_seconds"],
            "relative_to_freeze_seconds": sm["relative_to_freeze_seconds"],
            "transport_status": tr.get("status"), "transport_pass": bool(tr.get("pass")),
            "training_inliers": int(tr.get("training_inliers", 0)),
            "training_error": tr.get("training_error"), "withheld_error": tr.get("withheld_error"),
            "transport_gates": tr.get("gates"),
        }
        transfer_rows.append(row)
        if not tr.get("pass"):
            continue
        Hs2t = np.asarray(tr["H_source_to_target"], dtype=np.float64)
        Ht2s = np.linalg.inv(Hs2t)
        obs = transform_array(target_obs, Ht2s)
        landmarks = {n: [float(p[0]), float(p[1])] for n, p in zip(names, obs)}
        view = {
            "label": f"{args.camera_label} target clip {src.stem}",
            "principal_point_prior_sigma_px": view0.get("principal_point_prior_sigma_px", 160.0),
            "principal_point_bound_px": view0.get("principal_point_bound_px", 350.0),
            "focal_prior_px": view0.get("focal_prior_px", 900.0),
            "focal_prior_sigma_log": view0.get("focal_prior_sigma_log", 1.8),
            "landmarks": landmarks,
        }
        fr = {
            "frame_id": src.stem, "source": src,
            "decoded_time_seconds": float(sm["decoded_time_seconds"]),
            "relative_to_freeze_seconds": float(sm["relative_to_freeze_seconds"]),
            "H_target_to_source": Ht2s, "view": view, "obj": obj, "obs": obs,
        }
        try:
            ip, im = initialize_frame(fr, center, world)
        except Exception as exc:
            row["individual_metric_status"] = "solver_failed"
            row["individual_metric_error"] = repr(exc)
            continue
        row["individual_metric"] = {k: v for k, v in im.items() if k != "R"}
        good = (
            im["rmse_px"] <= args.max_individual_rmse_px
            and im["p95_px"] <= args.max_individual_p95_px
            and im["positive_depth"]
        )
        row["individual_metric_status"] = "accepted" if good else "rejected"
        if not good:
            continue
        fr["individual_params"] = np.asarray(ip, dtype=np.float64)
        fr["individual_metrics"] = im
        candidates.append(fr)

    if len(candidates) < args.min_accepted_frames:
        raise RuntimeError(f"Only {len(candidates)} individually valid target-clip frames")
    before = sum(f["relative_to_freeze_seconds"] < 0 for f in candidates)
    after = sum(f["relative_to_freeze_seconds"] > 0 for f in candidates)
    if before < 2 or after < 2:
        raise RuntimeError(f"Insufficient temporal support around freeze: before={before}, after={after}")

    pp_seed = np.median(np.asarray([f["individual_params"][4:6] for f in candidates]), axis=0)
    full = joint_solve(candidates, center, pp_seed)
    pp = full[:2]
    frame_results = [joint_frame_metrics(full, i, f, center) for i, f in enumerate(candidates)]
    max_joint_rmse = max(r["rmse_px"] for r in frame_results)
    focals = np.asarray([r["focal_px"] for r in frame_results], dtype=np.float64)
    focal_range_fraction = float((np.max(focals) - np.min(focals)) / np.median(focals))

    loo = []
    for hold in range(len(candidates)):
        subset = [f for i, f in enumerate(candidates) if i != hold]
        x = joint_solve(subset, center, pp, warm=subset_warm(full, candidates, subset))
        loo.append({
            "held_out_frame": candidates[hold]["frame_id"],
            "held_out_time_seconds": candidates[hold]["decoded_time_seconds"],
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    # Temporal block holdouts guard against many near-duplicate frames creating false confidence.
    ordered = sorted(candidates, key=lambda f: f["decoded_time_seconds"])
    blocks = np.array_split(np.arange(len(ordered)), 3)
    block_rows = []
    for bi, ids in enumerate(blocks):
        held = {ordered[int(i)]["frame_id"] for i in ids}
        subset = [f for f in candidates if f["frame_id"] not in held]
        if len(subset) < 3:
            continue
        x = joint_solve(subset, center, pp, warm=subset_warm(full, candidates, subset))
        block_rows.append({
            "block": bi, "held_out_frames": sorted(held),
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    rng = np.random.default_rng(310902)
    perturb = []
    for trial in range(args.perturbation_trials):
        # Coherent +/-0.5 px uncertainty on the manually trusted target landmarks,
        # propagated through every independently fitted target->source homography.
        target_pert = target_obs + rng.uniform(-0.5, 0.5, size=target_obs.shape)
        pert_frames = []
        for fr in candidates:
            pfr = dict(fr)
            obs = transform_array(target_pert, fr["H_target_to_source"])
            # Small independent transport residual captures subpixel homography uncertainty.
            obs = obs + rng.uniform(-0.20, 0.20, size=obs.shape)
            pfr["obs"] = obs
            pert_frames.append(pfr)
        x = joint_solve(pert_frames, center, pp, warm=full)
        perturb.append({
            "trial": trial,
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    max_loo = max(r["shift_px"] for r in loo)
    max_block = max(r["shift_px"] for r in block_rows) if block_rows else float("inf")
    max_pert = max(r["shift_px"] for r in perturb)
    gates = {
        "fixed_physical_center_prior_accepted": True,
        "immutable_frame_c_excluded_from_intrinsics_samples": True,
        "accepted_frame_count_at_least_minimum": len(candidates) >= args.min_accepted_frames,
        "at_least_two_frames_before_and_after_freeze": before >= 2 and after >= 2,
        "all_joint_frame_rmse_at_most_threshold": max_joint_rmse <= args.max_joint_frame_rmse_px,
        "all_joint_frames_positive_depth": all(r["positive_depth"] for r in frame_results),
        "focal_diversity_at_least_minimum": focal_range_fraction >= args.min_focal_range_fraction,
        "leave_one_frame_out_principal_point_stability": max_loo <= args.max_loo_frame_pp_shift_px,
        "leave_temporal_block_out_principal_point_stability": max_block <= args.max_temporal_block_pp_shift_px,
        "coherent_half_pixel_landmark_principal_point_stability": max_pert <= args.max_half_pixel_pp_shift_px,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_TARGET_CLIP_INTRINSICS_PRIOR" if passed else "FAIL_TARGET_CLIP_INTRINSICS_PRIOR",
        "version": "v31_target_clip_intrinsics",
        "game_id": reg["game_id"], "event_id": 489, "camera_label": args.camera_label,
        "method": "game-level fixed physical centre + many non-freeze frames from exact event clip + per-frame rotation/focal + one clip-level principal point",
        "guardrail": "Passing authorizes only the exact target-clip principal-point prior. Frame C metric pose and replay remain unpromoted.",
        "physical_camera_center_prior_cm": [float(x) for x in center],
        "shared_principal_point_px": [float(pp[0]), float(pp[1])],
        "accepted_frame_count": len(candidates), "accepted_before_freeze": before, "accepted_after_freeze": after,
        "frame_results": frame_results, "candidate_audit": transfer_rows,
        "individual_principal_points_px": [f["individual_metrics"]["principal_point_px"] for f in candidates],
        "individual_focals_px": [f["individual_metrics"]["focal_px"] for f in candidates],
        "joint_focal_range_fraction": focal_range_fraction,
        "leave_one_frame_out": loo, "leave_temporal_block_out": block_rows,
        "coherent_half_pixel_landmark_perturbation": perturb,
        "max_joint_frame_rmse_px": max_joint_rmse,
        "max_leave_one_frame_out_principal_point_shift_px": max_loo,
        "max_leave_temporal_block_out_principal_point_shift_px": max_block,
        "max_half_pixel_principal_point_shift_px": max_pert,
        "thresholds": {
            "max_individual_rmse_px": args.max_individual_rmse_px,
            "max_individual_p95_px": args.max_individual_p95_px,
            "max_joint_frame_rmse_px": args.max_joint_frame_rmse_px,
            "max_leave_one_frame_out_principal_point_shift_px": args.max_loo_frame_pp_shift_px,
            "max_leave_temporal_block_out_principal_point_shift_px": args.max_temporal_block_pp_shift_px,
            "max_half_pixel_principal_point_shift_px": args.max_half_pixel_pp_shift_px,
            "min_focal_range_fraction": args.min_focal_range_fraction,
        },
        "gates": gates,
        "principal_point_prior_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
    }
    (args.out / "left_above_rim_target_clip_intrinsics_v31.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "accepted_frames": len(candidates),
        "before": before, "after": after,
        "shared_principal_point_px": payload["shared_principal_point_px"],
        "max_joint_frame_rmse_px": max_joint_rmse,
        "focal_range_fraction": focal_range_fraction,
        "max_loo_frame_pp_shift_px": max_loo,
        "max_temporal_block_pp_shift_px": max_block,
        "max_half_pixel_pp_shift_px": max_pert,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
