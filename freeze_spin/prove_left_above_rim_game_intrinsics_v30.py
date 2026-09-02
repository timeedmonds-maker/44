from __future__ import annotations

"""Prove a same-game principal-point prior from many independent fixed-centre frames.

v29 showed that four single-frame event observations fit extremely well but left the
shared principal point slightly underconditioned under +/-0.5 px perturbation.  v30
does not relax that gate.  It increases independent evidence instead: every source
frame whose static-scene homography passes the existing held-out validation is used,
with each event weighted equally so three correlated samples from one clip cannot
outvote another event.

The already-proved physical centre is fixed.  Per-frame rotation and focal length are
free.  One principal point is shared across the game.  Player and ball pixels are
never metric anchors.  Passing authorizes only a same-game principal-point prior.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.audit_game_camera_registry_preflight_v1 import audit_pair
from freeze_spin.prove_frame_c_left_above_rim_metric_camera_v27 import project_fixed, solve_fixed
from freeze_spin.solve_nba_geometry_proof_v3 import solve_camera, world_landmarks

W, H = 960, 540


def safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def transform_points(points: dict[str, list[float]], H_target_to_source: np.ndarray) -> dict[str, list[float]]:
    names = list(points)
    arr = np.asarray([points[n] for n in names], dtype=np.float32)
    pred = cv2.perspectiveTransform(arr[:, None, :], H_target_to_source)[:, 0]
    return {n: [float(p[0]), float(p[1])] for n, p in zip(names, pred)}


def init_frame(frame: dict, center: np.ndarray, pp0: np.ndarray, world: dict[str, np.ndarray]) -> np.ndarray:
    view = frame["view"]
    *_, direct = solve_camera(view, world)
    if direct[0]:
        raise RuntimeError(f"Direct metric init implausible for {frame['frame_id']}")
    d = np.asarray(direct[2], dtype=np.float64)
    seed = np.r_[d[:3], d[6], pp0]
    fixed = solve_fixed(frame["obj"], frame["obs"], center, view, seed)
    return np.r_[fixed[:3], fixed[3]]


def joint_solve(frames: list[dict], center: np.ndarray, pp_seed: np.ndarray, world: dict[str, np.ndarray], *, warm: np.ndarray | None = None) -> np.ndarray:
    # vector = [shared cx,cy, frame0 rvec(3),logf, frame1 ...]
    if warm is None or len(warm) != 2 + 4 * len(frames):
        chunks = [init_frame(f, center, pp_seed, world) for f in frames]
        x0 = np.r_[pp_seed, *chunks]
    else:
        x0 = np.asarray(warm, dtype=np.float64).copy()

    counts = Counter(int(f["event_id"]) for f in frames)

    def residual(x: np.ndarray) -> np.ndarray:
        pp = x[:2]
        out = []
        for i, fr in enumerate(frames):
            off = 2 + 4 * i
            p = np.r_[x[off:off + 3], x[off + 3], pp]
            uv, cam, focal, _ = project_fixed(p, fr["obj"], center)
            # Equal total geometric influence per independent event.
            w = 1.0 / math.sqrt(float(counts[int(fr["event_id"])]))
            out.append((uv - fr["obs"]).ravel() * w)
            out.append(np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0 * w)
            fp = float(fr["view"].get("focal_prior_px", 900.0))
            fs = float(fr["view"].get("focal_prior_sigma_log", 1.8))
            out.append(np.asarray([(math.log(float(focal)) - math.log(fp)) / fs]) * w)
        # Weak optical-axis prior only; independent regulation geometry must dominate.
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


def frame_metrics(x: np.ndarray, idx: int, fr: dict, center: np.ndarray) -> dict:
    pp = x[:2]
    off = 2 + 4 * idx
    p = np.r_[x[off:off + 3], x[off + 3], pp]
    uv, cam, focal, _ = project_fixed(p, fr["obj"], center)
    err = np.linalg.norm(uv - fr["obs"], axis=1)
    return {
        "frame_id": fr["frame_id"],
        "event_id": int(fr["event_id"]),
        "source": str(fr["source"]),
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "focal_px": float(focal),
        "positive_depth": bool(float(np.min(cam[:, 2])) > 20.0),
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
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-independent-events", type=int, default=5)
    ap.add_argument("--min-passing-frames", type=int, default=8)
    ap.add_argument("--max-frame-rmse-px", type=float, default=1.5)
    ap.add_argument("--max-loo-event-pp-shift-px", type=float, default=8.0)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    ap.add_argument("--max-half-pixel-pp-shift-px", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    target_image = cv2.imread(str(args.target_frame))
    if target_image is None or target_image.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 target frame")

    spec = json.loads(args.target_landmarks.read_text(encoding="utf-8"))
    freeze = spec.get("freeze_lock", {})
    if freeze.get("authority_camera") != "Right Slash" or freeze.get("chooser_option") != "C":
        raise RuntimeError("Landmark spec not bound to immutable Frame C")
    view0 = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if view0 is None:
        raise RuntimeError("Missing target camera landmark view")
    if any(t in json.dumps(view0).lower() for t in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Dynamic metric anchors forbidden")

    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    camreg = reg["cameras"][args.camera_label]
    if camreg.get("status") != "FIXED_PHYSICAL_CENTRE_PRIOR_ACCEPTED" or not camreg["permissions"].get("center_prior_allowed"):
        raise RuntimeError("Fixed physical centre prior not accepted")
    center = np.asarray(camreg["physical_camera_center_prior_cm"], dtype=np.float64)

    world = world_landmarks()
    names = list(view0["landmarks"])
    obj = np.asarray([world[n] for n in names], dtype=np.float64)
    prefix = safe(args.camera_label)
    grouped: dict[int, list[Path]] = {}
    for p in sorted(args.samples.glob(f"{prefix}__event*__s*.png")):
        eid = int(p.stem.split("__event", 1)[1].split("__", 1)[0])
        grouped.setdefault(eid, []).append(p)

    frames = []
    transfer_rows = []
    for eid, paths in sorted(grouped.items()):
        for src in paths:
            tr = audit_pair(src, args.target_frame)
            rec = {
                "event_id": int(eid), "source": str(src), "status": tr.get("status"),
                "pass": bool(tr.get("pass")), "training_inliers": int(tr.get("training_inliers", 0)),
                "training_error": tr.get("training_error"), "withheld_error": tr.get("withheld_error"),
                "gates": tr.get("gates"),
            }
            transfer_rows.append(rec)
            if not tr.get("pass"):
                continue
            Hs2t = np.asarray(tr["H_source_to_target"], dtype=np.float64)
            lm = transform_points(view0["landmarks"], np.linalg.inv(Hs2t))
            obs = np.asarray([lm[n] for n in names], dtype=np.float64)
            view = {
                "label": f"{args.camera_label} event {eid} {src.stem}",
                "principal_point_prior_sigma_px": view0.get("principal_point_prior_sigma_px", 160.0),
                "principal_point_bound_px": view0.get("principal_point_bound_px", 350.0),
                "focal_prior_px": view0.get("focal_prior_px", 900.0),
                "focal_prior_sigma_log": view0.get("focal_prior_sigma_log", 1.8),
                "landmarks": lm,
            }
            frames.append({
                "frame_id": src.stem, "event_id": int(eid), "source": src,
                "view": view, "obj": obj, "obs": obs,
            })

    accepted_events = sorted(set(int(f["event_id"]) for f in frames))
    if len(accepted_events) < args.min_independent_events:
        raise RuntimeError(f"Only {len(accepted_events)} independent events supplied passing frames: {accepted_events}")
    if len(frames) < args.min_passing_frames:
        raise RuntimeError(f"Only {len(frames)} passing source frames")

    full = joint_solve(frames, center, np.asarray([W / 2.0, H / 2.0]), world)
    pp = full[:2]
    frame_results = [frame_metrics(full, i, fr, center) for i, fr in enumerate(frames)]
    max_rmse = max(r["rmse_px"] for r in frame_results)

    loo = []
    for eid in accepted_events:
        subset = [f for f in frames if int(f["event_id"]) != eid]
        warm = subset_warm(full, frames, subset)
        x = joint_solve(subset, center, pp, world, warm=warm)
        loo.append({
            "held_out_event": int(eid),
            "remaining_frame_count": len(subset),
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    rng = np.random.default_rng(300902)
    perturb = []
    for trial in range(args.perturbation_trials):
        pert = []
        for fr in frames:
            pfr = dict(fr)
            pfr["obs"] = fr["obs"] + rng.uniform(-0.5, 0.5, size=fr["obs"].shape)
            pert.append(pfr)
        x = joint_solve(pert, center, pp, world, warm=full)
        perturb.append({
            "trial": trial,
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    max_loo = max(r["shift_px"] for r in loo)
    max_pert = max(r["shift_px"] for r in perturb)
    gates = {
        "fixed_physical_center_prior_accepted": True,
        "independent_event_count_at_least_minimum": len(accepted_events) >= args.min_independent_events,
        "passing_frame_count_at_least_minimum": len(frames) >= args.min_passing_frames,
        "all_frame_fit_rmse_at_most_threshold": max_rmse <= args.max_frame_rmse_px,
        "all_frame_positive_depth": all(r["positive_depth"] for r in frame_results),
        "leave_one_whole_event_out_principal_point_stability": max_loo <= args.max_loo_event_pp_shift_px,
        "half_pixel_many_frame_principal_point_stability": max_pert <= args.max_half_pixel_pp_shift_px,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_GAME_INTRINSICS_PRIOR" if passed else "FAIL_GAME_INTRINSICS_PRIOR",
        "version": "v30_many_frame_equal_event_weight",
        "game_id": reg["game_id"], "camera_label": args.camera_label,
        "method": "many accepted same-game fixed-centre frames + equal event weighting + per-frame rotation/focal + one shared principal point",
        "guardrail": "Passing authorizes only a same-game principal-point prior; exact event camera and replay remain unpromoted.",
        "physical_camera_center_prior_cm": [float(x) for x in center],
        "shared_principal_point_px": [float(pp[0]), float(pp[1])],
        "accepted_independent_events": accepted_events,
        "accepted_independent_event_count": len(accepted_events),
        "accepted_frame_count": len(frames),
        "frame_results": frame_results,
        "transport_results": transfer_rows,
        "leave_one_event_out": loo,
        "half_pixel_joint_perturbation": perturb,
        "max_frame_rmse_px": max_rmse,
        "max_leave_one_event_out_principal_point_shift_px": max_loo,
        "max_half_pixel_principal_point_shift_px": max_pert,
        "thresholds": {
            "max_frame_rmse_px": args.max_frame_rmse_px,
            "max_leave_one_event_out_principal_point_shift_px": args.max_loo_event_pp_shift_px,
            "max_half_pixel_principal_point_shift_px": args.max_half_pixel_pp_shift_px,
        },
        "gates": gates,
        "principal_point_prior_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
    }
    (args.out / "left_above_rim_game_intrinsics_v30.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "events": accepted_events, "frames": len(frames),
        "shared_principal_point_px": payload["shared_principal_point_px"],
        "max_frame_rmse_px": max_rmse, "max_loo_event_pp_shift_px": max_loo,
        "max_half_pixel_pp_shift_px": max_pert,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
