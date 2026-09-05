from __future__ import annotations

"""Learn a same-game principal-point prior for the fixed Left Above Rim camera.

The physical camera centre is already independently proved and locked in the game
camera registry.  This proof uses only independent same-game event frames (not the
target Frame C pose) to determine whether the optical principal point can be shared
across pan/tilt/zoom states.  Per-event rotation and focal length remain free.

Regulation target/lane landmarks are transported from the trusted target view by a
held-out-validated static-scene homography.  Because this feed has already passed the
fixed-optical-centre preflight, pure camera rotation/zoom produces a scene-wide
projective homography; no player or ball points are used.

Passing this script may authorize a principal-point prior only.  It does not promote
an event camera or replay render.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin.audit_game_camera_registry_preflight_v1 import best_event_pair
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


def fit_metrics(rvec: np.ndarray, logf: float, pp: np.ndarray, center: np.ndarray, obj: np.ndarray, obs: np.ndarray) -> dict:
    p = np.r_[rvec, logf, pp]
    uv, cam, focal, R = project_fixed(p, obj, center)
    err = np.linalg.norm(uv - obs, axis=1)
    return {
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95)),
        "max_px": float(np.max(err)),
        "focal_px": float(focal),
        "R": R,
        "cam": cam,
        "uv": uv,
    }


def joint_solve(events: list[dict], center: np.ndarray, pp0: np.ndarray, *, warm: np.ndarray | None = None) -> np.ndarray:
    # vector: [cx, cy, event0_rvec(3), logf, event1_rvec(3), logf, ...]
    if warm is None:
        chunks = []
        for ev in events:
            view = ev["view"]
            obj = ev["obj"]
            obs = ev["obs"]
            *_, direct = solve_camera(view, world_landmarks())
            if direct[0]:
                raise RuntimeError(f"Direct init implausible for event {ev['event_id']}")
            d = np.asarray(direct[2], dtype=np.float64)
            p0 = np.r_[d[:3], d[6], pp0]
            sf = solve_fixed(obj, obs, center, view, p0)
            chunks.append(np.r_[sf[:3], sf[3]])
        x0 = np.r_[pp0, *chunks]
    else:
        x0 = np.asarray(warm, dtype=np.float64).copy()

    def residual(x: np.ndarray) -> np.ndarray:
        pp = x[:2]
        out = []
        for i, ev in enumerate(events):
            off = 2 + 4 * i
            rvec = x[off:off + 3]
            logf = float(x[off + 3])
            p = np.r_[rvec, logf, pp]
            uv, cam, focal, _ = project_fixed(p, ev["obj"], center)
            out.append((uv - ev["obs"]).ravel())
            out.append(np.minimum(cam[:, 2] - 20.0, 0.0) / 5.0)
            fp = float(ev["view"].get("focal_prior_px", 900.0))
            fs = float(ev["view"].get("focal_prior_sigma_log", 1.8))
            out.append(np.asarray([(math.log(focal) - math.log(fp)) / fs]))
        # Weak optical-axis prior only; the independent event geometry should dominate.
        out.append(np.asarray([(pp[0] - W / 2.0) / 160.0, (pp[1] - H / 2.0) / 160.0]))
        return np.concatenate(out)

    n = len(events)
    lower = [W / 2.0 - 250.0, H / 2.0 - 250.0]
    upper = [W / 2.0 + 250.0, H / 2.0 + 250.0]
    for _ in range(n):
        lower += [-np.inf, -np.inf, -np.inf, math.log(150.0)]
        upper += [np.inf, np.inf, np.inf, math.log(4000.0)]
    opt = least_squares(
        residual,
        x0,
        bounds=(np.asarray(lower), np.asarray(upper)),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=40000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--target-landmarks", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--camera-label", default="Left Above Rim")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-independent-events", type=int, default=3)
    ap.add_argument("--max-event-rmse-px", type=float, default=1.5)
    ap.add_argument("--max-loo-principal-point-shift-px", type=float, default=8.0)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    ap.add_argument("--max-half-pixel-principal-point-shift-px", type=float, default=5.0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target = cv2.imread(str(args.target_frame))
    if target is None or target.shape[:2] != (H, W):
        raise RuntimeError("Expected native 960x540 target frame")

    spec = json.loads(args.target_landmarks.read_text(encoding="utf-8"))
    freeze = spec.get("freeze_lock", {})
    if freeze.get("authority_camera") != "Right Slash" or freeze.get("chooser_option") != "C":
        raise RuntimeError("Target landmark spec is not bound to immutable Frame C")
    view0 = next((v for v in spec["views"] if v["label"] == args.camera_label), None)
    if view0 is None:
        raise RuntimeError("Missing target camera landmark view")
    if any(t in json.dumps(view0).lower() for t in ("player", "ball", "body", "hand", "elbow", "shoulder")):
        raise RuntimeError("Dynamic anchors forbidden")

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

    events = []
    transport_rows = []
    for eid, paths in sorted(grouped.items()):
        tr = best_event_pair(paths, args.target_frame)
        rec = {
            "event_id": eid,
            "status": tr.get("status"),
            "pass": bool(tr.get("pass")),
            "source": tr.get("source"),
            "training_inliers": int(tr.get("training_inliers", 0)),
            "training_error": tr.get("training_error"),
            "withheld_error": tr.get("withheld_error"),
        }
        transport_rows.append(rec)
        if not tr.get("pass"):
            continue
        Hs2t = np.asarray(tr["H_source_to_target"], dtype=np.float64)
        lm = transform_points(view0["landmarks"], np.linalg.inv(Hs2t))
        obs = np.asarray([lm[n] for n in names], dtype=np.float64)
        view = {
            "label": f"{args.camera_label} event {eid}",
            "principal_point_prior_sigma_px": view0.get("principal_point_prior_sigma_px", 160.0),
            "principal_point_bound_px": view0.get("principal_point_bound_px", 350.0),
            "focal_prior_px": view0.get("focal_prior_px", 900.0),
            "focal_prior_sigma_log": view0.get("focal_prior_sigma_log", 1.8),
            "landmarks": lm,
        }
        events.append({"event_id": eid, "view": view, "obj": obj, "obs": obs})

    if len(events) < args.min_independent_events:
        raise RuntimeError(f"Only {len(events)} accepted independent events")

    full = joint_solve(events, center, np.asarray([W / 2.0, H / 2.0]))
    pp = full[:2]
    event_results = []
    for i, ev in enumerate(events):
        off = 2 + 4 * i
        m = fit_metrics(full[off:off + 3], full[off + 3], pp, center, ev["obj"], ev["obs"])
        event_results.append({
            "event_id": ev["event_id"],
            "rmse_px": m["rmse_px"],
            "p95_px": m["p95_px"],
            "max_px": m["max_px"],
            "focal_px": m["focal_px"],
        })

    loo = []
    for hold in range(len(events)):
        subset = [ev for i, ev in enumerate(events) if i != hold]
        x = joint_solve(subset, center, pp)
        shift = float(np.linalg.norm(x[:2] - pp))
        loo.append({"held_out_event": events[hold]["event_id"], "principal_point_px": [float(x[0]), float(x[1])], "shift_px": shift})

    rng = np.random.default_rng(290902)
    perturb = []
    for trial in range(args.perturbation_trials):
        pert_events = []
        for ev in events:
            pert_events.append({**ev, "obs": ev["obs"] + rng.uniform(-0.5, 0.5, size=ev["obs"].shape)})
        x = joint_solve(pert_events, center, pp)
        perturb.append({
            "trial": trial,
            "principal_point_px": [float(x[0]), float(x[1])],
            "shift_px": float(np.linalg.norm(x[:2] - pp)),
        })

    max_loo = max(x["shift_px"] for x in loo)
    max_pert = max(x["shift_px"] for x in perturb)
    max_event_rmse = max(x["rmse_px"] for x in event_results)
    gates = {
        "fixed_physical_center_prior_accepted": True,
        "independent_event_count_at_least_minimum": len(events) >= args.min_independent_events,
        "all_event_fit_rmse_at_most_threshold": max_event_rmse <= args.max_event_rmse_px,
        "leave_one_event_out_principal_point_stability": max_loo <= args.max_loo_principal_point_shift_px,
        "half_pixel_joint_principal_point_stability": max_pert <= args.max_half_pixel_principal_point_shift_px,
    }
    passed = bool(all(gates.values()))
    payload = {
        "status": "PASS_GAME_INTRINSICS_PRIOR" if passed else "FAIL_GAME_INTRINSICS_PRIOR",
        "game_id": reg["game_id"],
        "camera_label": args.camera_label,
        "method": "independent same-game fixed-centre frames + per-event rotation/focal + one shared principal point",
        "guardrail": "Passing authorizes only a same-game principal-point prior. It does not promote any event camera or replay render.",
        "physical_camera_center_prior_cm": [float(x) for x in center],
        "shared_principal_point_px": [float(pp[0]), float(pp[1])],
        "independent_event_count": len(events),
        "event_results": event_results,
        "transport_results": transport_rows,
        "leave_one_event_out": loo,
        "half_pixel_joint_perturbation": perturb,
        "max_event_rmse_px": max_event_rmse,
        "max_leave_one_event_out_principal_point_shift_px": max_loo,
        "max_half_pixel_principal_point_shift_px": max_pert,
        "thresholds": {
            "max_event_rmse_px": args.max_event_rmse_px,
            "max_leave_one_event_out_principal_point_shift_px": args.max_loo_principal_point_shift_px,
            "max_half_pixel_principal_point_shift_px": args.max_half_pixel_principal_point_shift_px,
        },
        "gates": gates,
        "principal_point_prior_allowed": passed,
        "metric_event_camera_allowed": False,
        "replay_render_allowed": False,
    }
    (args.out / "left_above_rim_game_intrinsics_v29.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "shared_principal_point_px": payload["shared_principal_point_px"],
        "max_event_rmse_px": max_event_rmse,
        "max_loo_pp_shift_px": max_loo,
        "max_half_pixel_pp_shift_px": max_pert,
    }, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
