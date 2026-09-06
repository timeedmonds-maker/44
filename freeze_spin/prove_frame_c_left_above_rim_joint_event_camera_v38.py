from __future__ import annotations

"""Joint metric + exact-clip static optical proof for immutable Frame C.

v37 established an excellent metric pinhole event camera but missed the locked
principal-point perturbation gate by 0.496 px.  This stage does not weaken that
gate.  Instead it adds independent same-camera static-scene correspondences to
the event-camera solve itself.  Manual metric observations and static SIFT
observations retain separate perturbation tests and static held-out pixels are
never used by the joint fit.

Passing authorizes the exact Left Above Rim Frame C metric event camera only.
Replay rendering remains forbidden pending a second distinct metric camera,
exact-state multi-view alignment, and static novel-view QA.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import prove_frame_c_left_above_rim_metric_camera_v37 as v37

W, H = 960, 540
FT, IN = 30.48, 2.54


def spatial_cap(p: np.ndarray, q: np.ndarray, limit: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if len(p) <= limit:
        return p.astype(np.float64), q.astype(np.float64)
    key = p[:, 0] + 17.0 * p[:, 1]
    order = np.argsort(key)
    ids = np.linspace(0, len(order) - 1, limit).round().astype(int)
    pick = order[ids]
    return p[pick].astype(np.float64), q[pick].astype(np.float64)


def build_pairs(target: Path, samples: Path, manifest: dict) -> list[dict]:
    meta = {x["file"]: x for x in manifest["samples"]}
    out = []
    for src in sorted(samples.glob("Left_Above_Rim_target_event__*.png")):
        pr = v37.clip_pair(src, target)
        if pr is None:
            continue
        p, q = spatial_cap(pr["p"], pr["q"], 100)
        mm = meta[src.name]
        out.append({
            "name": src.name,
            "relative_seconds": float(mm["relative_to_freeze_seconds"]),
            "p": p,
            "q": q,
            "pw": np.asarray(pr["pw"], dtype=np.float64),
            "qw": np.asarray(pr["qw"], dtype=np.float64),
            "H": np.asarray(pr["H"], dtype=np.float64),
        })
    return out


def metric_residual(core: np.ndarray, C: np.ndarray, obs: dict[str, np.ndarray], held: dict[str, set[int]],
                    targetP: np.ndarray, targetO: np.ndarray, exclude_family: str | None = None) -> np.ndarray:
    z = []
    for name, oo in obs.items():
        if name == exclude_family:
            continue
        pred = v37.project(core, "pinhole", v37.CURVES[name], C)
        for i, x in enumerate(oo):
            if i not in held[name]:
                z.extend(v37.nearest_res(pred, x))
    z.extend((v37.project(core, "pinhole", targetP, C) - targetO).ravel())
    return np.asarray(z, dtype=np.float64)


def pair_project(p: np.ndarray, pp: np.ndarray, ft: float, fs: float, rv: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
    hm = v37.k(ft, pp) @ R @ np.linalg.inv(v37.k(fs, pp))
    ph = np.column_stack([p, np.ones(len(p))])
    qh = (hm @ ph.T).T
    return qh[:, :2] / qh[:, 2:3]


def initialize_pair(pr: dict, pp: np.ndarray, ft: float) -> np.ndarray:
    fs0 = ft
    rv0 = v37.nearest_rot(pr["H"], pp, ft, fs0)
    x0 = np.r_[math.log(fs0), rv0]

    def fun(x: np.ndarray) -> np.ndarray:
        return (pair_project(pr["p"], pp, ft, math.exp(float(x[0])), x[1:4]) - pr["q"]).ravel()

    o = least_squares(
        fun, x0,
        bounds=(np.r_[math.log(150.0), [-10.0] * 3], np.r_[math.log(4000.0), [10.0] * 3]),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=5000,
    )
    return np.asarray(o.x, dtype=np.float64)


def bounds_for(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.r_[[-10.0] * 3, math.log(250.0), 100.0, 50.0]
    hi = np.r_[[10.0] * 3, math.log(2500.0), 850.0, 520.0]
    for _ in pairs:
        lo = np.r_[lo, math.log(150.0), [-10.0] * 3]
        hi = np.r_[hi, math.log(4000.0), [10.0] * 3]
    return lo, hi


def initial_joint(metric_core: np.ndarray, pairs: list[dict]) -> np.ndarray:
    pp = metric_core[4:6]
    ft = math.exp(float(metric_core[3]))
    parts = [metric_core]
    for pr in pairs:
        parts.append(initialize_pair(pr, pp, ft))
    return np.concatenate(parts)


def subset_warm(full_x: np.ndarray, full_pairs: list[dict], subset: list[dict]) -> np.ndarray:
    by_name = {p["name"]: i for i, p in enumerate(full_pairs)}
    parts = [full_x[:6]]
    for pr in subset:
        i = by_name[pr["name"]]
        off = 6 + 4 * i
        parts.append(full_x[off:off + 4])
    return np.concatenate(parts)


def joint_residual(x: np.ndarray, pairs: list[dict], C: np.ndarray,
                   obs: dict[str, np.ndarray], held: dict[str, set[int]], targetP: np.ndarray, targetO: np.ndarray,
                   exclude_family: str | None = None) -> np.ndarray:
    core = x[:6]
    ft = math.exp(float(core[3]))
    pp = core[4:6]
    out = [metric_residual(core, C, obs, held, targetP, targetO, exclude_family)]
    off = 6
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        # Unit pixel weighting is deliberate: one static image residual has the
        # same units/scale as one metric source-pixel residual.  A deterministic
        # 100-point spatial cap prevents a textured frame from dominating.
        out.append((pair_project(pr["p"], pp, ft, fs, rv) - pr["q"]).ravel())
    return np.concatenate(out)


def fit_joint(warm: np.ndarray, pairs: list[dict], C: np.ndarray,
              obs: dict[str, np.ndarray], held: dict[str, set[int]], targetP: np.ndarray, targetO: np.ndarray,
              exclude_family: str | None = None, max_nfev: int = 5000) -> np.ndarray:
    lo, hi = bounds_for(pairs)
    o = least_squares(
        lambda x: joint_residual(x, pairs, C, obs, held, targetP, targetO, exclude_family),
        warm, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
    )
    return np.asarray(o.x, dtype=np.float64)


def static_metrics(x: np.ndarray, pairs: list[dict]) -> list[dict]:
    core = x[:6]
    pp = core[4:6]
    ft = math.exp(float(core[3]))
    rows = []
    off = 6
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        tr = np.linalg.norm(pair_project(pr["p"], pp, ft, fs, rv) - pr["q"], axis=1)
        wh = np.linalg.norm(pair_project(pr["pw"], pp, ft, fs, rv) - pr["qw"], axis=1)
        rows.append({
            "frame": pr["name"],
            "relative_seconds": pr["relative_seconds"],
            "source_focal_px": float(fs),
            "joint_train_points": int(len(pr["p"])),
            "heldout_points": int(len(pr["pw"])),
            "train_p95_px": float(np.percentile(tr, 95)),
            "heldout_median_px": float(np.median(wh)),
            "heldout_p95_px": float(np.percentile(wh, 95)),
        })
    return rows


def draw_overlay(image: Path, out: Path, core: np.ndarray, C: np.ndarray,
                 obs: dict[str, np.ndarray], targetP: np.ndarray, targetO: np.ndarray) -> None:
    im = cv2.imread(str(image))
    for oo in obs.values():
        for q in oo:
            cv2.circle(im, tuple(np.round(q).astype(int)), 2, (0, 255, 0), -1)
    for name in obs:
        pred = v37.project(core, "pinhole", v37.CURVES[name], C)
        ok = (pred[:, 0] >= -20) & (pred[:, 0] <= W + 20) & (pred[:, 1] >= -20) & (pred[:, 1] <= H + 20)
        pts = np.round(pred[ok]).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(im, [pts.reshape(-1, 1, 2)], False, (255, 0, 255), 1)
    tp = v37.project(core, "pinhole", targetP, C)
    for q in targetO:
        cv2.circle(im, tuple(np.round(q).astype(int)), 5, (0, 255, 0), 2)
    for q in tp:
        cv2.circle(im, tuple(np.round(q).astype(int)), 4, (255, 0, 255), 2)
    cv2.polylines(im, [np.round(tp).astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), 2)
    cv2.imwrite(str(out), im)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-frame", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--wide-court", type=Path, required=True)
    ap.add_argument("--floor-proof", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--legacy-landmarks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=24)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wide = json.loads(args.wide_court.read_text(encoding="utf-8"))
    floor = json.loads(args.floor_proof.read_text(encoding="utf-8"))
    reg = json.loads(args.registry.read_text(encoding="utf-8"))
    legacy = json.loads(args.legacy_landmarks.read_text(encoding="utf-8"))
    manifest = json.loads(args.sample_manifest.read_text(encoding="utf-8"))
    cam = reg["cameras"]["Left Above Rim"]
    if not cam["permissions"].get("physical_camera_center_allowed"):
        raise RuntimeError("v36 physical centre is not authorized")
    if floor.get("status") != "PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35":
        raise RuntimeError("v35 floor proof is not accepted")
    if abs(float(manifest["immutable_freeze_time_seconds"]) - 8.653093) > 5e-7:
        raise RuntimeError("Immutable Left Above Rim Frame C time changed")

    C = np.asarray(cam["physical_camera_center_prior_cm"], dtype=np.float64)
    Hfloor = np.asarray(floor["floor_homography_world_to_image"], dtype=np.float64)
    obs = {k: np.asarray(v, dtype=np.float64) for k, v in wide["observations_px"].items()}
    held = {k: set(v) for k, v in wide["held_out_indices"].items()}
    lv = next(v for v in legacy["views"] if v["label"] == "Left Above Rim")
    L = lv["landmarks"]
    targetO = np.asarray([L[x] for x in [
        "target_inner_top_left", "target_inner_top_right", "target_inner_bottom_right", "target_inner_bottom_left"
    ]], dtype=np.float64)
    # Inner opening of NBA 24x18-inch outside rectangle with 2-inch line:
    # 20x14 inches.  Bottom inner edge is 2 inches above the ring/top-line plane.
    targetP = np.asarray([
        [0, -10 * IN, 10 * FT + 16 * IN], [0, 10 * IN, 10 * FT + 16 * IN],
        [0, 10 * IN, 10 * FT + 2 * IN], [0, -10 * IN, 10 * FT + 2 * IN],
    ], dtype=np.float64)

    pairs = build_pairs(args.target_frame, args.samples, manifest)
    before = sum(p["relative_seconds"] < 0 for p in pairs)
    after = sum(p["relative_seconds"] > 0 for p in pairs)
    if len(pairs) < 6 or before < 2 or after < 2:
        raise RuntimeError(f"Insufficient exact-clip static support: n={len(pairs)}, before={before}, after={after}")

    # Reproduce v37 metric-only root as the neutral seed; no v37 result file is required.
    metric_seed = v37.fit_metric("pinhole", C, Hfloor, obs, held, targetP, targetO)
    x0 = initial_joint(metric_seed, pairs)
    full = fit_joint(x0, pairs, C, obs, held, targetP, targetO)
    core = full[:6]
    rv, focal, _, cx, cy, _ = v37.unpack(core, "pinhole")
    metric = v37.metrics(core, "pinhole", C, obs, held, targetP, targetO)
    static = static_metrics(full, pairs)

    # Whole metric-family holdout while retaining independent static optical evidence.
    family_loo = {}
    for fam in obs:
        q = fit_joint(full, pairs, C, obs, held, targetP, targetO, exclude_family=fam, max_nfev=3000)
        pred = v37.project(q[:6], "pinhole", v37.CURVES[fam], C)
        ee = [float(np.min(np.linalg.norm(pred - x, axis=1))) for x in obs[fam]]
        family_loo[fam] = {"p95_px": float(np.percentile(ee, 95)), "max_px": float(max(ee))}

    # Whole static-frame holdout: no single exact-clip frame may define the optical axis.
    static_loo = []
    for j, hold in enumerate(pairs):
        subset = [p for i, p in enumerate(pairs) if i != j]
        warm = subset_warm(full, pairs, subset)
        q = fit_joint(warm, subset, C, obs, held, targetP, targetO, max_nfev=3000)
        static_loo.append({
            "held_out_frame": hold["name"],
            "principal_point_shift_px": float(np.linalg.norm(q[4:6] - core[4:6])),
            "target_focal_fraction": float(abs(math.exp(float(q[3])) - focal) / focal),
        })

    # Combined measurement perturbation: manual geometry ±0.5 px AND independently
    # detected SIFT training coordinates ±0.25 px. Held-out static pixels remain untouched.
    rng_metric = np.random.default_rng(20260903)
    rng_static = np.random.default_rng(380038)
    Rb, _ = cv2.Rodrigues(rv.reshape(3, 1))
    perturb = []
    for trial in range(args.perturbation_trials):
        po = {k: v + rng_metric.uniform(-0.5, 0.5, v.shape) for k, v in obs.items()}
        to = targetO + rng_metric.uniform(-0.5, 0.5, targetO.shape)
        ppairs = []
        for pr in pairs:
            p2 = dict(pr)
            p2["p"] = pr["p"] + rng_static.uniform(-0.25, 0.25, pr["p"].shape)
            p2["q"] = pr["q"] + rng_static.uniform(-0.25, 0.25, pr["q"].shape)
            ppairs.append(p2)
        q = fit_joint(full, ppairs, C, po, held, targetP, to, max_nfev=1800)
        qc = q[:6]
        qrv, qf, _, qcx, qcy, _ = v37.unpack(qc, "pinhole")
        Rq, _ = cv2.Rodrigues(qrv.reshape(3, 1))
        ang = math.degrees(math.acos(float(np.clip((np.trace(Rq @ Rb.T) - 1.0) / 2.0, -1.0, 1.0))))
        mm = v37.metrics(qc, "pinhole", C, obs, held, targetP, targetO)
        perturb.append({
            "trial": trial,
            "focal_fraction": float(abs(qf - focal) / focal),
            "principal_point_shift_px": float(math.hypot(qcx - cx, qcy - cy)),
            "rotation_deg": float(ang),
            "heldout_floor_p95_px": float(mm["floor_heldout_p95_px"]),
            "target_p95_px": float(mm["target_p95_px"]),
        })

    gates = {
        "metric_train_floor": bool(metric["floor_train_p95_px"] <= 2.5),
        "metric_heldout_floor": bool(metric["floor_heldout_p95_px"] <= 1.5),
        "metric_target": bool(metric["target_p95_px"] <= 1.0),
        "leave_floor_family": bool(max(x["p95_px"] for x in family_loo.values()) <= 2.5),
        "static_frames": bool(len(static) >= 6 and before >= 2 and after >= 2),
        "static_joint_train": bool(max(x["train_p95_px"] for x in static) <= 2.5),
        "static_heldout": bool(max(x["heldout_p95_px"] for x in static) <= 3.0),
        "leave_static_frame_pp": bool(max(x["principal_point_shift_px"] for x in static_loo) <= 5.0),
        "perturb_floor": bool(max(x["heldout_floor_p95_px"] for x in perturb) <= 1.5),
        "perturb_target": bool(max(x["target_p95_px"] for x in perturb) <= 1.0),
        "perturb_focal": bool(max(x["focal_fraction"] for x in perturb) <= 0.005),
        "perturb_pp": bool(max(x["principal_point_shift_px"] for x in perturb) <= 10.0),
        "perturb_rotation": bool(max(x["rotation_deg"] for x in perturb) <= 0.8),
    }
    passed = bool(all(gates.values()))

    report = {
        "schema_version": 1,
        "status": "PASS_JOINT_METRIC_EVENT_CAMERA_V38" if passed else "FAIL_JOINT_METRIC_EVENT_CAMERA_V38",
        "game_id": "0022500301", "event_id": 489, "camera_label": "Left Above Rim",
        "physical_center_cm": C.tolist(),
        "model": "square-pixel pinhole + fixed v36 physical centre + exact-clip pure-rotation/zoom static constraints",
        "camera": {"rvec": rv.tolist(), "focal_px": float(focal), "principal_point_px": [float(cx), float(cy)]},
        "metric": metric,
        "exact_clip_static": {
            "accepted_frames": len(static), "before": before, "after": after,
            "max_train_p95_px": max(x["train_p95_px"] for x in static),
            "max_heldout_p95_px": max(x["heldout_p95_px"] for x in static),
            "frames": static,
        },
        "leave_one_floor_family_out": family_loo,
        "leave_one_static_frame_out": {
            "max_principal_point_shift_px": max(x["principal_point_shift_px"] for x in static_loo),
            "max_target_focal_fraction": max(x["target_focal_fraction"] for x in static_loo),
            "frames": static_loo,
        },
        "combined_perturbation": {
            "trials": len(perturb), "manual_metric_amplitude_px": 0.5, "static_training_amplitude_px": 0.25,
            "max_principal_point_shift_px": max(x["principal_point_shift_px"] for x in perturb),
            "max_focal_fraction": max(x["focal_fraction"] for x in perturb),
            "max_rotation_deg": max(x["rotation_deg"] for x in perturb),
            "max_heldout_floor_p95_px": max(x["heldout_floor_p95_px"] for x in perturb),
            "max_target_p95_px": max(x["target_p95_px"] for x in perturb),
            "trials_detail": perturb,
        },
        "v37_failure_addressed": "v37 missed the unchanged 10 px principal-point perturbation gate at 10.4957 px; v38 adds independent exact-clip static optical evidence rather than weakening the gate.",
        "gates": gates,
        "permissions": {"metric_event_camera_allowed": passed, "replay_render_allowed": False},
        "next_gate": "Calibrate a second distinct physical/metric camera at the same immutable state, then refine exact multi-view state alignment before any static virtual-view render." if passed else "Do not promote event camera; inspect failed v38 gate without weakening thresholds.",
    }
    (args.out / "frame_c_left_above_rim_joint_event_camera_v38.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    draw_overlay(args.target_frame, args.out / "frame_c_left_above_rim_joint_metric_overlay_v38.png", core, C, obs, targetP, targetO)
    print(json.dumps({
        "status": report["status"], "camera": report["camera"], "metric": metric,
        "static": report["exact_clip_static"], "static_loo": report["leave_one_static_frame_out"],
        "perturbation": {k: v for k, v in report["combined_perturbation"].items() if k != "trials_detail"},
        "gates": gates,
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
