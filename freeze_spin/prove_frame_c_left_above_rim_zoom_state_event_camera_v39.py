from __future__ import annotations

"""Zoom-state-aware exact Frame C event-camera proof.

v38 passed 12/13 gates. Its sole miss was a 5.2198 px target-principal-point
shift when the earliest accepted exact-clip frame was removed. That frame is
also the only strongly different zoom state (~776.6 px source focal versus
~824.7 px at/near Frame C). The v38 model forced the same encoded principal
point at every zoom state.

v39 keeps every accepted v38 metric/holdout/perturbation threshold. It adds one
shared, smooth zoom-dependent effective source principal-point term:

  pp_source = pp_target + beta * ((f_source - f_target) / f_target)

The target Frame C principal point remains the metric quantity being proved.
A 1 px smooth-crop regularization is applied to the *actual implied source PP
shift* in each calibration frame, not directly to beta. This is deliberately
small and physically interpretable: the source crop/optical axis may drift
slightly through the zoom ramp but cannot become an unconstrained nuisance.

Replay rendering remains forbidden even if this camera passes.
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
SOURCE_PP_DRIFT_SIGMA_PX = 1.0
BETA_BOUND_PX_PER_FOCAL_FRACTION = 100.0


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
    rows = []
    for src in sorted(samples.glob("Left_Above_Rim_target_event__*.png")):
        pr = v37.clip_pair(src, target)
        if pr is None:
            continue
        p, q = spatial_cap(pr["p"], pr["q"], 100)
        mm = meta[src.name]
        rows.append({
            "name": src.name,
            "relative_seconds": float(mm["relative_to_freeze_seconds"]),
            "p": p,
            "q": q,
            "pw": np.asarray(pr["pw"], dtype=np.float64),
            "qw": np.asarray(pr["qw"], dtype=np.float64),
            "H": np.asarray(pr["H"], dtype=np.float64),
        })
    return rows


def metric_residual(core: np.ndarray, C: np.ndarray, obs: dict[str, np.ndarray], held: dict[str, set[int]],
                    targetP: np.ndarray, targetO: np.ndarray, exclude_family: str | None = None) -> np.ndarray:
    out = []
    for name, oo in obs.items():
        if name == exclude_family:
            continue
        pred = v37.project(core, "pinhole", v37.CURVES[name], C)
        for i, x in enumerate(oo):
            if i not in held[name]:
                out.extend(v37.nearest_res(pred, x))
    out.extend((v37.project(core, "pinhole", targetP, C) - targetO).ravel())
    return np.asarray(out, dtype=np.float64)


def source_pp(core: np.ndarray, beta: np.ndarray, fs: float) -> np.ndarray:
    ft = math.exp(float(core[3]))
    return core[4:6] + beta * ((fs - ft) / ft)


def pair_project(p: np.ndarray, core: np.ndarray, beta: np.ndarray, fs: float, rv: np.ndarray) -> np.ndarray:
    ft = math.exp(float(core[3]))
    ppt = core[4:6]
    pps = source_pp(core, beta, fs)
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
    hm = v37.k(ft, ppt) @ R @ np.linalg.inv(v37.k(fs, pps))
    ph = np.column_stack([p, np.ones(len(p))])
    qh = (hm @ ph.T).T
    return qh[:, :2] / qh[:, 2:3]


def nearest_rotation(Hm: np.ndarray, ppt: np.ndarray, ft: float, pps: np.ndarray, fs: float) -> np.ndarray:
    M = np.linalg.inv(v37.k(ft, ppt)) @ Hm @ v37.k(fs, pps)
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
    return cv2.Rodrigues(R)[0].ravel()


def initialize_pair(pr: dict, core: np.ndarray) -> np.ndarray:
    ft = math.exp(float(core[3]))
    ppt = core[4:6]
    fs0 = ft
    rv0 = nearest_rotation(pr["H"], ppt, ft, ppt, fs0)
    x0 = np.r_[math.log(fs0), rv0]

    def fun(x: np.ndarray) -> np.ndarray:
        fs = math.exp(float(x[0]))
        # Neutral beta=0 initialization only; v39 beta is solved jointly later.
        return (pair_project(pr["p"], core, np.zeros(2), fs, x[1:4]) - pr["q"]).ravel()

    opt = least_squares(
        fun, x0,
        bounds=(np.r_[math.log(150.0), [-10.0] * 3], np.r_[math.log(4000.0), [10.0] * 3]),
        loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=5000,
    )
    return np.asarray(opt.x, dtype=np.float64)


def bounds_for(n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.r_[[-10.0] * 3, math.log(250.0), 100.0, 50.0,
               -BETA_BOUND_PX_PER_FOCAL_FRACTION, -BETA_BOUND_PX_PER_FOCAL_FRACTION]
    hi = np.r_[[10.0] * 3, math.log(2500.0), 850.0, 520.0,
               BETA_BOUND_PX_PER_FOCAL_FRACTION, BETA_BOUND_PX_PER_FOCAL_FRACTION]
    for _ in range(n_pairs):
        lo = np.r_[lo, math.log(150.0), [-10.0] * 3]
        hi = np.r_[hi, math.log(4000.0), [10.0] * 3]
    return lo, hi


def initial_joint(metric_core: np.ndarray, pairs: list[dict], beta_seed: np.ndarray) -> np.ndarray:
    parts = [metric_core, np.asarray(beta_seed, dtype=np.float64)]
    for pr in pairs:
        parts.append(initialize_pair(pr, metric_core))
    return np.concatenate(parts)


def subset_warm(full_x: np.ndarray, full_pairs: list[dict], subset: list[dict]) -> np.ndarray:
    by_name = {p["name"]: i for i, p in enumerate(full_pairs)}
    parts = [full_x[:8]]
    for pr in subset:
        i = by_name[pr["name"]]
        off = 8 + 4 * i
        parts.append(full_x[off:off + 4])
    return np.concatenate(parts)


def joint_residual(x: np.ndarray, pairs: list[dict], C: np.ndarray,
                   obs: dict[str, np.ndarray], held: dict[str, set[int]],
                   targetP: np.ndarray, targetO: np.ndarray,
                   exclude_family: str | None = None) -> np.ndarray:
    core = x[:6]
    beta = x[6:8]
    ft = math.exp(float(core[3]))
    out = [metric_residual(core, C, obs, held, targetP, targetO, exclude_family)]
    off = 8
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        out.append((pair_project(pr["p"], core, beta, fs, rv) - pr["q"]).ravel())
        # Broad physical smooth-crop prior in directly interpretable pixels.
        # It acts only on the source frame nuisance state, never directly on the
        # target Frame C principal point.
        out.append((source_pp(core, beta, fs) - core[4:6]) / SOURCE_PP_DRIFT_SIGMA_PX)
    return np.concatenate(out)


def fit_joint(warm: np.ndarray, pairs: list[dict], C: np.ndarray,
              obs: dict[str, np.ndarray], held: dict[str, set[int]],
              targetP: np.ndarray, targetO: np.ndarray,
              exclude_family: str | None = None, max_nfev: int = 5000) -> np.ndarray:
    lo, hi = bounds_for(len(pairs))
    opt = least_squares(
        lambda x: joint_residual(x, pairs, C, obs, held, targetP, targetO, exclude_family),
        warm, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
    )
    return np.asarray(opt.x, dtype=np.float64)


def solve_multistart(metric_seed: np.ndarray, pairs: list[dict], C: np.ndarray,
                     obs: dict[str, np.ndarray], held: dict[str, set[int]],
                     targetP: np.ndarray, targetO: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    roots = []
    for beta_seed in (np.asarray([0.0, 0.0]), np.asarray([0.0, 20.0]), np.asarray([0.0, -20.0])):
        x0 = initial_joint(metric_seed, pairs, beta_seed)
        x = fit_joint(x0, pairs, C, obs, held, targetP, targetO)
        r = joint_residual(x, pairs, C, obs, held, targetP, targetO)
        roots.append({"seed": beta_seed.tolist(), "x": x, "mean_square_residual": float(np.mean(r * r))})
    roots.sort(key=lambda z: z["mean_square_residual"])
    best = roots[0]["x"]
    summary = []
    for z in roots:
        x = z["x"]
        summary.append({
            "beta_seed": z["seed"],
            "mean_square_residual": z["mean_square_residual"],
            "target_principal_point_px": x[4:6].tolist(),
            "target_focal_px": float(math.exp(float(x[3]))),
            "beta_px_per_focal_fraction": x[6:8].tolist(),
        })
    return best, summary


def static_metrics(x: np.ndarray, pairs: list[dict]) -> list[dict]:
    core = x[:6]
    beta = x[6:8]
    ft = math.exp(float(core[3]))
    rows = []
    off = 8
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        tr = np.linalg.norm(pair_project(pr["p"], core, beta, fs, rv) - pr["q"], axis=1)
        wh = np.linalg.norm(pair_project(pr["pw"], core, beta, fs, rv) - pr["qw"], axis=1)
        pps = source_pp(core, beta, fs)
        rows.append({
            "frame": pr["name"], "relative_seconds": pr["relative_seconds"],
            "source_focal_px": float(fs), "source_principal_point_px": pps.tolist(),
            "source_pp_drift_from_target_px": float(np.linalg.norm(pps - core[4:6])),
            "joint_train_points": int(len(pr["p"])), "heldout_points": int(len(pr["pw"])),
            "train_p95_px": float(np.percentile(tr, 95)),
            "heldout_median_px": float(np.median(wh)),
            "heldout_p95_px": float(np.percentile(wh, 95)),
        })
    return rows


def max_root_disagreement(roots: list[dict]) -> dict:
    pp = [np.asarray(r["target_principal_point_px"], dtype=np.float64) for r in roots]
    ff = [float(r["target_focal_px"]) for r in roots]
    ppd = 0.0
    ffd = 0.0
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            ppd = max(ppd, float(np.linalg.norm(pp[i] - pp[j])))
            ffd = max(ffd, abs(ff[i] - ff[j]) / max((ff[i] + ff[j]) * 0.5, 1e-9))
    return {"max_target_pp_pairwise_px": ppd, "max_target_focal_pairwise_fraction": ffd}


def subset_diagnostics(full: np.ndarray, pairs: list[dict], drop_ids: set[int], C: np.ndarray,
                       obs: dict[str, np.ndarray], held: dict[str, set[int]],
                       targetP: np.ndarray, targetO: np.ndarray) -> dict:
    subset = [p for i, p in enumerate(pairs) if i not in drop_ids]
    q = fit_joint(subset_warm(full, pairs, subset), subset, C, obs, held, targetP, targetO, max_nfev=3000)
    return {
        "dropped_frames": [pairs[i]["name"] for i in sorted(drop_ids)],
        "target_pp_shift_px": float(np.linalg.norm(q[4:6] - full[4:6])),
        "target_focal_fraction": float(abs(math.exp(float(q[3])) - math.exp(float(full[3]))) / math.exp(float(full[3]))),
    }


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
    targetP = np.asarray([
        [0, -10 * IN, 10 * FT + 16 * IN], [0, 10 * IN, 10 * FT + 16 * IN],
        [0, 10 * IN, 10 * FT + 2 * IN], [0, -10 * IN, 10 * FT + 2 * IN],
    ], dtype=np.float64)

    pairs = build_pairs(args.target_frame, args.samples, manifest)
    before = sum(p["relative_seconds"] < 0 for p in pairs)
    after = sum(p["relative_seconds"] > 0 for p in pairs)
    if len(pairs) < 6 or before < 2 or after < 2:
        raise RuntimeError(f"Insufficient exact-clip static support: n={len(pairs)}, before={before}, after={after}")

    metric_seed = v37.fit_metric("pinhole", C, Hfloor, obs, held, targetP, targetO)
    full, roots = solve_multistart(metric_seed, pairs, C, obs, held, targetP, targetO)
    root_agreement = max_root_disagreement(roots)
    core = full[:6]
    beta = full[6:8]
    rv, focal, _, cx, cy, _ = v37.unpack(core, "pinhole")
    metric = v37.metrics(core, "pinhole", C, obs, held, targetP, targetO)
    static = static_metrics(full, pairs)

    family_loo = {}
    for fam in obs:
        q = fit_joint(full, pairs, C, obs, held, targetP, targetO, exclude_family=fam, max_nfev=3000)
        pred = v37.project(q[:6], "pinhole", v37.CURVES[fam], C)
        ee = [float(np.min(np.linalg.norm(pred - x, axis=1))) for x in obs[fam]]
        family_loo[fam] = {"p95_px": float(np.percentile(ee, 95)), "max_px": float(max(ee))}

    static_loo = []
    for j, hold in enumerate(pairs):
        subset = [p for i, p in enumerate(pairs) if i != j]
        q = fit_joint(subset_warm(full, pairs, subset), subset, C, obs, held, targetP, targetO, max_nfev=3000)
        sm = static_metrics(q, subset)
        static_loo.append({
            "held_out_frame": hold["name"],
            "target_pp_shift_px": float(np.linalg.norm(q[4:6] - core[4:6])),
            "target_focal_fraction": float(abs(math.exp(float(q[3])) - focal) / focal),
            "max_remaining_source_pp_drift_px": max(x["source_pp_drift_from_target_px"] for x in sm),
        })

    # Explicit temporal blocks: early zoom ramp and later stable tail may not be
    # required for the target Frame C solution to remain stable.
    early_count = min(2, len(pairs) - 4)
    early_block = subset_diagnostics(full, pairs, set(range(early_count)), C, obs, held, targetP, targetO)
    late_block = subset_diagnostics(full, pairs, set(range(max(0, len(pairs) - 3), len(pairs))), C, obs, held, targetP, targetO)

    rng_metric = np.random.default_rng(20260903)
    rng_static = np.random.default_rng(390039)
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
        sm = static_metrics(q, ppairs)
        perturb.append({
            "trial": trial,
            "target_focal_fraction": float(abs(qf - focal) / focal),
            "target_pp_shift_px": float(math.hypot(qcx - cx, qcy - cy)),
            "target_rotation_deg": float(ang),
            "heldout_floor_p95_px": float(mm["floor_heldout_p95_px"]),
            "target_p95_px": float(mm["target_p95_px"]),
            "max_source_pp_drift_px": max(x["source_pp_drift_from_target_px"] for x in sm),
        })

    max_base_source_pp_drift = max(x["source_pp_drift_from_target_px"] for x in static)
    gates = {
        "metric_train_floor": bool(metric["floor_train_p95_px"] <= 2.5),
        "metric_heldout_floor": bool(metric["floor_heldout_p95_px"] <= 1.5),
        "metric_target": bool(metric["target_p95_px"] <= 1.0),
        "leave_floor_family": bool(max(x["p95_px"] for x in family_loo.values()) <= 2.5),
        "static_frames": bool(len(static) >= 6 and before >= 2 and after >= 2),
        "static_joint_train": bool(max(x["train_p95_px"] for x in static) <= 2.5),
        "static_heldout": bool(max(x["heldout_p95_px"] for x in static) <= 3.0),
        "root_agreement_pp": bool(root_agreement["max_target_pp_pairwise_px"] <= 0.5),
        "root_agreement_focal": bool(root_agreement["max_target_focal_pairwise_fraction"] <= 0.001),
        "leave_static_frame_pp": bool(max(x["target_pp_shift_px"] for x in static_loo) <= 5.0),
        "leave_static_frame_source_drift": bool(max(x["max_remaining_source_pp_drift_px"] for x in static_loo) <= 1.0),
        "early_zoom_block_pp": bool(early_block["target_pp_shift_px"] <= 5.0),
        "late_block_pp": bool(late_block["target_pp_shift_px"] <= 5.0),
        "base_source_pp_drift": bool(max_base_source_pp_drift <= 1.0),
        "perturb_floor": bool(max(x["heldout_floor_p95_px"] for x in perturb) <= 1.5),
        "perturb_target": bool(max(x["target_p95_px"] for x in perturb) <= 1.0),
        "perturb_focal": bool(max(x["target_focal_fraction"] for x in perturb) <= 0.005),
        "perturb_pp": bool(max(x["target_pp_shift_px"] for x in perturb) <= 10.0),
        "perturb_rotation": bool(max(x["target_rotation_deg"] for x in perturb) <= 0.8),
        "perturb_source_pp_drift": bool(max(x["max_source_pp_drift_px"] for x in perturb) <= 1.0),
    }
    passed = bool(all(gates.values()))

    report = {
        "schema_version": 1,
        "status": "PASS_ZOOM_STATE_EVENT_CAMERA_V39" if passed else "FAIL_ZOOM_STATE_EVENT_CAMERA_V39",
        "game_id": "0022500301", "event_id": 489, "camera_label": "Left Above Rim",
        "physical_center_cm": C.tolist(),
        "model": {
            "target_camera": "square-pixel pinhole with fixed v36 physical centre",
            "source_clip_state": "per-frame focal/rotation plus shared linear zoom-dependent effective principal-point drift",
            "source_pp_equation": "pp_s = pp_t + beta * ((f_s - f_t) / f_t)",
            "source_pp_drift_sigma_px": SOURCE_PP_DRIFT_SIGMA_PX,
            "beta_bound_px_per_focal_fraction": BETA_BOUND_PX_PER_FOCAL_FRACTION,
        },
        "camera": {"rvec": rv.tolist(), "focal_px": float(focal), "principal_point_px": [float(cx), float(cy)]},
        "zoom_pp_beta_px_per_focal_fraction": beta.tolist(),
        "metric": metric,
        "exact_clip_static": {
            "accepted_frames": len(static), "before": before, "after": after,
            "max_train_p95_px": max(x["train_p95_px"] for x in static),
            "max_heldout_p95_px": max(x["heldout_p95_px"] for x in static),
            "max_source_pp_drift_px": max_base_source_pp_drift,
            "frames": static,
        },
        "multistart": {"roots": roots, **root_agreement},
        "leave_one_floor_family_out": family_loo,
        "leave_one_static_frame_out": {
            "max_target_pp_shift_px": max(x["target_pp_shift_px"] for x in static_loo),
            "max_target_focal_fraction": max(x["target_focal_fraction"] for x in static_loo),
            "max_remaining_source_pp_drift_px": max(x["max_remaining_source_pp_drift_px"] for x in static_loo),
            "frames": static_loo,
        },
        "temporal_blocks": {"drop_early_zoom_ramp": early_block, "drop_late_tail": late_block},
        "combined_perturbation": {
            "trials": len(perturb), "manual_metric_amplitude_px": 0.5, "static_training_amplitude_px": 0.25,
            "max_target_pp_shift_px": max(x["target_pp_shift_px"] for x in perturb),
            "max_target_focal_fraction": max(x["target_focal_fraction"] for x in perturb),
            "max_target_rotation_deg": max(x["target_rotation_deg"] for x in perturb),
            "max_heldout_floor_p95_px": max(x["heldout_floor_p95_px"] for x in perturb),
            "max_target_p95_px": max(x["target_p95_px"] for x in perturb),
            "max_source_pp_drift_px": max(x["max_source_pp_drift_px"] for x in perturb),
            "trials_detail": perturb,
        },
        "v38_failure_addressed": "v38 passed 12/13 gates but holding out the earliest ~776.6px-focal zoom-ramp frame moved target PP 5.2198px. v39 models smooth source crop/optical-axis drift through zoom while retaining the unchanged 5px target-PP leave-out gate.",
        "gates": gates,
        "permissions": {"metric_event_camera_allowed": passed, "replay_render_allowed": False},
        "next_gate": "Calibrate a second distinct physical/metric camera at the immutable state; then refine exact multi-view state alignment before any static virtual-view render." if passed else "Do not promote. Diagnose the failed v39 gate without weakening thresholds.",
    }
    (args.out / "frame_c_left_above_rim_zoom_state_event_camera_v39.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    draw_overlay(args.target_frame, args.out / "frame_c_left_above_rim_zoom_state_overlay_v39.png", core, C, obs, targetP, targetO)
    print(json.dumps({
        "status": report["status"], "camera": report["camera"], "beta": report["zoom_pp_beta_px_per_focal_fraction"],
        "metric": metric, "static_max": report["exact_clip_static"], "multistart": report["multistart"],
        "static_loo": report["leave_one_static_frame_out"], "temporal_blocks": report["temporal_blocks"],
        "perturbation": {k: v for k, v in report["combined_perturbation"].items() if k != "trials_detail"},
        "gates": gates,
    }, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
