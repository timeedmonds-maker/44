from __future__ import annotations

"""Settled-optical-state exact Frame C event-camera proof (v40).

v39 passed every geometric/static/perturbation gate except multistart target
principal-point root agreement. The ambiguity came from asking one target camera
plus a zoom-dependent PP nuisance to explain both the early zoom ramp and the
settled Frame-C optical state.

v40 determines the settled optical cluster automatically from source-derived
pairwise focal estimates, fits Frame C only from that cluster with ONE shared
principal point, and treats non-settled early zoom frames as validation-only.
No v26 floor anchors or retired camera centre are used. Replay rendering remains
forbidden even if the event camera passes.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import prove_frame_c_left_above_rim_metric_camera_v37 as v37
from freeze_spin import prove_frame_c_left_above_rim_zoom_state_event_camera_v39 as v39

W, H = 960, 540
FT, IN = 30.48, 2.54
SETTLED_FOCAL_FRACTION = 0.010
EARLY_SOURCE_PP_BOUND_PX = 1.0
EARLY_SOURCE_PP_SIGMA_PX = 0.5


def pair_project_constant_pp(p: np.ndarray, core: np.ndarray, fs: float, rv: np.ndarray) -> np.ndarray:
    ft = math.exp(float(core[3]))
    pp = core[4:6]
    R, _ = cv2.Rodrigues(np.asarray(rv, dtype=np.float64).reshape(3, 1))
    hm = v37.k(ft, pp) @ R @ np.linalg.inv(v37.k(fs, pp))
    ph = np.column_stack([p, np.ones(len(p))])
    qh = (hm @ ph.T).T
    return qh[:, :2] / qh[:, 2:3]


def bounds_for(n_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    lo = np.r_[[-10.0] * 3, math.log(250.0), 100.0, 50.0]
    hi = np.r_[[10.0] * 3, math.log(2500.0), 850.0, 520.0]
    for _ in range(n_pairs):
        lo = np.r_[lo, math.log(150.0), [-10.0] * 3]
        hi = np.r_[hi, math.log(4000.0), [10.0] * 3]
    return lo, hi


def initial_joint(metric_core: np.ndarray, pairs: list[dict]) -> np.ndarray:
    parts = [np.asarray(metric_core, dtype=np.float64)]
    for pr in pairs:
        parts.append(v39.initialize_pair(pr, metric_core))
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
                   obs: dict[str, np.ndarray], held: dict[str, set[int]],
                   targetP: np.ndarray, targetO: np.ndarray,
                   exclude_family: str | None = None) -> np.ndarray:
    core = x[:6]
    out = [v39.metric_residual(core, C, obs, held, targetP, targetO, exclude_family)]
    off = 6
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        out.append((pair_project_constant_pp(pr["p"], core, fs, rv) - pr["q"]).ravel())
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
    seeds = [(0.0, 0.0), (-20.0, 0.0), (20.0, 0.0), (0.0, -20.0), (0.0, 20.0)]
    roots = []
    for dx, dy in seeds:
        core0 = np.asarray(metric_seed, dtype=np.float64).copy()
        core0[4] += dx
        core0[5] += dy
        x0 = initial_joint(core0, pairs)
        x = fit_joint(x0, pairs, C, obs, held, targetP, targetO)
        r = joint_residual(x, pairs, C, obs, held, targetP, targetO)
        roots.append({"seed_pp_offset_px": [dx, dy], "x": x, "mean_square_residual": float(np.mean(r * r))})
    roots.sort(key=lambda z: z["mean_square_residual"])
    summary = []
    for z in roots:
        x = z["x"]
        summary.append({
            "seed_pp_offset_px": z["seed_pp_offset_px"],
            "mean_square_residual": z["mean_square_residual"],
            "target_principal_point_px": x[4:6].tolist(),
            "target_focal_px": float(math.exp(float(x[3]))),
        })
    return roots[0]["x"], summary


def root_agreement(roots: list[dict]) -> dict:
    pp = [np.asarray(r["target_principal_point_px"], dtype=np.float64) for r in roots]
    ff = [float(r["target_focal_px"]) for r in roots]
    ppd = 0.0
    ffd = 0.0
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            ppd = max(ppd, float(np.linalg.norm(pp[i] - pp[j])))
            ffd = max(ffd, abs(ff[i] - ff[j]) / max((ff[i] + ff[j]) * 0.5, 1e-9))
    return {"max_target_pp_pairwise_px": ppd, "max_target_focal_pairwise_fraction": ffd}


def initial_focal_estimates(metric_seed: np.ndarray, pairs: list[dict]) -> list[dict]:
    rows = []
    for pr in pairs:
        x = v39.initialize_pair(pr, metric_seed)
        rows.append({"name": pr["name"], "relative_seconds": pr["relative_seconds"], "focal_px": float(math.exp(float(x[0])))})
    return rows


def select_settled(pairs: list[dict], estimates: list[dict]) -> tuple[list[dict], list[dict], dict]:
    by_name = {p["name"]: p for p in pairs}
    focals = np.asarray([x["focal_px"] for x in estimates], dtype=np.float64)
    median = float(np.median(focals))
    settled_names = [x["name"] for x in estimates if abs(x["focal_px"] - median) / median <= SETTLED_FOCAL_FRACTION]
    settled_set = set(settled_names)
    settled = [by_name[n] for n in settled_names]
    excluded = [by_name[x["name"]] for x in estimates if x["name"] not in settled_set]
    sel = {
        "method": "source-derived initial focal cluster around robust median",
        "settled_fraction_threshold": SETTLED_FOCAL_FRACTION,
        "median_initial_focal_px": median,
        "estimates": estimates,
        "settled_frames": settled_names,
        "validation_only_frames": [p["name"] for p in excluded],
    }
    return settled, excluded, sel


def static_metrics(x: np.ndarray, pairs: list[dict]) -> list[dict]:
    core = x[:6]
    rows = []
    off = 6
    for pr in pairs:
        fs = math.exp(float(x[off]))
        rv = x[off + 1:off + 4]
        off += 4
        tr = np.linalg.norm(pair_project_constant_pp(pr["p"], core, fs, rv) - pr["q"], axis=1)
        wh = np.linalg.norm(pair_project_constant_pp(pr["pw"], core, fs, rv) - pr["qw"], axis=1)
        rows.append({
            "frame": pr["name"], "relative_seconds": pr["relative_seconds"], "source_focal_px": float(fs),
            "joint_train_points": int(len(pr["p"])), "heldout_points": int(len(pr["pw"])),
            "train_p95_px": float(np.percentile(tr, 95)),
            "heldout_median_px": float(np.median(wh)), "heldout_p95_px": float(np.percentile(wh, 95)),
        })
    return rows


def fit_validation_nuisance(pr: dict, core: np.ndarray) -> dict:
    ft = math.exp(float(core[3]))
    pp = core[4:6]
    init = v39.initialize_pair(pr, core)
    x0 = np.r_[init, 0.0, 0.0]
    lo = np.r_[math.log(150.0), [-10.0] * 3, -EARLY_SOURCE_PP_BOUND_PX, -EARLY_SOURCE_PP_BOUND_PX]
    hi = np.r_[math.log(4000.0), [10.0] * 3, EARLY_SOURCE_PP_BOUND_PX, EARLY_SOURCE_PP_BOUND_PX]

    def project(p: np.ndarray, x: np.ndarray) -> np.ndarray:
        fs = math.exp(float(x[0]))
        rv = x[1:4]
        pps = pp + x[4:6]
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        hm = v37.k(ft, pp) @ R @ np.linalg.inv(v37.k(fs, pps))
        ph = np.column_stack([p, np.ones(len(p))])
        qh = (hm @ ph.T).T
        return qh[:, :2] / qh[:, 2:3]

    def fun(x: np.ndarray) -> np.ndarray:
        return np.r_[(project(pr["p"], x) - pr["q"]).ravel(), x[4:6] / EARLY_SOURCE_PP_SIGMA_PX]

    opt = least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=3000)
    x = np.asarray(opt.x, dtype=np.float64)
    tr = np.linalg.norm(project(pr["p"], x) - pr["q"], axis=1)
    wh = np.linalg.norm(project(pr["pw"], x) - pr["qw"], axis=1)
    return {
        "frame": pr["name"], "relative_seconds": pr["relative_seconds"],
        "source_focal_px": float(math.exp(float(x[0]))),
        "source_pp_nuisance_px": x[4:6].tolist(),
        "source_pp_nuisance_norm_px": float(np.linalg.norm(x[4:6])),
        "train_p95_px": float(np.percentile(tr, 95)),
        "heldout_median_px": float(np.median(wh)),
        "heldout_p95_px": float(np.percentile(wh, 95)),
    }


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

    all_pairs = v39.build_pairs(args.target_frame, args.samples, manifest)
    metric_seed = v37.fit_metric("pinhole", C, Hfloor, obs, held, targetP, targetO)
    estimates = initial_focal_estimates(metric_seed, all_pairs)
    settled, validation_only, selection = select_settled(all_pairs, estimates)
    before = sum(p["relative_seconds"] < 0 for p in settled)
    after = sum(p["relative_seconds"] > 0 for p in settled)
    if len(settled) < 6 or before < 2 or after < 2:
        raise RuntimeError(f"Insufficient settled optical support: n={len(settled)}, before={before}, after={after}")

    full, roots = solve_multistart(metric_seed, settled, C, obs, held, targetP, targetO)
    rag = root_agreement(roots)
    core = full[:6]
    rv, focal, _, cx, cy, _ = v37.unpack(core, "pinhole")
    metric = v37.metrics(core, "pinhole", C, obs, held, targetP, targetO)
    static = static_metrics(full, settled)
    early_validation = [fit_validation_nuisance(p, core) for p in validation_only if p["relative_seconds"] < 0]

    family_loo = {}
    for fam in obs:
        q = fit_joint(full, settled, C, obs, held, targetP, targetO, exclude_family=fam, max_nfev=3000)
        pred = v37.project(q[:6], "pinhole", v37.CURVES[fam], C)
        ee = [float(np.min(np.linalg.norm(pred - x, axis=1))) for x in obs[fam]]
        family_loo[fam] = {"p95_px": float(np.percentile(ee, 95)), "max_px": float(max(ee))}

    static_loo = []
    for j, hold in enumerate(settled):
        subset = [p for i, p in enumerate(settled) if i != j]
        q = fit_joint(subset_warm(full, settled, subset), subset, C, obs, held, targetP, targetO, max_nfev=3000)
        static_loo.append({
            "held_out_frame": hold["name"],
            "target_pp_shift_px": float(np.linalg.norm(q[4:6] - core[4:6])),
            "target_focal_fraction": float(abs(math.exp(float(q[3])) - focal) / focal),
        })

    pre_ids = {i for i, p in enumerate(settled) if p["relative_seconds"] < 0}
    late_ids = set(range(max(0, len(settled) - 3), len(settled)))

    def block(drop_ids: set[int]) -> dict:
        subset = [p for i, p in enumerate(settled) if i not in drop_ids]
        q = fit_joint(subset_warm(full, settled, subset), subset, C, obs, held, targetP, targetO, max_nfev=3000)
        return {
            "dropped_frames": [settled[i]["name"] for i in sorted(drop_ids)],
            "target_pp_shift_px": float(np.linalg.norm(q[4:6] - core[4:6])),
            "target_focal_fraction": float(abs(math.exp(float(q[3])) - focal) / focal),
        }

    pre_block = block(pre_ids)
    late_block = block(late_ids)

    rng_metric = np.random.default_rng(20260903)
    rng_static = np.random.default_rng(400040)
    Rb, _ = cv2.Rodrigues(rv.reshape(3, 1))
    perturb = []
    for trial in range(args.perturbation_trials):
        po = {k: v + rng_metric.uniform(-0.5, 0.5, v.shape) for k, v in obs.items()}
        to = targetO + rng_metric.uniform(-0.5, 0.5, targetO.shape)
        ppairs = []
        for pr in settled:
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
            "target_focal_fraction": float(abs(qf - focal) / focal),
            "target_pp_shift_px": float(math.hypot(qcx - cx, qcy - cy)),
            "target_rotation_deg": float(ang),
            "heldout_floor_p95_px": float(mm["floor_heldout_p95_px"]),
            "target_p95_px": float(mm["target_p95_px"]),
        })

    gates = {
        "metric_train_floor": bool(metric["floor_train_p95_px"] <= 2.5),
        "metric_heldout_floor": bool(metric["floor_heldout_p95_px"] <= 1.5),
        "metric_target": bool(metric["target_p95_px"] <= 1.0),
        "leave_floor_family": bool(max(x["p95_px"] for x in family_loo.values()) <= 2.5),
        "settled_frames": bool(len(static) >= 6 and before >= 2 and after >= 2),
        "settled_joint_train": bool(max(x["train_p95_px"] for x in static) <= 2.5),
        "settled_heldout": bool(max(x["heldout_p95_px"] for x in static) <= 3.0),
        "root_agreement_pp": bool(rag["max_target_pp_pairwise_px"] <= 0.5),
        "root_agreement_focal": bool(rag["max_target_focal_pairwise_fraction"] <= 0.001),
        "leave_settled_frame_pp": bool(max(x["target_pp_shift_px"] for x in static_loo) <= 5.0),
        "pre_freeze_settled_block_pp": bool(pre_block["target_pp_shift_px"] <= 5.0),
        "late_block_pp": bool(late_block["target_pp_shift_px"] <= 5.0),
        "early_ramp_validation_present": bool(len(early_validation) >= 2),
        "early_ramp_train": bool(early_validation and max(x["train_p95_px"] for x in early_validation) <= 2.5),
        "early_ramp_heldout": bool(early_validation and max(x["heldout_p95_px"] for x in early_validation) <= 3.0),
        "early_ramp_source_pp_nuisance": bool(early_validation and max(x["source_pp_nuisance_norm_px"] for x in early_validation) <= EARLY_SOURCE_PP_BOUND_PX + 1e-9),
        "perturb_floor": bool(max(x["heldout_floor_p95_px"] for x in perturb) <= 1.5),
        "perturb_target": bool(max(x["target_p95_px"] for x in perturb) <= 1.0),
        "perturb_focal": bool(max(x["target_focal_fraction"] for x in perturb) <= 0.005),
        "perturb_pp": bool(max(x["target_pp_shift_px"] for x in perturb) <= 10.0),
        "perturb_rotation": bool(max(x["target_rotation_deg"] for x in perturb) <= 0.8),
    }
    passed = bool(all(gates.values()))

    report = {
        "schema_version": 1,
        "status": "PASS_SETTLED_OPTICAL_EVENT_CAMERA_V40" if passed else "FAIL_SETTLED_OPTICAL_EVENT_CAMERA_V40",
        "game_id": "0022500301", "event_id": 489, "camera_label": "Left Above Rim",
        "physical_center_cm": C.tolist(),
        "selection": selection,
        "model": {
            "target_camera": "square-pixel pinhole with fixed v36 physical centre",
            "training_state": "automatically selected settled optical cluster only",
            "settled_source_state": "per-frame focal/rotation, target-equivalent shared principal point",
            "early_zoom_state": "validation-only; per-frame focal/rotation plus <=1 px source PP nuisance; cannot move target K",
        },
        "camera": {"rvec": rv.tolist(), "focal_px": float(focal), "principal_point_px": [float(cx), float(cy)]},
        "metric": metric,
        "settled_static": {
            "accepted_frames": len(static), "before": before, "after": after,
            "max_train_p95_px": max(x["train_p95_px"] for x in static),
            "max_heldout_p95_px": max(x["heldout_p95_px"] for x in static),
            "frames": static,
        },
        "early_zoom_validation": early_validation,
        "multistart": {"roots": roots, **rag},
        "leave_one_floor_family_out": family_loo,
        "leave_one_settled_frame_out": {
            "max_target_pp_shift_px": max(x["target_pp_shift_px"] for x in static_loo),
            "max_target_focal_fraction": max(x["target_focal_fraction"] for x in static_loo),
            "frames": static_loo,
        },
        "temporal_blocks": {"drop_pre_freeze_settled": pre_block, "drop_late_tail": late_block},
        "combined_perturbation": {
            "trials": len(perturb), "manual_metric_amplitude_px": 0.5, "settled_static_training_amplitude_px": 0.25,
            "max_target_pp_shift_px": max(x["target_pp_shift_px"] for x in perturb),
            "max_target_focal_fraction": max(x["target_focal_fraction"] for x in perturb),
            "max_target_rotation_deg": max(x["target_rotation_deg"] for x in perturb),
            "max_heldout_floor_p95_px": max(x["heldout_floor_p95_px"] for x in perturb),
            "max_target_p95_px": max(x["target_p95_px"] for x in perturb),
            "trials_detail": perturb,
        },
        "gates": gates,
        "permissions": {"metric_event_camera_allowed": passed, "replay_render_allowed": False},
        "next_gate": "Promote exact event camera and calibrate a second distinct physical camera" if passed else "If PP roots remain ambiguous, build v41 independent same-game settled-state intrinsics prior; do not relax gates.",
    }
    out_json = args.out / "frame_c_left_above_rim_settled_optical_event_camera_v40.json"
    out_png = args.out / "frame_c_left_above_rim_settled_optical_overlay_v40.png"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    v39.draw_overlay(args.target_frame, out_png, core, C, obs, targetP, targetO)
    print(json.dumps({
        "status": report["status"], "camera": report["camera"], "selection": selection,
        "multistart": report["multistart"], "metric": metric,
        "settled_static": report["settled_static"], "early_zoom_validation": early_validation,
        "temporal_blocks": report["temporal_blocks"],
        "perturbation": {k: v for k, v in report["combined_perturbation"].items() if k != "trials_detail"},
        "gates": gates,
    }, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
