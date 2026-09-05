from __future__ import annotations

"""v89: Broadcast shared-centre PTZ bundle calibration.

Target Frame C remains anchored only by regulation NBA floor paint and the white
backboard target lines used by v87. Same-game Broadcast frames are auxiliary
self-calibration evidence: if they come from the same physical PTZ centre, their
static-scene mapping to Frame C must be explainable by a rotation-induced
homography K_i R_i K_0^-1. Auxiliary states that cannot satisfy that model across
broad image regions are rejected before the bundle fit.

The orange rim remains completely held out from fitting and is used only as an
independent post-fit validation. This stage is fail-closed: no camera promotion
occurs unless target-frame projection, auxiliary shared-centre compatibility,
and the original 75 cm half-pixel physical-centre stability gate all pass.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44

FT = 30.48
EXPECTED_SHA256 = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"
CENTER_LIMIT_CM = 75.0
IMAGE_W, IMAGE_H = 960, 540


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def K_from_q(q: np.ndarray) -> np.ndarray:
    f = float(np.exp(q[0]))
    return np.array([[f, 0.0, q[1]], [0.0, f, q[2]], [0.0, 0.0, 1.0]], dtype=np.float64)


def nearest_rotation(A: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def rotation_homography(q: np.ndarray, state_p: np.ndarray) -> np.ndarray:
    K0 = K_from_q(q)
    f = float(np.exp(state_p[3]))
    Ki = np.array([[f, 0.0, q[1]], [0.0, f, q[2]], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = cv2.Rodrigues(np.asarray(state_p[:3], dtype=np.float64).reshape(3, 1))[0]
    return Ki @ R @ np.linalg.inv(K0)


def apply_H(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h = np.column_stack([xy, np.ones(len(xy), dtype=np.float64)]) @ H.T
    return h[:, :2] / h[:, 2:3]


def grid_spread(points: np.ndarray, cols: int = 6, rows: int = 4) -> int:
    cells = set()
    for x, y in np.asarray(points, dtype=np.float64):
        gx = min(cols - 1, max(0, int(x / IMAGE_W * cols)))
        gy = min(rows - 1, max(0, int(y / IMAGE_H * rows)))
        cells.add((gx, gy))
    return len(cells)


def spatial_balance(x: np.ndarray, y: np.ndarray, max_per_cell: int = 12, cols: int = 6, rows: int = 4) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (px, py) in enumerate(x):
        gx = min(cols - 1, max(0, int(px / IMAGE_W * cols)))
        gy = min(rows - 1, max(0, int(py / IMAGE_H * rows)))
        buckets.setdefault((gx, gy), []).append(i)
    keep = []
    for key in sorted(buckets):
        ids = buckets[key]
        if len(ids) <= max_per_cell:
            keep.extend(ids)
        else:
            choose = np.linspace(0, len(ids) - 1, max_per_cell).round().astype(int)
            keep.extend([ids[j] for j in choose])
    keep = np.asarray(sorted(set(keep)), dtype=int)
    return x[keep], y[keep]


def state_matches(target_kp, target_desc, source: Path) -> dict:
    gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (IMAGE_H, IMAGE_W):
        return {"status": "bad_image"}
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=10)
    kp, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return {"status": "no_descriptors"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(target_desc, desc, k=2)
    good = [m for m, n in knn if m.distance < 0.72 * n.distance]
    if len(good) < 80:
        return {"status": "insufficient_matches", "good_matches": len(good)}
    x = np.asarray([target_kp[m.queryIdx].pt for m in good], dtype=np.float64)
    y = np.asarray([kp[m.trainIdx].pt for m in good], dtype=np.float64)
    H, mask = cv2.findHomography(x, y, cv2.RANSAC, 2.5, maxIters=20000, confidence=0.999)
    if H is None or mask is None:
        return {"status": "homography_failed", "good_matches": len(good)}
    sel = mask.ravel().astype(bool)
    x, y = x[sel], y[sel]
    xb, yb = spatial_balance(x, y)
    return {
        "status": "ok",
        "good_matches": len(good),
        "ransac_inliers": int(len(x)),
        "grid_cells_6x4": grid_spread(x),
        "vertical_band_counts": [int(np.sum(x[:, 1] < 180.0)), int(np.sum((x[:, 1] >= 180.0) & (x[:, 1] < 360.0))), int(np.sum(x[:, 1] >= 360.0))],
        "H_init": H,
        "x": xb,
        "y": yb,
    }


def fit_state_fixed_q(obs: dict, q: np.ndarray) -> tuple[np.ndarray, dict]:
    x, y, H = obs["x"], obs["y"], obs["H_init"]
    K0 = K_from_q(q)
    A = np.linalg.inv(K0) @ H @ K0
    det = float(np.linalg.det(A))
    scale = np.cbrt(abs(det)) if abs(det) > 1e-12 else 1.0
    R0 = nearest_rotation(A / scale)
    rv0 = cv2.Rodrigues(R0)[0].ravel()
    p0 = np.r_[rv0, q[0]]
    lo = np.r_[[-0.8] * 3, math.log(500.0)]
    hi = np.r_[[0.8] * 3, math.log(5000.0)]
    def residual(p):
        return (apply_H(rotation_homography(q, p), x) - y).ravel()
    opt = least_squares(residual, p0, bounds=(lo, hi), loss="soft_l1", f_scale=1.25, x_scale="jac", max_nfev=2500)
    pred = apply_H(rotation_homography(q, opt.x), x)
    e = np.linalg.norm(pred - y, axis=1)
    return np.asarray(opt.x, dtype=np.float64), {
        "balanced_match_count": int(len(x)),
        "median_px": float(np.median(e)),
        "p90_px": float(np.percentile(e, 90)),
        "p95_px": float(np.percentile(e, 95)),
        "max_px": float(np.max(e)),
        "focal_px": float(np.exp(opt.x[3])),
    }


def metric_root(floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray], target_spec: dict) -> np.ndarray:
    roots = []
    for start in v87.pnp_starts(target_spec):
        try:
            p = v87.solve_warm(start, floor_train, target_obs, max_nfev=7000)
        except Exception:
            continue
        r = v87.data_residual(p, floor_train, target_obs)
        score = float(np.median(np.abs(r[:-8]))) if len(r) > 8 else float(np.median(np.abs(r)))
        C = v87.camera_center(p)
        if np.all(np.isfinite(p)) and np.all(np.isfinite(C)):
            roots.append((score, np.asarray(p, dtype=np.float64)))
    if not roots:
        raise RuntimeError("No v87 metric seed")
    roots.sort(key=lambda row: row[0])
    return roots[0][1]


def pack(target_p: np.ndarray, states: list[dict]) -> np.ndarray:
    return np.r_[target_p, *[s["p"] for s in states]]


def unpack(z: np.ndarray, nstates: int) -> tuple[np.ndarray, list[np.ndarray]]:
    p = np.asarray(z[:9], dtype=np.float64)
    states = [np.asarray(z[9 + 4*i: 13 + 4*i], dtype=np.float64) for i in range(nstates)]
    return p, states


def bundle_residual(z: np.ndarray, floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray], states: list[dict]) -> np.ndarray:
    p, sp = unpack(z, len(states))
    metric = v87.data_residual(p, floor_train, target_obs)
    out = [metric / math.sqrt(max(1, len(metric)))]
    q = np.asarray([p[6], p[7], p[8]], dtype=np.float64)
    for s, ps in zip(states, sp):
        e = (apply_H(rotation_homography(q, ps), s["x"]) - s["y"]).ravel()
        out.append(0.9 * e / math.sqrt(max(1.0, len(e) / 2.0)))
    return np.concatenate(out)


def bundle_bounds(nstates: int) -> tuple[np.ndarray, np.ndarray]:
    lo_t = np.r_[[-10.0] * 3, [-20000.0] * 3, math.log(500.0), 150.0, -500.0]
    hi_t = np.r_[[10.0] * 3, [20000.0] * 3, math.log(5000.0), 1600.0, 1300.0]
    lo_s = np.array([-0.8, -0.8, -0.8, math.log(500.0)], dtype=np.float64)
    hi_s = np.array([0.8, 0.8, 0.8, math.log(5000.0)], dtype=np.float64)
    return np.r_[lo_t, *([lo_s] * nstates)], np.r_[hi_t, *([hi_s] * nstates)]


def solve_bundle(z0: np.ndarray, floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray], states: list[dict], max_nfev: int = 3500) -> np.ndarray:
    lo, hi = bundle_bounds(len(states))
    opt = least_squares(lambda z: bundle_residual(z, floor_train, target_obs, states), z0, bounds=(lo, hi), loss="soft_l1", f_scale=0.8, x_scale="jac", max_nfev=max_nfev)
    return np.asarray(opt.x, dtype=np.float64)


def state_metrics(target_p: np.ndarray, states: list[dict], state_params: list[np.ndarray]) -> list[dict]:
    q = np.asarray([target_p[6], target_p[7], target_p[8]], dtype=np.float64)
    out = []
    for s, ps in zip(states, state_params):
        e = np.linalg.norm(apply_H(rotation_homography(q, ps), s["x"]) - s["y"], axis=1)
        out.append({
            "name": s["name"], "event_probe": s.get("event_probe"),
            "match_count": int(len(e)), "grid_cells_6x4": int(s["grid_cells_6x4"]),
            "median_px": float(np.median(e)), "p90_px": float(np.percentile(e,90)), "p95_px": float(np.percentile(e,95)),
            "focal_px": float(np.exp(ps[3])), "relative_rotation_deg": float(np.linalg.norm(ps[:3]) * 180.0 / math.pi),
        })
    return out


def rim_metrics(p: np.ndarray, rim_obs: np.ndarray) -> dict:
    theta = np.linspace(0.0, 2.0 * math.pi, 4000, endpoint=False)
    IN = 2.54
    P = np.column_stack([15.0*IN + 9.0*IN*np.cos(theta), 9.0*IN*np.sin(theta), np.full_like(theta, 10.0*FT)])
    proj, depth = v87.project3(p, P)
    if float(np.min(depth[:,2])) <= 20.0:
        return {"median_px": float("inf"), "p95_px": float("inf"), "max_px": float("inf")}
    d = np.sqrt(np.sum((rim_obs[:,None,:] - proj[None,:,:])**2, axis=2)).min(axis=1)
    return {"count": int(len(d)), "median_px": float(np.median(d)), "p95_px": float(np.percentile(d,95)), "max_px": float(np.max(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--target-lines", type=Path, required=True)
    ap.add_argument("--rim", type=Path, required=True)
    ap.add_argument("--states-json", type=Path, required=True)
    ap.add_argument("--states-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual = sha256(args.frame)
    if actual != EXPECTED_SHA256:
        raise RuntimeError(f"Immutable Broadcast target changed: {actual}")
    target_gray = cv2.imread(str(args.frame), cv2.IMREAD_GRAYSCALE)
    if target_gray is None or target_gray.shape != (IMAGE_H, IMAGE_W):
        raise RuntimeError("Expected immutable native 960x540 Broadcast Frame C")

    v85.patch_line_aware_geometry()
    floor_spec = json.loads(args.floor.read_text())
    target_spec = json.loads(args.target_lines.read_text())
    rim_spec = json.loads(args.rim.read_text())
    state_spec = json.loads(args.states_json.read_text())
    if floor_spec.get("camera_label") != "Broadcast" or target_spec.get("camera_label") != "Broadcast" or rim_spec.get("camera_label") != "Broadcast":
        raise RuntimeError("Broadcast provenance changed")
    if rim_spec.get("independence", {}).get("used_by_v87_fit") is not False:
        raise RuntimeError("Rim is not declared independent")

    floor_train, floor_held = v44.split_groups(floor_spec["observations_px"], floor_spec["held_out_indices"])
    target_obs = {k: np.asarray(target_spec["observed_line_samples_px"][k], dtype=np.float64) for k in v87.TARGET_KEYS}
    rim_obs = np.asarray(rim_spec["rim_contour_samples_px"], dtype=np.float64)
    seed = metric_root(floor_train, target_obs, target_spec)
    q_seed = np.asarray([seed[6], seed[7], seed[8]], dtype=np.float64)

    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=10)
    target_kp, target_desc = sift.detectAndCompute(target_gray, None)
    if target_desc is None:
        raise RuntimeError("No target descriptors")

    candidate_rows = []
    accepted = []
    for row in state_spec.get("top_candidates", []):
        rel = Path(row["selected_frame"])
        path = args.states_dir / rel.name
        obs = state_matches(target_kp, target_desc, path)
        rec = {"event_probe": row.get("event_probe"), "name": path.name, "match_status": obs.get("status")}
        if obs.get("status") == "ok":
            ps, fm = fit_state_fixed_q(obs, q_seed)
            rec.update({
                "ransac_inliers": obs["ransac_inliers"], "grid_cells_6x4": obs["grid_cells_6x4"],
                "vertical_band_counts": obs["vertical_band_counts"], "seed_rotation_model": fm,
            })
            compatible = bool(
                obs["ransac_inliers"] >= 150
                and obs["grid_cells_6x4"] >= 10
                and min(obs["vertical_band_counts"]) >= 12
                and fm["p95_px"] <= 3.0
            )
            rec["same_center_seed_compatible"] = compatible
            if compatible:
                obs.update({"name": path.name, "event_probe": row.get("event_probe"), "p": ps})
                accepted.append(obs)
        candidate_rows.append(rec)

    if len(accepted) < 4:
        report = {
            "schema_version": 1, "status": "FAIL_CLOSED_INSUFFICIENT_SHARED_CENTER_STATES_V89",
            "accepted_state_count": len(accepted), "candidate_states": candidate_rows,
            "permissions": {"broadcast_physical_camera_center_allowed": False, "broadcast_metric_event_camera_allowed": False, "broadcast_freeview_camera_allowed": False},
        }
        (args.out / "broadcast_shared_center_bundle_v89.json").write_text(json.dumps(report, indent=2))
        print("V89_FAIL_CLOSED", len(accepted), "same-center states")
        return

    z0 = pack(seed, accepted)
    z = solve_bundle(z0, floor_train, target_obs, accepted, max_nfev=5000)
    pb, sp = unpack(z, len(accepted))
    Cb = v87.camera_center(pb)
    state_nominal = state_metrics(pb, accepted, sp)
    floor_metrics = v44.pixel_metrics(v87.floor_homography(pb), floor_held, v44.dense_features())
    floor_max = float(max(r["p95_px"] for r in floor_metrics.values()))
    target_metrics = v87.target_metrics(pb, target_obs)
    target_max = float(max(r["p95_px"] for r in target_metrics.values()))
    rim_nominal = rim_metrics(pb, rim_obs)
    state_max = float(max(r["p95_px"] for r in state_nominal))

    rng = np.random.default_rng(890905)
    perturb = []
    for trial in range(args.perturbation_trials):
        fg = {k: floor_train[k] + rng.uniform(-0.5, 0.5, size=floor_train[k].shape) for k in v44.GROUPS}
        tg = {k: target_obs[k] + rng.uniform(-0.5, 0.5, size=target_obs[k].shape) for k in v87.TARGET_KEYS}
        sg = []
        for s, ps in zip(accepted, sp):
            ss = dict(s)
            ss["x"] = s["x"] + rng.uniform(-0.5, 0.5, size=s["x"].shape)
            ss["y"] = s["y"] + rng.uniform(-0.5, 0.5, size=s["y"].shape)
            ss["p"] = ps
            sg.append(ss)
        try:
            zp = solve_bundle(z, fg, tg, sg, max_nfev=900)
            pp, spp = unpack(zp, len(sg))
            Cp = v87.camera_center(pp)
            sh = v87.projection_shift(pb, pp)
            sm = state_metrics(pp, sg, spp)
            perturb.append({
                "trial": trial, "failed": False,
                "center_shift_cm": float(np.linalg.norm(Cp - Cb)),
                "projection_p95_shift_px": float(v87.max_p95(sh)),
                "state_max_p95_px": float(max(r["p95_px"] for r in sm)),
                "focal_shift_fraction": float(abs(np.exp(pp[6]) - np.exp(pb[6])) / np.exp(pb[6])),
                "principal_point_shift_px": float(np.linalg.norm(pp[7:9] - pb[7:9])),
            })
        except Exception as e:
            perturb.append({"trial": trial, "failed": True, "error": repr(e)})

    good = [r for r in perturb if not r.get("failed")]
    failures = len(perturb) - len(good)
    max_center = float(max((r["center_shift_cm"] for r in good), default=float("inf")))
    max_proj = float(max((r["projection_p95_shift_px"] for r in good), default=float("inf")))
    max_state = float(max((r["state_max_p95_px"] for r in good), default=float("inf")))

    gates = {
        "at_least_4_broad_same_center_states": len(accepted) >= 4,
        "nominal_auxiliary_rotation_model_p95_at_most_3px": state_max <= 3.0,
        "nominal_heldout_floor_p95_at_most_2px": floor_max <= 2.0,
        "nominal_target_line_p95_at_most_1_5px": target_max <= 1.5,
        "nominal_independent_rim_p95_at_most_1_5px": float(rim_nominal["p95_px"]) <= 1.5,
        "all_half_pixel_trials_converged": failures == 0 and len(good) == args.perturbation_trials,
        "half_pixel_projection_p95_at_most_2px": max_proj <= 2.0,
        "half_pixel_auxiliary_state_p95_at_most_4px": max_state <= 4.0,
        "half_pixel_center_shift_at_most_75cm": max_center <= CENTER_LIMIT_CM,
        "pinhole_only_no_distortion_parameters": True,
    }
    all_pass = bool(all(gates.values()))
    report = {
        "schema_version": 1,
        "status": "PASS_BROADCAST_SHARED_CENTER_BUNDLE_V89" if all_pass else "FAIL_CLOSED_BROADCAST_SHARED_CENTER_BUNDLE_V89",
        "game_id": "0022500301", "event_id": 489, "camera_label": "Broadcast",
        "immutable_frame_sha256": actual,
        "method": "metric Frame C floor+target lines jointly constrained by same-centre PTZ rotation homographies from broad same-game static-scene correspondences; orange rim held out",
        "seed_v87": {"focal_px": float(np.exp(seed[6])), "principal_point_px": seed[7:9].tolist(), "center_cm": v87.camera_center(seed).tolist()},
        "camera_candidate": {"focal_px": float(np.exp(pb[6])), "principal_point_px": pb[7:9].tolist(), "center_cm": Cb.tolist(), "center_ft": (Cb/FT).tolist()},
        "candidate_states": candidate_rows,
        "accepted_same_center_states": state_nominal,
        "nominal": {"heldout_floor": floor_metrics, "heldout_floor_max_p95_px": floor_max, "target_lines": target_metrics, "target_line_max_p95_px": target_max, "independent_rim": rim_nominal, "auxiliary_state_max_p95_px": state_max},
        "half_pixel_bundle_perturbation": {"trial_count": len(perturb), "converged_count": len(good), "failed_count": failures, "max_center_shift_cm": max_center, "max_projection_p95_shift_px": max_proj, "max_auxiliary_state_p95_px": max_state, "worst_center_trials": sorted(good, key=lambda r:r["center_shift_cm"], reverse=True)[:10], "all_trials": perturb},
        "gates": gates,
        "candidate_passes_all_gates": all_pass,
        "permissions": {
            "broadcast_physical_camera_center_allowed": all_pass,
            "broadcast_metric_event_camera_allowed": all_pass,
            "broadcast_freeview_camera_allowed": False,
            "four_camera_gate_allowed_from_v89_alone": False,
        },
    }
    (args.out / "broadcast_shared_center_bundle_v89.json").write_text(json.dumps(report, indent=2))
    print("V89_STATUS", report["status"])
    print("V89_ACCEPTED_STATES", len(accepted), [r["event_probe"] for r in state_nominal])
    print("V89_CAMERA", report["camera_candidate"])
    print("V89_NOMINAL floor", floor_max, "target", target_max, "rim", rim_nominal["p95_px"], "state", state_max)
    print("V89_PERT max_center_cm", max_center, "max_projection_px", max_proj, "max_state_px", max_state, "failures", failures)
    print("V89_GATES", gates)


if __name__ == "__main__":
    main()
