from __future__ import annotations

"""v90: strict two-state Broadcast shared-optical-centre proof.

Frame C remains the metric authority. A second same-game Broadcast state is
introduced only through immutable source pixels: a source-recomputed court-region
homography transfers the regulation floor-line observations, while elevated target
stripe samples are independently source-annotated and an unoccluded rim segment is
held out from every fit. Promotion is fail-closed and requires nominal accuracy,
multistart uniqueness, support-reduction stability, and 64 +/-0.5 px perturbations.
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
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87

EXPECTED_FRAME_C_SHA = "7cd80d1c24c9eefa025e50a55a7cf6cdc3d64ea1ac168ff66bb7aadb307d5b3c"
TARGET_KEYS = ("target_top", "target_left", "target_right")
RIM_RADIUS_CM = 9.0 * 2.54
RIM_CENTER_X_CM = 15.0 * 2.54
RIM_Z_CM = 10.0 * 30.48


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perspective_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float64)
    q = cv2.perspectiveTransform(a.astype(np.float32)[:, None, :], H.astype(np.float64))
    return q[:, 0, :].astype(np.float64)


def verify_source_homography(frame_c: Path, event_frame: Path, expected_H: np.ndarray) -> dict:
    a = cv2.imread(str(frame_c), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(event_frame), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != (540, 960) or b.shape != (540, 960):
        raise RuntimeError("v90 source images missing or not native 960x540")
    ma = np.zeros_like(a); mb = np.zeros_like(b)
    cv2.rectangle(ma, (0, 180), (820, 500), 255, -1)
    cv2.rectangle(mb, (0, 180), (820, 500), 255, -1)
    cv2.rectangle(ma, (0, 430), (260, 540), 0, -1)
    cv2.rectangle(mb, (0, 430), (260, 540), 0, -1)
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.015, edgeThreshold=10)
    ka, da = sift.detectAndCompute(a, ma)
    kb, db = sift.detectAndCompute(b, mb)
    if da is None or db is None:
        raise RuntimeError("v90 SIFT descriptors unavailable")
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(900155)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=10000, confidence=0.999)
    if H is None or mask is None:
        raise RuntimeError("v90 source homography failed")
    H = H / H[2, 2]
    inl = mask.ravel().astype(bool)
    pred = cv2.perspectiveTransform(src[inl, None, :], H)[:, 0, :]
    res = np.linalg.norm(pred - dst[inl], axis=1)
    gx, gy = np.meshgrid(np.linspace(80, 780, 16), np.linspace(200, 470, 10))
    grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    p1 = cv2.perspectiveTransform(grid[:, None, :], H)[:, 0, :]
    p2 = cv2.perspectiveTransform(grid[:, None, :], expected_H)[:, 0, :]
    d = np.linalg.norm(p1 - p2, axis=1)
    return {
        "ratio_matches": int(len(good)),
        "ransac_inliers": int(inl.sum()),
        "inlier_median_px": float(np.median(res)),
        "inlier_p95_px": float(np.percentile(res, 95)),
        "inlier_max_px": float(np.max(res)),
        "committed_H_grid_p95_delta_px": float(np.percentile(d, 95)),
        "recomputed_H": H.tolist(),
    }


def state_from_center(C: np.ndarray, rvec: np.ndarray, logf: float, cx: float, cy: float) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))[0]
    t = -R @ np.asarray(C, dtype=np.float64)
    return np.r_[rvec, t, logf, cx, cy]


def state_residual(p: np.ndarray, floor_train: dict[str, np.ndarray], target_obs: dict[str, np.ndarray]) -> np.ndarray:
    H = v87.floor_homography(p)
    rows = []
    for key in v44.GROUPS:
        if len(floor_train[key]):
            rows.append(v44.signed_pixel_residual(H, key, floor_train[key]))
    for key, P in v87.world_target_lines().items():
        if len(target_obs[key]):
            uv, _ = v87.project3(p, P)
            rows.append(v87.signed_line_distance(target_obs[key], uv[0], uv[1]))
    check = np.vstack([
        np.array([[v44.FT_X_CM, -v44.PAINT_HALF_CM, 0.0], [v44.FT_X_CM, v44.PAINT_HALF_CM, 0.0]], dtype=np.float64),
        np.vstack(list(v87.world_target_lines().values())),
    ])
    _, q = v87.project3(p, check)
    rows.append(np.minimum(q[:, 2] - 20.0, 0.0) / 5.0)
    return np.concatenate(rows)


def unpack_joint(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = z[:3]
    p1 = state_from_center(C, z[3:6], z[6], z[11], z[12])
    p2 = state_from_center(C, z[7:10], z[10], z[11], z[12])
    return C, p1, p2


def joint_residual(z: np.ndarray, f1, t1, f2, t2) -> np.ndarray:
    _, p1, p2 = unpack_joint(z)
    return np.r_[state_residual(p1, f1, t1), state_residual(p2, f2, t2)]


def joint_bounds() -> tuple[np.ndarray, np.ndarray]:
    lo = np.r_[[-20000.0] * 3, [-10.0] * 3, math.log(150.0), [-10.0] * 3, math.log(150.0), -2000.0, -2000.0]
    hi = np.r_[[20000.0] * 3, [10.0] * 3, math.log(8000.0), [10.0] * 3, math.log(8000.0), 3000.0, 3000.0]
    return lo, hi


def solve_joint(z0, f1, t1, f2, t2, max_nfev=12000):
    lo, hi = joint_bounds()
    opt = least_squares(
        lambda z: joint_residual(z, f1, t1, f2, t2),
        np.asarray(z0, dtype=np.float64), bounds=(lo, hi), loss="soft_l1",
        f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
    )
    return np.asarray(opt.x, dtype=np.float64), float(opt.cost)


def floor_metrics(p, held) -> tuple[dict, float]:
    m = v44.pixel_metrics(v87.floor_homography(p), held, v44.dense_features())
    return m, float(max(v["p95_px"] for v in m.values()))


def target_metrics(p, obs) -> tuple[dict, float]:
    m = v87.target_metrics(p, obs)
    return m, float(max(v["p95_px"] for v in m.values()))


def rim_world() -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, 2001, endpoint=False)
    return np.column_stack([
        RIM_CENTER_X_CM + RIM_RADIUS_CM * np.cos(th),
        RIM_RADIUS_CM * np.sin(th),
        np.full_like(th, RIM_Z_CM),
    ])


def rim_metrics(p: np.ndarray, obs: np.ndarray) -> dict:
    uv, _ = v87.project3(p, rim_world())
    d = np.sqrt(np.sum((np.asarray(obs)[:, None, :] - uv[None, :, :]) ** 2, axis=2)).min(axis=1)
    return {
        "count": int(len(d)), "median_px": float(np.median(d)),
        "p95_px": float(np.percentile(d, 95)), "max_px": float(np.max(d)),
        "per_point_px": d.tolist(),
    }


def find_frame_c_seed(floor_train, target_obs, target_spec) -> np.ndarray:
    roots = []
    for start in v87.pnp_starts(target_spec):
        try:
            p = v87.solve_warm(start, floor_train, target_obs, max_nfev=7000)
        except Exception:
            continue
        r = state_residual(p, floor_train, target_obs)
        score = float(np.median(np.abs(r)))
        C = v87.camera_center(p)
        if np.all(np.isfinite(p)) and np.all(np.isfinite(C)):
            roots.append((score, p))
    if not roots:
        raise RuntimeError("v90 could not reproduce Frame C v87 seed")
    roots.sort(key=lambda x: x[0])
    return roots[0][1]


def evaluate_state(p, held, target_obs, rim_obs) -> dict:
    fm, fmax = floor_metrics(p, held)
    tm, tmax = target_metrics(p, target_obs)
    rm = rim_metrics(p, rim_obs)
    return {"floor": fm, "floor_max_p95_px": fmax, "target": tm, "target_max_p95_px": tmax, "rim": rm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-c", type=Path, required=True)
    ap.add_argument("--event-frame", type=Path, required=True)
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--target-c", type=Path, required=True)
    ap.add_argument("--rim-c", type=Path, required=True)
    ap.add_argument("--event-spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--perturbation-trials", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if sha256(args.frame_c) != EXPECTED_FRAME_C_SHA:
        raise RuntimeError("immutable Broadcast Frame C changed")
    event_spec = json.loads(args.event_spec.read_text())
    if sha256(args.event_frame) != event_spec["image_sha256"]:
        raise RuntimeError("immutable event155 v89 frame changed")

    v85.patch_line_aware_geometry()
    floor_spec = json.loads(args.floor.read_text())
    target_c_spec = json.loads(args.target_c.read_text())
    rim_c_spec = json.loads(args.rim_c.read_text())
    H_event = np.asarray(event_spec["court_region_homography_from_frame_c"]["frame_c_to_event155_H"], dtype=np.float64)
    source_H = verify_source_homography(args.frame_c, args.event_frame, H_event)

    train_c, held_c = v44.split_groups(floor_spec["observations_px"], floor_spec["held_out_indices"])
    event_floor_all = {k: perspective_points(H_event, np.asarray(v, dtype=np.float64)) for k, v in floor_spec["observations_px"].items()}
    train_e, held_e = v44.split_groups(event_floor_all, floor_spec["held_out_indices"])
    target_c = {k: np.asarray(target_c_spec["observed_line_samples_px"][k], dtype=np.float64) for k in TARGET_KEYS}
    target_e = {k: np.asarray(event_spec["observed_target_line_samples_px"][k], dtype=np.float64) for k in TARGET_KEYS}
    rim_c = np.asarray(rim_c_spec["rim_contour_samples_px"], dtype=np.float64)
    rim_e = np.asarray(event_spec["heldout_visible_rim_samples_px"], dtype=np.float64)

    pc = find_frame_c_seed(train_c, target_c, target_c_spec)
    pe = v87.solve_warm(pc, train_e, target_e, max_nfev=10000)
    C0 = (v87.camera_center(pc) + v87.camera_center(pe)) / 2.0
    z0 = np.r_[C0, pc[:3], pc[6], pe[:3], pe[6], (pc[7:9] + pe[7:9]) / 2.0]
    z, cost = solve_joint(z0, train_c, target_c, train_e, target_e, max_nfev=20000)
    C, p1, p2 = unpack_joint(z)
    nominal_c = evaluate_state(p1, held_c, target_c, rim_c)
    nominal_e = evaluate_state(p2, held_e, target_e, rim_e)

    mods = [
        ([0,0,0],1.0,1.0,[0,0]), ([300,0,0],0.8,1.2,[80,-40]),
        ([-300,0,0],1.2,0.8,[-80,40]), ([0,300,0],0.7,0.7,[120,80]),
        ([0,-300,0],1.4,1.4,[-120,-80]), ([0,0,200],0.9,1.1,[0,120]),
        ([0,0,-200],1.1,0.9,[0,-120]), ([200,200,100],0.6,1.5,[150,-100]),
    ]
    roots = []
    for i, (dc, m1, m2, dpp) in enumerate(mods):
        s = z.copy(); s[:3] += np.asarray(dc, float)
        s[6] = math.log(float(np.exp(z[6]) * m1)); s[10] = math.log(float(np.exp(z[10]) * m2))
        s[11:13] += np.asarray(dpp, float)
        zz, cc = solve_joint(s, train_c, target_c, train_e, target_e, max_nfev=20000)
        Cr, a, b = unpack_joint(zz)
        roots.append({
            "index": i, "cost": cc, "center_cm": Cr.tolist(),
            "center_shift_from_nominal_cm": float(np.linalg.norm(Cr-C)),
            "principal_point_px": zz[11:13].tolist(),
            "focal_frame_c_px": float(np.exp(a[6])), "focal_event155_px": float(np.exp(b[6])),
        })
    pairwise = [
        float(np.linalg.norm(np.asarray(roots[i]["center_cm"])-np.asarray(roots[j]["center_cm"])))
        for i in range(len(roots)) for j in range(i+1, len(roots))
    ]

    support = []
    families = [("c_floor", k) for k in v44.GROUPS] + [("c_target", k) for k in TARGET_KEYS] + [("e_floor", k) for k in v44.GROUPS] + [("e_target", k) for k in TARGET_KEYS]
    for typ, key in families:
        f1 = {k: v.copy() for k, v in train_c.items()}; t1 = {k: v.copy() for k, v in target_c.items()}
        f2 = {k: v.copy() for k, v in train_e.items()}; t2 = {k: v.copy() for k, v in target_e.items()}
        if typ == "c_floor": f1[key] = np.empty((0,2), dtype=float)
        if typ == "c_target": t1[key] = np.empty((0,2), dtype=float)
        if typ == "e_floor": f2[key] = np.empty((0,2), dtype=float)
        if typ == "e_target": t2[key] = np.empty((0,2), dtype=float)
        zz, _ = solve_joint(z, f1, t1, f2, t2, max_nfev=12000)
        Cs, a, b = unpack_joint(zz)
        ec = evaluate_state(a, held_c, target_c, rim_c); ee = evaluate_state(b, held_e, target_e, rim_e)
        support.append({
            "dropped_family": f"{typ}:{key}", "center_shift_cm": float(np.linalg.norm(Cs-C)),
            "frame_c_floor_p95_px": ec["floor_max_p95_px"], "frame_c_target_p95_px": ec["target_max_p95_px"],
            "event155_floor_p95_px": ee["floor_max_p95_px"], "event155_target_p95_px": ee["target_max_p95_px"],
            "frame_c_rim_p95_px": ec["rim"]["p95_px"], "event155_rim_p95_px": ee["rim"]["p95_px"],
        })

    rng = np.random.default_rng(900105)
    pert = []
    for trial in range(args.perturbation_trials):
        f1 = {k: train_c[k] + rng.uniform(-0.5,0.5,size=train_c[k].shape) for k in v44.GROUPS}
        t1 = {k: target_c[k] + rng.uniform(-0.5,0.5,size=target_c[k].shape) for k in TARGET_KEYS}
        f2 = {k: train_e[k] + rng.uniform(-0.5,0.5,size=train_e[k].shape) for k in v44.GROUPS}
        t2 = {k: target_e[k] + rng.uniform(-0.5,0.5,size=target_e[k].shape) for k in TARGET_KEYS}
        try:
            zz, _ = solve_joint(z, f1, t1, f2, t2, max_nfev=7000)
        except Exception:
            pert.append({"trial": trial, "failed": True}); continue
        Cp, a, b = unpack_joint(zz)
        ec = evaluate_state(a, held_c, target_c, rim_c); ee = evaluate_state(b, held_e, target_e, rim_e)
        sh1 = v87.max_p95(v87.projection_shift(p1, a)); sh2 = v87.max_p95(v87.projection_shift(p2, b))
        pert.append({
            "trial": trial, "failed": False, "center_shift_cm": float(np.linalg.norm(Cp-C)),
            "principal_point_shift_px": float(np.linalg.norm(zz[11:13]-z[11:13])),
            "max_projection_p95_shift_px": float(max(sh1, sh2)),
            "frame_c_floor_p95_px": ec["floor_max_p95_px"], "frame_c_target_p95_px": ec["target_max_p95_px"], "frame_c_rim_p95_px": ec["rim"]["p95_px"],
            "event155_floor_p95_px": ee["floor_max_p95_px"], "event155_target_p95_px": ee["target_max_p95_px"], "event155_rim_p95_px": ee["rim"]["p95_px"],
        })
    good = [r for r in pert if not r.get("failed")]

    max_support_center = max(r["center_shift_cm"] for r in support)
    max_support_floor = max(max(r["frame_c_floor_p95_px"], r["event155_floor_p95_px"]) for r in support)
    max_support_target = max(max(r["frame_c_target_p95_px"], r["event155_target_p95_px"]) for r in support)
    max_pert_center = max((r["center_shift_cm"] for r in good), default=float("inf"))
    max_pert_proj = max((r["max_projection_p95_shift_px"] for r in good), default=float("inf"))
    max_pert_floor = max((max(r["frame_c_floor_p95_px"], r["event155_floor_p95_px"]) for r in good), default=float("inf"))
    max_pert_target = max((max(r["frame_c_target_p95_px"], r["event155_target_p95_px"]) for r in good), default=float("inf"))
    max_pert_rim_c = max((r["frame_c_rim_p95_px"] for r in good), default=float("inf"))
    max_pert_rim_e = max((r["event155_rim_p95_px"] for r in good), default=float("inf"))

    gates = {
        "source_homography_ratio_matches_at_least_100": source_H["ratio_matches"] >= 100,
        "source_homography_inliers_at_least_80": source_H["ransac_inliers"] >= 80,
        "source_homography_inlier_p95_at_most_2px": source_H["inlier_p95_px"] <= 2.0,
        "source_homography_reproduces_committed_H_at_most_0_5px": source_H["committed_H_grid_p95_delta_px"] <= 0.5,
        "frame_c_nominal_floor_p95_at_most_2px": nominal_c["floor_max_p95_px"] <= 2.0,
        "frame_c_nominal_target_p95_at_most_1_5px": nominal_c["target_max_p95_px"] <= 1.5,
        "frame_c_independent_rim_p95_at_most_1_5px": nominal_c["rim"]["p95_px"] <= 1.5,
        "event155_nominal_floor_p95_at_most_2px": nominal_e["floor_max_p95_px"] <= 2.0,
        "event155_nominal_target_p95_at_most_1_5px": nominal_e["target_max_p95_px"] <= 1.5,
        "event155_independent_visible_rim_p95_at_most_2px": nominal_e["rim"]["p95_px"] <= 2.0,
        "eight_wide_multistarts_converge_same_center_within_5cm": len(roots) == 8 and max(pairwise) <= 5.0,
        "support_reduction_center_shift_at_most_75cm": max_support_center <= 75.0,
        "support_reduction_heldout_floor_p95_at_most_2_5px": max_support_floor <= 2.5,
        "support_reduction_target_p95_at_most_2px": max_support_target <= 2.0,
        "all_64_half_pixel_trials_converged": len(good) == args.perturbation_trials,
        "half_pixel_center_shift_at_most_75cm": max_pert_center <= 75.0,
        "half_pixel_projection_p95_at_most_2px": max_pert_proj <= 2.0,
        "half_pixel_heldout_floor_p95_at_most_2_5px": max_pert_floor <= 2.5,
        "half_pixel_target_p95_at_most_1_5px": max_pert_target <= 1.5,
        "half_pixel_frame_c_rim_p95_at_most_1_5px": max_pert_rim_c <= 1.5,
        "half_pixel_event155_rim_p95_at_most_2px": max_pert_rim_e <= 2.0,
        "pinhole_only_no_brown_distortion": True,
    }
    passed = bool(all(gates.values()))
    report = {
        "status": "PASS_BROADCAST_SHARED_OPTICAL_CENTER_V90" if passed else "FAIL_BROADCAST_SHARED_OPTICAL_CENTER_V90",
        "game_id": "0022500301", "camera_label": "Broadcast",
        "method": "two immutable same-game Broadcast states; exact shared optical centre + exact shared principal point; state-specific rotation/focal; pinhole only",
        "shared_camera_center_cm": C.tolist(), "shared_camera_center_ft": (C/30.48).tolist(),
        "shared_principal_point_px": z[11:13].tolist(), "frame_c_focal_px": float(np.exp(p1[6])), "event155_focal_px": float(np.exp(p2[6])),
        "nominal_cost": cost, "source_homography": source_H,
        "nominal": {"frame_c": nominal_c, "event155": nominal_e},
        "multistart": {"roots": roots, "max_pairwise_center_spread_cm": max(pairwise)},
        "support_reduction": {"trials": support, "max_center_shift_cm": max_support_center, "max_heldout_floor_p95_px": max_support_floor, "max_target_p95_px": max_support_target},
        "half_pixel_perturbation": {
            "trial_count": args.perturbation_trials, "converged_count": len(good),
            "max_center_shift_cm": max_pert_center, "max_projection_p95_shift_px": max_pert_proj,
            "max_heldout_floor_p95_px": max_pert_floor, "max_target_p95_px": max_pert_target,
            "max_frame_c_rim_p95_px": max_pert_rim_c, "max_event155_rim_p95_px": max_pert_rim_e,
            "trials": pert,
        },
        "gates": gates,
        "permissions": {
            "broadcast_frame_c_metric_event_camera_allowed": passed,
            "broadcast_physical_camera_center_allowed": passed,
            "broadcast_counts_as_distinct_metric_camera": passed,
            "static_four_camera_free_view_allowed": False,
            "replay_render_allowed": False,
        },
        "guardrail": "Passing v90 promotes Broadcast as a physically identified event camera only. It does not authorize a four-camera replay until two additional distinct physical cameras independently pass their own metric proofs and cross-camera basket/court QA.",
    }
    (args.out / "broadcast_shared_optical_center_v90.json").write_text(json.dumps(report, indent=2) + "\n")
    print("status", report["status"])
    print("center_ft", report["shared_camera_center_ft"])
    print("nominal_frame_c", nominal_c["floor_max_p95_px"], nominal_c["target_max_p95_px"], nominal_c["rim"]["p95_px"])
    print("nominal_event155", nominal_e["floor_max_p95_px"], nominal_e["target_max_p95_px"], nominal_e["rim"]["p95_px"])
    print("multistart_center_spread_cm", max(pairwise))
    print("support_max_center_cm", max_support_center)
    print("perturb_max_center_cm", max_pert_center)
    print("perturb_max_projection_p95_px", max_pert_proj)
    print("perturb_max_floor_p95_px", max_pert_floor)
    print("perturb_max_target_p95_px", max_pert_target)
    print("perturb_max_event155_rim_p95_px", max_pert_rim_e)
    print("gates", gates)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
