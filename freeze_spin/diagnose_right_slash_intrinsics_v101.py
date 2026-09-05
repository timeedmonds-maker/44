from __future__ import annotations

"""Right Slash v101: bidirectional fixed-centre transfer + rotational self-calibration.

Diagnostic only.  This does not promote a metric camera.  It reuses the existing
static-scene transfer gates, adds an independent reverse-direction consistency gate,
and then asks whether one shared principal point is identifiable across independent
same-game PTZ states.  Per-state focal length remains free.  No player/ball/body
landmarks are used.
"""

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

W, H = 960, 540
EVENT_RE = re.compile(r"event_(\d+)_frames$")


def action_core(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    return (x > .20 * W) & (x < .80 * W) & (y > .48 * H) & (y < .98 * H)


def sift_points(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=.015)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < .72 * n.distance]
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    return (np.float32([ka[m.queryIdx].pt for m in good]),
            np.float32([kb[m.trainIdx].pt for m in good])) if good else (
            np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))


def estats(e: np.ndarray) -> dict:
    if not len(e):
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None}
    return {"n": int(len(e)), "median_px": float(np.median(e)),
            "p90_px": float(np.percentile(e, 90)), "p95_px": float(np.percentile(e, 95))}


def xerr(M: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if not len(p):
        return np.empty(0)
    pred = cv2.perspectiveTransform(p[:, None, :], M)[:, 0]
    return np.linalg.norm(pred - q, axis=1)


def audit_one(source: np.ndarray, target: np.ndarray) -> dict:
    p, q = sift_points(source, target)
    rec = {"pass": False, "match_count": int(len(p))}
    if len(p) < 30:
        rec["status"] = "insufficient_matches"; return rec
    xa, ya = p[:, 0], p[:, 1]; xb, yb = q[:, 0], q[:, 1]
    train_geom = ((ya < .46 * H) | (xa < .14 * W) | (xa > .86 * W))
    train_geom &= ((yb < .46 * H) | (xb < .14 * W) | (xb > .86 * W))
    train = train_geom & ~action_core(p) & ~action_core(q)
    held = ~train & ~action_core(p) & ~action_core(q)
    if int(train.sum()) < 12:
        rec["status"] = "insufficient_background_training"; return rec
    M, mask = cv2.findHomography(p[train], q[train], cv2.RANSAC, 1.5,
                                 maxIters=30000, confidence=.999)
    if M is None or mask is None:
        rec["status"] = "homography_failed"; return rec
    ii = mask.ravel().astype(bool)
    tr = xerr(M, p[train][ii], q[train][ii]); wh = xerr(M, p[held], q[held])
    rec.update({"training_inliers": int(ii.sum()), "training_error": estats(tr),
                "withheld_error": estats(wh), "H": M.tolist()})
    gates = {
        "training_inliers_at_least_24": int(ii.sum()) >= 24,
        "training_p95_at_most_1_5px": len(tr) > 0 and np.percentile(tr, 95) <= 1.5,
        "withheld_matches_at_least_10": len(wh) >= 10,
        "withheld_median_at_most_2_5px": len(wh) > 0 and np.median(wh) <= 2.5,
        "withheld_p90_at_most_4px": len(wh) > 0 and np.percentile(wh, 90) <= 4.0,
    }
    rec["gates"] = gates; rec["pass"] = bool(all(gates.values()))
    rec["status"] = "pass" if rec["pass"] else "reject"
    return rec


def composition_error(Hst: np.ndarray, Hts: np.ndarray) -> dict:
    # Test inverse consistency over a deterministic static grid, excluding action core.
    xs = np.linspace(40, W - 40, 9); ys = np.linspace(35, H - 35, 6)
    pts = np.asarray([[x, y] for y in ys for x in xs], np.float32)
    pts = pts[~action_core(pts)]
    q = cv2.perspectiveTransform(pts[:, None, :], Hst)[:, 0]
    r = cv2.perspectiveTransform(q[:, None, :], Hts)[:, 0]
    e = np.linalg.norm(r - pts, axis=1)
    return {"median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95))}


def bidirectional(source_path: Path, target_path: Path) -> dict:
    s = cv2.imread(str(source_path)); t = cv2.imread(str(target_path))
    if s is None or t is None or s.shape[:2] != (H, W) or t.shape[:2] != (H, W):
        return {"pass": False, "status": "unreadable"}
    fwd = audit_one(s, t); rev = audit_one(t, s)
    rec = {"pass": False, "forward": fwd, "reverse": rev}
    if not fwd.get("pass") or not rev.get("pass"):
        rec["status"] = "one_direction_rejected"; return rec
    Hst = np.asarray(fwd["H"], float); Hts = np.asarray(rev["H"], float)
    comp = composition_error(Hst, Hts)
    rec["composition"] = comp
    rec["bidirectional_composition_median_at_most_1px"] = comp["median_px"] <= 1.0
    rec["bidirectional_composition_p95_at_most_2px"] = comp["p95_px"] <= 2.0
    rec["pass"] = bool(rec["bidirectional_composition_median_at_most_1px"] and
                       rec["bidirectional_composition_p95_at_most_2px"])
    rec["status"] = "bidirectional_pass" if rec["pass"] else "inverse_inconsistent"
    if rec["pass"]:
        rec["H_source_to_target"] = fwd["H"]
    return rec


def K(f: float, pp: np.ndarray) -> np.ndarray:
    return np.asarray([[f, 0, pp[0]], [0, f, pp[1]], [0, 0, 1.]], float)


def rotation_residual(x: np.ndarray, rows: list[dict]) -> np.ndarray:
    pp = x[:2]; ft = math.exp(float(x[2])); out = []
    for i, row in enumerate(rows):
        fs = math.exp(float(x[3 + i]))
        M = np.linalg.inv(K(ft, pp)) @ np.asarray(row["H_source_to_target"], float) @ K(fs, pp)
        det = float(np.linalg.det(M))
        if abs(det) < 1e-12:
            out.extend([100.] * 7); continue
        M = M / np.cbrt(abs(det))
        if det < 0: M = -M
        A = M.T @ M - np.eye(3)
        out.extend((5.0 * A[np.triu_indices(3)]).tolist())
        out.append(5.0 * (np.linalg.det(M) - 1.0))
    # broad physical regularisation only, same spirit as v32.
    out.extend([(pp[0] - W / 2) / 350., (pp[1] - H / 2) / 350.,
                (math.log(ft) - math.log(550.)) / 1.8])
    return np.asarray(out, float)


def selfcal(rows: list[dict], seed: tuple[float, float, float] | None = None) -> np.ndarray:
    n = len(rows)
    seeds = [seed] if seed else [(480.,270.,350.),(480.,330.,550.),(520.,300.,700.),
                                 (440.,300.,700.),(500.,290.,1000.)]
    lo = np.r_[0.,0.,math.log(150.), np.repeat(math.log(150.), n)]
    hi = np.r_[960.,540.,math.log(4000.), np.repeat(math.log(4000.), n)]
    best = None; best_score = float("inf")
    for sx, sy, sf in seeds:
        x0 = np.r_[sx, sy, math.log(sf), np.repeat(math.log(sf), n)]
        opt = least_squares(lambda z: rotation_residual(z, rows), x0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=1., x_scale="jac", max_nfev=12000)
        score = float(np.mean(rotation_residual(opt.x, rows) ** 2))
        if np.isfinite(score) and score < best_score:
            best_score, best = score, opt.x
    if best is None: raise RuntimeError("no rotational self-calibration root")
    return best


def radial_residual_test(row: dict, pp: np.ndarray, ft: float, fs: float) -> dict:
    # Compare the accepted projective H to the closest pure-rotation H under fitted K.
    Hm = np.asarray(row["H_source_to_target"], float)
    M = np.linalg.inv(K(ft, pp)) @ Hm @ K(fs, pp)
    det = np.linalg.det(M); M = M / np.cbrt(abs(det)); M = -M if det < 0 else M
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1.; R = U @ Vt
    Hr = K(ft, pp) @ R @ np.linalg.inv(K(fs, pp)); Hr /= Hr[2,2]
    xs = np.linspace(45, W - 45, 15); ys = np.linspace(35, H - 35, 10)
    p = np.asarray([[x,y] for y in ys for x in xs], np.float32); p = p[~action_core(p)]
    q0 = cv2.perspectiveTransform(p[:,None,:], Hm)[:,0]
    q1 = cv2.perspectiveTransform(p[:,None,:], Hr)[:,0]
    r = q1 - q0; v = q0 - pp; radius = np.linalg.norm(v, axis=1)
    u = v / np.maximum(radius[:,None], 1e-9)
    rr = np.sum(r * u, axis=1); tt = r[:,0] * (-u[:,1]) + r[:,1] * u[:,0]
    corr = float(np.corrcoef(radius, rr)[0,1]) if len(radius) > 2 else 0.
    slope = float(np.polyfit(radius, rr, 1)[0])
    return {"radial_corr": corr, "radial_slope_px_per_px": slope,
            "median_abs_radial_px": float(np.median(np.abs(rr))),
            "median_abs_tangential_px": float(np.median(np.abs(tt)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-event", type=int, default=540)
    ap.add_argument("--target-frame", default="f02.png")
    ap.add_argument("--min-independent-events", type=int, default=4)
    ap.add_argument("--max-loo-pp-shift-px", type=float, default=8.0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    target = args.bank / f"event_{args.target_event}_frames" / args.target_frame
    if not target.exists(): raise RuntimeError(f"missing target {target}")

    all_rows = []; best = []
    for d in sorted(args.bank.glob("event_*_frames")):
        m = EVENT_RE.search(d.name)
        if not m: continue
        eid = int(m.group(1))
        if eid == args.target_event: continue
        candidates = []
        for p in sorted(d.glob("f*.png")):
            z = bidirectional(p, target)
            rec = {"event_id": eid, "frame": p.name, "source": str(p), **z}
            all_rows.append(rec)
            if z.get("pass"):
                f = z["forward"]; score = (float(f["withheld_error"]["median_px"]),
                                              float(f["withheld_error"]["p90_px"]),
                                              -int(f["training_inliers"]))
                candidates.append((score, rec))
        if candidates:
            candidates.sort(key=lambda x: x[0]); best.append(candidates[0][1])

    status = "FAIL_RIGHT_SLASH_INTRINSICS_V101"
    report = {"schema_version": 1, "game_id": "0022500301", "camera_label": "Right Slash",
              "target": str(target), "method": "existing transfer gates + independent reverse transfer + inverse consistency + rotational homography self-calibration",
              "guardrail": "Diagnostic only; cannot promote a metric camera or render.",
              "all_frames": all_rows,
              "best_passing_state_per_event": best,
              "independent_passing_event_count": len(best)}
    if len(best) >= args.min_independent_events:
        x = selfcal(best); pp = x[:2]; ft = math.exp(float(x[2]))
        loo = []
        for eid in [r["event_id"] for r in best]:
            sub = [r for r in best if r["event_id"] != eid]
            y = selfcal(sub, seed=(float(pp[0]), float(pp[1]), ft))
            loo.append({"held_out_event": eid, "principal_point_px": y[:2].tolist(),
                        "shift_px": float(np.linalg.norm(y[:2] - pp))})
        max_loo = max(r["shift_px"] for r in loo)
        radial = []
        for i, row in enumerate(best):
            radial.append({"event_id": row["event_id"], **radial_residual_test(
                row, pp, ft, math.exp(float(x[3+i])))})
        slopes = np.asarray([r["radial_slope_px_per_px"] for r in radial])
        same_sign = float(max(np.mean(slopes > 0), np.mean(slopes < 0))) if len(slopes) else 0.
        coherent_distortion = bool(same_sign >= .75 and np.median(np.abs(slopes)) >= .002)
        gates = {
            "independent_events_at_least_minimum": len(best) >= args.min_independent_events,
            "leave_one_event_out_pp_shift_at_most_8px": max_loo <= args.max_loo_pp_shift_px,
        }
        passed = bool(all(gates.values()))
        status = "PASS_RIGHT_SLASH_INTRINSICS_PRIOR_V101" if passed else "FAIL_RIGHT_SLASH_INTRINSICS_V101"
        report.update({"status": status, "shared_principal_point_px": pp.tolist(),
                       "target_focal_px": ft, "source_focal_px": [float(math.exp(v)) for v in x[3:]],
                       "leave_one_event_out": loo, "max_leave_one_event_out_pp_shift_px": max_loo,
                       "radial_residual_diagnostic": radial,
                       "radial_slope_same_sign_fraction": same_sign,
                       "coherent_radial_distortion_pattern_detected": coherent_distortion,
                       "gates": gates, "principal_point_prior_allowed": passed,
                       "metric_event_camera_allowed": False, "replay_render_allowed": False})
    else:
        report.update({"status": status, "gates": {"independent_events_at_least_minimum": False},
                       "principal_point_prior_allowed": False,
                       "metric_event_camera_allowed": False, "replay_render_allowed": False})
    (args.out / "right_slash_intrinsics_v101.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ["status","independent_passing_event_count","shared_principal_point_px","target_focal_px","max_leave_one_event_out_pp_shift_px","radial_slope_same_sign_fraction","coherent_radial_distortion_pattern_detected","gates"]}, indent=2), flush=True)

if __name__ == "__main__":
    main()
