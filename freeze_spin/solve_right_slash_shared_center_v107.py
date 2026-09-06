from __future__ import annotations

"""v107 shared-centre metric solver for the Right Slash physical camera.

The solver is deliberately observation-driven. It accepts only native-pixel
regulation geometry supplied in a JSON spec. Projective state registration may
supply initialization, but it is never consumed as metric residual evidence.

Model: one 3-D camera centre; per-state rotation and focal length; one bounded
principal point shared by the states. Floor families use the v44/v90 implicit
inverse-homography signed-pixel residual, avoiding dense curve projection in the
optimizer. Target lines provide non-coplanar metric support. Rims/boards may be
kept outside the fit and scored by the caller/auditor.

This file intentionally does not promote Camera #4. It reports nominal roots,
leave-one-family-out centre shifts, state-out centre shifts, and deterministic
+/-0.5 px perturbation stability. Promotion remains a separate fail-closed audit.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

FT = 30.48
IN = 2.54
RIM_X = 15.0 * IN
FT_X = 15.0 * FT
FT_R = 6.0 * FT
RESTRICT_R = 4.0 * FT
THREE_R = 23.75 * FT
PAINT_HALF = 8.0 * FT
TARGET_HALF_W = 11.0 * IN
TARGET_BOTTOM_Z = 10.0 * FT + 1.0 * IN
TARGET_TOP_Z = 10.0 * FT + 17.0 * IN

FLOOR_KEYS = {
    "three_point_arc",
    "free_throw_front_semicircle",
    "free_throw_line",
    "restricted_arc",
    "lane_negative_y",
    "lane_positive_y",
}
TARGET_KEYS = ("target_top", "target_left", "target_right")


def json_safe(v):
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def target_world():
    return {
        "target_top": np.asarray([[0.0, -TARGET_HALF_W, TARGET_TOP_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]]),
        "target_left": np.asarray([[0.0, -TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, -TARGET_HALF_W, TARGET_TOP_Z]]),
        "target_right": np.asarray([[0.0, TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]]),
    }


TARGET_WORLD = target_world()


def signed_line_distance(obs, a, b):
    d = np.asarray(b) - np.asarray(a)
    n = np.asarray([-d[1], d[0]], dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return (np.asarray(obs, dtype=np.float64) - np.asarray(a, dtype=np.float64)) @ n


def state(C, rvec, logf, cx, cy):
    return np.r_[np.asarray(C, dtype=np.float64), float(logf), float(cx), float(cy), np.asarray(rvec, dtype=np.float64)]


def project(p, P):
    C = np.asarray(p[:3], dtype=np.float64)
    f = float(np.exp(p[3]))
    cx, cy = map(float, p[4:6])
    R = cv2.Rodrigues(np.asarray(p[6:9], dtype=np.float64).reshape(3, 1))[0]
    Q = (R @ (np.asarray(P, dtype=np.float64) - C).T).T
    uv = np.column_stack([f * Q[:, 0] / Q[:, 2] + cx, f * Q[:, 1] / Q[:, 2] + cy])
    return uv, Q[:, 2]


def floor_homography(p):
    C = np.asarray(p[:3], dtype=np.float64)
    f = float(np.exp(p[3]))
    cx, cy = map(float, p[4:6])
    R = cv2.Rodrigues(np.asarray(p[6:9], dtype=np.float64).reshape(3, 1))[0]
    K = np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])


def project_h(Hm, xy):
    xy = np.asarray(xy, dtype=np.float64)
    q = (Hm @ np.column_stack([xy, np.ones(len(xy))]).T).T
    return q[:, :2] / q[:, 2:3]


def implicit_world_value(Hm, key, pixels):
    world = project_h(np.linalg.inv(Hm), pixels)
    if key == "three_point_arc":
        return np.hypot(world[:, 0] - RIM_X, world[:, 1]) - THREE_R
    if key == "free_throw_front_semicircle":
        return np.hypot(world[:, 0] - FT_X, world[:, 1]) - FT_R
    if key == "restricted_arc":
        return np.hypot(world[:, 0] - RIM_X, world[:, 1]) - RESTRICT_R
    if key == "free_throw_line":
        return world[:, 0] - FT_X
    if key == "lane_negative_y":
        return world[:, 1] + PAINT_HALF
    if key == "lane_positive_y":
        return world[:, 1] - PAINT_HALF
    raise KeyError(key)


def signed_pixel_residual(Hm, key, pixels):
    pixels = np.asarray(pixels, dtype=np.float64)
    eps = 0.25
    f = implicit_world_value(Hm, key, pixels)
    gx = (implicit_world_value(Hm, key, pixels + [eps, 0.0]) - implicit_world_value(Hm, key, pixels - [eps, 0.0])) / (2.0 * eps)
    gy = (implicit_world_value(Hm, key, pixels + [0.0, eps]) - implicit_world_value(Hm, key, pixels - [0.0, eps])) / (2.0 * eps)
    return f / np.maximum(np.hypot(gx, gy), 1e-6)


def lookat_rvec(C, aim=np.asarray([200.0, 0.0, 180.0])):
    C = np.asarray(C, dtype=np.float64)
    z = np.asarray(aim, dtype=np.float64) - C
    z /= np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.asarray([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return cv2.Rodrigues(np.vstack([x, y, z]))[0].ravel()


def validate_spec(spec):
    states = [str(s) for s in spec["states"]]
    if len(states) < 3 or len(set(states)) != len(states):
        raise RuntimeError("v107 needs at least three unique states")
    for s in states:
        ob = spec["observations"][s]
        if set(ob.get("target", {})) != set(TARGET_KEYS):
            raise RuntimeError(f"state {s}: all three target sides are required")
        for k, pts in ob.get("floor", {}).items():
            if k not in FLOOR_KEYS:
                raise RuntimeError(f"state {s}: unsupported floor family {k}")
            if len(pts) < 3:
                raise RuntimeError(f"state {s}: insufficient {k} support")
    return states


def unpack(z, states):
    C = np.asarray(z[:3], dtype=np.float64)
    cx, cy = map(float, z[3:5])
    off = 5
    out = {}
    for s in states:
        logf = z[off]
        rvec = z[off + 1:off + 4]
        off += 4
        out[s] = state(C, rvec, logf, cx, cy)
    return C, cx, cy, out


def bounds(states, spec):
    b = spec.get("bounds", {})
    C_lo = b.get("center_lo_cm", [500.0, -3000.0, 50.0])
    C_hi = b.get("center_hi_cm", [4500.0, 2500.0, 1500.0])
    pp_lo = b.get("principal_point_lo_px", [200.0, 80.0])
    pp_hi = b.get("principal_point_hi_px", [800.0, 480.0])
    f_lo, f_hi = b.get("focal_px", [400.0, 5000.0])
    lo = list(C_lo) + list(pp_lo)
    hi = list(C_hi) + list(pp_hi)
    for _ in states:
        lo += [math.log(float(f_lo)), -10.0, -10.0, -10.0]
        hi += [math.log(float(f_hi)), 10.0, 10.0, 10.0]
    return np.asarray(lo), np.asarray(hi)


def make_seed(states, spec, rng=None):
    seed = spec.get("seed", {})
    C = np.asarray(seed.get("center_cm", [2276.0, -788.0, 201.0]), dtype=np.float64)
    pp = np.asarray(seed.get("principal_point_px", [480.0, 270.0]), dtype=np.float64)
    z = list(C) + list(pp)
    for s in states:
        ss = seed.get("states", {}).get(s, {})
        f = float(ss.get("focal_px", 1800.0))
        rv = np.asarray(ss.get("rvec", lookat_rvec(C).tolist()), dtype=np.float64)
        if rng is not None:
            f *= float(np.exp(rng.normal(0.0, 0.12)))
            rv = rv + rng.normal(0.0, 0.04, 3)
        z += [math.log(f)] + rv.tolist()
    z = np.asarray(z, dtype=np.float64)
    if rng is not None:
        z[:3] += rng.normal(0.0, [120.0, 120.0, 50.0])
        z[3:5] += rng.normal(0.0, [30.0, 20.0])
    return z


def residual(z, states, obs, drop=None):
    _, _, _, st = unpack(z, states)
    rows = []
    for s in states:
        p = st[s]
        for key, P in TARGET_WORLD.items():
            uv, _ = project(p, P)
            pts = np.asarray(obs[s]["target"][key], dtype=np.float64)
            rows.append(signed_line_distance(pts, uv[0], uv[1]))
        Hm = floor_homography(p)
        for key, pts in obs[s].get("floor", {}).items():
            if drop == (s, key):
                continue
            rows.append(signed_pixel_residual(Hm, key, np.asarray(pts, dtype=np.float64)))
        check = np.vstack([np.asarray([[0, 0, 0], [FT_X, 0, 0], [RIM_X + THREE_R, 0, 0]], dtype=np.float64), np.vstack(list(TARGET_WORLD.values()))])
        _, depth = project(p, check)
        rows.append(np.minimum(depth - 20.0, 0.0) / 5.0)
    return np.concatenate(rows)


def solve(states, obs, spec, seed=None, drop=None, max_nfev=None):
    lo, hi = bounds(states, spec)
    if seed is None:
        seed = make_seed(states, spec)
    seed = np.minimum(np.maximum(np.asarray(seed, dtype=np.float64), lo + 1e-9), hi - 1e-9)
    cap = int(max_nfev or spec.get("max_nfev", 4000))
    return least_squares(lambda z: residual(z, states, obs, drop), seed, bounds=(lo, hi), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=cap)


def state_metrics(z, states, obs):
    C, cx, cy, st = unpack(z, states)
    out = {"center_cm": C, "principal_point_px": [cx, cy], "states": {}}
    for s in states:
        p = st[s]
        sm = {"focal_px": float(np.exp(p[3])), "target": {}, "floor": {}}
        for key, P in TARGET_WORLD.items():
            uv, _ = project(p, P)
            d = np.abs(signed_line_distance(np.asarray(obs[s]["target"][key]), uv[0], uv[1]))
            sm["target"][key] = {"median_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)), "max_px": float(np.max(d))}
        Hm = floor_homography(p)
        for key, pts in obs[s].get("floor", {}).items():
            d = np.abs(signed_pixel_residual(Hm, key, np.asarray(pts)))
            sm["floor"][key] = {"median_px": float(np.median(d)), "p95_px": float(np.percentile(d, 95)), "max_px": float(np.max(d))}
        out["states"][s] = sm
    return out


def perturb_obs(obs, rng, sigma=0.5):
    out = {}
    for s, so in obs.items():
        out[s] = {"target": {}, "floor": {}}
        for k, pts in so["target"].items():
            a = np.asarray(pts, dtype=np.float64)
            out[s]["target"][k] = (a + rng.uniform(-sigma, sigma, a.shape)).tolist()
        for k, pts in so.get("floor", {}).items():
            a = np.asarray(pts, dtype=np.float64)
            out[s]["floor"][k] = (a + rng.uniform(-sigma, sigma, a.shape)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--multistart", type=int, default=6)
    ap.add_argument("--perturbations", type=int, default=16)
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    states = validate_spec(spec)
    obs = spec["observations"]
    rng = np.random.default_rng(107031)

    roots = []
    for i in range(max(1, args.multistart)):
        z0 = make_seed(states, spec, None if i == 0 else rng)
        try:
            fit = solve(states, obs, spec, z0)
            roots.append((float(fit.cost), np.asarray(fit.x), int(fit.nfev), bool(fit.success), str(fit.message)))
        except Exception as exc:
            roots.append((float("inf"), None, 0, False, repr(exc)))
    good = [r for r in roots if r[1] is not None and np.isfinite(r[0])]
    if not good:
        raise RuntimeError("v107 all deterministic roots failed")
    good.sort(key=lambda x: x[0])
    best = good[0]
    z = best[1]
    C0 = z[:3].copy()

    support = []
    for s in states:
        for key in obs[s].get("floor", {}):
            fit = solve(states, obs, spec, z, drop=(s, key), max_nfev=max(800, int(spec.get("max_nfev", 4000)) // 2))
            support.append({"state": s, "family": key, "center_cm": fit.x[:3], "center_shift_cm": float(np.linalg.norm(fit.x[:3] - C0)), "cost": float(fit.cost), "nfev": int(fit.nfev)})

    state_out = []
    if len(states) >= 4:
        for hold in states:
            keep = [s for s in states if s != hold]
            C, cx, cy, st = unpack(z, states)
            zz = list(C) + [cx, cy]
            for s in keep:
                p = st[s]
                zz += [p[3]] + p[6:9].tolist()
            fit = solve(keep, obs, spec, np.asarray(zz), max_nfev=max(1000, int(spec.get("max_nfev", 4000)) // 2))
            state_out.append({"held_state": hold, "center_cm": fit.x[:3], "center_shift_cm": float(np.linalg.norm(fit.x[:3] - C0)), "cost": float(fit.cost), "nfev": int(fit.nfev)})

    perturb = []
    for i in range(max(0, args.perturbations)):
        po = perturb_obs(obs, rng, 0.5)
        fit = solve(states, po, spec, z, max_nfev=max(700, int(spec.get("max_nfev", 4000)) // 3))
        perturb.append({"trial": i, "center_cm": fit.x[:3], "center_shift_cm": float(np.linalg.norm(fit.x[:3] - C0)), "cost": float(fit.cost), "nfev": int(fit.nfev)})

    result = {
        "stage": "RIGHT_SLASH_GAME_LEVEL_SHARED_CENTER_V107_SOLVER",
        "promotion": False,
        "replay_render_allowed": False,
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "states": states,
        "nominal": {"cost": best[0], "nfev": best[2], "success": best[3], "message": best[4], "metrics": state_metrics(z, states, obs)},
        "multistart": [{"cost": r[0], "center_cm": None if r[1] is None else r[1][:3], "nfev": r[2], "success": r[3], "message": r[4]} for r in roots],
        "support_reduction": support,
        "state_out": state_out,
        "perturbation_half_pixel": perturb,
        "note": "Solver evidence only. A separate v107 auditor must apply the locked thresholds and independent held-out rim/board/court QA before any camera promotion.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2) + "\n")
    print(json.dumps(json_safe({"center_cm": C0, "cost": best[0], "max_support_shift_cm": max((x["center_shift_cm"] for x in support), default=None), "max_state_out_shift_cm": max((x["center_shift_cm"] for x in state_out), default=None), "max_perturb_shift_cm": max((x["center_shift_cm"] for x in perturb), default=None), "promotion": False}), indent=2))


if __name__ == "__main__":
    main()
