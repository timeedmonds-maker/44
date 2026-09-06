from __future__ import annotations

"""v106: discovery-only direct metric solve for Right Slash event 416.

This stage is deliberately not a camera-promotion gate.  v105 has already
established that the distributed Right Slash states are compatible with one
fixed optical centre.  v106 asks a narrower question: does the best direct
metric state (event 416 f06) admit a stable 3-D pinhole solution when the
source-visible geometry is interpreted correctly as separate regulation
families: target rectangle, rim, free-throw front circle, free-throw line and
restricted-area arc?

The blue curve nearer the basket is the restricted-area arc, not the back half
of the free-throw circle.  This correction is the reason v106 exists.

All image observations below are source-pixel annotations from the immutable
native 960x540 frame.  Players and ball are never used.  A v106 result can only
seed the subsequent shared-centre robustness solve; it cannot promote Right
Slash or authorize replay rendering.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

W, H = 960, 540
FT = 30.48
IN = 2.54
RIM_X = 15.0 * IN
RIM_R = 9.0 * IN
RIM_Z = 10.0 * FT
FT_X = 15.0 * FT
FT_R = 6.0 * FT
PAINT_HALF = 8.0 * FT
RESTRICT_R = 4.0 * FT
TARGET_HALF_W = 11.0 * IN
TARGET_BOTTOM_Z = 10.0 * FT + 1.0 * IN
TARGET_TOP_Z = 10.0 * FT + 17.0 * IN
FRAME_SHA = "325a02876fb09c89de6657a711e3241ef5382fbf39fcc1696c95686a642d2668"

# Source-derived event416 f06 observations.  Target samples follow the visible
# inner target stripe centre-lines.  The free-throw line is the long straight
# blue diameter visible through the foreground circle.  The front-circle and
# restricted-arc lists deliberately retain only source-visible support.
TARGET_OBS = {
    "target_top": np.column_stack([np.linspace(439.0, 493.0, 20), np.linspace(103.8, 103.0, 20)]),
    "target_left": np.column_stack([np.full(18, 437.5), np.linspace(103.0, 144.0, 18)]),
    "target_right": np.column_stack([np.full(18, 494.0), np.linspace(103.0, 142.0, 18)]),
}
FT_LINE_OBS = np.column_stack([np.linspace(420.0, 840.0, 40), np.linspace(497.6, 479.3, 40)])
FT_FRONT_OBS = np.asarray([
    [455,490],[480,490],[500,489],[525,488],[550,486],[575,483],
    [600,480],[625,477],[650,474],[675,472],[700,470],[725,470],
    [750,471],[775,473],[800,477],[825,482],[850,488],[875,494],
], dtype=np.float64)
RESTRICT_OBS = np.asarray([
    [515,429],[530,427],[535,427],[540,427],[545,427],[550,427],
    [555,427],[560,431],[565,425],[570,425],[575,425],[580,425],
    [585,436],[590,438],[595,425],[600,425],[605,430],[610,422],
    [615,423],[620,423],[625,423],[630,423],[635,423],[640,429],[650,431],
], dtype=np.float64)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_world() -> dict[str, np.ndarray]:
    return {
        "target_top": np.asarray([[0.0, -TARGET_HALF_W, TARGET_TOP_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]]),
        "target_left": np.asarray([[0.0, -TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, -TARGET_HALF_W, TARGET_TOP_Z]]),
        "target_right": np.asarray([[0.0, TARGET_HALF_W, TARGET_BOTTOM_Z], [0.0, TARGET_HALF_W, TARGET_TOP_Z]]),
    }


def circle_world(cx: float, r: float, z: float, n: int = 360) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), r * np.sin(t), np.full_like(t, z)])


FT_CURVE = circle_world(FT_X, FT_R, 0.0)
RESTRICT_CURVE = circle_world(RIM_X, RESTRICT_R, 0.0)
RIM_CURVE = circle_world(RIM_X, RIM_R, RIM_Z)
FT_LINE_WORLD = np.asarray([[FT_X, -PAINT_HALF, 0.0], [FT_X, PAINT_HALF, 0.0]], dtype=np.float64)
TARGET_WORLD = target_world()


def rotation(p: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(p[6:9], dtype=np.float64).reshape(3, 1))[0]


def project(p: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    C = np.asarray(p[:3], dtype=np.float64)
    f = float(np.exp(p[3]))
    cx, cy = map(float, p[4:6])
    R = rotation(p)
    Q = (R @ (np.asarray(P, dtype=np.float64) - C).T).T
    uv = np.column_stack([f * Q[:, 0] / Q[:, 2] + cx, f * Q[:, 1] / Q[:, 2] + cy])
    return uv, Q[:, 2]


def signed_line_distance(obs: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(b) - np.asarray(a)
    n = np.asarray([-d[1], d[0]], dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-12)
    return (np.asarray(obs, dtype=np.float64) - np.asarray(a, dtype=np.float64)) @ n


def nearest_curve_distance(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d2 = np.sum((np.asarray(obs)[:, None, :] - np.asarray(pred)[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def split_obs(a: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(a))
    held = ((idx + offset) % 4) == 0
    return a[~held], a[held]


def rim_observations(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    Y, X = np.indices(h.shape)
    mask = (((h <= 12) | (h >= 170)) & (s >= 90) & (v >= 70) &
            (X >= 452) & (X <= 520) & (Y >= 137) & (Y <= 165))
    y, x = np.where(mask)
    pts = np.column_stack([x, y]).astype(np.float64)
    if len(pts) < 200:
        raise RuntimeError(f"v106 rim source support collapsed: {len(pts)} pixels")
    # Deterministic thinning; the orange band has finite thickness and is used
    # at lower weight than line-centre observations.
    return pts[::4]


def lookat_rvec(C: np.ndarray, aim: np.ndarray = np.asarray([300.0, 0.0, 170.0])) -> np.ndarray:
    C = np.asarray(C, dtype=np.float64)
    z = np.asarray(aim, dtype=np.float64) - C
    z /= np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-8:
        x = np.asarray([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return cv2.Rodrigues(R)[0].ravel()


def residual(p: np.ndarray, train: dict[str, np.ndarray]) -> np.ndarray:
    rows = []
    for key, P in TARGET_WORLD.items():
        uv, _ = project(p, P)
        rows.append(signed_line_distance(train[key], uv[0], uv[1]))
    uv, _ = project(p, FT_LINE_WORLD)
    rows.append(0.9 * signed_line_distance(train["free_throw_line"], uv[0], uv[1]))
    uv, _ = project(p, FT_CURVE)
    rows.append(1.1 * nearest_curve_distance(train["free_throw_front"], uv))
    uv, _ = project(p, RESTRICT_CURVE)
    rows.append(1.1 * nearest_curve_distance(train["restricted_arc"], uv))
    uv, _ = project(p, RIM_CURVE)
    rows.append(0.5 * nearest_curve_distance(train["rim"], uv))

    check = np.vstack([
        FT_CURVE[::60], RESTRICT_CURVE[::60], RIM_CURVE[::60],
        np.vstack(list(TARGET_WORLD.values())),
    ])
    _, z = project(p, check)
    rows.append(np.minimum(z - 20.0, 0.0) / 5.0)
    return np.concatenate(rows)


def metric_line(p: np.ndarray, key: str, obs: np.ndarray) -> dict:
    if key in TARGET_WORLD:
        uv, _ = project(p, TARGET_WORLD[key])
    elif key == "free_throw_line":
        uv, _ = project(p, FT_LINE_WORLD)
    else:
        raise KeyError(key)
    d = np.abs(signed_line_distance(obs, uv[0], uv[1]))
    return {"count": int(len(d)), "median_px": float(np.median(d)), "p95_px": float(np.percentile(d,95)), "max_px": float(np.max(d))}


def metric_curve(p: np.ndarray, curve: np.ndarray, obs: np.ndarray) -> dict:
    uv, _ = project(p, curve)
    d = nearest_curve_distance(obs, uv)
    return {"count": int(len(d)), "median_px": float(np.median(d)), "p95_px": float(np.percentile(d,95)), "max_px": float(np.max(d))}


def dense_action_projection(p: np.ndarray) -> np.ndarray:
    P = np.asarray([[x,y,z] for x in np.linspace(-30,700,8) for y in np.linspace(-300,300,9) for z in np.linspace(0,360,7)], dtype=np.float64)
    return project(p, P)[0]


def draw_overlay(frame: np.ndarray, p: np.ndarray, out: Path, held: dict[str,np.ndarray]) -> None:
    ov = frame.copy()
    colors = {
        "free_throw_front": (255,255,0), "restricted_arc": (255,128,0),
        "rim": (0,255,255), "free_throw_line": (255,0,255),
    }
    for key, P in TARGET_WORLD.items():
        uv, _ = project(p, P)
        cv2.line(ov, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), (0,255,0), 2, cv2.LINE_AA)
    for key, curve in (("free_throw_front",FT_CURVE),("restricted_arc",RESTRICT_CURVE),("rim",RIM_CURVE)):
        uv, _ = project(p, curve)
        q = np.round(uv).astype(int)
        ok = (q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
        for x,y in q[ok][::2]:
            cv2.circle(ov,(int(x),int(y)),1,colors[key],-1,cv2.LINE_AA)
    uv, _ = project(p, FT_LINE_WORLD)
    cv2.line(ov, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), colors["free_throw_line"], 2, cv2.LINE_AA)
    for pts in held.values():
        for x,y in np.round(pts).astype(int):
            cv2.circle(ov,(int(x),int(y)),4,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(out), ov)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=Path, required=True)
    ap.add_argument("--v105", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if sha256(args.frame) != FRAME_SHA:
        raise RuntimeError("immutable event416 f06 SHA mismatch")
    auth = json.loads(args.v105.read_text())
    if auth.get("status") != "PASS_RIGHT_SLASH_FIXED_CENTER_AUTHORIZATION_V105" or not auth.get("permissions",{}).get("shared_center_metric_attempt_allowed"):
        raise RuntimeError("v105 has not authorized a metric shared-centre attempt")

    frame = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (H,W):
        raise RuntimeError("event416 source frame missing or non-native")
    rim = rim_observations(frame)

    train, held = {}, {}
    for key, arr, off in [
        ("target_top",TARGET_OBS["target_top"],0),
        ("target_left",TARGET_OBS["target_left"],1),
        ("target_right",TARGET_OBS["target_right"],2),
        ("free_throw_line",FT_LINE_OBS,3),
        ("free_throw_front",FT_FRONT_OBS,0),
        ("restricted_arc",RESTRICT_OBS,1),
        ("rim",rim,2),
    ]:
        train[key], held[key] = split_obs(np.asarray(arr,dtype=np.float64), off)

    lo = np.r_[[-5000.0,-5000.0,50.0], math.log(150.0), -2000.0,-1500.0, [-10.0]*3]
    hi = np.r_[[5000.0,5000.0,2000.0], math.log(5000.0), 3000.0,2000.0, [10.0]*3]
    specs = [
        ([1200,-1200,250],700),([1200,1200,250],900),([1600,-700,250],900),([1600,700,250],1200),
        ([1900,-300,200],1500),([1900,300,300],1800),([2200,-1000,350],1200),([2200,1000,350],1600),
        ([2600,-500,500],1800),([2600,500,500],1000),([1400,0,600],900),([2800,0,600],1800),
    ]
    roots = []
    for i,(C0,f0) in enumerate(specs):
        C0 = np.asarray(C0,dtype=np.float64)
        s = np.r_[C0, math.log(float(f0)), 480.0,270.0, lookat_rvec(C0)]
        try:
            fit = least_squares(lambda p: residual(p,train), s, bounds=(lo,hi), loss="soft_l1", f_scale=1.2, x_scale="jac", max_nfev=5000)
        except Exception as exc:
            roots.append({"index":i,"error":repr(exc)})
            continue
        p = np.asarray(fit.x,dtype=np.float64)
        _, depth = project(p, np.vstack([FT_CURVE[::60],RESTRICT_CURVE[::60],RIM_CURVE[::60],np.vstack(list(TARGET_WORLD.values()))]))
        if not np.isfinite(p).all() or not np.all(depth>20.0):
            roots.append({"index":i,"cost":float(fit.cost),"physical":False})
            continue
        roots.append({
            "index":i,"cost":float(fit.cost),"physical":True,"params":p.tolist(),
            "center_cm":p[:3].tolist(),"focal_px":float(np.exp(p[3])),"principal_point_px":p[4:6].tolist(),
            "median_abs_train_residual_px":float(np.median(np.abs(residual(p,train)))),
        })

    physical = [r for r in roots if r.get("physical")]
    if not physical:
        raise RuntimeError("v106 produced no physical roots")
    physical.sort(key=lambda r:r["cost"])
    best = physical[0]
    pb = np.asarray(best["params"],dtype=np.float64)
    ref = dense_action_projection(pb)

    competitive = []
    for r in physical:
        p = np.asarray(r["params"],dtype=np.float64)
        d = np.linalg.norm(dense_action_projection(p)-ref,axis=1)
        competitive.append({
            "index":r["index"],"cost":r["cost"],"center_cm":r["center_cm"],
            "center_shift_cm":float(np.linalg.norm(p[:3]-pb[:3])),
            "action_projection_p95_shift_px":float(np.percentile(d,95)),
            "focal_px":r["focal_px"],"principal_point_px":r["principal_point_px"],
        })

    held_metrics = {
        "target_top": metric_line(pb,"target_top",held["target_top"]),
        "target_left": metric_line(pb,"target_left",held["target_left"]),
        "target_right": metric_line(pb,"target_right",held["target_right"]),
        "free_throw_line": metric_line(pb,"free_throw_line",held["free_throw_line"]),
        "free_throw_front": metric_curve(pb,FT_CURVE,held["free_throw_front"]),
        "restricted_arc": metric_curve(pb,RESTRICT_CURVE,held["restricted_arc"]),
        "rim": metric_curve(pb,RIM_CURVE,held["rim"]),
    }
    draw_overlay(frame,pb,args.out/"right_slash_event416_metric_overlay_v106.png",held)

    report = {
        "schema_version":1,
        "status":"RIGHT_SLASH_EVENT416_METRIC_DISCOVERY_V106",
        "game_id":"0022500301","camera_label":"Right Slash","event_id":416,"frame":"f06.png","image_sha256":FRAME_SHA,
        "interpretation_correction":"The nearer blue court curve is modeled as the regulation restricted-area arc, not as the back half of the free-throw circle.",
        "best_root":best,
        "heldout_metrics":held_metrics,
        "all_roots":roots,
        "competitive_root_comparison":competitive,
        "guardrails":["native 960x540 source pixels only","no player or ball landmarks","v106 is discovery-only","no metric camera or replay promotion"],
        "permissions":{"shared_center_metric_attempt_allowed":True,"right_slash_metric_camera_allowed":False,"replay_render_allowed":False},
    }
    (args.out/"right_slash_event416_metric_v106.json").write_text(json.dumps(report,indent=2)+"\n")
    print("V106 BEST center_cm",np.round(pb[:3],3).tolist(),"f",round(float(np.exp(pb[3])),3),"pp",np.round(pb[4:6],3).tolist(),flush=True)
    print("V106 heldout p95",{k:round(v["p95_px"],3) for k,v in held_metrics.items()},flush=True)
    print("V106 physical roots",len(physical),"of",len(specs),flush=True)
    print("V106 DISCOVERY ONLY; no camera promotion",flush=True)


if __name__ == "__main__":
    main()
