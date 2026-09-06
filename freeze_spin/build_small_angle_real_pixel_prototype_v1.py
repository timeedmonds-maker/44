from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH = 960
HEIGHT = 540


def project(P: np.ndarray, pts: np.ndarray):
    h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    q = (P @ h.T).T
    valid = q[:, 2] > 1e-6
    uv = np.full((len(pts), 2), np.nan, dtype=np.float64)
    uv[valid] = q[valid, :2] / q[valid, 2:3]
    return uv, valid


def camera_rows(camera_json: Path, locked_dir: Path):
    payload = json.loads(camera_json.read_text(encoding="utf-8"))
    manifest = json.loads((locked_dir / "locked_exact_state.json").read_text(encoding="utf-8"))
    images = {v["label"]: v["image"] for v in manifest["views"]}
    rows = []
    for c in payload["cameras"]:
        label = c["label"]
        if label not in images:
            continue
        img = cv2.imread(str(locked_dir / images[label]))
        if img is None:
            raise RuntimeError(f"Could not read locked image for {label}")
        K = np.asarray(c["K"], dtype=np.float64)
        R = np.asarray(c["R_world_to_camera"], dtype=np.float64)
        t = np.asarray(c["t_world_to_camera_cm"], dtype=np.float64).reshape(3, 1)
        C = np.asarray(c["camera_center_world_cm"], dtype=np.float64)
        P = np.asarray(c["projection_matrix_KRt"], dtype=np.float64)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        feat = np.dstack([lab, grad[:, :, None]]).astype(np.float32)
        rows.append(dict(label=label, image=img, K=K, R=R, t=t, C=C, P=P, feat=feat))
    if len(rows) < 4:
        raise RuntimeError(f"Expected four locked calibrated views, got {len(rows)}")
    return rows


def read_ball(ball_report: Path):
    p = json.loads(ball_report.read_text(encoding="utf-8"))
    for key in ("ball_center_world_cm", "world_point_cm", "ball_world_cm"):
        if key in p:
            return np.asarray(p[key], dtype=np.float64)
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("ball_center_world_cm", "world_point_cm", "ball_world_cm") and isinstance(v, list) and len(v) == 3:
                    return np.asarray(v, dtype=np.float64)
                z = walk(v)
                if z is not None:
                    return z
        elif isinstance(x, list):
            for v in x:
                z = walk(v)
                if z is not None:
                    return z
        return None
    out = walk(p)
    if out is None:
        raise RuntimeError("Could not locate 3D ball centre in report")
    return out


def bilinear(image, u, v):
    return cv2.remap(image, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def select_reference(rows, target):
    ranked = []
    for r in rows:
        uv, ok = project(r["P"], target[None, :])
        if not ok[0]:
            continue
        u, v = uv[0]
        margin = min(u, WIDTH - 1 - u, v, HEIGHT - 1 - v)
        ranked.append((margin, r))
    if not ranked:
        raise RuntimeError("Ball target is outside all calibrated views")
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1]


def reconstruct(rows, target, stride=2, depth_steps=120):
    ref = select_reference(rows, target)
    sources = [r for r in rows if r is not ref]
    target_uv, ok = project(ref["P"], target[None, :])
    if not ok[0]:
        raise RuntimeError("Target does not project in reference")
    hu, hv = target_uv[0]
    x0 = max(0, int(hu - 260)); x1 = min(WIDTH - 1, int(hu + 260))
    y0 = max(0, int(hv - 245)); y1 = min(HEIGHT - 1, int(hv + 250))
    xs = np.arange(x0, x1 + 1, stride, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, stride, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    H, W = uu.shape

    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(H * W)], axis=0)
    rays_cam = np.linalg.inv(ref["K"]) @ pix
    rays_world = ref["R"].T @ rays_cam
    rays_world /= np.linalg.norm(rays_world, axis=0, keepdims=True) + 1e-12
    rays = rays_world.T.reshape(H, W, 3)

    centre_depth = float(np.linalg.norm(target - ref["C"]))
    depths = np.linspace(max(80.0, centre_depth - 500.0), centre_depth + 500.0, depth_steps)
    ref_feat = ref["feat"][vv.astype(np.int32), uu.astype(np.int32)]
    ref_rgb = ref["image"][vv.astype(np.int32), uu.astype(np.int32)]
    best = np.full((H, W), np.inf, dtype=np.float32)
    second = np.full((H, W), np.inf, dtype=np.float32)
    best_depth = np.zeros((H, W), dtype=np.float32)
    support_best = np.zeros((H, W), dtype=np.uint8)

    for depth in depths:
        X = ref["C"][None, None, :] + rays * depth
        # Conservative action volume: keep player/ball/basket vicinity and exclude most court/crowd.
        physical = ((X[:, :, 2] >= 20.0) & (X[:, :, 2] <= 380.0) &
                    (np.abs(X[:, :, 0] - target[0]) <= 360.0) &
                    (np.abs(X[:, :, 1] - target[1]) <= 360.0))
        costs = []
        valids = []
        flat = X.reshape(-1, 3)
        for src in sources:
            suv, sok = project(src["P"], flat)
            su = suv[:, 0].reshape(H, W); sv = suv[:, 1].reshape(H, W)
            inside = sok.reshape(H, W) & physical & (su >= 1) & (su < WIDTH - 2) & (sv >= 1) & (sv < HEIGHT - 2)
            sf = bilinear(src["feat"], su, sv)
            dl = np.abs(sf[:, :, :3] - ref_feat[:, :, :3]).mean(axis=2)
            dg = np.abs(sf[:, :, 3] - ref_feat[:, :, 3])
            c = np.minimum(dl, 0.7) * 0.78 + np.minimum(dg, 0.7) * 0.22
            costs.append(c)
            valids.append(inside)
        stack = np.stack(costs, axis=2)
        vm = np.stack(valids, axis=2)
        stack[~vm] = np.nan
        support = vm.sum(axis=2).astype(np.uint8)
        # Median makes a single occluded camera less destructive.
        with np.errstate(all="ignore"):
            cost = np.nanmedian(stack, axis=2)
        cost = np.where(support >= 2, cost, np.inf)
        improved = cost < best
        second = np.where(improved, best, np.minimum(second, cost))
        best = np.where(improved, cost, best)
        best_depth = np.where(improved, depth, best_depth)
        support_best = np.where(improved, support, support_best)

    margin = second - best
    textured = ref_feat[:, :, 3] > 0.025
    good = np.isfinite(best) & (best < 0.30) & (margin > 0.004) & (support_best >= 2) & textured
    X = ref["C"][None, None, :] + rays * best_depth[:, :, None]
    pts = X[good]
    colors = ref_rgb[good]
    q = {
        "reference": ref["label"], "sources": [s["label"] for s in sources],
        "roi": [x0, y0, x1, y1], "accepted_points": int(len(pts)),
        "candidate_grid_points": int(H * W), "accepted_fraction": round(float(good.mean()), 6),
        "median_cost": None if len(pts) == 0 else round(float(np.median(best[good])), 6),
        "median_depth_margin": None if len(pts) == 0 else round(float(np.median(margin[good])), 6),
    }
    return ref, pts, colors, q


def rz(deg):
    a = math.radians(deg); c = math.cos(a); s = math.sin(a)
    return np.asarray([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)


def render(points, colors, ref, target, deg, scale=4):
    Q = rz(deg)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3,1)
    S = np.diag([scale, scale, 1.0])
    P = (S @ ref["K"]) @ np.hstack([R,t])
    uv, valid = project(P, points)
    z = (R @ (points - C).T).T[:,2]
    valid &= z > 1
    ow, oh = WIDTH*scale, HEIGHT*scale
    valid &= (uv[:,0] >= 0) & (uv[:,0] < ow) & (uv[:,1] >= 0) & (uv[:,1] < oh)
    canvas = np.zeros((oh,ow,3), np.uint8)
    cover = np.zeros((oh,ow), np.uint8)
    ids = np.where(valid)[0]
    ids = ids[np.argsort(z[ids])[::-1]]
    radius = 5
    for i in ids:
        x,y = int(round(uv[i,0])), int(round(uv[i,1]))
        col = tuple(int(v) for v in colors[i])
        cv2.circle(canvas,(x,y),radius,col,-1,cv2.LINE_AA)
        cv2.circle(cover,(x,y),radius,255,-1,cv2.LINE_AA)
    return canvas, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth-steps", type=int, default=120)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    rows = camera_rows(args.cameras, args.locked_images)
    ball = read_ball(args.ball_report)
    ref, points, colors, recon = reconstruct(rows, ball, args.stride, args.depth_steps)
    if len(points) < 800:
        raise RuntimeError(f"Insufficient geometrically supported action surface: {len(points)} points")

    degrees = [0,3,5,8]
    metrics = []
    for d in degrees:
        frame, cover = render(points, colors, ref, ball, d, scale=4)
        cv2.imwrite(str(args.out / f"real_pixel_virtual_{d:02d}deg_uhd.png"), frame)
        cv2.imwrite(str(args.out / f"coverage_{d:02d}deg_uhd.png"), cover)
        metrics.append({"degrees": d, "covered_pixels": int((cover > 0).sum()),
                        "coverage_fraction_of_4k_frame": round(float((cover > 0).mean()), 6)})

    qa = {
        "prototype": "small_angle_real_pixel_free_view_v1",
        "source_resolution": [960,540],
        "render_resolution": [3840,2160],
        "resolution_policy": "native source pixels only; 4K canvas is diagnostic render space, not detail enhancement",
        "identity_policy": "no player identity labels required; geometry is accepted/rejected per visible surface support",
        "source_policy": "real synchronized official NBA pixels only; no diffusion, generative fill, crossfade, optical-flow morph, or hallucinated body completion",
        "target_ball_world_cm": ball.tolist(),
        "degrees": degrees,
        "reconstruction": recon,
        "renders": metrics,
        "interpretation": "Black pixels are unresolved geometry and are intentionally exposed. Prototype success requires coherent player/ball silhouettes at 3-5 degrees before attempting larger angles or hole filling.",
    }
    (args.out / "small_angle_real_pixel_qa_v1.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))

if __name__ == "__main__":
    main()
