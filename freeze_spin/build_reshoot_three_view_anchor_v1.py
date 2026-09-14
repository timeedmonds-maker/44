from __future__ import annotations

"""Reshoot-inspired deterministic three-view NBA free-view anchor.

This prototype deliberately stops before any generative completion. It keeps
real pixels only, builds one dense world-space point cloud from exactly three
locked NBA camera frames, suppresses depth-boundary/outlier points before
reprojection, fuses the three real views with target-view visibility logic,
and emits an explicit hole mask.

The point is to answer one question cleanly: is the geometry good enough to
support a real camera move before a generator is allowed to hide mistakes?
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

import build_portable_moge_pnp_freeview_v12 as base
from build_portable_moge_pnp_freeview_v13 import solve_target_from_reference_reciprocal
from build_portable_moge_true_orbit_v16 import true_orbit_pose

W, H = 960, 540
GAME_ID = "0022500301"
EVENT_ID = 489
REFERENCE = "Left Above Rim"
LOCKED_FRAMES = {
    "Left Above Rim": 259,
    "Right Above Rim": 250,
    "Broadcast": 294,
}
# Accepted v33e ball observations used only to define the orbit pivot. The
# renderer never paints a synthetic ball and does not alter source timing.
BALL_PIXELS = {
    "Right Above Rim": (601.0, 191.0),
    "Broadcast": (531.0, 105.0),
}


def safe(label: str) -> str:
    return label.replace(" ", "_")


def high_recall_sift_matches(im1, im2, mask1, mask2):
    """v23-style candidate recall; reciprocal geometry gates remain unchanged."""
    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.015,
        edgeThreshold=14,
        sigma=1.3,
    )
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, mask1.astype(np.uint8) * 255)
    k2, d2 = sift.detectAndCompute(g2, mask2.astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return [], np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = []
    for pair in matcher.knnMatch(d1, d2, k=2):
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    p1 = np.array([k1[m.queryIdx].pt for m in good], np.float64) if good else np.empty((0, 2))
    p2 = np.array([k2[m.trainIdx].pt for m in good], np.float64) if good else np.empty((0, 2))
    return good, p1, p2


def find_clip(clips: Path, label: str) -> Path:
    token = safe(label)
    hits = sorted(clips.glob(f"*_{EVENT_ID}_{token}_SOURCE.mp4"))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {label} event-{EVENT_ID} source; found {hits}")
    return hits[0]


def read_exact_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    decoded = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
    cap.release()
    if not ok or frame is None or decoded != int(frame_index):
        raise RuntimeError(f"Exact frame decode failed for {path.name}: wanted {frame_index}, got {decoded}")
    if frame.shape[:2] != (H, W):
        raise RuntimeError(f"Native source changed for {path.name}: {frame.shape}")
    return frame


def extract_locked_three(clips: Path, out: Path) -> Dict[str, np.ndarray]:
    out.mkdir(parents=True, exist_ok=True)
    images = {}
    manifest = []
    for label, frame_index in LOCKED_FRAMES.items():
        source = find_clip(clips, label)
        image = read_exact_frame(source, frame_index)
        name = f"{safe(label)}_F{frame_index:04d}.png"
        cv2.imwrite(str(out / name), image)
        images[label] = image
        manifest.append({
            "camera": label,
            "frame_index": frame_index,
            "source": source.name,
            "image": name,
        })
    payload = {
        "event": {"game_id": GAME_ID, "event_id": EVENT_ID, "date": "2025-11-30"},
        "lock": "v33e exact three-camera state",
        "camera_count": 3,
        "cameras": manifest,
        "policy": "exactly LAR/RAR/Broadcast; native 960x540; frame indices immutable",
    }
    (out / "locked_three_view_manifest.json").write_text(json.dumps(payload, indent=2))
    return images


def local_depth_keep(depth: np.ndarray, valid: np.ndarray, dynamic: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Suppress flying pixels at depth discontinuities and isolated local outliers."""
    d = np.asarray(depth, np.float32)
    good = valid & np.isfinite(d) & (d > 0.25)
    inv = np.zeros_like(d, np.float32)
    inv[good] = 1.0 / np.maximum(d[good], 1e-4)
    gx = cv2.Sobel(inv, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(inv, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    vals = grad[good]
    if len(vals) == 0:
        return good, {"valid": 0}
    static_thr = float(np.percentile(vals, 90.0))
    dynamic_thr = float(np.percentile(vals, 96.0))
    edge_ok = grad <= np.where(dynamic, dynamic_thr, static_thr)

    med = cv2.medianBlur(d, 5)
    rel = np.abs(d - med) / np.maximum(np.abs(med), 1e-3)
    local_ok = (~np.isfinite(rel)) | (rel <= np.where(dynamic, 0.16, 0.09))
    keep = good & edge_ok & local_ok
    return keep, {
        "raw_valid": int(good.sum()),
        "kept": int(keep.sum()),
        "removed": int((good & ~keep).sum()),
        "static_disparity_edge_threshold": static_thr,
        "dynamic_disparity_edge_threshold": dynamic_thr,
    }


def infer_views(images: Dict[str, np.ndarray], out: Path, tokens: int):
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    detector = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        progress=True,
    ).eval()
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    views = {}
    for label, image in images.items():
        dynamic, balls = base.detect_dynamic_and_ball(detector, image)
        depth, points, valid, K, Kn = base.moge_infer(moge, image, tokens)
        keep, edge_qa = local_depth_keep(depth, valid, dynamic)
        view = {
            "label": label,
            "image": image,
            "dynamic": dynamic,
            "balls": balls,
            "depth": depth,
            "points": points,
            "valid": valid,
            "keep": keep,
            "K": K,
            "Kn": Kn,
            "edge_qa": edge_qa,
        }
        views[label] = view
        cv2.imwrite(str(out / f"{safe(label)}_dynamic.png"), dynamic.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{safe(label)}_depth_keep.png"), keep.astype(np.uint8) * 255)
    return views


def solve_cameras(views: dict) -> dict:
    # Only the candidate generator changes; v13 acceptance thresholds stay intact.
    base.sift_matches = high_recall_sift_matches
    ref = views[REFERENCE]
    solves = {
        REFERENCE: {
            "passed": True,
            "R": np.eye(3, dtype=np.float64),
            "t": np.zeros(3, dtype=np.float64),
            "C": np.zeros(3, dtype=np.float64),
            "depth_scale": 1.0,
            "reason": "reference camera",
        }
    }
    for label in ("Right Above Rim", "Broadcast"):
        s = solve_target_from_reference_reciprocal(ref, views[label])
        if not s.get("passed"):
            serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            raise RuntimeError(f"{label} reciprocal camera solve failed: {serial}")
        solves[label] = s
    return solves


def build_cloud(view: dict, solve: dict) -> dict:
    keep = view["keep"]
    ys, xs = np.where(keep)
    Xc = view["points"][ys, xs].astype(np.float64) * float(solve["depth_scale"])
    R = solve["R"].astype(np.float64)
    t = solve["t"].astype(np.float64)
    Xw = (R.T @ (Xc - t).T).T.astype(np.float32)
    return {
        "label": view["label"],
        "points": Xw,
        "colors": view["image"][ys, xs].copy(),
        "dynamic": view["dynamic"][ys, xs].copy(),
        "C": solve["C"].astype(np.float64),
    }


def projection_matrix(K: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    t = -R @ C
    return K @ np.column_stack([R, t])


def triangulate_two(P1: np.ndarray, uv1: Tuple[float, float], P2: np.ndarray, uv2: Tuple[float, float]) -> np.ndarray:
    x1 = np.array(uv1, np.float64).reshape(2, 1)
    x2 = np.array(uv2, np.float64).reshape(2, 1)
    Xh = cv2.triangulatePoints(P1, P2, x1, x2).reshape(4)
    if abs(float(Xh[3])) < 1e-9:
        raise RuntimeError("Ball triangulation is degenerate")
    X = Xh[:3] / Xh[3]
    return X.astype(np.float64)


def ball_target(views: dict, solves: dict) -> Tuple[np.ndarray, dict]:
    label1, label2 = "Right Above Rim", "Broadcast"
    s1, s2 = solves[label1], solves[label2]
    P1 = projection_matrix(views[label1]["K"], s1["R"], s1["C"])
    P2 = projection_matrix(views[label2]["K"], s2["R"], s2["C"])
    X = triangulate_two(P1, BALL_PIXELS[label1], P2, BALL_PIXELS[label2])
    repro = {}
    for label, uv0 in BALL_PIXELS.items():
        s = solves[label]
        P = projection_matrix(views[label]["K"], s["R"], s["C"])
        q = P @ np.r_[X, 1.0]
        uv = q[:2] / q[2]
        repro[label] = {
            "observed": list(uv0),
            "projected": uv.tolist(),
            "error_px": float(np.linalg.norm(uv - np.asarray(uv0))),
        }
    return X, {"world_xyz_reference_units": X.tolist(), "reprojection": repro}


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))))


def source_view_angle(C_source: np.ndarray, C_target: np.ndarray, pivot: np.ndarray) -> float:
    return angle_between(C_source - pivot, C_target - pivot)


def project_points(points: np.ndarray, K: np.ndarray, R: np.ndarray, C: np.ndarray):
    X = points.astype(np.float64)
    Xc = (R @ (X - C).T).T
    z = Xc[:, 2]
    q = (K @ Xc.T).T
    uv = np.zeros((len(X), 2), np.float64)
    ok = z > 0.05
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    return uv, z, ok


def raster_one(cloud: dict, K: np.ndarray, R: np.ndarray, C: np.ndarray, radius: int = 1):
    """Depth-aware point splat with a small deterministic screen-space radius."""
    uv, z, ok = project_points(cloud["points"], K, R, C)
    u0 = np.rint(uv[:, 0]).astype(np.int32)
    v0 = np.rint(uv[:, 1]).astype(np.int32)
    ok &= (u0 >= -radius) & (u0 < W + radius) & (v0 >= -radius) & (v0 < H + radius)
    ids = np.where(ok)[0]
    zbuf = np.full(H * W, np.inf, np.float32)
    if not len(ids):
        return (
            np.zeros((H, W, 3), np.uint8),
            np.zeros((H, W), bool),
            np.full((H, W), np.inf, np.float32),
            np.zeros((H, W), bool),
        )

    offsets = [(0, 0)]
    for r in range(1, radius + 1):
        offsets += [
            (dx, dy)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if max(abs(dx), abs(dy)) == r
        ]

    for dx, dy in offsets:
        u = u0[ids] + dx
        v = v0[ids] + dy
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ii = ids[inside]
        pix = v[inside] * W + u[inside]
        np.minimum.at(zbuf, pix, z[ii].astype(np.float32))

    image = np.zeros((H, W, 3), np.uint8)
    dyn = np.zeros((H, W), bool)
    assigned = np.zeros(H * W, bool)
    # Centre-first assignment gives exact samples priority when depths tie.
    for dx, dy in offsets:
        u = u0[ids] + dx
        v = v0[ids] + dy
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ii = ids[inside]
        uu = u[inside]
        vv = v[inside]
        pix = vv * W + uu
        winners = (z[ii] <= zbuf[pix] + 1e-4) & (~assigned[pix])
        if not np.any(winners):
            continue
        jj = ii[winners]
        pp = pix[winners]
        xu = uu[winners]
        yv = vv[winners]
        image[yv, xu] = cloud["colors"][jj]
        dyn[yv, xu] = cloud["dynamic"][jj]
        assigned[pp] = True
    mask = np.isfinite(zbuf).reshape(H, W)
    return image, mask, zbuf.reshape(H, W), dyn


def fuse_renders(clouds: Iterable[dict], K: np.ndarray, R: np.ndarray, C: np.ndarray, pivot: np.ndarray, radius: int = 1):
    renders = []
    for cloud in clouds:
        image, mask, depth, dynamic = raster_one(cloud, K, R, C, radius=radius)
        renders.append({
            "label": cloud["label"],
            "image": image,
            "mask": mask,
            "depth": depth,
            "dynamic": dynamic,
            "angle": source_view_angle(cloud["C"], C, pivot),
        })

    stack_z = np.stack([r["depth"] for r in renders], axis=0)
    front = np.min(stack_z, axis=0)
    canvas = np.zeros((H, W, 3), np.uint8)
    resolved = np.zeros((H, W), bool)
    owner = np.full((H, W), -1, np.int8)
    # Prefer source closest in viewing direction, but only among surfaces agreeing
    # with the target-view front depth. This replaces v11's first-source-wins fill.
    order = np.argsort([r["angle"] for r in renders])
    reports = []
    for idx in order:
        r = renders[int(idx)]
        tol = np.where(r["dynamic"], 0.14, 0.08) * np.maximum(front, 1.0)
        visible = r["mask"] & np.isfinite(front) & (r["depth"] <= front + tol)
        take = (~resolved) & visible
        canvas[take] = r["image"][take]
        owner[take] = int(idx)
        resolved[take] = True
        reports.append({
            "label": r["label"],
            "view_angle_deg": float(r["angle"]),
            "accepted_pixels": int(take.sum()),
            "visible_candidates": int(visible.sum()),
        })
    return canvas, resolved, owner, reports


def pose_for_real(solve: dict):
    return solve["R"].astype(np.float64), solve["C"].astype(np.float64)


def leave_one_out_qa(clouds: dict, views: dict, solves: dict, pivot: np.ndarray, out: Path, radius: int):
    rows = []
    for held in LOCKED_FRAMES:
        R, C = pose_for_real(solves[held])
        others = [clouds[k] for k in LOCKED_FRAMES if k != held]
        synth, mask, _, _ = fuse_renders(others, views[held]["K"], R, C, pivot, radius=radius)
        real = views[held]["image"]
        cv2.imwrite(str(out / f"loo_{safe(held)}_from_other_two.png"), synth)
        cv2.imwrite(str(out / f"loo_{safe(held)}_holes.png"), (~mask).astype(np.uint8) * 255)
        if mask.any():
            diff = np.abs(synth.astype(np.float32) - real.astype(np.float32)).mean(axis=2)
            med = float(np.median(diff[mask]))
            p90 = float(np.percentile(diff[mask], 90))
        else:
            med = float("inf")
            p90 = float("inf")
        rows.append({
            "held_out_camera": held,
            "coverage_fraction": float(mask.mean()),
            "median_abs_bgr_error": med,
            "p90_abs_bgr_error": p90,
        })
    return rows


def serialise_solve(s: dict) -> dict:
    out = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=1600)
    ap.add_argument("--frames", type=int, default=31)
    ap.add_argument("--max-degree", type=float, default=8.0)
    ap.add_argument("--splat-radius", type=int, default=1)
    args = ap.parse_args()
    if args.splat_radius < 0 or args.splat_radius > 2:
        raise ValueError("splat-radius must be 0..2")
    args.out.mkdir(parents=True, exist_ok=True)
    locked = args.out / "locked"
    images = extract_locked_three(args.clips, locked)
    views = infer_views(images, args.out, args.tokens)
    solves = solve_cameras(views)
    clouds = {label: build_cloud(views[label], solves[label]) for label in LOCKED_FRAMES}
    pivot, pivot_qa = ball_target(views, solves)

    # Pick the closest real solved secondary endpoint to minimise unsupported
    # extrapolation. The virtual camera never moves past a real physical baseline.
    endpoint_rows = []
    for label in ("Right Above Rim", "Broadcast"):
        baseline = source_view_angle(solves[REFERENCE]["C"], solves[label]["C"], pivot)
        endpoint_rows.append((baseline, label))
    viable = sorted((b, l) for b, l in endpoint_rows if b >= 3.0)
    if not viable:
        raise RuntimeError(f"No secondary camera proves >=3deg baseline: {endpoint_rows}")
    baseline_deg, endpoint = viable[0]
    render_max = min(float(args.max_degree), float(baseline_deg) * 0.85)
    if render_max < 3.0:
        raise RuntimeError(f"Supported orbit only {render_max:.3f} degrees")

    ref = views[REFERENCE]
    K = ref["K"].astype(np.float64)
    C_endpoint = solves[endpoint]["C"].astype(np.float64)
    cloud_list = [clouds[REFERENCE], clouds["Right Above Rim"], clouds["Broadcast"]]

    def render(degree: float):
        if abs(degree) < 1e-12:
            return ref["image"].copy(), np.ones((H, W), bool), [
                {"label": REFERENCE, "accepted_pixels": H * W, "view_angle_deg": 0.0}
            ]
        R, C, _ = true_orbit_pose(pivot, C_endpoint, degree)
        image, mask, _, reports = fuse_renders(cloud_list, K, R, C, pivot, radius=args.splat_radius)
        return image, mask, reports

    stills = []
    for nominal in (0.0, 2.0, 4.0, 6.0, 8.0):
        degree = min(nominal, render_max)
        frame, mask, reports = render(degree)
        tag = int(round(nominal))
        cv2.imwrite(str(args.out / f"reshoot_anchor_{tag:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"reshoot_anchor_{tag:02d}deg_holes.png"), (~mask).astype(np.uint8) * 255)
        stills.append({
            "nominal_degree": nominal,
            "actual_degree": degree,
            "resolved_fraction": float(mask.mean()),
            "unresolved_pixels": int((~mask).sum()),
            "source_ownership": reports,
        })

    motion = []
    for i in range(args.frames):
        phase = i / max(1, args.frames - 1)
        eased = 0.5 - 0.5 * math.cos(2.0 * math.pi * phase)
        degree = render_max * eased
        frame, mask, _ = render(degree)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)
        motion.append({"frame": i, "degree": degree, "resolved_fraction": float(mask.mean())})

    loo = leave_one_out_qa(clouds, views, solves, pivot, args.out, args.splat_radius)
    qa = {
        "prototype": "reshoot_inspired_three_view_real_pixel_anchor_v1",
        "event": {"game_id": GAME_ID, "event_id": EVENT_ID, "date": "2025-11-30"},
        "state_lock": {
            "camera_count": 3,
            "cameras": LOCKED_FRAMES,
            "reference": REFERENCE,
            "ball_pixels": {k: list(v) for k, v in BALL_PIXELS.items()},
            "native_resolution": [W, H],
        },
        "method": {
            "depth": "MoGe-2 per real locked frame",
            "camera_solve": "reciprocal static-SIFT/PnP closure; v13 hard gates; v23 candidate recall",
            "reshoot_transplants": [
                "disparity-edge suppression before point rendering",
                "local depth-outlier suppression",
                "dense point-cloud reprojection",
                "explicit disocclusion/hole mask",
            ],
            "new_three_view_fusion": "all three real views share one reference world; target-view z-consistency plus view-angle ownership",
            "pixel_policy": "real source pixels only; no inpainting, diffusion, crossfade, morph, or generative completion",
        },
        "camera_solves": {k: serialise_solve(v) for k, v in solves.items()},
        "depth_filter": {k: views[k]["edge_qa"] for k in LOCKED_FRAMES},
        "cloud_points": {k: int(len(clouds[k]["points"])) for k in LOCKED_FRAMES},
        "pivot": pivot_qa,
        "endpoint": endpoint,
        "endpoint_real_baseline_deg": baseline_deg,
        "render_max_degree": render_max,
        "stills": stills,
        "motion": motion,
        "leave_one_out_real_view_qa": loo,
        "success_interpretation": (
            "Do not judge by Actions success. Inspect stills/MP4 and leave-one-out QA. "
            "If body topology, basket geometry, or ball identity is wrong, stop here; a generator must not hide it."
        ),
    }
    (args.out / "reshoot_three_view_anchor_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({
        "endpoint": endpoint,
        "baseline_deg": baseline_deg,
        "render_max_degree": render_max,
        "stills": stills,
        "leave_one_out": loo,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
