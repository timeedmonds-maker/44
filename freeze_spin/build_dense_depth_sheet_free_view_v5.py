from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

import build_small_angle_real_pixel_prototype_v1 as v1
import build_small_angle_real_pixel_prototype_v2 as v2

WIDTH = 960
HEIGHT = 540
REFERENCE_LABEL = "In Arena"


def safe(label: str) -> str:
    return label.replace(" ", "_")


def load_rows(cameras: Path, locked: Path, ball_report: Path, masks: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    out = []
    for r in rows:
        mask_path = masks / f"{safe(r['label'])}_body_mask_anchor.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        rr = dict(r)
        rr["mask"] = mask > 0
        lab = cv2.cvtColor(r["image"], cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        rr["lab"] = lab
        out.append(rr)
    return out, alignment_qa


def bilinear_rows(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    mapx = uv[:, 0].astype(np.float32).reshape(-1, 1)
    mapy = uv[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, image.shape[2])


def nearest_mask_lookup(mask: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    inside = valid & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    hit = np.zeros(len(uv), dtype=bool)
    ids = np.where(inside)[0]
    if len(ids):
        hit[ids] = mask[vv[ids], u[ids]]
    return hit


def make_reference_pixels(ref):
    mask = ref["mask"].copy()
    ys, xs = np.where(mask)
    if len(xs) < 3000:
        raise RuntimeError(f"Reference action mask unexpectedly small: {len(xs)} px")
    pix = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    colors = ref["image"][ys, xs]
    lab = ref["lab"][ys, xs]
    ones = np.ones((len(pix), 1), dtype=np.float64)
    h = np.hstack([pix, ones]).T
    rays_cam = np.linalg.inv(ref["K"]) @ h
    rays_world = ref["R"].T @ rays_cam
    rays_world /= np.linalg.norm(rays_world, axis=0, keepdims=True) + 1e-12
    rays = rays_world.T
    return mask, xs, ys, pix, colors, lab, rays


def solve_dense_depth(rows, ref, target: np.ndarray, depth_steps: int):
    ref_mask, xs, ys, pix, colors, ref_lab, rays = make_reference_pixels(ref)
    sources = [r for r in rows if r["label"] != ref["label"]]
    n = len(xs)
    centre_depth = float(np.linalg.norm(target - ref["C"]))
    depths = np.linspace(max(120.0, centre_depth - 430.0), centre_depth + 430.0, depth_steps)

    best_score = np.full(n, np.inf, dtype=np.float32)
    second_score = np.full(n, np.inf, dtype=np.float32)
    best_depth = np.full(n, centre_depth, dtype=np.float32)
    best_support = np.zeros(n, dtype=np.uint8)
    best_color_cost = np.full(n, np.inf, dtype=np.float32)

    for depth in depths:
        X = ref["C"][None, :] + rays * depth
        physical = (
            (X[:, 2] >= 10.0) & (X[:, 2] <= 365.0) &
            (np.abs(X[:, 0] - target[0]) <= 420.0) &
            (np.abs(X[:, 1] - target[1]) <= 420.0)
        )
        support = np.zeros(n, dtype=np.uint8)
        color_sum = np.zeros(n, dtype=np.float32)

        for src in sources:
            uv, valid = v1.project(src["P"], X)
            hit = nearest_mask_lookup(src["mask"], uv, valid) & physical
            if not np.any(hit):
                continue
            sampled = bilinear_rows(src["lab"], uv)
            d = np.abs(sampled - ref_lab).mean(axis=1)
            d = np.minimum(d, 0.60)
            color_sum += np.where(hit, d, 0.0).astype(np.float32)
            support += hit.astype(np.uint8)

        mean_color = np.where(support > 0, color_sum / np.maximum(support, 1), 1.0)
        # Silhouette agreement is primary. Colour agreement resolves ambiguous depth intervals.
        score = mean_color + (3.0 - support.astype(np.float32)) * 0.105
        score = np.where((support > 0) & physical, score, np.inf)
        improved = score < best_score
        second_score = np.where(improved, best_score, np.minimum(second_score, score))
        best_score = np.where(improved, score, best_score)
        best_depth = np.where(improved, depth, best_depth)
        best_support = np.where(improved, support, best_support)
        best_color_cost = np.where(improved, mean_color, best_color_cost)

    raw_depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    raw_support = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    raw_score = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    raw_depth[ys, xs] = best_depth
    raw_support[ys, xs] = best_support
    raw_score[ys, xs] = best_score

    supported = ref_mask & (raw_support >= 1)
    strong = ref_mask & (raw_support >= 2)
    if int(supported.sum()) < 1000:
        raise RuntimeError("Too few reference-body pixels receive multi-view silhouette depth support")

    # Preserve every reference body pixel. Unsupported pixels borrow the nearest supported depth;
    # this fills disocclusion-risk limbs without deleting them from the prototype.
    filled = raw_depth.copy()
    unsupported = ref_mask & ~supported
    if np.any(unsupported):
        _, indices = ndimage.distance_transform_edt(~supported, return_indices=True)
        nearest = raw_depth[indices[0], indices[1]]
        filled[unsupported] = nearest[unsupported]

    # Smooth only inside the body silhouette to suppress staircase depth while retaining component shape.
    m = ref_mask.astype(np.float32)
    num = ndimage.gaussian_filter(filled * m, sigma=1.15)
    den = ndimage.gaussian_filter(m, sigma=1.15) + 1e-6
    smooth = np.where(ref_mask, num / den, 0.0).astype(np.float32)

    # Keep strong measurements close to the measured solution while still reducing isolated spikes.
    smooth[strong] = 0.72 * raw_depth[strong] + 0.28 * smooth[strong]

    final_depth = smooth[ys, xs]
    points = ref["C"][None, :] + rays * final_depth[:, None]
    margin = second_score - best_score
    qa = {
        "reference": ref["label"],
        "reference_mask_pixels": int(n),
        "depth_steps": int(depth_steps),
        "depth_search_cm": [round(float(depths[0]), 3), round(float(depths[-1]), 3)],
        "supported_pixels_ge_1_other_view": int(supported.sum()),
        "strong_pixels_ge_2_other_views": int(strong.sum()),
        "supported_fraction": round(float(supported.sum() / n), 6),
        "strong_fraction": round(float(strong.sum() / n), 6),
        "best_support_histogram": {str(i): int((best_support == i).sum()) for i in range(4)},
        "median_best_score_supported": round(float(np.median(best_score[best_support > 0])), 6),
        "median_colour_cost_supported": round(float(np.median(best_color_cost[best_support > 0])), 6),
        "median_depth_margin_supported": round(float(np.median(margin[np.isfinite(margin) & (best_support > 0)])), 6),
        "final_depth_cm_percentiles": [round(float(v), 3) for v in np.percentile(final_depth, [1, 10, 50, 90, 99])],
        "policy": "Every reference foreground pixel is retained. Other calibrated silhouettes estimate depth; unsupported pixels use nearest supported body depth rather than being deleted or hallucinated.",
    }
    return ref_mask, xs, ys, points, colors, raw_support, smooth, qa


def rotate_about_target(ref, target: np.ndarray, degree: float):
    Q = v1.rz(degree)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    P = ref["K"] @ np.hstack([R, t])
    return C, R, P


def render_dense(points, colors, ref, target: np.ndarray, degree: float):
    C, R, P = rotate_about_target(ref, target, degree)
    uv, valid = v1.project(P, points)
    z = (R @ (points - C).T).T[:, 2]
    valid &= z > 1.0
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    valid &= (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cover = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    ids = np.where(valid)[0]
    # Far to near: nearest real body pixels overwrite farther ones.
    ids = ids[np.argsort(z[ids])[::-1]]
    radius = 1 if abs(degree) <= 3.0 else 2
    for i in ids:
        x, y = int(u[i]), int(vv[i])
        cv2.circle(canvas, (x, y), radius, tuple(int(c) for c in colors[i]), -1, cv2.LINE_AA)
        cv2.circle(cover, (x, y), radius, 255, -1, cv2.LINE_AA)
    return canvas, cover, int(len(ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-steps", type=int, default=150)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, alignment_qa = load_rows(args.cameras, args.locked_images, args.ball_report, args.masks)
    by_label = {r["label"]: r for r in rows}
    if REFERENCE_LABEL not in by_label:
        raise RuntimeError(f"Missing required reference {REFERENCE_LABEL}")
    ref = by_label[REFERENCE_LABEL]
    target = v1.read_ball(args.ball_report)

    ref_mask, xs, ys, points, colors, support_map, depth_map, depth_qa = solve_dense_depth(
        rows, ref, target, args.depth_steps
    )

    ref_cut = np.zeros_like(ref["image"])
    ref_cut[ys, xs] = colors
    cv2.imwrite(str(args.out / "dense_real_reference_native.png"), ref_cut)
    cv2.imwrite(str(args.out / "dense_depth_support_native.png"), support_map * 80)
    depth_vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    vals = depth_map[ref_mask]
    lo, hi = np.percentile(vals, [2, 98])
    depth_vis[ref_mask] = np.clip((depth_map[ref_mask] - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.out / "dense_depth_native.png"), depth_vis)

    degrees = [0, 3, 5, 8]
    renders = []
    for degree in degrees:
        if degree == 0:
            native = ref_cut.copy()
            cover = (ref_mask.astype(np.uint8) * 255)
            visible = int(len(points))
        else:
            native, cover, visible = render_dense(points, colors, ref, target, degree)
        cv2.imwrite(str(args.out / f"dense_virtual_{degree:02d}deg_native.png"), native)
        cv2.imwrite(str(args.out / f"dense_virtual_{degree:02d}deg_uhd.png"),
                    cv2.resize(native, (3840, 2160), interpolation=cv2.INTER_LANCZOS4))
        cv2.imwrite(str(args.out / f"dense_coverage_{degree:02d}deg_native.png"), cover)
        renders.append({
            "degree": degree,
            "visible_reference_pixels": visible,
            "covered_native_pixels": int((cover > 0).sum()),
            "covered_fraction_native_frame": round(float((cover > 0).mean()), 6),
        })

    qa = {
        "prototype": "dense_reference_pixel_depth_sheet_v5",
        "method": "dense depth-image-based rendering from the exact-state In Arena body mask; depth is solved from three other exact-state-aligned metric-camera silhouettes with photometric tie-breaking",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "identity_policy": "none; player names and cross-camera player identity correspondence are not used",
        "source_policy": "real official NBA foreground pixels only; no diffusion, generated fill, optical-flow morph, or synthesized anatomy",
        "zero_degree_policy": "exact aligned reference cutout is emitted directly, so any 0-degree degradation is forbidden by construction",
        "exact_state_alignment_qa": alignment_qa,
        "target_ball_world_cm": target.tolist(),
        "depth": depth_qa,
        "renders": renders,
        "success_gate": "3-degree and 5-degree body silhouettes must remain continuous and photographic enough to justify adding static-arena compositing; 8 degrees is exploratory.",
    }
    (args.out / "dense_depth_sheet_qa_v5.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
