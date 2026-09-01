from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import build_small_angle_real_pixel_prototype_v1 as v1
import build_small_angle_real_pixel_prototype_v2 as v2

WIDTH = 960
HEIGHT = 540


def mask_name(label: str) -> str:
    return f"{label.replace(' ', '_')}_body_mask_anchor.png"


def attach_masks(rows, masks_dir: Path, ball: np.ndarray):
    out = []
    mask_stats = {}
    for row in rows:
        mask = cv2.imread(str(masks_dir / mask_name(row["label"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(masks_dir / mask_name(row["label"]))
        mask = (mask > 0).astype(np.uint8) * 255
        # Keep the known ball even though COCO person segmentation intentionally excludes it.
        uv, ok = v1.project(row["P"], ball[None, :])
        if ok[0]:
            p = tuple(int(round(x)) for x in uv[0])
            cv2.circle(mask, p, 11, 255, -1, cv2.LINE_AA)
        r = dict(row)
        r["mask"] = mask
        r["mask_float"] = (mask.astype(np.float32) / 255.0)
        out.append(r)
        mask_stats[row["label"]] = int((mask > 0).sum())
    return out, mask_stats


def reconstruct_masked(rows, target, stride=1, depth_steps=140):
    ref = v1.select_reference(rows, target)
    sources = [r for r in rows if r is not ref]
    target_uv, ok = v1.project(ref["P"], target[None, :])
    if not ok[0]:
        raise RuntimeError("Target does not project in reference")
    hu, hv = target_uv[0]

    ys0, xs0 = np.where(ref["mask"] > 0)
    if len(xs0) < 100:
        raise RuntimeError(f"Reference action mask too small: {len(xs0)}")
    x0 = max(0, max(int(xs0.min()) - 12, int(hu - 300)))
    x1 = min(WIDTH - 1, min(int(xs0.max()) + 12, int(hu + 300)))
    y0 = max(0, max(int(ys0.min()) - 12, int(hv - 285)))
    y1 = min(HEIGHT - 1, min(int(ys0.max()) + 12, int(hv + 285)))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"Invalid masked action ROI {(x0, y0, x1, y1)}")

    xs = np.arange(x0, x1 + 1, stride, dtype=np.float64)
    ys = np.arange(y0, y1 + 1, stride, dtype=np.float64)
    uu, vv = np.meshgrid(xs, ys)
    Hh, Ww = uu.shape
    ui = uu.astype(np.int32); vi = vv.astype(np.int32)
    ref_support = ref["mask"][vi, ui] > 0

    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(Hh * Ww)], axis=0)
    rays_cam = np.linalg.inv(ref["K"]) @ pix
    rays_world = ref["R"].T @ rays_cam
    rays_world /= np.linalg.norm(rays_world, axis=0, keepdims=True) + 1e-12
    rays = rays_world.T.reshape(Hh, Ww, 3)

    centre_depth = float(np.linalg.norm(target - ref["C"]))
    depths = np.linspace(max(80.0, centre_depth - 500.0), centre_depth + 500.0, depth_steps)
    ref_feat = ref["feat"][vi, ui]
    ref_rgb = ref["image"][vi, ui]
    best = np.full((Hh, Ww), np.inf, dtype=np.float32)
    second = np.full((Hh, Ww), np.inf, dtype=np.float32)
    best_depth = np.zeros((Hh, Ww), dtype=np.float32)
    support_best = np.zeros((Hh, Ww), dtype=np.uint8)

    for depth in depths:
        X = ref["C"][None, None, :] + rays * depth
        physical = ((X[:, :, 2] >= 15.0) & (X[:, :, 2] <= 390.0) &
                    (np.abs(X[:, :, 0] - target[0]) <= 390.0) &
                    (np.abs(X[:, :, 1] - target[1]) <= 390.0) & ref_support)
        costs = []
        valids = []
        flat = X.reshape(-1, 3)
        for src in sources:
            suv, sok = v1.project(src["P"], flat)
            su = suv[:, 0].reshape(Hh, Ww); sv = suv[:, 1].reshape(Hh, Ww)
            inside = sok.reshape(Hh, Ww) & physical & (su >= 1) & (su < WIDTH - 2) & (sv >= 1) & (sv < HEIGHT - 2)
            src_mask = v1.bilinear(src["mask_float"], su, sv)
            inside &= src_mask > 0.45
            sf = v1.bilinear(src["feat"], su, sv)
            dl = np.abs(sf[:, :, :3] - ref_feat[:, :, :3]).mean(axis=2)
            dg = np.abs(sf[:, :, 3] - ref_feat[:, :, 3])
            c = np.minimum(dl, 0.7) * 0.80 + np.minimum(dg, 0.7) * 0.20
            costs.append(c)
            valids.append(inside)
        stack = np.stack(costs, axis=2)
        vm = np.stack(valids, axis=2)
        stack[~vm] = np.nan
        support = vm.sum(axis=2).astype(np.uint8)
        with np.errstate(all="ignore"):
            cost = np.nanmedian(stack, axis=2)
        cost = np.where(support >= 1, cost, np.inf)
        improved = cost < best
        second = np.where(improved, best, np.minimum(second, cost))
        best = np.where(improved, cost, best)
        best_depth = np.where(improved, depth, best_depth)
        support_best = np.where(improved, support, support_best)

    margin = second - best
    textured = ref_feat[:, :, 3] > 0.018
    # Core points have agreement from at least two source cameras (three views including reference).
    core = (ref_support & np.isfinite(best) & (support_best >= 2) &
            (best < 0.30) & (margin > 0.004) & textured)
    # Extended points still have genuine stereo (reference + one source), but must pass a stricter
    # photometric/uniqueness test. This is reported separately rather than silently weakening core QA.
    extended = (ref_support & np.isfinite(best) & (support_best == 1) &
                (best < 0.17) & (margin > 0.008) & textured)
    good = core | extended

    Xbest = ref["C"][None, None, :] + rays * best_depth[:, :, None]
    pts = Xbest[good]
    colors = ref_rgb[good]
    levels = np.where(core[good], 2, 1).astype(np.uint8)
    masked_candidates = int(ref_support.sum())
    qa = {
        "reference": ref["label"],
        "sources": [s["label"] for s in sources],
        "roi": [x0, y0, x1, y1],
        "grid_points": int(Hh * Ww),
        "masked_candidate_pixels": masked_candidates,
        "accepted_points": int(len(pts)),
        "core_three_view_points": int(core.sum()),
        "extended_two_view_points": int(extended.sum()),
        "accepted_fraction_of_mask": round(float(len(pts) / max(masked_candidates, 1)), 6),
        "median_cost": None if len(pts) == 0 else round(float(np.median(best[good])), 6),
        "median_depth_margin": None if len(pts) == 0 else round(float(np.median(margin[good])), 6),
    }
    return ref, pts, colors, levels, qa


def render_dense(points, colors, levels, ref, target, deg, scale=4):
    Q = v1.rz(deg)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    S = np.diag([scale, scale, 1.0])
    P = (S @ ref["K"]) @ np.hstack([R, t])
    uv, valid = v1.project(P, points)
    z = (R @ (points - C).T).T[:, 2]
    valid &= z > 1.0
    ow, oh = WIDTH * scale, HEIGHT * scale
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < ow) & (uv[:, 1] >= 0) & (uv[:, 1] < oh)
    canvas = np.zeros((oh, ow, 3), np.uint8)
    cover = np.zeros((oh, ow), np.uint8)
    ids = np.where(valid)[0]
    ids = ids[np.argsort(z[ids])[::-1]]
    for i in ids:
        x, y = int(round(uv[i, 0])), int(round(uv[i, 1]))
        # One native pixel spans roughly 4x4 output pixels. A 3 px radius closes only
        # sub-pixel projection gaps while retaining holes caused by unsupported geometry.
        radius = 3 if levels[i] >= 2 else 2
        cv2.circle(canvas, (x, y), radius, tuple(int(v) for v in colors[i]), -1, cv2.LINE_AA)
        cv2.circle(cover, (x, y), radius, 255, -1, cv2.LINE_AA)
    return canvas, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depth-steps", type=int, default=140)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, alignment_qa = v2.aligned_camera_rows(args.cameras, args.locked_images, args.ball_report)
    ball = v1.read_ball(args.ball_report)
    rows, mask_stats = attach_masks(rows, args.masks, ball)
    ref, points, colors, levels, recon = reconstruct_masked(rows, ball, args.stride, args.depth_steps)
    if len(points) < 600:
        raise RuntimeError(f"Body-masked reconstruction too sparse: {len(points)} supported pixels")

    degrees = [0, 3, 5, 8]
    render_qa = []
    for d in degrees:
        frame, cover = render_dense(points, colors, levels, ref, ball, d, scale=4)
        cv2.imwrite(str(args.out / f"body_virtual_{d:02d}deg_uhd.png"), frame)
        cv2.imwrite(str(args.out / f"body_coverage_{d:02d}deg_uhd.png"), cover)
        render_qa.append({
            "degrees": d,
            "covered_pixels": int((cover > 0).sum()),
            "coverage_fraction_of_4k_frame": round(float((cover > 0).mean()), 6),
        })

    # Ground-truth visual reference for the selected body mask in the same F28 metric coordinate system.
    ref_cutout = cv2.bitwise_and(ref["image"], ref["image"], mask=ref["mask"])
    cv2.imwrite(str(args.out / "body_reference_real_00deg_native.png"), ref_cutout)
    cv2.imwrite(str(args.out / "body_reference_real_00deg_uhd.png"),
                cv2.resize(ref_cutout, (3840, 2160), interpolation=cv2.INTER_NEAREST))

    qa = {
        "prototype": "small_angle_identity_free_body_masked_v3",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "identity_policy": "no player identity matching; all action-body pixels near the known ball are treated as foreground support",
        "segmentation_role": "2D support only; segmentation does not determine 3D depth or camera pose",
        "geometry_policy": "3D depth must be supported by calibrated multi-view photometric agreement",
        "source_policy": "real official NBA pixels only; no generative fill or hallucinated body completion",
        "mask_pixels_by_view": mask_stats,
        "exact_state_alignment_qa": alignment_qa,
        "reconstruction": recon,
        "renders": render_qa,
        "degrees": degrees,
        "success_gate": "0deg reconstruction should resemble the real masked body cutout; 3-5deg should preserve coherent continuous body silhouettes before any background compositing or hole filling",
    }
    (args.out / "body_masked_prototype_qa_v3.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
