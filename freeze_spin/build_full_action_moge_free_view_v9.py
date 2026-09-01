from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import build_small_angle_real_pixel_prototype_v1 as v1
import build_small_angle_real_pixel_prototype_v2 as v2
import build_moge_depth_free_view_v7 as v7
import build_moge_motion_ball_preview_v8 as v8

WIDTH = 960
HEIGHT = 540
REF_LABEL = "In Arena"


def safe(label: str) -> str:
    return label.replace(" ", "_")


def mask_hit(mask: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    u = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    inside = valid & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    out = np.zeros(len(uv), dtype=bool)
    ids = np.where(inside)[0]
    if len(ids):
        out[ids] = mask[vv[ids], u[ids]]
    return out


def load_scene(cameras: Path, locked: Path, ball_report: Path, body_masks: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    for r in rows:
        p = body_masks / f"{safe(r['label'])}_body_mask_anchor.png"
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(p)
        r["body_mask"] = m > 0
    ref = next(r for r in rows if r["label"] == REF_LABEL)
    sources = [r for r in rows if r["label"] != REF_LABEL]
    target = v1.read_ball(ball_report)
    return rows, ref, sources, target, alignment_qa


def ball_camera_z(ref, ball_report: Path) -> float:
    payload = json.loads(ball_report.read_text(encoding="utf-8"))
    X = np.asarray(payload["ball_center_world_cm"], dtype=np.float64)
    return float((ref["R"] @ X.reshape(3, 1) + ref["t"])[2, 0])


def choose_global_depth_mapping(ref, sources, target_mask: np.ndarray, registered_depth: np.ndarray,
                                z_ball: float, sample_limit: int = 5000):
    ys, xs = np.where(target_mask & np.isfinite(registered_depth) & (registered_depth > 0))
    if len(xs) < 5000:
        raise RuntimeError(f"Full action mask too small for v9: {len(xs)} pixels")
    stride = max(1, int(math.ceil(len(xs) / sample_limit)))
    ys = ys[::stride]
    xs = xs[::stride]
    raw = registered_depth[ys, xs].astype(np.float64)
    uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    gains = np.linspace(0.05, 0.60, 23)
    offsets = np.linspace(-55.0, 55.0, 23)
    best = None
    curve = []
    for gain in gains:
        base = z_ball + gain * (raw - z_ball)
        for offset in offsets:
            z = base + offset
            physically_plausible = (z >= z_ball - 190.0) & (z <= z_ball + 190.0)
            world = v7.backproject_z(ref, uv, z)
            support = np.zeros(len(uv), dtype=np.uint8)
            per_view = []
            for src in sources:
                suv, valid = v1.project(src["P"], world)
                hit = mask_hit(src["body_mask"], suv, valid) & physically_plausible
                support += hit.astype(np.uint8)
                per_view.append(float(hit.mean()))
            any_hit = support >= 1
            two_hit = support >= 2
            three_hit = support >= 3
            score = (
                float(any_hit.mean())
                + 0.70 * float(two_hit.mean())
                + 0.20 * float(three_hit.mean())
                - 0.0008 * abs(float(offset))
                - 0.05 * float((gain - 0.25) ** 2)
            )
            row = {
                "gain": float(gain),
                "offset_cm": float(offset),
                "score": float(score),
                "any_hit_fraction": float(any_hit.mean()),
                "two_hit_fraction": float(two_hit.mean()),
                "three_hit_fraction": float(three_hit.mean()),
                "per_source_hit_fraction": per_view,
            }
            curve.append(row)
            if best is None or score > best[0]:
                best = (score, gain, offset, row)
    assert best is not None
    return float(best[1]), float(best[2]), best[3], curve


def build_full_action_points(ref, body_mask: np.ndarray, registered_depth: np.ndarray,
                             z_ball: float, gain: float, offset: float):
    mask = body_mask & np.isfinite(registered_depth) & (registered_depth > 0)
    ys, xs = np.where(mask)
    raw = registered_depth[ys, xs].astype(np.float64)
    z = z_ball + gain * (raw - z_ball) + offset
    z = np.clip(z, z_ball - 190.0, z_ball + 190.0)
    uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    points = v7.backproject_z(ref, uv, z)
    colors = ref["image"][ys, xs]
    return points, colors, mask, z


def render_full_action(points, colors, exact_mask, ref, target, degree: float):
    if abs(degree) < 1e-9:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[exact_mask] = ref["image"][exact_mask]
        coverage = exact_mask.astype(np.uint8) * 255
        return frame, coverage
    radius = 1 if degree <= 3.0 else 2
    return v7.render_points(points, colors, ref, target, degree, radius)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--body-masks", type=Path, required=True)
    ap.add_argument("--moge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--max-degree", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, ref, sources, target, alignment_qa = load_scene(
        args.cameras, args.locked_images, args.ball_report, args.body_masks
    )
    depth_m = np.load(args.moge_dir / "moge_depth_m.npy").astype(np.float32)
    valid = np.load(args.moge_dir / "moge_valid.npy").astype(bool)
    mqa = json.loads((args.moge_dir / "moge_reference_qa_v1.json").read_text(encoding="utf-8"))
    scale = float(mqa["metric_scale_multiplier_from_ball"])
    registered = np.where(valid & np.isfinite(depth_m), depth_m * 100.0 * scale, 0.0).astype(np.float32)
    registered = cv2.bilateralFilter(registered, d=7, sigmaColor=18.0, sigmaSpace=4.0)

    ref_mask = ref["body_mask"] & valid & np.isfinite(registered) & (registered > 0)
    z_ball = ball_camera_z(ref, args.ball_report)
    gain, offset, mapping_qa, _ = choose_global_depth_mapping(
        ref, sources, ref_mask, registered, z_ball
    )
    points, colors, exact_mask, final_z = build_full_action_points(
        ref, ref_mask, registered, z_ball, gain, offset
    )

    if len(points) < 20000:
        raise RuntimeError(f"v9 failed to preserve enough full-action pixels: {len(points)}")

    depth_vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    lo, hi = np.percentile(final_z, [1, 99])
    ys, xs = np.where(exact_mask)
    depth_vis[ys, xs] = np.clip((final_z - lo) * 255.0 / max(float(hi - lo), 1e-6), 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.out / "full_action_depth_native.png"), depth_vis)

    ball_tex, ball_mask, ball_world = v8.ball_patch(ref, args.ball_report)

    stills = []
    for degree in [0, 1, 2, 3, 5]:
        frame, cover = render_full_action(points, colors, exact_mask, ref, target, float(degree))
        frame = v8.composite_ball(frame, ref, target, float(degree), ball_tex, ball_mask, ball_world)
        cv2.imwrite(str(args.out / f"full_action_{degree:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"full_action_coverage_{degree:02d}deg_native.png"), cover)
        stills.append({
            "degree": degree,
            "covered_native_pixels": int((cover > 0).sum()),
            "coverage_fraction_native": round(float((cover > 0).mean()), 6),
        })

    degrees = []
    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        degree = float(args.max_degree * math.sin(math.pi * phase))
        degrees.append(degree)
        frame, _ = render_full_action(points, colors, exact_mask, ref, target, degree)
        frame = v8.composite_ball(frame, ref, target, degree, ball_tex, ball_mask, ball_world)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    qa = {
        "prototype": "full_action_moge_native_free_view_v9",
        "source_resolution": [960, 540],
        "render_resolution": [960, 540],
        "resolution_policy": "native only; no enlargement or upscaling in reconstruction or QA",
        "reference": REF_LABEL,
        "identity_policy": "none; uses the union action-body silhouette rather than per-player matching",
        "appearance_policy": "all rendered body and ball pixels originate from the exact-state official NBA reference image",
        "geometry_policy": "MoGe relative depth over the complete reference action mask, globally compressed/offset by calibrated silhouette agreement in three independent metric cameras; ball remains independently triangulated",
        "reference_action_mask_pixels": int(ref_mask.sum()),
        "body_points": int(len(points)),
        "metric_ball_camera_z_cm": round(float(z_ball), 6),
        "depth_mapping": {
            "relative_depth_gain": round(float(gain), 6),
            "offset_cm": round(float(offset), 6),
            "selected_support": mapping_qa,
            "final_depth_percentiles_cm": [round(float(x), 4) for x in np.percentile(final_z, [1, 10, 50, 90, 99])],
        },
        "exact_state_alignment_qa": alignment_qa,
        "stills": stills,
        "motion_frames": args.frames,
        "motion_degrees_min_max": [round(float(min(degrees)), 4), round(float(max(degrees)), 4)],
        "success_gate": "0-degree must preserve the full visible action cutout; 1-5 degree motion must remain continuous with no detached body layers, checkerboard tearing, or detached ball."
    }
    (args.out / "full_action_moge_qa_v9.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
