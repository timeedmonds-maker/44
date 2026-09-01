from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import build_small_angle_real_pixel_prototype_v1 as v1
import build_moge_depth_free_view_v7 as v7
import build_moge_motion_ball_preview_v8 as v8
import build_full_action_moge_free_view_v9 as v9

WIDTH = 960
HEIGHT = 540


def build_full_frame_world(ref, depth_cm: np.ndarray):
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    uv = np.column_stack([xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)])
    z = depth_cm.ravel().astype(np.float64)
    valid = np.isfinite(z) & (z > 50.0) & (z < 12000.0)
    world = np.zeros((len(z), 3), dtype=np.float64)
    world[valid] = v7.backproject_z(ref, uv[valid], z[valid])
    return world, valid, xx.astype(np.float32), yy.astype(np.float32)


def forward_flow(world: np.ndarray, valid_world: np.ndarray, ref, target: np.ndarray, degree: float,
                 grid_x: np.ndarray, grid_y: np.ndarray):
    _, _, P = v7.novel_camera(ref, target, degree)
    uv_new = np.zeros((len(world), 2), dtype=np.float64)
    valid = np.zeros(len(world), dtype=bool)
    projected, ok = v1.project(P, world[valid_world])
    ids = np.where(valid_world)[0]
    uv_new[ids] = projected
    valid[ids] = ok
    fx = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    fy = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    flat_x = grid_x.ravel().astype(np.float64)
    flat_y = grid_y.ravel().astype(np.float64)
    fx.ravel()[valid] = (uv_new[valid, 0] - flat_x[valid]).astype(np.float32)
    fy.ravel()[valid] = (uv_new[valid, 1] - flat_y[valid]).astype(np.float32)
    return fx, fy


def geometry_inverse_remap(source: np.ndarray, source_valid: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray,
                           grid_x: np.ndarray, grid_y: np.ndarray, iterations: int = 3):
    map_x = grid_x.copy()
    map_y = grid_y.copy()
    for _ in range(iterations):
        sx = cv2.remap(flow_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sy = cv2.remap(flow_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        map_x = grid_x - sx
        map_y = grid_y - sy
    warped = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(source_valid, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, valid


def alpha_composite(base: np.ndarray, layer: np.ndarray, mask: np.ndarray):
    a = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[:, :, None]
    return (base.astype(np.float32) * (1.0 - a) + layer.astype(np.float32) * a).astype(np.uint8)


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

    rows, ref, sources, target, alignment_qa = v9.load_scene(
        args.cameras, args.locked_images, args.ball_report, args.body_masks
    )
    depth_m = np.load(args.moge_dir / "moge_depth_m.npy").astype(np.float32)
    valid_moge = np.load(args.moge_dir / "moge_valid.npy").astype(bool)
    mqa = json.loads((args.moge_dir / "moge_reference_qa_v1.json").read_text(encoding="utf-8"))
    scale = float(mqa["metric_scale_multiplier_from_ball"])
    registered = np.where(valid_moge & np.isfinite(depth_m), depth_m * 100.0 * scale, 0.0).astype(np.float32)
    registered = cv2.bilateralFilter(registered, d=7, sigmaColor=18.0, sigmaSpace=4.0)

    ref_mask = ref["body_mask"] & valid_moge & np.isfinite(registered) & (registered > 0)
    z_ball = v9.ball_camera_z(ref, args.ball_report)
    gain, offset, mapping_qa, _ = v9.choose_global_depth_mapping(ref, sources, ref_mask, registered, z_ball)
    body_points, body_colors, exact_mask, final_z = v9.build_full_action_points(
        ref, ref_mask, registered, z_ball, gain, offset
    )

    world_bg, valid_bg_depth, grid_x, grid_y = build_full_frame_world(ref, registered)

    hole = cv2.dilate((exact_mask.astype(np.uint8) * 255), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    ball_payload = json.loads(args.ball_report.read_text(encoding="utf-8"))
    bv = next(v for v in ball_payload["views"] if v["label"] == "In Arena")
    bx, by = [int(round(float(x))) for x in bv["observed_ball_anchor_px"]]
    cv2.circle(hole, (bx, by), 12, 255, -1)
    source_background = ref["image"].copy()
    source_background[hole > 0] = 0
    source_valid = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    source_valid[hole > 0] = 0
    source_valid[~valid_bg_depth.reshape(HEIGHT, WIDTH)] = 0

    ball_tex, ball_mask, ball_world = v8.ball_patch(ref, args.ball_report)

    def render_scene(degree: float):
        if abs(degree) < 1e-9:
            return ref["image"].copy(), np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
        fx, fy = forward_flow(world_bg, valid_bg_depth, ref, target, degree, grid_x, grid_y)
        background, bg_valid = geometry_inverse_remap(source_background, source_valid, fx, fy, grid_x, grid_y, 3)
        action, action_cov = v9.render_full_action(body_points, body_colors, exact_mask, ref, target, degree)
        frame = alpha_composite(background, action, action_cov)
        frame = v8.composite_ball(frame, ref, target, degree, ball_tex, ball_mask, ball_world)
        resolved = np.maximum(bg_valid, action_cov)
        return frame, resolved

    stills = []
    for degree in [0, 1, 2, 3, 5]:
        frame, resolved = render_scene(float(degree))
        cv2.imwrite(str(args.out / f"full_scene_{degree:02d}deg_native.png"), frame)
        unresolved = (resolved == 0).astype(np.uint8) * 255
        cv2.imwrite(str(args.out / f"full_scene_unresolved_{degree:02d}deg_native.png"), unresolved)
        stills.append({
            "degree": degree,
            "resolved_fraction_native": round(float((resolved > 0).mean()), 6),
            "unresolved_pixels": int((resolved == 0).sum()),
        })

    degrees = []
    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        degree = float(args.max_degree * math.sin(math.pi * phase))
        degrees.append(degree)
        frame, _ = render_scene(degree)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    qa = {
        "prototype": "full_scene_moge_native_free_view_v10",
        "source_resolution": [960, 540],
        "render_resolution": [960, 540],
        "resolution_policy": "native only",
        "appearance_policy": "all visible appearance pixels come from the exact-state official NBA In Arena frame; no generated fill",
        "background_geometry": "MoGe-2 full-frame depth registered to the accepted metric ball depth, rendered by geometry-derived inverse remap rather than optical flow",
        "foreground_geometry": "v9 complete action mask with MoGe relative depth compressed/offset by three calibrated silhouette views",
        "occlusion_policy": "source action pixels are removed from the background before reprojection; newly exposed regions remain unresolved rather than hallucinated",
        "reference_action_mask_pixels": int(exact_mask.sum()),
        "body_depth_mapping": {
            "gain": round(float(gain), 6),
            "offset_cm": round(float(offset), 6),
            "support": mapping_qa,
            "depth_percentiles_cm": [round(float(x), 4) for x in np.percentile(final_z, [1, 10, 50, 90, 99])],
        },
        "exact_state_alignment_qa": alignment_qa,
        "stills": stills,
        "motion_frames": args.frames,
        "motion_degrees_min_max": [round(float(min(degrees)), 4), round(float(max(degrees)), 4)],
        "success_gate": "0 degrees must exactly reproduce the reference frame; 1-5 degrees must read as coherent scene camera travel with stable basket/background and continuous action, with unresolved disocclusions explicitly visible rather than synthesized."
    }
    (args.out / "full_scene_moge_qa_v10.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
