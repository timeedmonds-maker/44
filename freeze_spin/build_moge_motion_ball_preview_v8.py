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

WIDTH = 960
HEIGHT = 540
REF_LABEL = "In Arena"


def build_body_points(cameras: Path, locked: Path, ball_report: Path, instances_dir: Path, layered_qa: Path, moge_dir: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    ref = next(r for r in rows if r["label"] == REF_LABEL)
    target = v1.read_ball(ball_report)
    layers, owner_masks = v7.build_owner_masks(instances_dir, 2)
    layered = json.loads(layered_qa.read_text(encoding="utf-8"))
    centres = [float(r["depth_fit"]["z_cam_cm"]) for r in layered["layers"][:len(layers)]]
    depth_m = np.load(moge_dir / "moge_depth_m.npy").astype(np.float32)
    valid_moge = np.load(moge_dir / "moge_valid.npy").astype(bool)
    mqa = json.loads((moge_dir / "moge_reference_qa_v1.json").read_text(encoding="utf-8"))
    scale = float(mqa["metric_scale_multiplier_from_ball"])
    smooth = cv2.bilateralFilter(np.where(np.isfinite(depth_m), depth_m * 100.0 * scale, 0.0).astype(np.float32), 7, 18.0, 4.0)
    pts_all, col_all = [], []
    union = np.zeros((HEIGHT, WIDTH), bool)
    for owned, centre in zip(owner_masks, centres):
        mask = owned & valid_moge & np.isfinite(smooth) & (smooth > 0)
        ys, xs = np.where(mask)
        vals = smooth[ys, xs]
        med = float(np.median(vals))
        z = centre + np.clip((vals - med) * 0.85, -65.0, 65.0)
        uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        pts_all.append(v7.backproject_z(ref, uv, z.astype(np.float64)))
        col_all.append(ref["image"][ys, xs])
        union[ys, xs] = True
    return ref, target, np.concatenate(pts_all), np.concatenate(col_all), union, alignment_qa


def ball_patch(ref, ball_report: Path, radius_px: int = 8):
    ball = json.loads(ball_report.read_text(encoding="utf-8"))
    bv = next(v for v in ball["views"] if v["label"] == REF_LABEL)
    cx, cy = [float(x) for x in bv["observed_ball_anchor_px"]]
    x0, x1 = int(round(cx)) - radius_px, int(round(cx)) + radius_px
    y0, y1 = int(round(cy)) - radius_px, int(round(cy)) + radius_px
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(WIDTH - 1, x1), min(HEIGHT - 1, y1)
    tex = ref["image"][y0:y1+1, x0:x1+1].copy()
    yy, xx = np.ogrid[:tex.shape[0], :tex.shape[1]]
    ccx, ccy = cx - x0, cy - y0
    mask = (((xx - ccx) ** 2 + (yy - ccy) ** 2) <= radius_px ** 2).astype(np.uint8) * 255
    corners = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=np.float64)
    X = np.asarray(ball["ball_center_world_cm"], dtype=np.float64)
    z = float((ref["R"] @ X.reshape(3,1) + ref["t"])[2,0])
    world = v7.backproject_z(ref, corners, np.full(4, z, dtype=np.float64))
    return tex, mask, world


def composite_ball(frame, ref, target, degree, tex, mask, world):
    _, _, P = v7.novel_camera(ref, target, degree)
    dst, valid = v1.project(P, world)
    if not np.all(valid):
        return frame
    src = np.array([[0,0],[tex.shape[1]-1,0],[tex.shape[1]-1,tex.shape[0]-1],[0,tex.shape[0]-1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    warped = cv2.warpPerspective(tex, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wm = cv2.warpPerspective(mask, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(wm.astype(np.float32)/255.0,0,1)[:,:,None]
    return (frame.astype(np.float32)*(1-a)+warped.astype(np.float32)*a).astype(np.uint8)


def render_body(points, colors, ref, target, degree):
    if abs(degree) < 1e-9:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        uv, valid = v1.project(ref["P"], points)
        u = np.rint(uv[:,0]).astype(np.int32); vv = np.rint(uv[:,1]).astype(np.int32)
        ok = valid & (u>=0)&(u<WIDTH)&(vv>=0)&(vv<HEIGHT)
        frame[vv[ok],u[ok]] = colors[ok]
        return frame
    radius = 1 if abs(degree) <= 2.0 else 2
    frame, _ = v7.render_points(points, colors, ref, target, float(degree), radius)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--reference-instances", type=Path, required=True)
    ap.add_argument("--layered-qa", type=Path, required=True)
    ap.add_argument("--moge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=61)
    ap.add_argument("--max-degree", type=float, default=5.0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    ref, target, points, colors, union, alignment_qa = build_body_points(args.cameras,args.locked_images,args.ball_report,args.reference_instances,args.layered_qa,args.moge_dir)
    tex, bmask, bworld = ball_patch(ref,args.ball_report)
    degrees=[]
    for i in range(args.frames):
        phase = i / max(args.frames-1,1)
        d = args.max_degree * math.sin(math.pi * phase)
        degrees.append(float(d))
        frame = render_body(points, colors, ref, target, d)
        frame = composite_ball(frame, ref, target, d, tex, bmask, bworld)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)
    qa={
      "prototype":"moge_motion_metric_ball_v8",
      "source_resolution":[960,540],
      "render_target":[3840,2160],
      "degrees_min_max":[round(min(degrees),4),round(max(degrees),4)],
      "frames":args.frames,
      "body_points":int(len(points)),
      "appearance_policy":"real In Arena exact-state NBA pixels only",
      "geometry_policy":"MoGe-2 relative body depth recentered to calibrated layer depths plus independently triangulated metric ball plane",
      "ball_policy":"real reference ball-area pixels projected at the accepted four-camera 3D ball centre; no generated ball",
      "alignment_qa":alignment_qa,
      "success_gate":"0-5-0 motion must read as one continuous frozen action with no layer separation, checkerboard tearing, or detached ball."
    }
    (args.out/"motion_preview_qa_v8.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps(qa,indent=2),flush=True)

if __name__ == "__main__":
    main()
