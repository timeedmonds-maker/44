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
REF_LABEL = "In Arena"


def safe(label: str) -> str:
    return label.replace(" ", "_")


def view_azimuth(C: np.ndarray, target: np.ndarray) -> float:
    d = C[:2] - target[:2]
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def angle_delta(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def build_source_clouds(rows, ball_report: Path, body_masks: Path, moge_dir: Path):
    q = json.loads((moge_dir / "moge_all_views_qa_v2.json").read_text(encoding="utf-8"))
    ball = json.loads(ball_report.read_text(encoding="utf-8"))
    ball_views = {v["label"]: v for v in ball["views"]}
    clouds = []
    mapping_reports = {}
    for row in rows:
        label = row["label"]
        depth_m = np.load(moge_dir / f"{safe(label)}_depth_m.npy").astype(np.float32)
        valid = np.load(moge_dir / f"{safe(label)}_valid.npy").astype(bool)
        scale = float(q["views"][label]["metric_scale_multiplier_from_ball"])
        registered = np.where(valid & np.isfinite(depth_m), depth_m * 100.0 * scale, 0.0).astype(np.float32)
        registered = cv2.bilateralFilter(registered, d=7, sigmaColor=18.0, sigmaSpace=4.0)
        zball = float(q["views"][label]["metric_ball_camera_z_cm"])
        sources = [r for r in rows if r["label"] != label]
        gain, offset, mqa, _ = v9.choose_global_depth_mapping(row, sources, row["body_mask"], registered, zball)
        final_depth = registered.astype(np.float64)
        bm = row["body_mask"] & valid & np.isfinite(registered) & (registered > 0)
        final_depth[bm] = zball + gain * (registered[bm] - zball) + offset
        final_depth[bm] = np.clip(final_depth[bm], zball - 190.0, zball + 190.0)

        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        keep = valid & np.isfinite(final_depth) & (final_depth > 50.0) & (final_depth < 12000.0)
        bp = np.asarray(ball_views[label]["observed_ball_anchor_px"], dtype=np.float64)
        ball_exclude = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.circle(ball_exclude, tuple(int(round(x)) for x in bp), 12, 1, -1)
        keep &= ball_exclude == 0
        ys, xs = np.where(keep)
        uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        z = final_depth[ys, xs]
        points = v7.backproject_z(row, uv, z).astype(np.float32)
        colors = row["image"][ys, xs].copy()
        clouds.append({
            "label": label,
            "C": row["C"].copy(),
            "points": points,
            "colors": colors,
        })
        mapping_reports[label] = {
            "body_pixels": int(bm.sum()),
            "relative_depth_gain": round(float(gain), 6),
            "offset_cm": round(float(offset), 6),
            "support": mqa,
            "cloud_points": int(len(points)),
        }
    return clouds, mapping_reports, q


def raster_cloud(points: np.ndarray, colors: np.ndarray, C: np.ndarray, R: np.ndarray, P: np.ndarray, radius: int = 1):
    uv, valid = v1.project(P, points.astype(np.float64))
    z = (R @ (points.astype(np.float64) - C).T).T[:, 2]
    u = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    valid &= (z > 1.0) & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    ids = np.where(valid)[0]
    if not len(ids):
        return np.zeros((HEIGHT, WIDTH, 3), np.uint8), np.zeros((HEIGHT, WIDTH), np.uint8)
    pix = vv[ids] * WIDTH + u[ids]
    zbuf = np.full(WIDTH * HEIGHT, np.inf, dtype=np.float32)
    np.minimum.at(zbuf, pix, z[ids].astype(np.float32))
    winners = ids[z[ids] <= zbuf[pix] + 1e-3]
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    image[vv[winners], u[winners]] = colors[winners]
    mask[vv[winners], u[winners]] = 255

    if radius > 0:
        base_img = image.copy()
        base_mask = mask.copy()
        offsets = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for dx, dy in offsets:
            shifted_img = np.zeros_like(image)
            shifted_mask = np.zeros_like(mask)
            x_src0 = max(0, -dx); x_src1 = min(WIDTH, WIDTH - dx)
            y_src0 = max(0, -dy); y_src1 = min(HEIGHT, HEIGHT - dy)
            x_dst0 = x_src0 + dx; x_dst1 = x_src1 + dx
            y_dst0 = y_src0 + dy; y_dst1 = y_src1 + dy
            shifted_img[y_dst0:y_dst1, x_dst0:x_dst1] = base_img[y_src0:y_src1, x_src0:x_src1]
            shifted_mask[y_dst0:y_dst1, x_dst0:x_dst1] = base_mask[y_src0:y_src1, x_src0:x_src1]
            take = (mask == 0) & (shifted_mask > 0)
            image[take] = shifted_img[take]
            mask[take] = 255
    return image, mask


def render_multiview(clouds, ref, target: np.ndarray, degree: float, ball_tex, ball_mask, ball_world):
    if abs(degree) < 1e-9:
        frame = ref["image"].copy()
        return v8.composite_ball(frame, ref, target, 0.0, ball_tex, ball_mask, ball_world), np.full((HEIGHT, WIDTH), 255, np.uint8), [REF_LABEL]
    C, R, P = v7.novel_camera(ref, target, degree)
    vaz = view_azimuth(C, target)
    ranked = sorted(clouds, key=lambda c: angle_delta(view_azimuth(c["C"], target), vaz))
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    resolved = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    used = []
    for ci, cloud in enumerate(ranked):
        image, mask = raster_cloud(cloud["points"], cloud["colors"], C, R, P, radius=1)
        take = (resolved == 0) & (mask > 0)
        canvas[take] = image[take]
        resolved[take] = 255
        used.append({"label": cloud["label"], "new_pixels": int(take.sum())})
    canvas = v8.composite_ball(canvas, ref, target, degree, ball_tex, ball_mask, ball_world)
    return canvas, resolved, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--body-masks", type=Path, required=True)
    ap.add_argument("--moge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=31)
    ap.add_argument("--max-degree", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, ref, _, target, alignment_qa = v9.load_scene(args.cameras, args.locked_images, args.ball_report, args.body_masks)
    clouds, mapping_reports, moge_qa = build_source_clouds(rows, args.ball_report, args.body_masks, args.moge_dir)
    ball_tex, ball_mask, ball_world = v8.ball_patch(ref, args.ball_report)

    still_reports = []
    for degree in [0, 1, 2, 3, 5]:
        frame, resolved, used = render_multiview(clouds, ref, target, float(degree), ball_tex, ball_mask, ball_world)
        cv2.imwrite(str(args.out / f"multiview_{degree:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"multiview_unresolved_{degree:02d}deg_native.png"), (resolved == 0).astype(np.uint8) * 255)
        still_reports.append({
            "degree": degree,
            "resolved_fraction_native": round(float((resolved > 0).mean()), 6),
            "unresolved_pixels": int((resolved == 0).sum()),
            "source_fill": used,
        })

    degrees = []
    for i in range(args.frames):
        phase = i / max(args.frames - 1, 1)
        degree = float(args.max_degree * math.sin(math.pi * phase))
        degrees.append(degree)
        frame, _, _ = render_multiview(clouds, ref, target, degree, ball_tex, ball_mask, ball_world)
        cv2.imwrite(str(args.out / f"motion_{i:03d}.png"), frame)

    qa = {
        "prototype": "multiview_moge_real_pixel_holefill_v11",
        "source_resolution": [960, 540],
        "render_resolution": [960, 540],
        "resolution_policy": "native only",
        "source_policy": "four official exact-state synchronized NBA camera views; no generated pixels",
        "geometry_policy": "per-camera MoGe geometry registered independently to the same accepted metric 3D ball; action depth in each camera compressed/offset by the other calibrated silhouettes",
        "compositing_policy": "view nearest the virtual camera owns a pixel first; farther real cameras may only fill pixels still unresolved, preventing crossfade/ghost blending",
        "ball_policy": "one real reference ball patch at the accepted metric 3D ball plane; ball pixels are removed from source clouds to prevent duplicates",
        "per_view_geometry": mapping_reports,
        "moge_all_views": moge_qa,
        "exact_state_alignment_qa": alignment_qa,
        "stills": still_reports,
        "motion_frames": args.frames,
        "motion_degrees_min_max": [round(float(min(degrees)), 4), round(float(max(degrees)), 4)],
        "success_gate": "secondary cameras must materially reduce v10 disocclusion holes without introducing doubled players/ball, gross basket misregistration, or source-view seams."
    }
    (args.out / "multiview_moge_qa_v11.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
