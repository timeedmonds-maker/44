from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import build_small_angle_real_pixel_prototype_v1 as v1
import build_small_angle_real_pixel_prototype_v2 as v2
import build_layered_body_planes_free_view_v6 as v6

WIDTH = 960
HEIGHT = 540
REF_LABEL = "In Arena"


def load_reference(cameras: Path, locked: Path, ball_report: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    ref = next(r for r in rows if r["label"] == REF_LABEL)
    return ref, alignment_qa


def build_owner_masks(instances_dir: Path, max_layers: int):
    payload = json.loads((instances_dir / "reference_instances_v2.json").read_text(encoding="utf-8"))
    layers = []
    for item in payload["instances"][:max_layers]:
        mask = cv2.imread(str(instances_dir / item["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(item["anchor_mask"])
        layers.append({"meta": item, "mask": mask > 0})
    owner = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    # Smaller masks get first ownership in overlap regions, preserving narrow action components.
    for i in sorted(range(len(layers)), key=lambda j: int(layers[j]["mask"].sum())):
        owner[layers[i]["mask"] & (owner < 0)] = i
    return layers, [(owner == i) for i in range(len(layers))]


def backproject_z(ref, uv: np.ndarray, z_cam: np.ndarray) -> np.ndarray:
    h = np.column_stack([uv, np.ones(len(uv), dtype=np.float64)])
    rays = (np.linalg.inv(ref["K"]) @ h.T).T
    xc = rays * (z_cam[:, None] / np.maximum(rays[:, 2:3], 1e-9))
    return (ref["R"].T @ (xc.T - ref["t"])).T


def novel_camera(ref, target: np.ndarray, degree: float):
    Q = v1.rz(degree)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    P = ref["K"] @ np.hstack([R, t])
    return C, R, P


def render_points(points, colors, ref, target: np.ndarray, degree: float, radius: int):
    C, R, P = novel_camera(ref, target, degree)
    uv, valid = v1.project(P, points)
    cam_z = (R @ (points - C).T).T[:, 2]
    valid &= cam_z > 1.0
    u = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    valid &= (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    ids = np.where(valid)[0]
    ids = ids[np.argsort(cam_z[ids])[::-1]]
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    coverage = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for i in ids:
        x, y = int(u[i]), int(vv[i])
        col = tuple(int(c) for c in colors[i])
        cv2.circle(canvas, (x, y), radius, col, -1, cv2.LINE_AA)
        cv2.circle(coverage, (x, y), radius, 255, -1, cv2.LINE_AA)
    return canvas, coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--reference-instances", type=Path, required=True)
    ap.add_argument("--layered-qa", type=Path, required=True)
    ap.add_argument("--moge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-layers", type=int, default=2)
    ap.add_argument("--detail-gain", type=float, default=0.85)
    ap.add_argument("--max-depth-deviation-cm", type=float, default=65.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ref, alignment_qa = load_reference(args.cameras, args.locked_images, args.ball_report)
    target = v1.read_ball(args.ball_report)
    layers, owner_masks = build_owner_masks(args.reference_instances, args.max_layers)
    layered = json.loads(args.layered_qa.read_text(encoding="utf-8"))
    layer_centres = [float(row["depth_fit"]["z_cam_cm"]) for row in layered["layers"][:len(layers)]]

    depth_m = np.load(args.moge_dir / "moge_depth_m.npy").astype(np.float32)
    valid_moge = np.load(args.moge_dir / "moge_valid.npy").astype(bool)
    moge_qa = json.loads((args.moge_dir / "moge_reference_qa_v1.json").read_text(encoding="utf-8"))
    scale = moge_qa.get("metric_scale_multiplier_from_ball")
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("MoGe ball registration scale is unavailable")
    depth_cm = depth_m * 100.0 * float(scale)
    # Edge-preserving regularization removes tiny depth speckle but retains body relief.
    finite_depth = np.where(np.isfinite(depth_cm), depth_cm, 0.0).astype(np.float32)
    smooth_depth = cv2.bilateralFilter(finite_depth, d=7, sigmaColor=18.0, sigmaSpace=4.0)

    points_all = []
    colors_all = []
    layer_reports = []
    final_depth_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    union = np.zeros((HEIGHT, WIDTH), dtype=bool)

    for i, (layer, owned, centre_z) in enumerate(zip(layers, owner_masks, layer_centres)):
        mask = owned & valid_moge & np.isfinite(smooth_depth) & (smooth_depth > 0)
        ys, xs = np.where(mask)
        if len(xs) < 500:
            raise RuntimeError(f"Too few valid MoGe pixels for layer {i}: {len(xs)}")
        vals = smooth_depth[ys, xs]
        median = float(np.median(vals))
        deviations = (vals - median) * float(args.detail_gain)
        deviations = np.clip(deviations, -args.max_depth_deviation_cm, args.max_depth_deviation_cm)
        z = centre_z + deviations
        uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        pts = backproject_z(ref, uv, z.astype(np.float64))
        points_all.append(pts)
        colors_all.append(ref["image"][ys, xs])
        final_depth_map[ys, xs] = z
        union[ys, xs] = True
        layer_reports.append({
            "layer": i,
            "owned_pixels": int(owned.sum()),
            "moge_valid_owned_pixels": int(len(xs)),
            "calibrated_layer_centre_z_cm": round(float(centre_z), 6),
            "moge_registered_median_z_cm_before_recentering": round(median, 6),
            "final_z_percentiles_cm": [round(float(v), 4) for v in np.percentile(z, [1, 10, 50, 90, 99])],
            "detail_gain": float(args.detail_gain),
            "depth_deviation_clip_cm": float(args.max_depth_deviation_cm),
        })

    points = np.concatenate(points_all, axis=0)
    colors = np.concatenate(colors_all, axis=0)
    if len(points) < 3000:
        raise RuntimeError(f"Insufficient textured body points: {len(points)}")

    dvis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    vals = final_depth_map[union]
    lo, hi = np.percentile(vals, [1, 99])
    dvis[union] = np.clip((final_depth_map[union] - lo) * 255.0 / max(float(hi - lo), 1e-6), 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.out / "moge_registered_body_depth.png"), dvis)

    # 0 degrees remains exact real reference pixels; novel views use the learned geometry prior only.
    zero = np.zeros_like(ref["image"])
    zero[union] = ref["image"][union]
    degrees = [0, 1, 2, 3, 5, 8]
    renders = []
    for degree in degrees:
        if degree == 0:
            frame = zero.copy()
            cover = union.astype(np.uint8) * 255
        else:
            radius = 1 if degree <= 2 else 2
            frame, cover = render_points(points, colors, ref, target, float(degree), radius)
        cv2.imwrite(str(args.out / f"moge_virtual_{degree:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"moge_virtual_{degree:02d}deg_uhd.png"), cv2.resize(frame, (3840, 2160), interpolation=cv2.INTER_LANCZOS4))
        cv2.imwrite(str(args.out / f"moge_coverage_{degree:02d}deg_native.png"), cover)
        renders.append({"degree": degree, "covered_native_pixels": int((cover > 0).sum()),
                        "coverage_fraction_native": round(float((cover > 0).mean()), 6)})

    qa = {
        "prototype": "moge2_geometry_prior_real_pixel_free_view_v7",
        "method": "MoGe-2 small model supplies smooth within-body relative depth; each identity-free reference layer is recentered to its calibrated multi-view silhouette depth; only original NBA reference pixels are rendered",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "identity_policy": "none",
        "appearance_policy": "100% real reference NBA pixels; learned model affects geometry only",
        "generation_policy": "no diffusion, generated texture, frame synthesis, optical-flow morph, or hidden-body texture completion",
        "moge": moge_qa,
        "exact_state_alignment_qa": alignment_qa,
        "layers": layer_reports,
        "renders": renders,
        "success_gate": "1-5 degree outputs must remain continuous while showing smoother non-planar body parallax than v6; unsupported hidden surfaces remain absent.",
    }
    (args.out / "moge_free_view_qa_v7.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
