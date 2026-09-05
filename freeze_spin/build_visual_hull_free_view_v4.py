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


def safe(label: str) -> str:
    return label.replace(" ", "_")


def load_rows(cameras: Path, locked: Path, ball_report: Path, masks: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    out = []
    for r in rows:
        mask = cv2.imread(str(masks / f"{safe(r['label'])}_body_mask_anchor.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(masks / f"{safe(r['label'])}_body_mask_anchor.png")
        rr = dict(r)
        rr["mask"] = mask > 0
        out.append(rr)
    return out, alignment_qa


def project_mask_membership(P: np.ndarray, pts: np.ndarray, mask: np.ndarray, chunk=250_000):
    hit = np.zeros(len(pts), dtype=bool)
    for start in range(0, len(pts), chunk):
        end = min(len(pts), start + chunk)
        uv, valid = v1.project(P, pts[start:end])
        u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
        vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
        inside = valid & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
        ids = np.where(inside)[0]
        local = np.zeros(end - start, dtype=bool)
        if len(ids):
            local[ids] = mask[vv[ids], u[ids]]
        hit[start:end] = local
    return hit


def build_visual_hull(rows, target: np.ndarray, voxel_cm: float, min_views: int):
    # The action is directly in front of the board. A deliberately compact volume prevents
    # spectator silhouettes from creating remote phantom geometry and keeps the test focused
    # on the two airborne/contact bodies rather than the entire court.
    xs = np.arange(target[0] - 80.0, target[0] + 220.0 + voxel_cm * 0.5, voxel_cm, dtype=np.float64)
    ys = np.arange(target[1] - 170.0, target[1] + 170.0 + voxel_cm * 0.5, voxel_cm, dtype=np.float64)
    zs = np.arange(20.0, 355.0 + voxel_cm * 0.5, voxel_cm, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    counts = np.zeros(len(pts), dtype=np.uint8)
    per_view_hits = {}
    for r in rows:
        hit = project_mask_membership(r["P"], pts, r["mask"])
        counts += hit.astype(np.uint8)
        per_view_hits[r["label"]] = int(hit.sum())

    shape = X.shape
    occ = (counts.reshape(shape) >= min_views)
    if int(occ.sum()) < 500:
        raise RuntimeError(f"Visual hull too small: {int(occ.sum())} occupied voxels")

    # Remove tiny disconnected phantom islands while preserving all substantial action components.
    labels, n = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())
    keep_labels = []
    for li in range(1, n + 1):
        if sizes[li] >= 120:
            keep_labels.append(li)
    if not keep_labels:
        raise RuntimeError("No substantial visual-hull components survived")
    occ = np.isin(labels, keep_labels)

    eroded = ndimage.binary_erosion(occ, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    surface = occ & ~eroded
    surface_idx = np.flatnonzero(surface.ravel())
    surface_pts = pts[surface_idx]
    support_counts = counts[surface_idx]
    qa = {
        "voxel_cm": voxel_cm,
        "grid_shape": list(shape),
        "grid_voxels": int(len(pts)),
        "volume_bounds_cm": {
            "x": [round(float(xs[0]), 3), round(float(xs[-1]), 3)],
            "y": [round(float(ys[0]), 3), round(float(ys[-1]), 3)],
            "z": [round(float(zs[0]), 3), round(float(zs[-1]), 3)],
        },
        "minimum_silhouette_views": min_views,
        "per_view_voxels_inside_mask": per_view_hits,
        "occupied_voxels_after_component_filter": int(occ.sum()),
        "surface_voxels": int(len(surface_pts)),
        "component_count_before_filter": int(n),
        "kept_component_count": len(keep_labels),
        "kept_component_sizes": [int(sizes[i]) for i in keep_labels],
        "surface_support_histogram": {str(i): int((support_counts == i).sum()) for i in range(1, len(rows) + 1)},
    }
    return surface_pts, support_counts, qa


def source_visibility(row, pts: np.ndarray, voxel_cm: float):
    uv, valid = v1.project(row["P"], pts)
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    z = (row["R"] @ (pts - row["C"]).T).T[:, 2]
    inside = valid & (z > 1.0) & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    ids = np.where(inside)[0]
    masked = np.zeros(len(pts), dtype=bool)
    if len(ids):
        masked[ids] = row["mask"][vv[ids], u[ids]]
    inside &= masked

    pix = vv * WIDTH + u
    zbuf = np.full(WIDTH * HEIGHT, np.inf, dtype=np.float32)
    ids = np.where(inside)[0]
    if len(ids):
        np.minimum.at(zbuf, pix[ids], z[ids].astype(np.float32))
    visible = np.zeros(len(pts), dtype=bool)
    if len(ids):
        visible[ids] = z[ids] <= zbuf[pix[ids]] + max(5.0, voxel_cm * 2.0)

    # Sample real source colour once; rendering can select among source-visible colours by view angle.
    mapx = uv[:, 0].astype(np.float32).reshape(-1, 1)
    mapy = uv[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(row["image"], mapx, mapy, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, 3)
    return visible, sampled


def wrap_deg(x: float) -> float:
    return ((x + 180.0) % 360.0) - 180.0


def view_azimuth(C: np.ndarray, target: np.ndarray) -> float:
    d = C[:2] - target[:2]
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def render_hull(rows, ref, pts, source_vis, source_colors, target, degree, voxel_cm):
    Q = v1.rz(degree)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    P = ref["K"] @ np.hstack([R, t])
    uv, valid = v1.project(P, pts)
    z = (R @ (pts - C).T).T[:, 2]
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    valid &= (z > 1.0) & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)

    virtual_az = view_azimuth(C, target)
    ranked = sorted(range(len(rows)), key=lambda i: abs(wrap_deg(view_azimuth(rows[i]["C"], target) - virtual_az)))
    chosen = np.full(len(pts), -1, dtype=np.int16)
    for ri in ranked:
        take = (chosen < 0) & source_vis[ri]
        chosen[take] = ri
    valid &= chosen >= 0

    # Far-to-near splatting gives a deterministic approximate z-buffer while filling the
    # sub-pixel gaps between 2.5 cm surface voxels at native 960x540 resolution.
    ids = np.where(valid)[0]
    ids = ids[np.argsort(z[ids])[::-1]]
    native = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    coverage = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    radius = 2
    for i in ids:
        col = source_colors[int(chosen[i])][i]
        cv2.circle(native, (int(u[i]), int(vv[i])), radius, tuple(int(c) for c in col), -1, cv2.LINE_AA)
        cv2.circle(coverage, (int(u[i]), int(vv[i])), radius, 255, -1, cv2.LINE_AA)

    uhd = cv2.resize(native, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
    cov_uhd = cv2.resize(coverage, (3840, 2160), interpolation=cv2.INTER_NEAREST)
    qa = {
        "degree": degree,
        "visible_surface_voxels": int(len(ids)),
        "native_covered_pixels": int((coverage > 0).sum()),
        "native_coverage_fraction": round(float((coverage > 0).mean()), 6),
        "source_rank_nearest_to_virtual": [rows[i]["label"] for i in ranked],
    }
    return native, uhd, cov_uhd, qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--voxel-cm", type=float, default=2.5)
    ap.add_argument("--min-views", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, alignment_qa = load_rows(args.cameras, args.locked_images, args.ball_report, args.masks)
    target = v1.read_ball(args.ball_report)
    ref = v1.select_reference(rows, target)
    pts, support_counts, hull_qa = build_visual_hull(rows, target, args.voxel_cm, args.min_views)

    source_vis = []
    source_colors = []
    visibility_qa = {}
    for r in rows:
        vis, colors = source_visibility(r, pts, args.voxel_cm)
        source_vis.append(vis)
        source_colors.append(colors)
        visibility_qa[r["label"]] = int(vis.sum())

    degrees = [0, 3, 5, 8]
    renders = []
    for d in degrees:
        native, uhd, cov, qa = render_hull(rows, ref, pts, source_vis, source_colors, target, d, args.voxel_cm)
        cv2.imwrite(str(args.out / f"visual_hull_{d:02d}deg_native.png"), native)
        cv2.imwrite(str(args.out / f"visual_hull_{d:02d}deg_uhd.png"), uhd)
        cv2.imwrite(str(args.out / f"visual_hull_coverage_{d:02d}deg_uhd.png"), cov)
        renders.append(qa)

    # Real aligned reference-body cutout for a stringent 0-degree visual comparison.
    ref_mask = (ref["mask"].astype(np.uint8) * 255)
    ref_cut = cv2.bitwise_and(ref["image"], ref["image"], mask=ref_mask)
    cv2.imwrite(str(args.out / "visual_hull_real_reference_native.png"), ref_cut)
    cv2.imwrite(str(args.out / "visual_hull_real_reference_uhd.png"),
                cv2.resize(ref_cut, (3840, 2160), interpolation=cv2.INTER_LANCZOS4))

    qa = {
        "prototype": "identity_free_calibrated_visual_hull_v4",
        "method": "shape from silhouette / visual hull using union action-body masks from four exact-state-aligned metric cameras; view-dependent real-pixel texture selection",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "identity_policy": "none; no cross-camera player identity correspondence is required",
        "segmentation_role": "binary foreground silhouette only",
        "geometry_policy": "voxel occupancy requires projection inside action-body silhouettes from multiple independently calibrated cameras",
        "texture_policy": "real source pixels sampled only from source-visible hull surfaces; no generated texture",
        "reference": ref["label"],
        "target_ball_world_cm": target.tolist(),
        "exact_state_alignment_qa": alignment_qa,
        "hull": hull_qa,
        "source_visible_surface_voxels": visibility_qa,
        "renders": renders,
        "success_gate": "0-degree hull must reproduce continuous action-body silhouettes substantially better than sparse plane sweep; 3-5 degrees must remain coherent without anatomy warping before background compositing.",
    }
    (args.out / "visual_hull_qa_v4.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
