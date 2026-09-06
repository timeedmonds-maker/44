from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import build_small_angle_real_pixel_prototype_v1 as v1
import build_small_angle_real_pixel_prototype_v2 as v2

WIDTH = 960
HEIGHT = 540
REF_LABEL = "In Arena"


def safe(label: str) -> str:
    return label.replace(" ", "_")


def load_inputs(cameras: Path, locked: Path, ball_report: Path, body_masks: Path, instances_dir: Path):
    rows, alignment_qa = v2.aligned_camera_rows(cameras, locked, ball_report)
    by_label = {r["label"]: r for r in rows}
    if REF_LABEL not in by_label:
        raise RuntimeError(f"Missing reference camera {REF_LABEL}")
    for r in rows:
        m = cv2.imread(str(body_masks / f"{safe(r['label'])}_body_mask_anchor.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Missing source body mask for {r['label']}")
        r["body_mask"] = m > 0
        r["lab"] = cv2.cvtColor(r["image"], cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    inst = json.loads((instances_dir / "reference_instances_v2.json").read_text(encoding="utf-8"))
    layers = []
    for item in inst["instances"]:
        mask = cv2.imread(str(instances_dir / item["anchor_mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(item["anchor_mask"])
        layers.append({"meta": item, "mask": mask > 0})
    return rows, by_label[REF_LABEL], layers, alignment_qa


def camera_z_for_world(ref, X: np.ndarray) -> float:
    return float((ref["R"] @ X.reshape(3, 1) + ref["t"])[2, 0])


def backproject_plane(ref, uv: np.ndarray, z_cam: float) -> np.ndarray:
    h = np.column_stack([uv, np.ones(len(uv), dtype=np.float64)])
    rays = (np.linalg.inv(ref["K"]) @ h.T).T
    xc = rays * (z_cam / np.maximum(rays[:, 2:3], 1e-9))
    world = (ref["R"].T @ (xc.T - ref["t"])).T
    return world


def sample_lab(image: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(uv), 3), dtype=np.float32)
    ids = np.where(valid)[0]
    if not len(ids):
        return out
    mapx = uv[ids, 0].astype(np.float32).reshape(-1, 1)
    mapy = uv[ids, 1].astype(np.float32).reshape(-1, 1)
    vals = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, 3)
    out[ids] = vals
    return out


def mask_hits(mask: np.ndarray, uv: np.ndarray, valid: np.ndarray) -> np.ndarray:
    u = np.rint(uv[:, 0]).astype(np.int32)
    vv = np.rint(uv[:, 1]).astype(np.int32)
    inside = valid & (u >= 0) & (u < WIDTH) & (vv >= 0) & (vv < HEIGHT)
    hit = np.zeros(len(uv), dtype=bool)
    ids = np.where(inside)[0]
    if len(ids):
        hit[ids] = mask[vv[ids], u[ids]]
    return hit


def estimate_layer_depth(ref, sources, layer_mask: np.ndarray, target: np.ndarray, steps: int = 140):
    ys, xs = np.where(layer_mask)
    if len(xs) < 500:
        raise RuntimeError(f"Reference layer too small: {len(xs)}")
    # Deterministic spatial subsample keeps the search fast while retaining the full silhouette extent.
    stride = max(1, int(math.ceil(len(xs) / 2400)))
    xs_s = xs[::stride].astype(np.float64)
    ys_s = ys[::stride].astype(np.float64)
    uv = np.column_stack([xs_s, ys_s])
    ref_lab = ref["lab"][ys[::stride], xs[::stride]]

    z_ball = camera_z_for_world(ref, target)
    z_min = max(120.0, z_ball - 360.0)
    z_max = z_ball + 360.0
    depths = np.linspace(z_min, z_max, steps)
    best = None
    curve = []
    for z in depths:
        world = backproject_plane(ref, uv, float(z))
        support = np.zeros(len(uv), dtype=np.uint8)
        color_sum = np.zeros(len(uv), dtype=np.float32)
        color_n = np.zeros(len(uv), dtype=np.uint8)
        per_source_hit = []
        for src in sources:
            suv, valid = v1.project(src["P"], world)
            hit = mask_hits(src["body_mask"], suv, valid)
            support += hit.astype(np.uint8)
            per_source_hit.append(float(hit.mean()))
            if np.any(hit):
                sampled = sample_lab(src["lab"], suv, valid)
                d = np.abs(sampled - ref_lab).mean(axis=1)
                color_sum += np.where(hit, np.minimum(d, 0.55), 0.0).astype(np.float32)
                color_n += hit.astype(np.uint8)
        any_hit = support > 0
        two_hit = support >= 2
        mean_color = np.where(color_n > 0, color_sum / np.maximum(color_n, 1), 0.55)
        # Stable layer objective: silhouette agreement dominates, color only breaks broad plateaus.
        loss = 1.0 - float(any_hit.mean())
        loss += 0.55 * (1.0 - float(two_hit.mean()))
        loss += 0.25 * float(np.mean(mean_color[any_hit])) if np.any(any_hit) else 0.25
        # Weak physical prior: action layers should not drift hundreds of cm from the validated ball plane.
        loss += 0.0000015 * float((z - z_ball) ** 2)
        row = {
            "z_cam_cm": float(z),
            "loss": float(loss),
            "any_hit_fraction": float(any_hit.mean()),
            "two_hit_fraction": float(two_hit.mean()),
            "mean_color_cost": float(np.mean(mean_color[any_hit])) if np.any(any_hit) else None,
            "per_source_hit_fraction": per_source_hit,
        }
        curve.append(row)
        if best is None or loss < best[0]:
            best = (loss, z, row)
    assert best is not None
    # Uncertainty proxy: depth range within 3% loss of optimum.
    threshold = best[0] * 1.03
    near = [r["z_cam_cm"] for r in curve if r["loss"] <= threshold]
    qa = dict(best[2])
    qa["z_ball_cam_cm"] = round(float(z_ball), 4)
    qa["near_optimal_depth_span_cm"] = [round(float(min(near)), 3), round(float(max(near)), 3)] if near else None
    qa["search_range_cm"] = [round(float(z_min), 3), round(float(z_max), 3)]
    qa["sample_pixels"] = int(len(uv))
    return float(best[1]), qa


def novel_camera(ref, target: np.ndarray, degrees: float):
    Q = v1.rz(degrees)
    C = target + Q @ (ref["C"] - target)
    R = ref["R"] @ Q.T
    t = -R @ C.reshape(3, 1)
    P = ref["K"] @ np.hstack([R, t])
    return C, R, P


def plane_homography(ref, P_new: np.ndarray, z_cam: float) -> np.ndarray:
    corners = np.array([[0.0, 0.0], [WIDTH - 1.0, 0.0], [WIDTH - 1.0, HEIGHT - 1.0], [0.0, HEIGHT - 1.0]], dtype=np.float64)
    world = backproject_plane(ref, corners, z_cam)
    dst, valid = v1.project(P_new, world)
    if not np.all(valid):
        raise RuntimeError("Reference plane corner projects behind novel camera")
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst.astype(np.float32))


def build_owner_masks(layers):
    # Partition overlapping detector masks so every reference pixel is emitted exactly once.
    order = sorted(range(len(layers)), key=lambda i: int(layers[i]["mask"].sum()))
    owner = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    for i in order:
        m = layers[i]["mask"] & (owner < 0)
        owner[m] = i
    return [(owner == i) for i in range(len(layers))], owner


def render(ref, layers, owner_masks, layer_depths, target: np.ndarray, degrees: float):
    if degrees == 0:
        union = np.any(np.stack(owner_masks, axis=0), axis=0)
        out = np.zeros_like(ref["image"])
        out[union] = ref["image"][union]
        return out, union.astype(np.uint8) * 255

    _, _, Pnew = novel_camera(ref, target, degrees)
    warped = []
    # Larger camera-z is farther from reference camera; composite far to near.
    for i, (mask, depth) in enumerate(zip(owner_masks, layer_depths)):
        tex = np.zeros_like(ref["image"])
        tex[mask] = ref["image"][mask]
        src_mask = mask.astype(np.uint8) * 255
        H = plane_homography(ref, Pnew, depth)
        wtex = cv2.warpPerspective(tex, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wmask = cv2.warpPerspective(src_mask, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped.append((depth, wtex, wmask))
    warped.sort(key=lambda x: x[0], reverse=True)
    canvas = np.zeros_like(ref["image"])
    coverage = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for _, tex, mask in warped:
        alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
        # Keep edges antialiased but do not invent pixels outside the projected source silhouette.
        canvas = (canvas.astype(np.float32) * (1.0 - alpha[:, :, None]) + tex.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
        coverage = np.maximum(coverage, mask)
    return canvas, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--body-masks", type=Path, required=True)
    ap.add_argument("--reference-instances", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, ref, layers, alignment_qa = load_inputs(args.cameras, args.locked_images, args.ball_report, args.body_masks, args.reference_instances)
    sources = [r for r in rows if r["label"] != REF_LABEL]
    target = v1.read_ball(args.ball_report)
    owner_masks, owner = build_owner_masks(layers)

    layer_depths = []
    layer_qa = []
    for i, layer in enumerate(layers):
        # Fit the unpartitioned detector silhouette for robust geometry; render only its owned pixels.
        z, qa = estimate_layer_depth(ref, sources, layer["mask"], target)
        layer_depths.append(z)
        q = {"layer": i, "detector": layer["meta"], "owned_pixels": int(owner_masks[i].sum()), "depth_fit": qa}
        layer_qa.append(q)

    degrees = [0, 2, 3, 5, 8]
    render_qa = []
    for d in degrees:
        frame, cover = render(ref, layers, owner_masks, layer_depths, target, float(d))
        cv2.imwrite(str(args.out / f"layered_virtual_{d:02d}deg_native.png"), frame)
        cv2.imwrite(str(args.out / f"layered_virtual_{d:02d}deg_uhd.png"), cv2.resize(frame, (3840, 2160), interpolation=cv2.INTER_LANCZOS4))
        cv2.imwrite(str(args.out / f"layered_coverage_{d:02d}deg_native.png"), cover)
        render_qa.append({"degree": d, "covered_native_pixels": int((cover > 0).sum()),
                          "coverage_fraction_native": round(float((cover > 0).mean()), 6)})

    owner_vis = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    palette = [(255,255,255), (180,180,180), (100,100,100), (220,220,220)]
    for i, m in enumerate(owner_masks):
        owner_vis[m] = palette[i % len(palette)]
    cv2.imwrite(str(args.out / "reference_layer_partition.png"), owner_vis)

    qa = {
        "prototype": "identity_free_layered_body_planes_v6",
        "method": "real reference-camera person-instance pixels partitioned into stable fronto-parallel metric layers; each layer depth is selected by calibrated multi-view silhouette agreement with photometric tie-breaking; novel views are plane-induced homographies",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "identity_policy": "none; no player names or cross-camera identity matching",
        "source_policy": "official NBA pixels only; no generative fill, diffusion, optical-flow interpolation, or synthesized anatomy",
        "stability_policy": "no independent per-pixel depth labels; each visible reference instance uses one smooth geometric plane to eliminate checkerboard tearing",
        "target_ball_world_cm": target.tolist(),
        "exact_state_alignment_qa": alignment_qa,
        "layers": layer_qa,
        "renders": render_qa,
        "success_gate": "2-5 degree body motion must be continuous and photographic. Cardboard-like parallax is acceptable as a diagnostic limitation; tearing/checkerboard is not.",
    }
    (args.out / "layered_body_planes_qa_v6.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
