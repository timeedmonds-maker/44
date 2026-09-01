from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def read_noncomment(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip() and not line.startswith("#")]


def parse_model(path: Path):
    cameras = {}
    for line in read_noncomment(path / "cameras.txt"):
        p = line.split()
        cameras[int(p[0])] = {
            "model": p[1], "width": int(p[2]), "height": int(p[3]),
            "params": [float(v) for v in p[4:]],
        }

    images = {}
    lines = read_noncomment(path / "images.txt")
    # images.txt alternates image metadata and POINTS2D rows. POINTS2D rows can be
    # empty; after comment/empty removal, distinguish metadata by token structure.
    for line in lines:
        p = line.split()
        if len(p) >= 10 and p[-1].lower().endswith((".png", ".jpg", ".jpeg")):
            image_id = int(p[0])
            images[image_id] = {
                "qvec": [float(v) for v in p[1:5]],
                "tvec": [float(v) for v in p[5:8]],
                "camera_id": int(p[8]),
                "name": p[9],
            }

    points = {}
    for line in read_noncomment(path / "points3D.txt"):
        p = line.split()
        if len(p) < 8:
            continue
        pid = int(p[0])
        track = p[8:]
        obs = []
        for i in range(0, len(track) - 1, 2):
            try:
                obs.append((int(track[i]), int(track[i + 1])))
            except ValueError:
                pass
        points[pid] = {
            "xyz": [float(v) for v in p[1:4]],
            "rgb": [int(v) for v in p[4:7]],
            "error": float(p[7]),
            "track": obs,
        }
    return cameras, images, points


def qvec_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def camera_center(image):
    R = qvec_to_rotmat(image["qvec"])
    t = np.asarray(image["tvec"], dtype=np.float64)
    return -R.T @ t


def write_ply(path: Path, points):
    rows = [p for p in points.values() if p["error"] < 4.0 and len(p["track"]) >= 2]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in rows:
            x, y, z = p["xyz"]
            r, g, b = p["rgb"]
            f.write(f"{x:.7f} {y:.7f} {z:.7f} {r} {g} {b}\n")


def layout_image(images, points, labels_by_name, out_path: Path):
    centers = np.array([camera_center(im) for im in images.values()], dtype=np.float64)
    if len(centers) < 2:
        return
    scene = np.array([p["xyz"] for p in points.values() if p["error"] < 4.0 and len(p["track"]) >= 2], dtype=np.float64)
    origin = np.median(scene, axis=0) if len(scene) else np.mean(centers, axis=0)
    cloud = np.vstack([centers - origin, scene - origin]) if len(scene) else centers - origin
    _, _, vh = np.linalg.svd(cloud - np.mean(cloud, axis=0), full_matrices=False)
    basis = vh[:2].T
    c2 = (centers - origin) @ basis
    s2 = (scene - origin) @ basis if len(scene) else np.empty((0,2))

    W, H = 2400, 1800
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    all2 = np.vstack([c2, s2]) if len(s2) else c2
    lo = np.percentile(all2, 2, axis=0)
    hi = np.percentile(all2, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def xy(p):
        x = int(120 + (p[0] - lo[0]) / span[0] * (W - 240))
        y = int(H - 120 - (p[1] - lo[1]) / span[1] * (H - 240))
        return x, y

    if len(s2):
        step = max(1, len(s2)//5000)
        for p in s2[::step]:
            cv2.circle(canvas, xy(p), 1, (85,85,85), -1)

    image_rows = list(images.items())
    for idx, (iid, im) in enumerate(image_rows):
        x, y = xy(c2[idx])
        cv2.circle(canvas, (x,y), 13, (255,255,255), -1, cv2.LINE_AA)
        label = labels_by_name.get(im["name"], im["name"])
        cv2.putText(canvas, label, (x+18, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "JOINT MULTI-VIEW SfM CAMERA LAYOUT", (80, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "White = recovered cameras | Gray = sparse real-image 3D points | arbitrary SfM scale", (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190,190,190), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["views"]
    labels_by_name = {m["image"]: m["label"] for m in mapping}
    model_rows = []
    best = None
    for d in sorted(p for p in args.models.iterdir() if p.is_dir()):
        cameras, images, points = parse_model(d)
        row = {"model": d.name, "registered_images": len(images), "points3D": len(points)}
        model_rows.append(row)
        score = (len(images), len(points))
        if best is None or score > best[0]:
            best = (score, d, cameras, images, points)
    if best is None:
        raise RuntimeError("COLMAP produced no model")

    _, d, cameras, images, points = best
    registered = []
    centers = {}
    for iid, im in images.items():
        label = labels_by_name.get(im["name"], im["name"])
        C = camera_center(im)
        centers[label] = C
        cam = cameras[im["camera_id"]]
        registered.append({
            "label": label,
            "image": im["name"],
            "camera_model": cam["model"],
            "camera_params": cam["params"],
            "center_sfm": [round(float(v), 6) for v in C],
        })

    labels = list(centers)
    baseline = []
    if len(labels) >= 2:
        vals = []
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                dist = float(np.linalg.norm(centers[labels[i]] - centers[labels[j]]))
                vals.append(dist)
                baseline.append({"a": labels[i], "b": labels[j], "distance_sfm": round(dist, 6)})
        med = float(np.median(vals)) if vals else 1.0
        for row in baseline:
            row["distance_over_median_baseline"] = round(row["distance_sfm"] / max(med, 1e-9), 4)

    useful_names = {"Broadcast", "In Arena", "Left Slash", "Right Slash", "Left HandHeld", "Right HandHeld", "Left Above Rim", "Right Above Rim"}
    useful_registered = sorted(useful_names.intersection(centers))
    point_errors = [p["error"] for p in points.values()]
    tracks = [len(p["track"]) for p in points.values()]
    qa = {
        "mode": "joint_uncalibrated_multiview_sfm",
        "best_model": d.name,
        "all_models": model_rows,
        "registered_count": len(images),
        "registered": registered,
        "useful_distinct_view_count": len(useful_registered),
        "useful_distinct_views": useful_registered,
        "points3D": len(points),
        "median_reprojection_error_px": None if not point_errors else round(float(np.median(point_errors)), 4),
        "median_track_length": None if not tracks else round(float(np.median(tracks)), 3),
        "baseline_pairs": baseline,
        "quality_gate": {
            "pass_camera_geometry": len(useful_registered) >= 4 and len(points) >= 300,
            "minimum_useful_distinct_views": 4,
            "minimum_sparse_points": 300,
            "note": "This gate tests recoverable joint camera geometry only. It does not yet claim photorealistic novel-view quality.",
        },
    }
    (args.out / "colmap_static_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    write_ply(args.out / "colmap_sparse_real_pixels.ply", points)
    layout_image(images, points, labels_by_name, args.out / "colmap_camera_layout_uhd.png")
    print(json.dumps(qa, indent=2), flush=True)
    if not qa["quality_gate"]["pass_camera_geometry"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
