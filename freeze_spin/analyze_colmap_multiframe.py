from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

USEFUL_VIEWS = {
    "Broadcast", "Mobile Broadcast", "In Arena", "Left Slash", "Right Slash",
    "Left HandHeld", "Right HandHeld", "Left Above Rim", "Right Above Rim",
}


def read_noncomment(path: Path):
    if not path.exists():
        return []
    return [
        line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def parse_model(path: Path):
    cameras = {}
    for line in read_noncomment(path / "cameras.txt"):
        p = line.split()
        cameras[int(p[0])] = {
            "model": p[1], "width": int(p[2]), "height": int(p[3]),
            "params": [float(v) for v in p[4:]],
        }

    images = {}
    for line in read_noncomment(path / "images.txt"):
        p = line.split()
        if len(p) >= 10 and p[-1].lower().endswith((".png", ".jpg", ".jpeg")):
            iid = int(p[0])
            images[iid] = {
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
        track = []
        for i in range(8, len(p) - 1, 2):
            try:
                track.append((int(p[i]), int(p[i + 1])))
            except ValueError:
                pass
        points[int(p[0])] = {
            "xyz": [float(v) for v in p[1:4]],
            "rgb": [int(v) for v in p[4:7]],
            "error": float(p[7]),
            "track": track,
        }
    return cameras, images, points


def qvec_to_rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qz*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def camera_center(image):
    R = qvec_to_rotmat(image["qvec"])
    t = np.asarray(image["tvec"], dtype=np.float64)
    return -R.T @ t


def write_ply(path: Path, points, cross_ids):
    rows = [points[pid] for pid in cross_ids if points[pid]["error"] < 4.0]
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in rows:
            x, y, z = p["xyz"]
            r, g, b = p["rgb"]
            f.write(f"{x:.7f} {y:.7f} {z:.7f} {r} {g} {b}\n")


def layout_image(representatives, points, cross_ids, out_path: Path):
    if len(representatives) < 2:
        return
    labels = list(representatives)
    centers = np.array([representatives[k] for k in labels], dtype=np.float64)
    scene = np.array(
        [points[pid]["xyz"] for pid in cross_ids if points[pid]["error"] < 4.0],
        dtype=np.float64,
    )
    origin = np.median(scene, axis=0) if len(scene) else np.mean(centers, axis=0)
    cloud = np.vstack([centers - origin, scene - origin]) if len(scene) else centers - origin
    _, _, vh = np.linalg.svd(cloud - np.mean(cloud, axis=0), full_matrices=False)
    basis = vh[:2].T
    c2 = (centers - origin) @ basis
    s2 = (scene - origin) @ basis if len(scene) else np.empty((0, 2))

    W, H = 2400, 1800
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    all2 = np.vstack([c2, s2]) if len(s2) else c2
    lo = np.percentile(all2, 2, axis=0)
    hi = np.percentile(all2, 98, axis=0)
    span = np.maximum(hi - lo, 1e-6)

    def xy(p):
        return (
            int(120 + (p[0] - lo[0]) / span[0] * (W - 240)),
            int(H - 120 - (p[1] - lo[1]) / span[1] * (H - 240)),
        )

    if len(s2):
        step = max(1, len(s2) // 5000)
        for p in s2[::step]:
            cv2.circle(canvas, xy(p), 1, (85, 85, 85), -1)
    for idx, label in enumerate(labels):
        x, y = xy(c2[idx])
        cv2.circle(canvas, (x, y), 13, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 18, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "TEMPORAL-BRIDGE JOINT SfM", (80, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Gray = points triangulated across >=2 physical feeds", (80, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (190, 190, 190), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    mapping = payload["views"]
    meta_by_name = {m["image"]: m for m in mapping}

    model_rows = []
    best = None
    for d in sorted(p for p in args.models.iterdir() if p.is_dir()):
        cameras, images, points = parse_model(d)
        labels = {meta_by_name[im["name"]]["label"] for im in images.values() if im["name"] in meta_by_name}
        cross_count = 0
        for p in points.values():
            track_labels = {
                meta_by_name[images[iid]["name"]]["label"]
                for iid, _ in p["track"]
                if iid in images and images[iid]["name"] in meta_by_name
            }
            if len(track_labels) >= 2:
                cross_count += 1
        row = {
            "model": d.name,
            "registered_images": len(images),
            "physical_views": len(labels),
            "points3D": len(points),
            "cross_camera_points3D": cross_count,
        }
        model_rows.append(row)
        score = (len(labels), cross_count, len(images), len(points))
        if best is None or score > best[0]:
            best = (score, d, cameras, images, points)
    if best is None:
        raise RuntimeError("COLMAP produced no model")

    _, d, cameras, images, points = best
    registered_by_label = defaultdict(list)
    anchor_registered = set()
    registered_rows = []
    for iid, im in images.items():
        meta = meta_by_name.get(im["name"])
        if not meta:
            continue
        label = meta["label"]
        C = camera_center(im)
        registered_by_label[label].append((im["name"], C))
        if meta.get("impact_anchor"):
            anchor_registered.add(label)
        cam = cameras[im["camera_id"]]
        registered_rows.append({
            "label": label,
            "image": im["name"],
            "impact_anchor": bool(meta.get("impact_anchor")),
            "camera_id": im["camera_id"],
            "camera_model": cam["model"],
            "camera_params": cam["params"],
            "center_sfm": [round(float(v), 6) for v in C],
        })

    representatives = {}
    for label, rows in registered_by_label.items():
        anchor_name = next((m["image"] for m in mapping if m["label"] == label and m.get("impact_anchor")), None)
        anchor = next((C for name, C in rows if name == anchor_name), None)
        representatives[label] = anchor if anchor is not None else np.median(np.array([C for _, C in rows]), axis=0)

    cross_ids = []
    useful_cross_ids = []
    track_physical_counts = []
    for pid, p in points.items():
        labels = {
            meta_by_name[images[iid]["name"]]["label"]
            for iid, _ in p["track"]
            if iid in images and images[iid]["name"] in meta_by_name
        }
        track_physical_counts.append(len(labels))
        if len(labels) >= 2:
            cross_ids.append(pid)
        if len(labels.intersection(USEFUL_VIEWS)) >= 2:
            useful_cross_ids.append(pid)

    physical = sorted(registered_by_label)
    useful = sorted(set(physical).intersection(USEFUL_VIEWS))
    useful_anchors = sorted(anchor_registered.intersection(USEFUL_VIEWS))
    errors = [points[pid]["error"] for pid in cross_ids]
    cross_tracks = [len(points[pid]["track"]) for pid in cross_ids]

    qa = {
        "mode": "temporal_bridge_joint_multiview_sfm",
        "input_view_count": payload.get("view_count"),
        "input_image_count": payload.get("image_count"),
        "images_per_view": payload.get("images_per_view"),
        "best_model": d.name,
        "all_models": model_rows,
        "registered_image_count": len(images),
        "registered_physical_view_count": len(physical),
        "registered_physical_views": physical,
        "useful_distinct_view_count": len(useful),
        "useful_distinct_views": useful,
        "impact_anchor_useful_view_count": len(useful_anchors),
        "impact_anchor_useful_views": useful_anchors,
        "registered": registered_rows,
        "points3D_total": len(points),
        "cross_camera_points3D": len(cross_ids),
        "useful_cross_camera_points3D": len(useful_cross_ids),
        "median_cross_camera_reprojection_error_px": None if not errors else round(float(np.median(errors)), 4),
        "median_cross_camera_track_length": None if not cross_tracks else round(float(np.median(cross_tracks)), 3),
        "median_physical_views_per_point": None if not track_physical_counts else round(float(np.median(track_physical_counts)), 3),
        "quality_gate": {
            "pass_camera_geometry": len(useful) >= 4 and len(useful_cross_ids) >= 300,
            "minimum_useful_distinct_views": 4,
            "minimum_cross_camera_sparse_points": 300,
            "point_definition": "A qualifying point must be observed by at least two useful physical camera feeds; temporal duplicates from one feed cannot count.",
        },
    }
    (args.out / "colmap_multiframe_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    write_ply(args.out / "colmap_cross_camera_real_pixels.ply", points, useful_cross_ids)
    layout_image(representatives, points, useful_cross_ids, args.out / "colmap_camera_layout_uhd.png")
    print(json.dumps(qa, indent=2), flush=True)
    if not qa["quality_gate"]["pass_camera_geometry"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
