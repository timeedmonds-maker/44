from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_colmap_static import camera_center, parse_model, write_ply

USEFUL_VIEWS = {
    "Broadcast", "Mobile Broadcast", "In Arena", "Left Slash", "Right Slash",
    "Left HandHeld", "Right HandHeld", "Left Above Rim", "Right Above Rim",
}


def labels_for_point(point, images, meta_by_name):
    return {
        meta_by_name[images[iid]["name"]]["label"]
        for iid, _ in point["track"]
        if iid in images and images[iid]["name"] in meta_by_name
    }


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
        labels = {
            meta_by_name[im["name"]]["label"]
            for im in images.values() if im["name"] in meta_by_name
        }
        useful = labels.intersection(USEFUL_VIEWS)
        cross = 0
        useful_cross = 0
        for p in points.values():
            plabels = labels_for_point(p, images, meta_by_name)
            if len(plabels) >= 2:
                cross += 1
            if len(plabels.intersection(USEFUL_VIEWS)) >= 2:
                useful_cross += 1
        row = {
            "model": d.name,
            "registered_images": len(images),
            "physical_views": len(labels),
            "useful_physical_views": len(useful),
            "points3D_total": len(points),
            "cross_camera_points3D": cross,
            "useful_cross_camera_points3D": useful_cross,
        }
        model_rows.append(row)
        score = (len(useful), useful_cross, len(labels), len(images), len(points))
        if best is None or score > best[0]:
            best = (score, d, cameras, images, points)

    if best is None:
        raise RuntimeError("COLMAP produced no model")

    _, model_dir, cameras, images, points = best
    registered_by_label = defaultdict(list)
    camera_ids_by_label = defaultdict(set)
    anchor_registered = set()
    registered_rows = []
    for iid, im in images.items():
        meta = meta_by_name.get(im["name"])
        if not meta:
            continue
        label = meta["label"]
        C = camera_center(im)
        registered_by_label[label].append(C)
        camera_ids_by_label[label].add(im["camera_id"])
        if meta.get("impact_anchor"):
            anchor_registered.add(label)
        registered_rows.append({
            "label": label,
            "image": im["name"],
            "impact_anchor": bool(meta.get("impact_anchor")),
            "camera_id": im["camera_id"],
            "center_sfm": [round(float(v), 6) for v in C],
        })

    cross_ids = []
    useful_cross_ids = []
    physical_views_per_point = []
    for pid, p in points.items():
        plabels = labels_for_point(p, images, meta_by_name)
        physical_views_per_point.append(len(plabels))
        if len(plabels) >= 2:
            cross_ids.append(pid)
        if len(plabels.intersection(USEFUL_VIEWS)) >= 2:
            useful_cross_ids.append(pid)

    physical = sorted(registered_by_label)
    useful = sorted(set(physical).intersection(USEFUL_VIEWS))
    useful_anchors = sorted(anchor_registered.intersection(USEFUL_VIEWS))
    useful_errors = [points[pid]["error"] for pid in useful_cross_ids]
    useful_tracks = [len(points[pid]["track"]) for pid in useful_cross_ids]
    intrinsics_shared_per_feed = all(len(ids) == 1 for ids in camera_ids_by_label.values())

    qa = {
        "mode": "temporal_bridge_joint_multiview_sfm",
        "input_view_count": payload.get("view_count"),
        "input_image_count": payload.get("image_count"),
        "images_per_view": payload.get("images_per_view"),
        "best_model": model_dir.name,
        "all_models": model_rows,
        "registered_image_count": len(images),
        "registered_physical_view_count": len(physical),
        "registered_physical_views": physical,
        "useful_distinct_view_count": len(useful),
        "useful_distinct_views": useful,
        "impact_anchor_useful_view_count": len(useful_anchors),
        "impact_anchor_useful_views": useful_anchors,
        "camera_ids_by_physical_feed": {k: sorted(v) for k, v in sorted(camera_ids_by_label.items())},
        "intrinsics_shared_per_physical_feed": intrinsics_shared_per_feed,
        "registered": registered_rows,
        "points3D_total": len(points),
        "cross_camera_points3D": len(cross_ids),
        "useful_cross_camera_points3D": len(useful_cross_ids),
        "median_useful_cross_camera_reprojection_error_px": None if not useful_errors else round(float(np.median(useful_errors)), 4),
        "median_useful_cross_camera_track_length": None if not useful_tracks else round(float(np.median(useful_tracks)), 3),
        "median_physical_views_per_point": None if not physical_views_per_point else round(float(np.median(physical_views_per_point)), 3),
        "quality_gate": {
            "pass_camera_geometry": len(useful) >= 4 and len(useful_cross_ids) >= 300,
            "minimum_useful_distinct_views": 4,
            "minimum_cross_camera_sparse_points": 300,
            "point_definition": "A qualifying point is observed by at least two useful physical camera feeds; multiple temporal frames from one feed cannot inflate the count.",
        },
    }

    (args.out / "colmap_multiframe_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    subset = {pid: points[pid] for pid in useful_cross_ids}
    write_ply(args.out / "colmap_cross_camera_real_pixels.ply", subset)
    print(json.dumps(qa, indent=2), flush=True)
    if not qa["quality_gate"]["pass_camera_geometry"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
