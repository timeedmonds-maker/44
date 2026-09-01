from __future__ import annotations

import argparse
import json
from pathlib import Path


def distinct_cameras(cluster: dict) -> list[str]:
    cameras: set[str] = set()
    for pair in cluster.get("camera_pairs", []):
        cameras.update(pair)
    return sorted(cameras)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cloud = json.loads(args.cloud.read_text(encoding="utf-8"))
    clusters = cloud.get("clusters", [])

    audited = []
    for cluster in clusters:
        cameras = distinct_cameras(cluster)
        row = dict(cluster)
        row["distinct_camera_count"] = len(cameras)
        row["distinct_cameras"] = cameras
        audited.append(row)

    multi_pair = [c for c in audited if int(c.get("camera_pair_count", 0)) >= 2]
    three_camera = [c for c in audited if c["distinct_camera_count"] >= 3]
    torso_three_camera = [
        c for c in three_camera
        if 160.0 <= float(c["centroid_cm"][2]) <= 230.0
    ]

    # A cluster with several members from one pair is not independent multi-view
    # evidence.  The strict gate requires spatial convergence from multiple pair
    # families / at least three physical cameras before a point cloud may be
    # described as a reconstructed player surface.
    gate = {
        "minimum_3_multi_pair_clusters": len(multi_pair) >= 3,
        "minimum_2_three_camera_clusters": len(three_camera) >= 2,
        "minimum_1_three_camera_torso_cluster": len(torso_three_camera) >= 1,
    }
    gate["pass"] = bool(all(gate.values()))

    payload = {
        "purpose": "Prevent same-pair duplicate feature matches from masquerading as multi-view player reconstruction",
        "source_cloud": str(args.cloud),
        "raw_dynamic_3d_point_count": int(cloud.get("raw_dynamic_3d_point_count", 0)),
        "spatial_cluster_count": len(audited),
        "legacy_repeated_cluster_count": int(cloud.get("repeated_cluster_count", 0)),
        "multi_pair_cluster_count": len(multi_pair),
        "three_camera_cluster_count": len(three_camera),
        "three_camera_torso_cluster_count": len(torso_three_camera),
        "gate": gate,
        "multi_pair_clusters": multi_pair,
        "three_camera_clusters": three_camera,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
