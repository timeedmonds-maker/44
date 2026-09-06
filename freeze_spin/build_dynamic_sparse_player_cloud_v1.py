from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

VIEW_INFO = {
    "In Arena": {"index": 4, "dir": "04_In_Arena", "roi": (300, 120, 650, 540), "static_roi": (250, 30, 600, 220)},
    "Left Slash": {"index": 6, "dir": "06_Left_Slash", "roi": (350, 120, 760, 540), "static_roi": (250, 20, 700, 210)},
    "Left HandHeld": {"index": 8, "dir": "08_Left_HandHeld", "roi": (500, 120, 960, 540), "static_roi": (300, 20, 850, 260)},
    "Left Above Rim": {"index": 10, "dir": "10_Left_Above_Rim", "roi": (300, 100, 700, 420), "static_roi": (250, 20, 720, 190)},
}


def camera_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {c["label"]: c for c in payload["cameras"]}


def ball_map(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {v["label"]: v for v in payload["views"]}


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def estimate_homography(source: np.ndarray, reference: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, dict]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = roi
    mask = np.zeros_like(ref_gray)
    mask[y1:y2, x1:x2] = 255
    points_ref = cv2.goodFeaturesToTrack(ref_gray, 500, 0.01, 7, mask=mask, blockSize=7)
    if points_ref is None or len(points_ref) < 20:
        raise RuntimeError("Insufficient static features")
    points_src, status, err = cv2.calcOpticalFlowPyrLK(
        ref_gray, src_gray, points_ref, None,
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    )
    good = (status.ravel() == 1) & (err.ravel() < 15)
    ref = points_ref[good].reshape(-1, 2)
    src = points_src[good].reshape(-1, 2)
    H, inliers = cv2.findHomography(src, ref, cv2.RANSAC, 1.0)
    if H is None or inliers is None:
        raise RuntimeError("Static homography failed")
    keep = inliers.ravel().astype(bool)
    pred = cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), H).reshape(-1, 2)
    residual = np.linalg.norm(pred - ref[keep], axis=1)
    qa = {
        "tracked": int(len(ref)),
        "inliers": int(np.sum(keep)),
        "median_px": float(np.median(residual)),
        "p95_px": float(np.percentile(residual, 95)),
    }
    if qa["inliers"] < 30 or qa["median_px"] > 0.8 or qa["p95_px"] > 1.6:
        raise RuntimeError(f"Static homography QA failed: {qa}")
    return H, qa


def dynamic_mask(selected: np.ndarray, previous: np.ndarray, following: np.ndarray, roi: tuple[int, int, int, int], static_roi: tuple[int, int, int, int]) -> np.ndarray:
    H_prev, _ = estimate_homography(previous, selected, static_roi)
    H_next, _ = estimate_homography(following, selected, static_roi)
    prev_warp = cv2.warpPerspective(previous, H_prev, (selected.shape[1], selected.shape[0]))
    next_warp = cv2.warpPerspective(following, H_next, (selected.shape[1], selected.shape[0]))
    diff = np.maximum(cv2.absdiff(selected, prev_warp), cv2.absdiff(selected, next_warp)).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = (diff > 18).astype(np.uint8) * 255
    region = np.zeros_like(mask)
    x1, y1, x2, y2 = roi
    region[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, region)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    q = P @ np.r_[X, 1.0]
    return q[:2] / q[2]


def triangulate(cameras: dict[str, dict], label1: str, p1: np.ndarray, label2: str, p2: np.ndarray):
    P1 = np.asarray(cameras[label1]["projection_matrix_KRt"], dtype=np.float64)
    P2 = np.asarray(cameras[label2]["projection_matrix_KRt"], dtype=np.float64)
    Xh = cv2.triangulatePoints(P1, P2, p1.reshape(2, 1), p2.reshape(2, 1)).ravel()
    if abs(Xh[3]) < 1e-9:
        return None
    X = Xh[:3] / Xh[3]
    for label in (label1, label2):
        R = np.asarray(cameras[label]["R_world_to_camera"], dtype=np.float64)
        t = np.asarray(cameras[label]["t_world_to_camera_cm"], dtype=np.float64)
        if (R @ X + t)[2] <= 20.0:
            return None
    errors = []
    for label, point in ((label1, p1), (label2, p2)):
        P = np.asarray(cameras[label]["projection_matrix_KRt"], dtype=np.float64)
        errors.append(float(np.linalg.norm(project(P, X) - point)))
    return X, errors


def write_ply(path: Path, clusters: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(clusters)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for cluster in clusters:
            x, y, z = cluster["centroid_cm"]
            r, g, b = cluster["rgb"]
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cameras = camera_map(args.cameras)
    ball_views = ball_map(args.ball_report)
    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    images = {}
    masks = {}
    view_qa = {}
    for label, info in VIEW_INFO.items():
        selected_index = int(state["selected_frames"][label])
        previous_index = int(state["qa_neighbors"]["pre_contact_or_contact_start"][label])
        following_index = int(state["qa_neighbors"]["post_initial_deflection"][label])
        directory = args.candidates / info["dir"]
        selected = read_image(directory / f"f{selected_index:03d}.png")
        previous = read_image(directory / f"f{previous_index:03d}.png")
        following = read_image(directory / f"f{following_index:03d}.png")
        mask = dynamic_mask(selected, previous, following, info["roi"], info["static_roi"])
        H = np.asarray(ball_views[label]["camera_motion_homography_selected_to_anchor"], dtype=np.float64)
        images[label] = cv2.warpPerspective(selected, H, (960, 540))
        masks[label] = cv2.warpPerspective(mask, H, (960, 540), flags=cv2.INTER_NEAREST)
        view_qa[label] = {"dynamic_pixels": int(np.sum(masks[label] > 0))}
        cv2.imwrite(str(args.out / f"{info['index']:02d}_{label.replace(' ', '_')}_dynamic_mask.png"), masks[label])

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015, edgeThreshold=12)
    features = {}
    for label in VIEW_INFO:
        keypoints, descriptors = sift.detectAndCompute(images[label], masks[label])
        features[label] = (keypoints, descriptors)
        view_qa[label]["sift_keypoints"] = len(keypoints)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    labels = list(VIEW_INFO)
    raw_points = []
    pair_counts = Counter()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label1, label2 = labels[i], labels[j]
            kp1, d1 = features[label1]
            kp2, d2 = features[label2]
            if d1 is None or d2 is None:
                continue
            for first, second in matcher.knnMatch(d1, d2, k=2):
                if first.distance >= 0.86 * second.distance:
                    continue
                p1 = np.asarray(kp1[first.queryIdx].pt, dtype=np.float64)
                p2 = np.asarray(kp2[first.trainIdx].pt, dtype=np.float64)
                result = triangulate(cameras, label1, p1, label2, p2)
                if result is None:
                    continue
                X, errors = result
                if max(errors) > 5.0:
                    continue
                if not (-150 <= X[0] <= 250 and -300 <= X[1] <= 300 and 20 <= X[2] <= 400):
                    continue
                # Suppress obvious residual matches on the rigid backboard plane.
                if abs(X[0]) < 15 and abs(X[1]) < 120 and X[2] > 240:
                    continue
                px = images[label1][int(round(p1[1])), int(round(p1[0]))]
                raw_points.append({
                    "pair": [label1, label2],
                    "p1": [float(x) for x in p1],
                    "p2": [float(x) for x in p2],
                    "xyz_cm": [float(x) for x in X],
                    "max_pair_reproj_px": float(max(errors)),
                    "rgb": [int(px[2]), int(px[1]), int(px[0])],
                })
                pair_counts[(label1, label2)] += 1

    clusters = []
    for record in sorted(raw_points, key=lambda r: r["max_pair_reproj_px"]):
        X = np.asarray(record["xyz_cm"], dtype=np.float64)
        target = None
        for cluster in clusters:
            if np.linalg.norm(X - np.asarray(cluster["centroid_cm"], dtype=np.float64)) < 8.0:
                target = cluster
                break
        if target is None:
            clusters.append({"centroid_cm": record["xyz_cm"][:], "members": [record]})
        else:
            target["members"].append(record)
            target["centroid_cm"] = np.mean([m["xyz_cm"] for m in target["members"]], axis=0).tolist()

    output_clusters = []
    for cluster in clusters:
        pairs = sorted({tuple(member["pair"]) for member in cluster["members"]})
        rgb = np.mean([member["rgb"] for member in cluster["members"]], axis=0)
        output_clusters.append({
            "centroid_cm": [round(float(x), 5) for x in cluster["centroid_cm"]],
            "support_count": len(cluster["members"]),
            "camera_pair_count": len(pairs),
            "camera_pairs": [list(pair) for pair in pairs],
            "rgb": [int(round(x)) for x in rgb],
            "member_max_reproj_px": round(max(m["max_pair_reproj_px"] for m in cluster["members"]), 4),
        })

    repeated = [c for c in output_clusters if c["support_count"] >= 2]
    torso_band = [c for c in repeated if 160 <= c["centroid_cm"][2] <= 230]
    gate = {
        "minimum_25_dynamic_points": len(raw_points) >= 25,
        "minimum_15_spatial_clusters": len(output_clusters) >= 15,
        "minimum_4_camera_pair_families": len(pair_counts) >= 4,
        "minimum_3_repeated_torso_band_clusters": len(torso_band) >= 3,
    }
    gate["pass"] = bool(all(gate.values()))

    payload = {
        "method": "same-state dynamic foreground masks + calibrated cross-view SIFT + metric triangulation + 8cm spatial clustering",
        "raw_dynamic_3d_point_count": len(raw_points),
        "spatial_cluster_count": len(output_clusters),
        "repeated_cluster_count": len(repeated),
        "repeated_torso_band_cluster_count": len(torso_band),
        "camera_pair_counts": {f"{a} <-> {b}": n for (a, b), n in pair_counts.items()},
        "view_qa": view_qa,
        "gate": gate,
        "clusters": sorted(output_clusters, key=lambda c: (-c["support_count"], c["member_max_reproj_px"])),
        "raw_points": raw_points,
    }
    (args.out / "dynamic_sparse_player_cloud_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_ply(args.out / "dynamic_sparse_player_cloud_v1.ply", output_clusters)
    print(json.dumps({
        "raw_dynamic_3d_point_count": payload["raw_dynamic_3d_point_count"],
        "spatial_cluster_count": payload["spatial_cluster_count"],
        "repeated_cluster_count": payload["repeated_cluster_count"],
        "repeated_torso_band_cluster_count": payload["repeated_torso_band_cluster_count"],
        "camera_pair_counts": payload["camera_pair_counts"],
        "gate": payload["gate"],
    }, indent=2), flush=True)
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
