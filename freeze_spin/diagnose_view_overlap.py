from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.frames / "manifest.json").read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in manifest["angles"]}

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    diagnostics = []

    order = config["orbit_order"]
    for left_label, right_label in zip(order, order[1:]):
        left = cv2.imread(str(args.frames / by_label[left_label]["frame"]))
        right = cv2.imread(str(args.frames / by_label[right_label]["frame"]))
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(left_gray, None)
        kp2, des2 = sift.detectAndCompute(right_gray, None)

        good = []
        if des1 is not None and des2 is not None:
            for first, second in matcher.knnMatch(des1, des2, k=2):
                if first.distance < 0.72 * second.distance:
                    good.append(first)

        inliers = 0
        inlier_ratio = 0.0
        median_error = None
        if len(good) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if homography is not None and mask is not None:
                mask_flat = mask.ravel().astype(bool)
                inliers = int(mask_flat.sum())
                inlier_ratio = float(mask_flat.mean())
                projected = cv2.perspectiveTransform(src, homography)
                errors = np.linalg.norm(projected[:, 0] - dst[:, 0], axis=1)
                if inliers:
                    median_error = float(np.median(errors[mask_flat]))

        diagnostics.append(
            {
                "from": left_label,
                "to": right_label,
                "keypoints_from": len(kp1),
                "keypoints_to": len(kp2),
                "ratio_test_matches": len(good),
                "homography_inliers": inliers,
                "homography_inlier_ratio": round(inlier_ratio, 4),
                "median_inlier_reprojection_error_px": (
                    None if median_error is None else round(median_error, 3)
                ),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pairs": diagnostics}, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
