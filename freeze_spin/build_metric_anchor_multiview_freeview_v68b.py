from __future__ import annotations

"""v68b: high-recall static correspondence front-end for v68.

The v68 geometry, metric anchors, PnP thresholds, renderer and permissions are
unchanged. Only correspondence proposal changes: detect many more static SIFT
features, use a permissive one-way ratio for recall, then require a one-to-one
target assignment, Fundamental-matrix RANSAC consistency and broad spatial
support before any match can enter metric-depth sampling or PnP.
"""

import cv2
import numpy as np

import build_metric_anchor_multiview_freeview_v68 as v68


def _cells(points: np.ndarray, gx: int = 8, gy: int = 5) -> int:
    if len(points) == 0:
        return 0
    ix = np.clip((points[:, 0] / v68.W * gx).astype(int), 0, gx - 1)
    iy = np.clip((points[:, 1] / v68.H * gy).astype(int), 0, gy - 1)
    return len(set(zip(ix.tolist(), iy.tolist())))


def high_recall_static_matches(im1, im2, mask1, mask2):
    sift = cv2.SIFT_create(
        nfeatures=25000,
        contrastThreshold=0.004,
        edgeThreshold=28,
        sigma=1.1,
    )
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, mask1.astype(np.uint8) * 255)
    k2, d2 = sift.detectAndCompute(g2, mask2.astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 100 or len(k2) < 100:
        raise RuntimeError("v68b insufficient static SIFT keypoints")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(d1, d2, k=2)
    proposed = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.90 * b.distance:
            proposed.append(a)

    # Exact one-to-one target assignment before geometry filtering.
    best = {}
    for m in proposed:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    proposed = list(best.values())
    if len(proposed) < 80:
        raise RuntimeError(f"v68b only {len(proposed)} one-to-one static proposals")

    p1 = np.asarray([k1[m.queryIdx].pt for m in proposed], dtype=np.float64)
    p2 = np.asarray([k2[m.trainIdx].pt for m in proposed], dtype=np.float64)
    F, inlier = cv2.findFundamentalMat(
        p1, p2, cv2.FM_RANSAC, 2.5, 0.999, 25000
    )
    if F is None or inlier is None:
        raise RuntimeError("v68b Fundamental-matrix RANSAC failed")
    keep = inlier.reshape(-1).astype(bool)
    p1 = p1[keep]
    p2 = p2[keep]
    kept_matches = [m for m, ok in zip(proposed, keep) if ok]

    c1, c2 = _cells(p1), _cells(p2)
    print(
        "V68B_MATCH_DIAGNOSTIC",
        {
            "keypoints_lar": len(k1),
            "keypoints_pbp": len(k2),
            "one_to_one_proposals": len(proposed),
            "fundamental_inliers": len(p1),
            "lar_cells_8x5": c1,
            "pbp_cells_8x5": c2,
        },
        flush=True,
    )
    if len(p1) < 35:
        raise RuntimeError(f"v68b only {len(p1)} epipolar-consistent static matches")
    if c1 < 10 or c2 < 10:
        raise RuntimeError(f"v68b static matches too clustered cells={c1},{c2}")
    return kept_matches, p1, p2


if __name__ == "__main__":
    v68.sift_matches = high_recall_static_matches
    v68.main()
