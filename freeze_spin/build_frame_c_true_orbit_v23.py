from __future__ import annotations

"""Frame-C true-orbit v23.

v22 correctly failed because its correspondence *candidate generator* was too
recall-starved: Left Slash -> Left HandHeld reached 50 valid depth matches and
22 forward PnP inliers, just below the unchanged 25-inlier geometry gate.

v23 changes only SIFT candidate generation: more features and Lowe 0.75 rather
than 0.70.  It does NOT relax any acceptance criterion.  The same forward PnP,
reverse PnP, reprojection, cheirality, scale-consistency and SE(3) closure gates
from v13 remain authoritative.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import build_portable_moge_pnp_freeview_v13 as reciprocal
import build_frame_c_true_orbit_v22 as v22


def calibration_sift_matches(im1, im2, mask1, mask2):
    """Higher-recall candidate generator; downstream geometry gates unchanged."""
    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.015,
        edgeThreshold=14,
        sigma=1.3,
    )
    g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, mask1.astype(np.uint8) * 255)
    k2, d2 = sift.detectAndCompute(g2, mask2.astype(np.uint8) * 255)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return [], np.empty((0, 2)), np.empty((0, 2))

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(d1, d2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < 0.75 * b.distance:
            good.append(a)

    # Preserve v12/v13's approximate one-to-one target assignment.
    best = {}
    for m in good:
        if m.trainIdx not in best or m.distance < best[m.trainIdx].distance:
            best[m.trainIdx] = m
    good = list(best.values())
    p1 = np.array([k1[m.queryIdx].pt for m in good], np.float64) if good else np.empty((0, 2))
    p2 = np.array([k2[m.trainIdx].pt for m in good], np.float64) if good else np.empty((0, 2))
    return good, p1, p2


def output_dir_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--out":
            return Path(sys.argv[i + 1])
    return None


if __name__ == "__main__":
    # solve_target_from_reference_reciprocal dereferences reciprocal.base.sift_matches
    # at runtime, so this changes only candidate correspondence generation.
    reciprocal.base.sift_matches = calibration_sift_matches
    out = output_dir_from_argv()
    v22.main()
    if out is not None:
        p = out / "frame_c_true_orbit_qa_v22.json"
        if p.exists():
            q = json.loads(p.read_text())
            q["prototype"] = "frame_c_reciprocal_static_true_orbit_v23"
            q["calibration_candidate_matcher"] = {
                "method": "SIFT static-region candidates, one-to-one target assignment",
                "nfeatures": 10000,
                "contrast_threshold": 0.015,
                "lowe_ratio": 0.75,
                "change_from_v22": "candidate recall only",
                "acceptance_gate_change": "none",
                "hard_geometry_gates": "unchanged v13 forward/reverse PnP, reprojection, cheirality, depth-scale consistency and SE(3) closure",
            }
            p.write_text(json.dumps(q, indent=2), encoding="utf-8")
