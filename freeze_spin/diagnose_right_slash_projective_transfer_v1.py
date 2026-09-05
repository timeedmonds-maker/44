from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def sift_matches(a: np.ndarray, b: np.ndarray):
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.70 * n.distance]
    p = np.float32([ka[m.queryIdx].pt for m in good])
    q = np.float32([kb[m.trainIdx].pt for m in good])
    return p, q


def errors(H: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(p) == 0:
        return np.empty(0, dtype=np.float64)
    pred = cv2.perspectiveTransform(p[:, None, :], H)[:, 0]
    return np.linalg.norm(pred - q, axis=1)


def stats(e: np.ndarray) -> dict:
    if len(e) == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None, "within_2px": 0}
    return {
        "n": int(len(e)),
        "median_px": float(np.median(e)),
        "p90_px": float(np.percentile(e, 90)),
        "p95_px": float(np.percentile(e, 95)),
        "within_2px": int(np.sum(e <= 2.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True, help="Cleaner Right Slash frame")
    ap.add_argument("--target", type=Path, required=True, help="Candidate impact-state Right Slash frame")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref = cv2.imread(str(args.reference))
    target = cv2.imread(str(args.target))
    if ref is None or target is None or ref.shape != target.shape:
        raise RuntimeError("Expected two readable same-size native Right Slash frames")

    p, q = sift_matches(ref, target)
    if len(p) < 30:
        raise RuntimeError(f"Insufficient wide-scene matches: {len(p)}")

    # Estimate the transfer without using the basket/action core, then test that
    # withheld region independently. A calibration matrix may only be transported
    # if one image homography explains fixed geometry across materially different
    # scene depths; a low global residual alone is not sufficient.
    x, y = p[:, 0], p[:, 1]
    withheld_basket = (x > 330) & (x < 800) & (y > 20) & (y < 220)
    action_core = (x > 280) & (x < 760) & (y > 180) & (y < 520)
    train = ~withheld_basket & ~action_core
    if int(np.sum(train)) < 30:
        raise RuntimeError(f"Insufficient training matches outside held-out geometry: {int(np.sum(train))}")

    H, inlier = cv2.findHomography(
        p[train], q[train], cv2.RANSAC, 1.5, maxIters=20000, confidence=0.999
    )
    if H is None or inlier is None:
        raise RuntimeError("Homography estimation failed")
    H /= H[2, 2]

    train_inlier = inlier.ravel().astype(bool)
    global_e = errors(H, p, q)
    basket_e = errors(H, p[withheld_basket], q[withheld_basket])
    train_e = errors(H, p[train][train_inlier], q[train][train_inlier])

    gate = {
        "training_inliers_at_least_60": bool(np.sum(train_inlier) >= 60),
        "training_p95_at_most_1_5px": bool(len(train_e) and np.percentile(train_e, 95) <= 1.5),
        "withheld_basket_matches_at_least_12": bool(np.sum(withheld_basket) >= 12),
        "withheld_basket_median_at_most_2px": bool(len(basket_e) and np.median(basket_e) <= 2.0),
        "withheld_basket_p90_at_most_3px": bool(len(basket_e) and np.percentile(basket_e, 90) <= 3.0),
    }
    gate["pass"] = bool(all(gate.values()))

    payload = {
        "status": "projective_transfer_supported" if gate["pass"] else "projective_transfer_rejected",
        "purpose": "Test whether a clean-frame Right Slash metric camera could be transported to the impact frame by a single source-image homography.",
        "guardrail": "This is a camera-transfer diagnostic only. Player matches are not calibration inputs, and the basket/action core is withheld from homography estimation.",
        "reference": args.reference.name,
        "target": args.target.name,
        "match_count": int(len(p)),
        "training_count": int(np.sum(train)),
        "training_inlier_count": int(np.sum(train_inlier)),
        "H_reference_to_target": H.tolist(),
        "training_inlier_error": stats(train_e),
        "withheld_basket_error": stats(basket_e),
        "global_match_error": stats(global_e),
        "gate": gate,
        "next_action": (
            "A clean-frame calibration may be transported only after a separate metric solve and exact-state lock."
            if gate["pass"]
            else "Do not transport a clean-frame camera matrix. Solve the impact frame independently or add stronger fixed-geometry constraints."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "right_slash_projective_transfer_v1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    warped = cv2.warpPerspective(ref, H, (target.shape[1], target.shape[0]))
    blend = cv2.addWeighted(warped, 0.5, target, 0.5, 0.0)
    cv2.imwrite(str(args.out / "right_slash_projective_transfer_blend_v1.png"), blend)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
