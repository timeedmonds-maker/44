from __future__ import annotations

"""Audit whether clean same-game Right Slash frames can be transported to impact.

This intentionally reuses the same hard projective-transfer thresholds as
``diagnose_right_slash_projective_transfer_v1.py``.  A shared camera label is
not evidence of a shared metric camera matrix: the operator may pan, tilt and
zoom.  The audit therefore requires enough independent static-scene feature
support and a held-out basket-region test before a clean-frame calibration can
be considered transferable.

No player, ball or body landmarks are used.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MIN_TRAINING_INLIERS = 60
MAX_TRAINING_P95_PX = 1.5
MIN_WITHHELD_BASKET_MATCHES = 12
MAX_WITHHELD_BASKET_MEDIAN_PX = 2.0
MAX_WITHHELD_BASKET_P90_PX = 3.0


def sift_matches(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=8000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    raw = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in raw if m.distance < 0.70 * n.distance]
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    return pa, pb


def transfer_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) == 0:
        return np.empty(0, dtype=np.float64)
    pred = cv2.perspectiveTransform(src[:, None, :], H)[:, 0]
    return np.linalg.norm(pred - dst, axis=1)


def stats(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"n": 0, "median_px": None, "p90_px": None, "p95_px": None, "within_2px": 0}
    return {
        "n": int(len(err)),
        "median_px": float(np.median(err)),
        "p90_px": float(np.percentile(err, 90)),
        "p95_px": float(np.percentile(err, 95)),
        "within_2px": int(np.sum(err <= 2.0)),
    }


def audit_pair(reference: Path, target: Path) -> dict:
    a = cv2.imread(str(reference))
    b = cv2.imread(str(target))
    if a is None or b is None:
        return {"reference": str(reference), "status": "unreadable", "gate": {"pass": False}}
    if a.shape[:2] != b.shape[:2]:
        return {
            "reference": str(reference),
            "status": "shape_mismatch",
            "reference_shape": list(a.shape),
            "target_shape": list(b.shape),
            "gate": {"pass": False},
        }

    p, q = sift_matches(a, b)
    # These are the same source-pixel regions used by the existing v1 transfer
    # diagnostic at 960x540.  The basket/action core is never used to fit H.
    x, y = (p[:, 0], p[:, 1]) if len(p) else (np.empty(0), np.empty(0))
    withheld_basket = (x > 330) & (x < 800) & (y > 20) & (y < 220)
    action_core = (x > 280) & (x < 760) & (y > 180) & (y < 520)
    training = ~withheld_basket & ~action_core

    rec: dict = {
        "reference": str(reference),
        "target": str(target),
        "match_count": int(len(p)),
        "training_count": int(np.sum(training)),
        "withheld_basket_count": int(np.sum(withheld_basket)),
    }

    if int(np.sum(training)) < 4:
        rec.update({"status": "insufficient_training_correspondence", "gate": {"pass": False}})
        return rec

    H, mask = cv2.findHomography(
        p[training], q[training], cv2.RANSAC, 1.5, maxIters=20000, confidence=0.999
    )
    if H is None or mask is None:
        rec.update({"status": "homography_failed", "gate": {"pass": False}})
        return rec

    inlier = mask.ravel().astype(bool)
    train_err = transfer_errors(H, p[training][inlier], q[training][inlier])
    basket_err = transfer_errors(H, p[withheld_basket], q[withheld_basket])
    ts = stats(train_err)
    bs = stats(basket_err)
    gate = {
        "training_inliers_at_least_60": int(np.sum(inlier)) >= MIN_TRAINING_INLIERS,
        "training_p95_at_most_1_5px": ts["p95_px"] is not None and ts["p95_px"] <= MAX_TRAINING_P95_PX,
        "withheld_basket_matches_at_least_12": bs["n"] >= MIN_WITHHELD_BASKET_MATCHES,
        "withheld_basket_median_at_most_2px": bs["median_px"] is not None and bs["median_px"] <= MAX_WITHHELD_BASKET_MEDIAN_PX,
        "withheld_basket_p90_at_most_3px": bs["p90_px"] is not None and bs["p90_px"] <= MAX_WITHHELD_BASKET_P90_PX,
    }
    gate["pass"] = bool(all(gate.values()))
    rec.update(
        {
            "status": "transfer_passed" if gate["pass"] else "transfer_rejected",
            "training_inlier_count": int(np.sum(inlier)),
            "training_inlier_error": ts,
            "withheld_basket_error": bs,
            "gate": gate,
        }
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    candidates = sorted(args.candidate_root.glob("event_*_frames/*.png"))
    if not candidates:
        raise SystemExit(f"No native candidate frames under {args.candidate_root}")

    rows = [audit_pair(p, args.target) for p in candidates]
    passed = [r for r in rows if r.get("gate", {}).get("pass")]
    payload = {
        "status": "transferable_prior_found" if passed else "no_transferable_same_game_prior",
        "purpose": "Strictly test clean same-game Right Slash native frames against the exact-state Right Slash impact frame.",
        "guardrail": "Camera name alone never promotes a metric prior. Player/ball/body points are excluded. Thresholds are unchanged from the v1 projective-transfer gate.",
        "thresholds": {
            "min_training_inliers": MIN_TRAINING_INLIERS,
            "max_training_p95_px": MAX_TRAINING_P95_PX,
            "min_withheld_basket_matches": MIN_WITHHELD_BASKET_MATCHES,
            "max_withheld_basket_median_px": MAX_WITHHELD_BASKET_MEDIAN_PX,
            "max_withheld_basket_p90_px": MAX_WITHHELD_BASKET_P90_PX,
        },
        "candidate_count": len(rows),
        "pass_count": len(passed),
        "promotion_allowed": bool(passed),
        "results": rows,
        "next_action": (
            "Use only a passing candidate as a metric-transfer source."
            if passed
            else "Do not transport a clean-frame camera matrix. Solve the impact camera from fixed geometry, potentially using a separately calibrated same-camera centre as a constrained prior."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
