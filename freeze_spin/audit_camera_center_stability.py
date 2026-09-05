#!/usr/bin/env python3
"""Audit whether independently calibrated frames support a reusable physical camera centre.

This is deliberately narrower than a homography/pose transfer. Pan, tilt and zoom may
change between frames; only the recovered optical centre is tested for stability.

Inputs must be calibration reports produced from fixed, regulation/arena geometry only.
Player, pose and ball anchors are forbidden. A report that was not independently
promoted by its own source-pixel reprojection gate cannot contribute to this audit.

The script never promotes an impact camera. It only decides whether a *centre prior*
is admissible for a later impact-frame solve, which must still pass its own withheld
fixed-geometry validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


ALLOWED_ANCHOR_FAMILIES = {
    "fixed_regulation_geometry",
    "fixed_arena_geometry",
    "basket_and_court_fixed_geometry",
}
FORBIDDEN_ANCHOR_TOKENS = ("player", "pose", "body", "shoulder", "elbow", "wrist", "hand", "ball")


def _as_vec3(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must be a 3-vector")
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{label} contains non-finite values")
    return out


def _extract_center(report: dict[str, Any]) -> list[float]:
    qa = report.get("qa") if isinstance(report.get("qa"), dict) else {}
    candidates = [
        report.get("camera_center_cm"),
        report.get("center_cm"),
        report.get("camera", {}).get("center_cm") if isinstance(report.get("camera"), dict) else None,
        report.get("solution", {}).get("camera_center_cm") if isinstance(report.get("solution"), dict) else None,
        report.get("solution", {}).get("center_cm") if isinstance(report.get("solution"), dict) else None,
        qa.get("camera_center_world_cm"),
        qa.get("camera_center_cm"),
    ]
    for candidate in candidates:
        if candidate is not None:
            return _as_vec3(candidate, "camera centre")
    raise ValueError("report does not expose a camera centre in centimetres")


def _extract_rms(report: dict[str, Any]) -> float:
    qa = report.get("qa") if isinstance(report.get("qa"), dict) else {}
    candidates = [
        report.get("source_curve_rms_px"),
        report.get("source_rms_px"),
        report.get("reprojection_rms_px"),
        report.get("metrics", {}).get("source_curve_rms_px") if isinstance(report.get("metrics"), dict) else None,
        report.get("metrics", {}).get("reprojection_rms_px") if isinstance(report.get("metrics"), dict) else None,
        qa.get("combined_curve_rms_px"),
        qa.get("source_curve_rms_px"),
        qa.get("reprojection_rms_px"),
    ]
    for candidate in candidates:
        if candidate is not None:
            value = float(candidate)
            if not math.isfinite(value):
                raise ValueError("source RMS is non-finite")
            return value
    raise ValueError("report does not expose an independently measured source-pixel RMS")


def _promotion_passed(report: dict[str, Any]) -> bool:
    for key in ("promotion_allowed", "metric_promotion_allowed", "passed"):
        if key in report:
            return bool(report[key])
    gate = report.get("gate")
    if isinstance(gate, dict):
        for key in ("promotion_allowed", "passed", "pass"):
            if key in gate:
                return bool(gate[key])
    return False


def _anchor_family(report: dict[str, Any]) -> str:
    value = report.get("anchor_family")
    if value is None and isinstance(report.get("provenance"), dict):
        value = report["provenance"].get("anchor_family")
    return str(value or "").strip().lower()


def _anchor_text(report: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in ("anchor_family", "anchors", "landmarks", "provenance", "notes"):
        value = report.get(key)
        if value is not None:
            pieces.append(json.dumps(value, sort_keys=True).lower())
    return " ".join(pieces)


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = list(a), list(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(aa, bb)))


def audit(paths: list[Path], max_source_rms_px: float, max_center_deviation_cm: float, min_frames: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text())
        family = _anchor_family(report)
        text = _anchor_text(report)
        forbidden_hits = sorted({token for token in FORBIDDEN_ANCHOR_TOKENS if token in text})
        try:
            center = _extract_center(report)
            rms = _extract_rms(report)
            promoted = _promotion_passed(report)
            geometry_ok = family in ALLOWED_ANCHOR_FAMILIES and not forbidden_hits
            eligible = promoted and geometry_ok and rms <= max_source_rms_px
            error = None
        except (TypeError, ValueError, KeyError) as exc:
            center, rms, promoted, geometry_ok, eligible = None, None, False, False, False
            error = str(exc)
        rows.append({
            "path": str(path),
            "anchor_family": family,
            "forbidden_anchor_hits": forbidden_hits,
            "source_rms_px": rms,
            "input_promotion_allowed": promoted,
            "fixed_geometry_only": geometry_ok,
            "eligible": eligible,
            "camera_center_cm": center,
            "error": error,
        })

    eligible_rows = [r for r in rows if r["eligible"]]
    result: dict[str, Any] = {
        "gate": "physical_camera_center_stability_v1",
        "semantics": "centre-prior eligibility only; never impact-camera promotion",
        "thresholds": {
            "min_independent_frames": min_frames,
            "max_source_rms_px": max_source_rms_px,
            "max_center_deviation_cm": max_center_deviation_cm,
        },
        "candidate_count": len(rows),
        "eligible_input_count": len(eligible_rows),
        "inputs": rows,
        "center_prior_allowed": False,
        "promotion_allowed": False,
    }

    if len(eligible_rows) < min_frames:
        result["status"] = "FAIL_INSUFFICIENT_INDEPENDENT_FIXED_GEOMETRY_CALIBRATIONS"
        return result

    centers = [r["camera_center_cm"] for r in eligible_rows]
    med = [median([c[i] for c in centers]) for i in range(3)]
    deviations = [_distance(c, med) for c in centers]
    pairwise = [_distance(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers))]
    max_dev = max(deviations)
    result.update({
        "median_camera_center_cm": med,
        "center_deviation_cm": deviations,
        "max_center_deviation_cm": max_dev,
        "median_center_deviation_cm": median(deviations),
        "max_pairwise_center_distance_cm": max(pairwise) if pairwise else 0.0,
    })

    allowed = max_dev <= max_center_deviation_cm
    result["center_prior_allowed"] = allowed
    # Deliberately stays false: a stable centre is a prior, not a calibrated impact view.
    result["promotion_allowed"] = False
    result["status"] = "PASS_CENTER_PRIOR_ONLY" if allowed else "FAIL_CENTER_UNSTABLE"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--max-source-rms-px", type=float, default=3.0)
    ap.add_argument("--max-center-deviation-cm", type=float, required=True)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.min_frames < 3:
        raise SystemExit("min-frames must be >=3; two frames cannot establish a stable centre prior")
    if args.max_source_rms_px <= 0 or args.max_center_deviation_cm <= 0:
        raise SystemExit("thresholds must be positive")
    result = audit(args.reports, args.max_source_rms_px, args.max_center_deviation_cm, args.min_frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "eligible_input_count", "center_prior_allowed", "promotion_allowed")}, indent=2))


if __name__ == "__main__":
    main()
