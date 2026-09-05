from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def slug(label: str) -> str:
    return label.replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--shoulders", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    shoulders = json.loads(args.shoulders.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    min_views = int(audit.get("minimum_valid_views_per_shoulder", 3))
    border = float(audit.get("minimum_border_margin_px", 30.0))

    point_results: dict[str, dict] = {}
    overlays = []

    for view_label, view_audit in audit["views"].items():
        image_path = args.locked_images / view_audit["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Missing locked-state image: {image_path}")
        h, w = image.shape[:2]
        overlay = image.copy()

        for point_name, pa in view_audit["points"].items():
            obs = shoulders["shoulders"][point_name]["views"].get(view_label)
            if obs is None:
                raise KeyError(f"No shoulder observation for {point_name} in {view_label}")
            x, y = map(float, obs["pixel_xy_selected_frame"])
            margin = min(x, y, (w - 1.0) - x, (h - 1.0) - y)
            manual_ok = bool(pa.get("visual_identity_valid", False))
            border_ok = margin >= border
            valid = manual_ok and border_ok

            row = point_results.setdefault(point_name, {"valid_views": [], "invalid_views": {}})
            detail = {
                "pixel_xy": [x, y],
                "manual_visual_identity_valid": manual_ok,
                "border_margin_px": round(float(margin), 3),
                "border_margin_gate_passed": border_ok,
                "reason": pa.get("reason", ""),
            }
            if valid:
                row["valid_views"].append(view_label)
            else:
                row["invalid_views"][view_label] = detail

            p = (int(round(x)), int(round(y)))
            color = (0, 220, 0) if valid else (0, 0, 255)
            cv2.circle(overlay, p, 8, color, 2, cv2.LINE_AA)
            if not valid:
                cv2.line(overlay, (p[0]-7, p[1]-7), (p[0]+7, p[1]+7), color, 2, cv2.LINE_AA)
                cv2.line(overlay, (p[0]-7, p[1]+7), (p[0]+7, p[1]-7), color, 2, cv2.LINE_AA)
            text = f"{point_name}: {'VALID' if valid else 'INVALID'}"
            cv2.putText(overlay, text, (max(5, p[0]-180), max(20, p[1]-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        title = f"{view_label} | manual identity audit"
        cv2.rectangle(overlay, (0, 0), (min(w, 430), 28), (0, 0, 0), -1)
        cv2.putText(overlay, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = args.out / f"{slug(view_label)}_identity_audit.png"
        cv2.imwrite(str(out_path), overlay)
        overlays.append(overlay)

    per_point_gate = {}
    for point_name, row in point_results.items():
        count = len(row["valid_views"])
        per_point_gate[point_name] = {
            "valid_view_count": count,
            "required_valid_view_count": min_views,
            "valid_views": row["valid_views"],
            "invalid_views": row["invalid_views"],
            "pass": count >= min_views,
        }

    overall_pass = all(v["pass"] for v in per_point_gate.values())
    report = {
        "status": "pass" if overall_pass else "fail_visual_identity",
        "source_status_under_test": shoulders.get("status"),
        "principle": "Low residual triangulation cannot override source-pixel identity failure.",
        "minimum_valid_views_per_shoulder": min_views,
        "minimum_border_margin_px": border,
        "points": per_point_gate,
        "gate": {"pass": overall_pass},
        "interpretation": (
            "Shoulder identity has enough independently visible calibrated views."
            if overall_pass
            else "The v2 numerical shoulder result is not a valid identity lock; acquire additional metric identity-bearing views rather than loosening the gate."
        ),
    }
    (args.out / "block_identity_visual_gate_v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if overlays:
        target_h = 360
        resized = []
        for im in overlays:
            scale = target_h / im.shape[0]
            resized.append(cv2.resize(im, (int(round(im.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA))
        montage = np.hstack(resized)
        cv2.imwrite(str(args.out / "block_identity_visual_audit_montage.png"), montage)

    print(json.dumps(report, indent=2), flush=True)
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
