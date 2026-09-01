from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor

VIEWS = {
    "In Arena": "In_Arena_F26.png",
    "Left Slash": "Left_Slash_F27.png",
    "Left HandHeld": "Left_HandHeld_F25.png",
    "Left Above Rim": "Left_Above_Rim_F24.png",
}


def safe(label: str) -> str:
    return label.replace(" ", "_")


def box_distance_to_point(box: np.ndarray, point: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    px, py = [float(v) for v in point]
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-threshold", type=float, default=0.28)
    ap.add_argument("--mask-threshold", type=float, default=0.42)
    ap.add_argument("--max-box-distance-from-ball", type=float, default=150.0)
    ap.add_argument("--min-box-height", type=float, default=105.0)
    ap.add_argument("--min-box-bottom-below-ball", type=float, default=85.0)
    ap.add_argument("--min-mask-pixels", type=int, default=650)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ball = json.loads(args.ball_report.read_text(encoding="utf-8"))
    ball_views = {v["label"]: v for v in ball["views"]}

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True).eval()

    report = {
        "purpose": "Identity-free foreground support for free-view reconstruction. Segmentation selects court-body pixels only; metric cameras remain authoritative for 3D.",
        "model": "torchvision MaskRCNN ResNet50 FPN V2 default COCO weights",
        "person_class": 1,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "max_box_distance_from_ball_px": args.max_box_distance_from_ball,
        "court_player_gate": {
            "min_box_height_px": args.min_box_height,
            "min_box_bottom_below_ball_px": args.min_box_bottom_below_ball,
            "min_mask_pixels": args.min_mask_pixels,
            "reason": "At these four validated basket-facing views, action players extend down into the court below the airborne ball; nearby spectator false positives do not. This is geometry/context filtering, not identity matching.",
        },
        "views": {},
    }

    with torch.inference_mode():
        for label, filename in VIEWS.items():
            path = args.locked_images / filename
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            if label not in ball_views:
                raise KeyError(f"Ball report missing {label}")
            ball_px = np.asarray(ball_views[label]["observed_ball_selected_px"], dtype=np.float64)

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out = model([to_tensor(rgb)])[0]
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            masks = out["masks"].detach().cpu().numpy()[:, 0]

            union = np.zeros(image.shape[:2], dtype=np.uint8)
            selected = []
            all_persons = []
            for i in range(len(scores)):
                if int(labels[i]) != 1 or float(scores[i]) < args.score_threshold:
                    continue
                dist = box_distance_to_point(boxes[i], ball_px)
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                box_h = y2 - y1
                binary = (masks[i] >= args.mask_threshold).astype(np.uint8) * 255
                mask_pixels = int((binary > 0).sum())
                row = {
                    "det_index": int(i),
                    "score": round(float(scores[i]), 6),
                    "box_xyxy": [round(float(v), 2) for v in boxes[i]],
                    "box_height_px": round(box_h, 3),
                    "box_bottom_minus_ball_y_px": round(y2 - float(ball_px[1]), 3),
                    "mask_pixels": mask_pixels,
                    "box_distance_to_ball_px": round(dist, 3),
                }
                all_persons.append(row)
                if dist > args.max_box_distance_from_ball:
                    continue
                if box_h < args.min_box_height:
                    continue
                if y2 < float(ball_px[1]) + args.min_box_bottom_below_ball:
                    continue
                if mask_pixels < args.min_mask_pixels:
                    continue
                union = cv2.bitwise_or(union, binary)
                selected.append(row)

            # Preserve thin limbs and close small holes without substantially expanding silhouettes.
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            union = cv2.dilate(union, kernel_dilate, iterations=1)

            # Drop residual tiny disconnected islands after unioning accepted court-player instances.
            count, comp, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), 8)
            clean = np.zeros_like(union)
            component_areas = []
            for ci in range(1, count):
                area = int(stats[ci, cv2.CC_STAT_AREA])
                component_areas.append(area)
                if area >= 500:
                    clean[comp == ci] = 255
            union = clean

            if int((union > 0).sum()) < 1200:
                raise RuntimeError(f"Action-body mask too small for {label}: {(union > 0).sum()} px")

            H = np.asarray(ball_views[label]["camera_motion_homography_selected_to_anchor"], dtype=np.float64)
            aligned = cv2.warpPerspective(union, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            aligned = (aligned > 0).astype(np.uint8) * 255

            native_path = args.out / f"{safe(label)}_body_mask_selected.png"
            aligned_path = args.out / f"{safe(label)}_body_mask_anchor.png"
            cv2.imwrite(str(native_path), union)
            cv2.imwrite(str(aligned_path), aligned)

            overlay = image.copy()
            tint = np.zeros_like(image)
            tint[:, :, 1] = union
            overlay = cv2.addWeighted(overlay, 1.0, tint, 0.38, 0.0)
            cv2.circle(overlay, tuple(int(round(v)) for v in ball_px), 7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"{safe(label)}_body_overlay_selected.png"), overlay)

            report["views"][label] = {
                "image": filename,
                "observed_ball_selected_px": [round(float(v), 3) for v in ball_px],
                "all_person_detections": all_persons,
                "selected_action_persons": selected,
                "selected_person_count": len(selected),
                "post_union_component_areas_px": sorted(component_areas, reverse=True),
                "native_mask_pixels": int((union > 0).sum()),
                "anchor_mask_pixels": int((aligned > 0).sum()),
                "native_mask": native_path.name,
                "anchor_mask": aligned_path.name,
                "overlay": f"{safe(label)}_body_overlay_selected.png",
            }

    (args.out / "action_body_segmentation_v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: {"persons": v["selected_person_count"], "mask_px": v["native_mask_pixels"]}
                      for k, v in report["views"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
