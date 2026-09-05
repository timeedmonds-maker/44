from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor


def box_distance_to_point(box: np.ndarray, point: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    px, py = [float(v) for v in point]
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def mask_distance_to_point(binary: np.ndarray, point: np.ndarray) -> float:
    x = int(np.clip(round(float(point[0])), 0, binary.shape[1] - 1))
    y = int(np.clip(round(float(point[1])), 0, binary.shape[0] - 1))
    if binary[y, x] > 0:
        return 0.0
    inv = (binary == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    return float(dist[y, x])


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
    ap.add_argument("--max-action-instances", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    image_path = args.locked_images / "In_Arena_F26.png"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    ball = json.loads(args.ball_report.read_text(encoding="utf-8"))
    row = next(v for v in ball["views"] if v["label"] == "In Arena")
    ball_px = np.asarray(row["observed_ball_selected_px"], dtype=np.float64)
    H = np.asarray(row["camera_motion_homography_selected_to_anchor"], dtype=np.float64)

    circle = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(circle, tuple(int(round(v)) for v in ball_px), 90, 1, -1)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=True).eval()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        out = model([to_tensor(rgb)])[0]

    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    boxes = out["boxes"].detach().cpu().numpy()
    masks = out["masks"].detach().cpu().numpy()[:, 0]

    candidates = []
    for i in range(len(scores)):
        if int(labels[i]) != 1 or float(scores[i]) < args.score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        box_h = y2 - y1
        dist_box = box_distance_to_point(boxes[i], ball_px)
        binary = (masks[i] >= args.mask_threshold).astype(np.uint8) * 255
        pixels = int((binary > 0).sum())
        dist_mask = mask_distance_to_point(binary, ball_px)
        overlap90 = int(((binary > 0) & (circle > 0)).sum())
        if dist_box > args.max_box_distance_from_ball:
            continue
        if box_h < args.min_box_height:
            continue
        if y2 < float(ball_px[1]) + args.min_box_bottom_below_ball:
            continue
        if pixels < args.min_mask_pixels:
            continue
        meta = {
            "det_index": int(i),
            "score": round(float(scores[i]), 6),
            "box_xyxy": [round(float(v), 2) for v in boxes[i]],
            "mask_pixels": pixels,
            "mask_distance_to_ball_px": round(dist_mask, 3),
            "mask_overlap_ball90_px": overlap90,
        }
        candidates.append((dist_mask, -overlap90, -pixels, -float(scores[i]), meta, binary))

    if len(candidates) < 2:
        raise RuntimeError(f"Need at least two reference action instances, got {len(candidates)}")
    candidates.sort(key=lambda x: x[:4])
    chosen = candidates[:args.max_action_instances]

    entries = []
    union_selected = np.zeros(image.shape[:2], dtype=np.uint8)
    union_anchor = np.zeros(image.shape[:2], dtype=np.uint8)
    for rank, (_, _, _, _, meta, binary) in enumerate(chosen):
        aligned = cv2.warpPerspective(binary, H, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned = (aligned > 0).astype(np.uint8) * 255
        selected_name = f"reference_instance_{rank:02d}_selected.png"
        anchor_name = f"reference_instance_{rank:02d}_anchor.png"
        cv2.imwrite(str(args.out / selected_name), binary)
        cv2.imwrite(str(args.out / anchor_name), aligned)
        union_selected = cv2.bitwise_or(union_selected, binary)
        union_anchor = cv2.bitwise_or(union_anchor, aligned)
        e = dict(meta)
        e.update({"rank": rank, "selected_mask": selected_name, "anchor_mask": anchor_name,
                  "anchor_mask_pixels": int((aligned > 0).sum())})
        entries.append(e)

    cv2.imwrite(str(args.out / "reference_instances_union_selected.png"), union_selected)
    cv2.imwrite(str(args.out / "reference_instances_union_anchor.png"), union_anchor)
    payload = {
        "purpose": "Identity-free per-instance reference layers for stable small-angle 2.5D rendering",
        "reference": "In Arena",
        "identity_matching": False,
        "observed_ball_selected_px": [round(float(v), 3) for v in ball_px],
        "instances": entries,
    }
    (args.out / "reference_instances_v2.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
