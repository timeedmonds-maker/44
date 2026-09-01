from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

VIEWS = {
    "In Arena": "In_Arena_F26.png",
    "Left Slash": "Left_Slash_F27.png",
    "Left HandHeld": "Left_HandHeld_F25.png",
    "Left Above Rim": "Left_Above_Rim_F24.png",
}


def torso_colour(image: np.ndarray, box: np.ndarray, keypoints: np.ndarray) -> dict:
    h, w = image.shape[:2]
    ls = keypoints[5, :2]
    rs = keypoints[6, :2]
    lh = keypoints[11, :2]
    rh = keypoints[12, :2]
    xs = [ls[0], rs[0], lh[0], rh[0]]
    ys = [ls[1], rs[1], lh[1], rh[1]]
    if not np.all(np.isfinite(xs + ys)):
        x1, y1, x2, y2 = box
    else:
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        pad_x = max(4.0, (x2 - x1) * 0.15)
        pad_y = max(4.0, (y2 - y1) * 0.10)
        x1, x2 = x1 - pad_x, x2 + pad_x
        y1, y2 = y1 - pad_y, y2 + pad_y
    x1 = max(0, min(w - 1, int(round(x1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y1 = max(0, min(h - 1, int(round(y1))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return {"bgr_median": None, "red_fraction": 0.0, "white_fraction": 0.0}
    bgr = np.median(patch.reshape(-1, 3), axis=0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 45), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 70, 45), (179, 255, 255))
    red_fraction = float(np.mean((red1 > 0) | (red2 > 0)))
    white = cv2.inRange(hsv, (0, 0, 145), (179, 90, 255))
    white_fraction = float(np.mean(white > 0))
    return {
        "bgr_median": [round(float(v), 2) for v in bgr],
        "red_fraction": round(red_fraction, 5),
        "white_fraction": round(white_fraction, 5),
        "patch_xyxy": [x1, y1, x2, y2],
    }


def draw_detection(image: np.ndarray, det: dict) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"P{det['person_index']} {det['score']:.2f}", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    kp = det["keypoints"]
    for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                 "left_wrist", "right_wrist", "left_hip", "right_hip"):
        row = kp[name]
        if row["score"] < 0.25:
            continue
        p = tuple(int(round(v)) for v in row["xy"])
        cv2.circle(image, p, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(image, name.replace("left_", "L").replace("right_", "R")[:3],
                    (p[0] + 3, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (0, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--score-threshold", type=float, default=0.35)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    model = keypointrcnn_resnet50_fpn(weights=weights, progress=True).eval()

    report = {
        "purpose": "Pose model is annotation assistance only; calibrated multi-view geometry remains authoritative.",
        "model": "torchvision keypointrcnn_resnet50_fpn COCO_V1",
        "views": {},
    }

    with torch.inference_mode():
        for label, filename in VIEWS.items():
            path = args.locked_images / filename
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = model([to_tensor(rgb)])[0]
            overlay = image.copy()
            detections = []
            scores = output["scores"].detach().cpu().numpy()
            boxes = output["boxes"].detach().cpu().numpy()
            kps = output["keypoints"].detach().cpu().numpy()
            kp_scores = output.get("keypoints_scores")
            kp_scores = None if kp_scores is None else kp_scores.detach().cpu().numpy()
            keep = np.where(scores >= args.score_threshold)[0]
            for person_index, idx in enumerate(keep.tolist()):
                keypoints = {}
                for j, name in enumerate(KEYPOINT_NAMES):
                    score = float(kp_scores[idx, j]) if kp_scores is not None else float(kps[idx, j, 2])
                    keypoints[name] = {
                        "xy": [round(float(kps[idx, j, 0]), 3), round(float(kps[idx, j, 1]), 3)],
                        "score": round(score, 5),
                    }
                colour = torso_colour(image, boxes[idx], kps[idx])
                det = {
                    "person_index": person_index,
                    "score": round(float(scores[idx]), 6),
                    "box_xyxy": [round(float(v), 3) for v in boxes[idx]],
                    "torso_colour": colour,
                    "keypoints": keypoints,
                }
                detections.append(det)
                draw_detection(overlay, det)
            report["views"][label] = {"image": filename, "detections": detections}
            cv2.imwrite(str(args.out / f"{label.replace(' ', '_')}_pose_overlay.png"), overlay)

    (args.out / "pose_candidates_v1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({label: len(row["detections"]) for label, row in report["views"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
