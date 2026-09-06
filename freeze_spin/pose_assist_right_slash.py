from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

# This tool is deliberately annotation-only. Its 2D keypoints must never be
# consumed as metric truth without source-image review and calibrated reprojection
# QA in a separate stage.

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    args = ap.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    rows = []

    for path in sorted(args.images.glob("f*.png")):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        result = model.predict(source=image, conf=args.conf, verbose=False, device="cpu")[0]
        detections = []
        boxes = [] if result.boxes is None else result.boxes.xyxy.cpu().numpy()
        scores = [] if result.boxes is None else result.boxes.conf.cpu().numpy()
        keypoints = [] if result.keypoints is None else result.keypoints.data.cpu().numpy()

        for i, kp in enumerate(keypoints):
            points = {}
            for name, item in zip(COCO_KEYPOINTS, kp):
                x, y, conf = (float(item[0]), float(item[1]), float(item[2]))
                points[name] = {"xy": [round(x, 3), round(y, 3)], "confidence": round(conf, 5)}
            box = boxes[i].tolist() if i < len(boxes) else [0.0, 0.0, 0.0, 0.0]
            score = float(scores[i]) if i < len(scores) else 0.0
            detections.append({
                "detection_index": i,
                "box_xyxy": [round(float(x), 3) for x in box],
                "box_confidence": round(score, 5),
                "keypoints": points,
            })

        annotated = result.plot()
        overlay_path = args.out / f"{path.stem}_pose.png"
        if not cv2.imwrite(str(overlay_path), annotated):
            raise RuntimeError(f"Could not write {overlay_path}")
        rows.append({
            "image": path.name,
            "pose_overlay": overlay_path.name,
            "detection_count": len(detections),
            "detections": detections,
        })

    payload = {
        "status": "annotation_assistant_only_not_geometry_truth",
        "model": args.model,
        "confidence_threshold": args.conf,
        "guardrail": "No detected joint is accepted as an Adams/Cissoko landmark until manually identified on the source image and validated by calibrated multi-view reprojection.",
        "frames": rows,
    }
    (args.out / "right_slash_pose_assist.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"frame_count": len(rows), "detections_per_frame": [r["detection_count"] for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
