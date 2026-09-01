from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

VIEWS = [
    (4, "In Arena", "05_In_Arena_contact.mp4"),
    (6, "Left Slash", "07_Left_Slash_contact.mp4"),
    (8, "Left HandHeld", "09_Left_HandHeld_contact.mp4"),
    (10, "Left Above Rim", "11_Left_Above_Rim_contact.mp4"),
]


def safe_name(value: str) -> str:
    return value.replace(" ", "_")


def extract_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_index}")
    return frame


def make_sheet(frames: list[tuple[int, np.ndarray]], label: str) -> np.ndarray:
    thumb_w, thumb_h = 480, 270
    tile_h = thumb_h + 34
    sheet = np.zeros((tile_h * 3, thumb_w * 3, 3), dtype=np.uint8)
    for k, (fi, frame) in enumerate(frames):
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y = (k // 3) * tile_h
        x = (k % 3) * thumb_w
        sheet[y:y + thumb_h, x:x + thumb_w] = thumb
        cv2.putText(sheet, f"{label}  F{fi}", (x + 10, y + thumb_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=24)
    ap.add_argument("--end", type=int, default=32)
    args = ap.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    expected = args.end - args.start + 1
    if expected != 9:
        raise SystemExit("This QA exporter expects exactly nine candidate frames for a 3x3 sheet")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, label, filename in VIEWS:
        source = args.windows / filename
        if not source.exists():
            raise FileNotFoundError(source)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {source}")
        rows = []
        sheet_frames = []
        view_dir = args.out / f"{index:02d}_{safe_name(label)}"
        view_dir.mkdir(parents=True, exist_ok=True)
        try:
            for fi in range(args.start, args.end + 1):
                frame = extract_frame(cap, fi)
                target = view_dir / f"f{fi:03d}.png"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"Could not write {target}")
                rows.append({"frame_index": fi, "timestamp_seconds": round(fi / 30.0, 6), "image": str(target.relative_to(args.out))})
                sheet_frames.append((fi, frame))
        finally:
            cap.release()
        sheet = make_sheet(sheet_frames, label)
        sheet_path = args.out / f"{index:02d}_{safe_name(label)}_F24_F32_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        manifest.append({"index": index, "label": label, "source": filename, "center_legacy_frame": 28, "preferred_reference_candidate": 27, "candidates": rows, "sheet": sheet_path.name})

    payload = {
        "purpose": "Exact visual-state refinement after strict four-camera metric calibration",
        "rule": "Choose the same physical basketball state independently per camera; do not assume equal local frame index.",
        "candidate_range": [args.start, args.end],
        "views": manifest,
    }
    (args.out / "visual_state_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
