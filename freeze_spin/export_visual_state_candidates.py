from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# Keep the original four calibrated proof cameras first, then export the
# complementary right-side views as diagnostics for the next body-coverage
# milestone. Downstream four-camera geometry code keys by label and therefore
# continues to use only the calibrated views until the new cameras are solved.
VIEWS = [
    (4, "In Arena", "05_In_Arena_contact.mp4"),
    (6, "Left Slash", "07_Left_Slash_contact.mp4"),
    (8, "Left HandHeld", "09_Left_HandHeld_contact.mp4"),
    (10, "Left Above Rim", "11_Left_Above_Rim_contact.mp4"),
    (7, "Right Slash", "08_Right_Slash_contact.mp4"),
    (9, "Right HandHeld", "10_Right_HandHeld_contact.mp4"),
    (11, "Right Above Rim", "12_Right_Above_Rim_contact.mp4"),
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
    if not frames:
        raise ValueError("No frames supplied for QA sheet")
    thumb_w, thumb_h = 480, 270
    tile_h = thumb_h + 34
    cols = min(4, len(frames))
    rows = int(math.ceil(len(frames) / cols))
    sheet = np.zeros((tile_h * rows, thumb_w * cols, 3), dtype=np.uint8)
    for k, (fi, frame) in enumerate(frames):
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        y = (k // cols) * tile_h
        x = (k % cols) * thumb_w
        sheet[y:y + thumb_h, x:x + thumb_w] = thumb
        cv2.putText(sheet, f"{label}  F{fi}", (x + 10, y + thumb_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=23)
    ap.add_argument("--end", type=int, default=32)
    ap.add_argument("--reference-frame", type=int, default=28)
    args = ap.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be >= --start")
    if not (args.start <= args.reference_frame <= args.end):
        raise SystemExit("--reference-frame must lie inside the exported candidate range")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    calibrated_labels = {"In Arena", "Left Slash", "Left HandHeld", "Left Above Rim"}
    for index, label, filename in VIEWS:
        source = args.windows / filename
        if not source.exists():
            raise FileNotFoundError(source)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {source}")
        rows = []
        sheet_frames = []
        reference_image = None
        view_dir = args.out / f"{index:02d}_{safe_name(label)}"
        view_dir.mkdir(parents=True, exist_ok=True)
        try:
            for fi in range(args.start, args.end + 1):
                frame = extract_frame(cap, fi)
                target = view_dir / f"f{fi:03d}.png"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"Could not write {target}")
                rows.append({
                    "frame_index": fi,
                    "timestamp_seconds": round(fi / 30.0, 6),
                    "image": str(target.relative_to(args.out)),
                })
                sheet_frames.append((fi, frame))
                if fi == args.reference_frame:
                    reference_image = frame.copy()
        finally:
            cap.release()
        sheet = make_sheet(sheet_frames, label)
        sheet_path = args.out / f"{index:02d}_{safe_name(label)}_F{args.start}_F{args.end}_sheet.png"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError(f"Could not write {sheet_path}")
        # The packaging workflow already collects *_sheet.png files. Store a
        # single native-resolution reference as a one-frame QA sheet so camera
        # calibration can be annotated against the original 960x540 pixels.
        reference_path = args.out / f"{index:02d}_{safe_name(label)}_F{args.reference_frame}_reference_sheet.png"
        if reference_image is None or not cv2.imwrite(str(reference_path), reference_image):
            raise RuntimeError(f"Could not write {reference_path}")
        manifest.append({
            "index": index,
            "label": label,
            "source": filename,
            "calibration_status": "calibrated" if label in calibrated_labels else "diagnostic_unscaled",
            "center_legacy_frame": 28,
            "preferred_reference_candidate": 27,
            "native_reference_frame": args.reference_frame,
            "native_reference_image": reference_path.name,
            "candidates": rows,
            "sheet": sheet_path.name,
        })

    payload = {
        "purpose": "Exact visual-state refinement plus complementary right-side coverage diagnostics after strict four-camera metric calibration",
        "rule": "Choose the same physical basketball state independently per camera; do not assume equal local frame index. Diagnostic right-side views are not metric cameras until independently calibrated.",
        "candidate_range": [args.start, args.end],
        "candidate_count_per_view": args.end - args.start + 1,
        "native_reference_frame": args.reference_frame,
        "views": manifest,
    }
    (args.out / "visual_state_candidates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
