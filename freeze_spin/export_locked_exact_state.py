from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SOURCE_FILES = {
    "In Arena": "05_In_Arena_contact.mp4",
    "Left Slash": "07_Left_Slash_contact.mp4",
    "Left HandHeld": "09_Left_HandHeld_contact.mp4",
    "Left Above Rim": "11_Left_Above_Rim_contact.mp4",
}


def safe_name(value: str) -> str:
    return value.replace(" ", "_")


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
    return frame


def make_montage(rows: list[tuple[str, int, np.ndarray]]) -> np.ndarray:
    tile_w, tile_h = 480, 270
    footer = 42
    canvas = np.zeros((tile_h + footer, tile_w * len(rows), 3), dtype=np.uint8)
    for i, (label, frame_index, frame) in enumerate(rows):
        thumb = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        x = i * tile_w
        canvas[:tile_h, x:x + tile_w] = thumb
        cv2.putText(canvas, f"{label}  F{frame_index}", (x + 10, tile_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    selected = state["selected_frames"]
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    montage_rows = []
    for label in ("In Arena", "Left Slash", "Left HandHeld", "Left Above Rim"):
        if label not in selected:
            raise KeyError(f"Missing selected frame for {label}")
        source = args.windows / SOURCE_FILES[label]
        frame_index = int(selected[label])
        frame = read_frame(source, frame_index)
        target = args.out / f"{safe_name(label)}_F{frame_index}.png"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"Could not write {target}")
        manifest.append({
            "label": label,
            "frame_index": frame_index,
            "timestamp_seconds_assuming_30fps": round(frame_index / 30.0, 6),
            "source": SOURCE_FILES[label],
            "image": target.name,
        })
        montage_rows.append((label, frame_index, frame))

    montage = make_montage(montage_rows)
    montage_path = args.out / "locked_exact_state_montage.png"
    cv2.imwrite(str(montage_path), montage)

    payload = {
        "purpose": "Locked same-physical-moment four-camera freeze state",
        "state_version": state.get("state_version"),
        "selected_state": state.get("selected_state"),
        "source_state_file": str(args.state),
        "views": manifest,
        "montage": montage_path.name,
    }
    (args.out / "locked_exact_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
