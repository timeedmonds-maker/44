from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

FPS = 30.0
# Five frames spanning 0.267 s around the current impact anchor. This gives every
# physical feed strong same-camera temporal tracks while retaining cross-camera overlap.
FRAME_INDICES = [24, 26, 28, 30, 32]
VIEWS = [
    ("Broadcast", "01_Broadcast_contact.mp4"),
    ("Other Broadcast", "02_Other_Broadcast_contact.mp4"),
    ("Mobile Broadcast", "03_Mobile_Broadcast_contact.mp4"),
    ("Play by Play", "04_Play_by_Play_contact.mp4"),
    ("In Arena", "05_In_Arena_contact.mp4"),
    ("High Tight", "06_High_Tight_contact.mp4"),
    ("Left Slash", "07_Left_Slash_contact.mp4"),
    ("Right Slash", "08_Right_Slash_contact.mp4"),
    ("Left HandHeld", "09_Left_HandHeld_contact.mp4"),
    ("Right HandHeld", "10_Right_HandHeld_contact.mp4"),
    ("Left Above Rim", "11_Left_Above_Rim_contact.mp4"),
    ("Right Above Rim", "12_Right_Above_Rim_contact.mp4"),
]


def safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for view_index, (label, filename) in enumerate(VIEWS):
        src = args.windows / filename
        if not src.exists():
            raise FileNotFoundError(src)
        folder = args.out / f"{view_index:02d}_{safe(label)}"
        folder.mkdir(parents=True, exist_ok=True)
        for frame_index in FRAME_INDICES:
            t = frame_index / FPS
            target = folder / f"f{frame_index:03d}.png"
            subprocess.run([
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{t:.6f}", "-i", str(src), "-frames:v", "1", str(target),
            ], check=True)
            records.append({
                "view_index": view_index,
                "label": label,
                "source": filename,
                "frame_index": frame_index,
                "timestamp_in_window": t,
                "image": str(target.relative_to(args.out)).replace("\\", "/"),
                "impact_anchor": frame_index == 28,
            })

    payload = {
        "fps": FPS,
        "frame_indices": FRAME_INDICES,
        "images_per_view": len(FRAME_INDICES),
        "view_count": len(VIEWS),
        "image_count": len(records),
        "views": records,
    }
    (args.out / "mapping.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
