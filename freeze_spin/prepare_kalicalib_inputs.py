from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

IMPACT_FRAME = 28
FPS = 30.0

# Broad set first. The calibration model itself will determine which views contain
# enough court geometry to return a valid camera matrix.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    mapping = []
    timestamp = IMPACT_FRAME / FPS
    for index, (label, filename) in enumerate(VIEWS):
        source = args.windows / filename
        if not source.exists():
            raise FileNotFoundError(source)
        target = args.out / f"{index}.png"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{timestamp:.6f}", "-i", str(source),
                "-frames:v", "1", str(target),
            ],
            check=True,
        )
        mapping.append(
            {
                "index": index,
                "label": label,
                "source": filename,
                "impact_frame": IMPACT_FRAME,
                "impact_timestamp_in_window": timestamp,
                "image": target.name,
            }
        )

    (args.out / "mapping.json").write_text(
        json.dumps({"views": mapping}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"views": mapping}, indent=2), flush=True)


if __name__ == "__main__":
    main()
