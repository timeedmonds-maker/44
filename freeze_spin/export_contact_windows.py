from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sync", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--before", type=float, default=0.40)
    parser.add_argument("--after", type=float, default=0.40)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sync = json.loads(args.sync.read_text(encoding="utf-8"))
    by_sync = {row["label"]: row for row in sync["angles"]}
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, angle in enumerate(config["angles"], 1):
        label = angle["label"]
        predicted = float(by_sync[label]["predicted_freeze_time"])
        start = max(0.0, predicted - args.before)
        duration = args.before + args.after
        source = args.clips / angle["file"]
        output = args.out / f"{index:02d}_{safe_name(label)}_contact.mp4"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{start:.5f}", "-t", f"{duration:.5f}",
                "-i", str(source),
                "-an", "-vf", "fps=30,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
        manifest.append(
            {
                "label": label,
                "file": output.name,
                "window_start_source_seconds": round(start, 5),
                "predicted_contact_offset_in_window_seconds": round(predicted - start, 5),
                "predicted_source_time": round(predicted, 5),
                "manual_source_time_for_validation_only": angle["freeze_time"],
            }
        )

    (args.out / "manifest.json").write_text(
        json.dumps({"event": config["event"], "angles": manifest}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
