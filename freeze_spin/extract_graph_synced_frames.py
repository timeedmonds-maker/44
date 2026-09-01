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
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    sync = json.loads(args.sync.read_text(encoding="utf-8"))
    by_sync = {row["label"]: row for row in sync["angles"]}
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, angle in enumerate(config["angles"], 1):
        label = angle["label"]
        selected = by_sync[label]
        time_value = float(selected["predicted_freeze_time"])
        source = args.clips / angle["file"]
        output = args.out / f"{index:02d}_{safe_name(label)}.png"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{time_value:.5f}", "-i", str(source),
                "-frames:v", "1", str(output),
            ],
            check=True,
        )
        manifest.append(
            {
                **angle,
                "manual_freeze_time": angle["freeze_time"],
                "freeze_time": time_value,
                "sync_confidence": selected["confidence"],
                "audio_graph_offset_seconds": selected["offset_seconds_vs_reference"],
                "frame": output.name,
            }
        )

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "event": config["event"],
                "synchronization": sync["method"],
                "angles": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
