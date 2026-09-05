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
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, angle in enumerate(payload["angles"], 1):
        source = args.clips / angle["file"]
        output = args.out / f"{index:02d}_{safe_name(angle['label'])}.png"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-ss", f"{float(angle['freeze_time']):.3f}",
                "-i", str(source), "-frames:v", "1", str(output),
            ],
            check=True,
        )
        manifest.append({**angle, "frame": output.name})

    (args.out / "manifest.json").write_text(
        json.dumps({"event": payload["event"], "angles": manifest}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
