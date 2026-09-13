#!/usr/bin/env python3
"""Single approved HD/UHD presentation render for NBA video work.

This repository intentionally uses one deterministic presentation treatment:
light hqdn3d cleanup -> Lanczos resize -> conservative CAS sharpening -> 30 fps.
It does not create new basketball detail; it only creates a cleaner delivery
master from the official source imagery.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FILTER_BASE = "hqdn3d=0.6:0.6:2.0:2.0,scale={w}:{h}:flags=lanczos,cas=0.22,fps=30"
PROFILES = {
    "hd": {
        "width": 1280,
        "height": 720,
        "maxrate": "18M",
        "bufsize": "36M",
        "label": "HD",
    },
    "uhd": {
        "width": 3840,
        "height": 2160,
        "maxrate": "36M",
        "bufsize": "72M",
        "label": "UHD",
    },
}


def run(cmd: list[str], timeout: int = 900) -> None:
    subprocess.run(cmd, check=True, timeout=timeout)


def probe(path: Path) -> dict:
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt",
            "-show_entries", "format=duration,size",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(p.stdout)


def render(src: Path, dst: Path, profile: str, *, preset: str = "veryfast") -> dict:
    cfg = PROFILES[profile]
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = FILTER_BASE.format(w=cfg["width"], h=cfg["height"])
    run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", "16",
        "-maxrate", cfg["maxrate"], "-bufsize", cfg["bufsize"],
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ])
    info = probe(dst)
    stream = (info.get("streams") or [{}])[0]
    if int(stream.get("width") or 0) != cfg["width"] or int(stream.get("height") or 0) != cfg["height"]:
        raise RuntimeError(f"{cfg['label']} QA failed for {dst}: {info}")
    return {
        "profile": profile,
        "width": cfg["width"],
        "height": cfg["height"],
        "filter": vf,
        "codec": "libx264",
        "crf": 16,
        "maxrate": cfg["maxrate"],
        "bufsize": cfg["bufsize"],
        "audio": "aac 192k",
        "source": str(src),
        "output": str(dst),
        "probe": info,
        "method": "deterministic presentation resize; no new source detail claimed",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="uhd")
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--qa-json", type=Path)
    a = ap.parse_args()
    qa = render(a.input, a.output, a.profile, preset=a.preset)
    if a.qa_json:
        a.qa_json.parent.mkdir(parents=True, exist_ok=True)
        a.qa_json.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
