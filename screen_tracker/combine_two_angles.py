#!/usr/bin/env python3
"""Assemble the mandatory two-angle Screen Tracker delivery.

Important length invariant: each angle is encoded independently to the final
presentation profile, then the final-profile angle files are concat-copied.
Never re-encode a concat-demuxed two-angle master: that can retain timestamp
discontinuities and truncate angle 2.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def probe(path: Path):
    obj = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height,r_frame_rate,duration:format=duration",
        "-of", "json", str(path),
    ], text=True))
    streams = obj.get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    return {
        "video": video,
        "audio": audio,
        "format_duration": float(obj.get("format", {}).get("duration") or video.get("duration") or 0),
    }


def video_duration(meta):
    return float(meta["video"].get("duration") or meta["format_duration"])


def run(cmd):
    subprocess.run(cmd, check=True)


def concat_copy(paths, out):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for path in paths:
            f.write("file '%s'\n" % str(path).replace("'", "'\\''"))
    try:
        run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy", str(out),
        ])
    finally:
        list_path.unlink(missing_ok=True)


def concat_native(paths, out):
    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    for path in paths:
        args += ["-i", str(path)]
    graph = ";".join(
        f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}]"
        for i in range(len(paths))
    )
    graph += ";" + "".join(f"[v{i}][a{i}]" for i in range(len(paths)))
    graph += f"concat=n={len(paths)}:v=1:a=1[v][a]"
    args += [
        "-filter_complex", graph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(out),
    ]
    run(args)


def make_streamable(src, out):
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-maxrate", "14M", "-bufsize", "28M", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-g", "60", "-keyint_min", "60",
        "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(out),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--angle", action="append", nargs=4,
        metavar=("LABEL", "NATIVE", "UHD", "QA"), required=True,
        help="Exactly two entries: label native.mp4 uhd.mp4 qa.json",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--basename", required=True)
    args = ap.parse_args()

    if len(args.angle) != 2:
        raise SystemExit("Screen Tracker two-angle default requires exactly two angles")
    args.out.mkdir(parents=True, exist_ok=True)
    labels = [x[0] for x in args.angle]
    if labels[0].strip().lower() == labels[1].strip().lower():
        raise SystemExit("Two-angle output requires distinct angle labels")

    natives = [Path(x[1]) for x in args.angle]
    uhds = [Path(x[2]) for x in args.angle]
    qas = [json.loads(Path(x[3]).read_text()) for x in args.angle]
    for path in natives + uhds:
        if not path.exists():
            raise SystemExit(f"Missing two-angle input: {path}")

    angle_meta = [probe(path) for path in uhds]
    for label, meta in zip(labels, angle_meta):
        video = meta["video"]
        expected = (
            video.get("codec_name") == "h264"
            and int(video.get("width", 0)) == 3840
            and int(video.get("height", 0)) == 2160
            and video.get("r_frame_rate") == "30/1"
        )
        if not expected:
            raise SystemExit(f"Angle {label} UHD profile mismatch: {video}")

    expected_duration = sum(video_duration(x) for x in angle_meta)
    native_out = args.out / f"{args.basename}_native.mp4"
    concat_native(natives, native_out)

    uhd_out = args.out / f"{args.basename}_UHD.mp4"
    concat_copy(uhds, uhd_out)

    per_angle_streamable = []
    for i, path in enumerate(uhds):
        rendered = args.out / f"angle_{i + 1}_streamable.mp4"
        make_streamable(path, rendered)
        per_angle_streamable.append(rendered)
    stream_out = args.out / f"{args.basename}_4k_streamable.mp4"
    concat_copy(per_angle_streamable, stream_out)

    combined_meta = probe(uhd_out)
    stream_meta = probe(stream_out)
    actual_duration = video_duration(combined_meta)
    actual_stream_duration = video_duration(stream_meta)
    tolerance = 0.12
    if abs(actual_duration - expected_duration) > tolerance:
        raise SystemExit(
            f"Combined UHD duration failed: expected {expected_duration:.3f}, got {actual_duration:.3f}"
        )
    if abs(actual_stream_duration - expected_duration) > tolerance:
        raise SystemExit(
            f"Combined streamable duration failed: expected {expected_duration:.3f}, got {actual_stream_duration:.3f}"
        )

    first_duration = video_duration(angle_meta[0])
    second_duration = video_duration(angle_meta[1])
    qa = {
        "tool_id": "SCREEN_TRACKER",
        "angle_policy": {
            "mode": "two_angle_default",
            "angle_count": 2,
            "labels": labels,
            "primary": labels[0],
            "secondary": labels[1],
            "require_distinct_angles": True,
            "require_full_length_each_angle": True,
        },
        "angle_qa": qas,
        "length_qa": {
            "angle_video_durations_s": [first_duration, second_duration],
            "expected_combined_video_duration_s": expected_duration,
            "actual_combined_video_duration_s": actual_duration,
            "actual_streamable_video_duration_s": actual_stream_duration,
            "second_angle_start_s": first_duration,
            "second_angle_expected_end_s": expected_duration,
            "second_angle_full_length": abs(actual_duration - expected_duration) <= tolerance,
            "tolerance_s": tolerance,
            "assembly_rule": "encode each angle independently; concat-copy final-profile angle files; reset timestamps for native concat",
        },
        "native": str(native_out),
        "uhd": str(uhd_out),
        "streamable_4k": str(stream_out),
    }
    (args.out / "qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
