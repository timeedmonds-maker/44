from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


WIDTH = 960
HEIGHT = 540
FPS = 30


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def smoothstep(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def locked_view(
    image: np.ndarray,
    focus: tuple[float, float],
    zoom: float,
    lock_strength: float,
    drift: float,
) -> np.ndarray:
    center_x = WIDTH / 2.0
    center_y = HEIGHT / 2.0
    focus_x, focus_y = focus
    scale = zoom * (1.0 + 0.012 * drift)
    locked = np.array(
        [
            [scale, 0.0, center_x - scale * focus_x],
            [0.0, scale, center_y - scale * focus_y],
        ],
        dtype=np.float32,
    )
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    matrix = identity * (1.0 - lock_strength) + locked * lock_strength
    matrix[0, 2] += 8.0 * drift
    matrix[1, 2] -= 2.5 * abs(drift)
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def travel_warp(image: np.ndarray, amount: float) -> np.ndarray:
    shift = 115.0 * amount
    shear = 0.055 * amount
    matrix = np.array([[1.025, shear, shift - 12.0], [0.0, 1.025, -7.0]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def horizontal_blur(image: np.ndarray, strength: float) -> np.ndarray:
    kernel = max(1, int(round(1 + 28 * strength)))
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(image, (kernel, 1), 0)


def apply_finish(image: np.ndarray, vignette: np.ndarray) -> np.ndarray:
    frame = image.astype(np.float32)
    frame = (frame - 128.0) * 1.035 + 128.0
    frame *= vignette[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def make_vignette() -> np.ndarray:
    y, x = np.ogrid[-1.0:1.0:HEIGHT * 1j, -1.0:1.0:WIDTH * 1j]
    radius = np.sqrt(x * x + y * y)
    return np.clip(1.03 - 0.16 * radius**1.7, 0.80, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    frame_manifest = json.loads((args.frames / "manifest.json").read_text(encoding="utf-8"))
    by_label = {row["label"]: row for row in frame_manifest["angles"]}
    source_rows = {row["label"]: row for row in config["angles"]}
    order = config["orbit_order"]

    shutil.rmtree(args.work, ignore_errors=True)
    orbit_frames = args.work / "orbit_frames"
    orbit_frames.mkdir(parents=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    images: dict[str, np.ndarray] = {}
    for label in set(order):
        image = cv2.imread(str(args.frames / by_label[label]["frame"]))
        if image is None:
            raise RuntimeError(f"Could not read synchronized frame for {label}")
        if image.shape[1] != WIDTH or image.shape[0] != HEIGHT:
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        images[label] = image

    vignette = make_vignette()
    hold_frames = 12
    transition_frames = 4
    rendered_views: list[list[np.ndarray]] = []

    for index, label in enumerate(order):
        row = source_rows[label]
        focus = tuple(map(float, row.get("focus", [WIDTH / 2, HEIGHT / 2])))
        target_zoom = float(row.get("zoom", 1.0))
        view_frames = []
        for frame_index in range(hold_frames):
            progress = frame_index / max(hold_frames - 1, 1)
            lock_strength = 1.0
            zoom = target_zoom
            if index == 0:
                lock_strength = smoothstep(progress)
                zoom = 1.0 + (target_zoom - 1.0) * lock_strength
            elif index == len(order) - 1:
                lock_strength = 1.0 - smoothstep(progress)
                zoom = 1.0 + (target_zoom - 1.0) * lock_strength
            drift = (progress - 0.5) * 0.9
            frame = locked_view(images[label], focus, zoom, lock_strength, drift)
            view_frames.append(apply_finish(frame, vignette))
        rendered_views.append(view_frames)

    output_index = 0
    for index, view_frames in enumerate(rendered_views):
        for frame in view_frames:
            cv2.imwrite(str(orbit_frames / f"{output_index:05d}.png"), frame)
            output_index += 1
        if index == len(rendered_views) - 1:
            continue
        outgoing = view_frames[-1]
        incoming = rendered_views[index + 1][0]
        for transition_index in range(1, transition_frames + 1):
            linear = transition_index / (transition_frames + 1)
            blur_strength = math.sin(math.pi * linear) ** 0.7
            left = horizontal_blur(travel_warp(outgoing, -linear), blur_strength)
            right = horizontal_blur(travel_warp(incoming, 1.0 - linear), blur_strength)
            frame = left if linear < 0.5 else right
            frame = apply_finish(frame, vignette)
            cv2.imwrite(str(orbit_frames / f"{output_index:05d}.png"), frame)
            output_index += 1

    broadcast = args.clips / source_rows["Broadcast"]["file"]
    freeze_time = float(source_rows["Broadcast"]["freeze_time"])
    slow_start = 8.20
    pre_start = 6.20
    post_end = 12.70
    pre_lossless = args.work / "pre.mkv"
    orbit_lossless = args.work / "orbit.mkv"
    post_lossless = args.work / "post.mkv"

    run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(broadcast),
            "-filter_complex",
            (
                f"[0:v]trim=start={pre_start}:end={slow_start},setpts=PTS-STARTPTS[v0];"
                f"[0:v]trim=start={slow_start}:end={freeze_time},"
                "setpts=1.85*(PTS-STARTPTS)[v1];"
                "[v0][v1]concat=n=2:v=1:a=0,fps=30,format=yuv420p[v]"
            ),
            "-map", "[v]", "-an", "-c:v", "ffv1", str(pre_lossless),
        ]
    )
    run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-framerate", str(FPS),
            "-i", str(orbit_frames / "%05d.png"), "-an", "-c:v", "ffv1",
            "-pix_fmt", "yuv420p", str(orbit_lossless),
        ]
    )
    run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{freeze_time:.3f}",
            "-to", f"{post_end:.3f}", "-i", str(broadcast), "-vf", "fps=30,format=yuv420p",
            "-an", "-c:v", "ffv1", str(post_lossless),
        ]
    )
    run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-i", str(pre_lossless), "-i", str(orbit_lossless), "-i", str(post_lossless),
            "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
            "-profile:v", "high", "-level", "4.0", "-movflags", "+faststart", str(args.out),
        ]
    )

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration,size", "-of", "json", str(args.out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    qa = {
        "event": config["event"],
        "mode": "deterministic_level_c_action_locked_whip_orbit",
        "source_policy": "native official NBA frames only; no generative fill, AI enhancement, or synthetic action",
        "synchronization": config["synchronization"],
        "orbit_order": order,
        "hold_frames_per_view": hold_frames,
        "transition_frames_per_pair": transition_frames,
        "output_probe": json.loads(probe.stdout),
    }
    args.out.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
