from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

WIDTH = 960
HEIGHT = 540
FPS = 30
TARGET = (520, 310)
IMPACT_FRAME = 28


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    return frames


def normalize(image: np.ndarray, focus: tuple[float, float], scale: float) -> np.ndarray:
    fx, fy = focus
    matrix = np.array(
        [[scale, 0.0, TARGET[0] - scale * fx], [0.0, scale, TARGET[1] - scale * fy]],
        np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--orbit", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    left_slash = read_video(args.windows / "07_Left_Slash_contact.mp4")
    right_slash = read_video(args.windows / "08_Right_Slash_contact.mp4")
    orbit = read_video(args.orbit)
    if len(left_slash) <= IMPACT_FRAME or len(right_slash) <= IMPACT_FRAME:
        raise RuntimeError("Contact windows do not contain the corrected impact frame")

    rendered: list[np.ndarray] = []

    # Native real motion into the event, normalized to the same framing as the first frozen view.
    for index, frame in enumerate(left_slash[: IMPACT_FRAME + 1]):
        normalized = normalize(frame, (500, 350), 0.94)
        rendered.append(normalized)
        # Deterministic slow-in during the final ~0.23 s. No interpolated basketball frames.
        if 21 <= index < IMPACT_FRAME:
            rendered.append(normalized.copy())

    # Frozen multi-view travel.
    rendered.extend(orbit)

    # Resume real motion immediately after the same synchronized instant on the exit view.
    for frame in right_slash[IMPACT_FRAME + 1 :]:
        rendered.append(normalize(frame, (470, 320), 0.96))

    args.work.mkdir(parents=True, exist_ok=True)
    frames_dir = args.work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(rendered):
        cv2.imwrite(str(frames_dir / f"{index:05d}.png"), frame)

    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-framerate", str(FPS), "-i", str(frames_dir / "%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "14",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
        ],
        check=True,
    )

    print(
        f"REFINED_FULL_REPLAY frames={len(rendered)} duration={len(rendered) / FPS:.3f}s ",
        f"impact_frame={IMPACT_FRAME}",
        flush=True,
    )


if __name__ == "__main__":
    main()
