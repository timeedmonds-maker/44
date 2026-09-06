from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

WIDTH = 960
HEIGHT = 540
FPS = 30
REFERENCE_FRAME = 28
TARGET = (520, 310)

# Horizontal sweep only. Above-rim and extreme handheld views are deliberately excluded
# until camera calibration can place them correctly in 3D.
VIEWS = [
    dict(label="Left Slash", file="07_Left_Slash_contact.mp4", focus=(500, 350), scale=0.94, roi=(160, 175, 850, 540)),
    dict(label="In Arena", file="05_In_Arena_contact.mp4", focus=(455, 330), scale=1.00, roi=(140, 165, 720, 540)),
    dict(label="Mobile Broadcast", file="03_Mobile_Broadcast_contact.mp4", focus=(585, 310), scale=1.00, roi=(260, 130, 900, 540)),
    dict(label="Broadcast", file="01_Broadcast_contact.mp4", focus=(565, 285), scale=1.30, roi=(360, 120, 740, 430)),
    dict(label="Right Slash", file="08_Right_Slash_contact.mp4", focus=(470, 320), scale=0.96, roi=(140, 130, 850, 540)),
]


def read_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    if len(frames) <= REFERENCE_FRAME:
        raise RuntimeError(f"Contact window too short: {path} frames={len(frames)}")
    return frames


def stabilize_background(frames: list[np.ndarray], roi: tuple[int, int, int, int]):
    reference = frames[REFERENCE_FRAME]
    h, w = reference.shape[:2]
    feature_mask = np.full((h, w), 255, np.uint8)
    x1, y1, x2, y2 = roi
    cv2.rectangle(feature_mask, (x1, y1), (x2, y2), 0, -1)

    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.025)
    kp_ref, des_ref = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), feature_mask)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    aligned: list[np.ndarray] = []
    diagnostics = []
    for index, frame in enumerate(frames):
        # A symmetric subset is enough for a robust plate and keeps Actions runtime low.
        if index != REFERENCE_FRAME and (abs(index - REFERENCE_FRAME) > 18 or index % 2):
            continue
        if index == REFERENCE_FRAME:
            aligned.append(frame)
            diagnostics.append({"frame": index, "good": 999, "inliers": 999, "ratio": 1.0})
            continue

        kp, des = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), feature_mask)
        good = []
        if des is not None and des_ref is not None:
            for first, second in matcher.knnMatch(des, des_ref, k=2):
                if first.distance < 0.72 * second.distance:
                    good.append(first)

        homography = None
        inliers = None
        if len(good) >= 8:
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)

        if homography is None or inliers is None or int(inliers.sum()) < 6:
            diagnostics.append({"frame": index, "good": len(good), "inliers": 0, "ratio": 0.0})
            continue

        warped = cv2.warpPerspective(
            frame,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aligned.append(warped)
        diagnostics.append(
            {
                "frame": index,
                "good": len(good),
                "inliers": int(inliers.sum()),
                "ratio": float(inliers.mean()),
            }
        )

    if len(aligned) < 6:
        raise RuntimeError(f"Insufficient stabilized frames: {len(aligned)}")
    plate = np.median(np.stack(aligned), axis=0).astype(np.uint8)
    return reference, plate, diagnostics


def motion_mask(reference: np.ndarray, plate: np.ndarray, roi: tuple[int, int, int, int]):
    difference = cv2.absdiff(reference, plate)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    threshold = max(22, min(38, int(np.percentile(gray.reshape(-1), 84))))
    mask = (gray > threshold).astype(np.uint8) * 255

    roi_mask = np.zeros_like(mask)
    x1, y1, x2, y2 = roi
    cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.bitwise_and(mask, roi_mask)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    kept = np.zeros_like(mask)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= 180:
            kept[labels == component] = 255

    kept = cv2.dilate(
        kept,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    soft = cv2.GaussianBlur(kept, (0, 0), 2.2).astype(np.float32) / 255.0
    return soft, threshold


def normalize(image: np.ndarray, focus: tuple[float, float], scale: float, mask: bool = False):
    fx, fy = focus
    matrix = np.array(
        [[scale, 0.0, TARGET[0] - scale * fx], [0.0, scale, TARGET[1] - scale * fy]],
        np.float32,
    )
    if mask:
        return cv2.warpAffine(
            image,
            matrix,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def rigid(image: np.ndarray, dx: float, dy: float, scale: float = 1.0, shear: float = 0.0, mask: bool = False):
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    matrix = np.array(
        [[scale, shear, dx + cx - scale * cx - shear * cy], [0.0, scale, dy + cy - scale * cy]],
        np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR if mask else cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT if mask else cv2.BORDER_REFLECT_101,
        borderValue=0,
    )


def render_view(view: dict, travel: float) -> np.ndarray:
    # No local mesh deformation: each scene layer is rigid.
    background = rigid(
        view["plate"],
        -28.0 * travel,
        -1.5 * abs(travel),
        1.0 + 0.008 * abs(travel),
        -0.010 * travel,
    )
    foreground = rigid(
        view["reference"],
        -52.0 * travel,
        -3.0 * abs(travel),
        1.0 + 0.013 * abs(travel),
        -0.014 * travel,
    )
    mask = rigid(
        view["mask"],
        -52.0 * travel,
        -3.0 * abs(travel),
        1.0 + 0.013 * abs(travel),
        -0.014 * travel,
        mask=True,
    )
    output = background.astype(np.float32) * (1.0 - mask[..., None])
    output += foreground.astype(np.float32) * mask[..., None]
    return np.clip(output, 0, 255).astype(np.uint8)


def shutter_streak(image: np.ndarray, amount: float) -> np.ndarray:
    if amount < 0.01:
        return image
    accumulator = np.zeros_like(image, np.float32)
    total = 0.0
    for offset, weight in [(-2, 0.15), (-1, 0.22), (0, 0.26), (1, 0.22), (2, 0.15)]:
        accumulator += rigid(image, offset * 9.0 * amount, 0.0).astype(np.float32) * weight
        total += weight
    return np.clip(accumulator / total, 0, 255).astype(np.uint8)


def finish(image: np.ndarray) -> np.ndarray:
    frame = image.astype(np.float32)
    frame = (frame - 128.0) * 1.02 + 128.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    frame_dir = args.work / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    prepared = []
    qa_views = []
    for specification in VIEWS:
        frames = read_video(args.windows / specification["file"])
        reference, plate, diagnostics = stabilize_background(frames, specification["roi"])
        mask, threshold = motion_mask(reference, plate, specification["roi"])

        # Background-only cleanup. It never changes visible player/ball source pixels.
        binary_mask = (mask > 0.55).astype(np.uint8) * 255
        cleaned_plate = cv2.inpaint(plate, binary_mask, 5, cv2.INPAINT_TELEA)

        view = {
            **specification,
            "reference": normalize(reference, specification["focus"], specification["scale"]),
            "plate": normalize(cleaned_plate, specification["focus"], specification["scale"]),
            "mask": normalize(mask, specification["focus"], specification["scale"], mask=True),
        }
        prepared.append(view)
        qa_views.append(
            {
                "label": specification["label"],
                "threshold": threshold,
                "video_frames": len(frames),
                "median_stabilization_inliers": float(np.median([row["inliers"] for row in diagnostics])),
            }
        )

    rendered = []
    for _ in range(5):
        rendered.append(finish(render_view(prepared[0], 0.0)))

    for index in range(len(prepared) - 1):
        outgoing = prepared[index]
        incoming = prepared[index + 1]
        transition_frames = 10
        for frame_index in range(transition_frames):
            progress = frame_index / (transition_frames - 1)
            eased = progress * progress * (3.0 - 2.0 * progress)
            if eased < 0.5:
                travel = eased / 0.5
                frame = render_view(outgoing, travel)
            else:
                travel = (eased - 0.5) / 0.5
                frame = render_view(incoming, -(1.0 - travel))
            streak = math.exp(-((eased - 0.5) / 0.14) ** 2)
            rendered.append(finish(shutter_streak(frame, streak)))
        for _ in range(2):
            rendered.append(finish(render_view(incoming, 0.0)))

    for _ in range(7):
        rendered.append(finish(render_view(prepared[-1], 0.0)))

    for index, frame in enumerate(rendered):
        cv2.imwrite(str(frame_dir / f"{index:05d}.png"), frame)

    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-framerate", str(FPS), "-i", str(frame_dir / "%05d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "14",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out),
        ],
        check=True,
    )

    qa = {
        "prototype": "refined_v3_temporal_background_layered_orbit",
        "impact_frame_in_contact_window": REFERENCE_FRAME,
        "view_order": [view["label"] for view in prepared],
        "view_qa": qa_views,
        "source_policy": "native official NBA frames; visible player/ball pixels come only from the synchronized impact frame",
        "background_method": "same-camera SIFT/RANSAC stabilization + temporal median + deterministic background-only inpaint",
        "prohibited": [
            "piecewise mesh warp",
            "optical-flow morph",
            "generative model",
            "AI enhancement",
            "synthetic player/ball frames",
        ],
        "render_transforms": [
            "rigid affine background motion",
            "rigid foreground layer parallax",
            "deterministic shutter streak at real-camera handoff",
        ],
        "frames": len(rendered),
        "fps": FPS,
    }
    args.out.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
