from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_small_angle_real_pixel_prototype_v1 as v1

WIDTH = 960
HEIGHT = 540


def aligned_camera_rows(camera_json: Path, locked_dir: Path, ball_report: Path):
    cameras = json.loads(camera_json.read_text(encoding="utf-8"))
    manifest = json.loads((locked_dir / "locked_exact_state.json").read_text(encoding="utf-8"))
    ball = json.loads(ball_report.read_text(encoding="utf-8"))
    images = {v["label"]: v["image"] for v in manifest["views"]}
    ball_views = {v["label"]: v for v in ball["views"]}

    rows = []
    alignment_qa = {}
    for c in cameras["cameras"]:
        label = c["label"]
        if label not in images or label not in ball_views:
            continue
        img = cv2.imread(str(locked_dir / images[label]))
        if img is None:
            raise RuntimeError(f"Could not read locked image for {label}")

        H = np.asarray(ball_views[label]["camera_motion_homography_selected_to_anchor"], dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            raise RuntimeError(f"Invalid selected-to-anchor homography for {label}")
        aligned = cv2.warpPerspective(
            img,
            H,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        K = np.asarray(c["K"], dtype=np.float64)
        R = np.asarray(c["R_world_to_camera"], dtype=np.float64)
        t = np.asarray(c["t_world_to_camera_cm"], dtype=np.float64).reshape(3, 1)
        C = np.asarray(c["camera_center_world_cm"], dtype=np.float64)
        P = np.asarray(c["projection_matrix_KRt"], dtype=np.float64)

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        feat = np.dstack([lab, grad[:, :, None]]).astype(np.float32)

        qa = ball_views[label]["static_motion_compensation_qa"]
        if int(qa["inlier_features"]) < 30 or float(qa["median_static_residual_px"]) > 0.75 or float(qa["p95_static_residual_px"]) > 1.5:
            raise RuntimeError(f"Existing exact-state motion compensation gate no longer passes for {label}: {qa}")
        alignment_qa[label] = qa
        rows.append(dict(label=label, image=aligned, K=K, R=R, t=t, C=C, P=P, feat=feat))

    if len(rows) < 4:
        raise RuntimeError(f"Expected four exact-state aligned calibrated views, got {len(rows)}")
    return rows, alignment_qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--depth-steps", type=int, default=120)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows, alignment_qa = aligned_camera_rows(args.cameras, args.locked_images, args.ball_report)
    ball = v1.read_ball(args.ball_report)
    ref, points, colors, recon = v1.reconstruct(rows, ball, args.stride, args.depth_steps)
    if len(points) < 800:
        raise RuntimeError(f"Insufficient geometrically supported action surface after exact-state alignment: {len(points)} points")

    degrees = [0, 3, 5, 8]
    metrics = []
    for d in degrees:
        frame, cover = v1.render(points, colors, ref, ball, d, scale=4)
        cv2.imwrite(str(args.out / f"real_pixel_virtual_{d:02d}deg_uhd.png"), frame)
        cv2.imwrite(str(args.out / f"coverage_{d:02d}deg_uhd.png"), cover)
        metrics.append({
            "degrees": d,
            "covered_pixels": int((cover > 0).sum()),
            "coverage_fraction_of_4k_frame": round(float((cover > 0).mean()), 6),
        })

    qa = {
        "prototype": "small_angle_real_pixel_free_view_v2_exact_state_aligned",
        "source_resolution": [960, 540],
        "render_resolution": [3840, 2160],
        "resolution_policy": "native 960x540 source pixels; 4K diagnostic canvas only",
        "identity_policy": "no player identity labels required",
        "source_policy": "real synchronized official NBA pixels only; no diffusion, generative fill, crossfade, optical-flow morph, or hallucinated body completion",
        "exact_state_alignment": {
            "method": "warp locked selected frame into its accepted F28 metric-camera coordinate system using the already-validated static selected-to-anchor homography",
            "qa": alignment_qa,
        },
        "target_ball_world_cm": ball.tolist(),
        "degrees": degrees,
        "reconstruction": recon,
        "renders": metrics,
        "interpretation": "This v2 is a controlled correction of v1 camera/image-state mismatch. Judge body coherence before adding player masks or hole filling.",
    }
    (args.out / "small_angle_real_pixel_qa_v2.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
