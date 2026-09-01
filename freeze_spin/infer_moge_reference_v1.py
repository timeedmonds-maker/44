from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--locked-images", type=Path, required=True)
    ap.add_argument("--ball-report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pretrained", default="Ruicheng/moge-2-vits-normal")
    ap.add_argument("--num-tokens", type=int, default=1400)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cams = json.loads(args.cameras.read_text(encoding="utf-8"))
    cam = next(c for c in cams["cameras"] if c["label"] == "In Arena")
    K = np.asarray(cam["K"], dtype=np.float64)
    fx = float(K[0, 0])
    width, height = 960, 540
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx)))

    ball = json.loads(args.ball_report.read_text(encoding="utf-8"))
    bv = next(v for v in ball["views"] if v["label"] == "In Arena")
    H = np.asarray(bv["camera_motion_homography_selected_to_anchor"], dtype=np.float64)

    selected = cv2.imread(str(args.locked_images / "In_Arena_F26.png"))
    if selected is None:
        raise FileNotFoundError(args.locked_images / "In_Arena_F26.png")
    aligned = cv2.warpPerspective(selected, H, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cv2.imwrite(str(args.out / "moge_input_in_arena_anchor.png"), aligned)

    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.inference_mode():
        output = model.infer(image, num_tokens=args.num_tokens, fov_x=fov_x, use_fp16=False, apply_mask=False)

    depth_m = output["depth"].detach().cpu().numpy().astype(np.float32)
    points_m = output["points"].detach().cpu().numpy().astype(np.float32)
    valid = output["mask"].detach().cpu().numpy().astype(bool) if "mask" in output else np.isfinite(depth_m)
    normal = output.get("normal")
    normal_np = normal.detach().cpu().numpy().astype(np.float32) if normal is not None else None
    intr = output["intrinsics"].detach().cpu().numpy().astype(np.float32)

    np.save(args.out / "moge_depth_m.npy", depth_m)
    np.save(args.out / "moge_points_m.npy", points_m)
    np.save(args.out / "moge_valid.npy", valid)
    if normal_np is not None:
        np.save(args.out / "moge_normal.npy", normal_np)

    finite = np.isfinite(depth_m) & valid & (depth_m > 0)
    vis = np.zeros((height, width), dtype=np.uint8)
    if np.any(finite):
        lo, hi = np.percentile(depth_m[finite], [2, 98])
        vis[finite] = np.clip((depth_m[finite] - lo) * 255.0 / max(float(hi - lo), 1e-6), 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.out / "moge_depth_visual.png"), vis)
    if normal_np is not None:
        nvis = np.clip((normal_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.out / "moge_normal_visual.png"), cv2.cvtColor(nvis, cv2.COLOR_RGB2BGR))

    # Exact ball depth in the accepted metric camera, used only as a scale QA / registration anchor.
    R = np.asarray(cam["R_world_to_camera"], dtype=np.float64)
    t = np.asarray(cam["t_world_to_camera_cm"], dtype=np.float64).reshape(3, 1)
    Xball = np.asarray(ball["ball_center_world_cm"], dtype=np.float64)
    ball_z_cm = float((R @ Xball.reshape(3, 1) + t)[2, 0])
    ball_anchor_px = np.asarray(bv["observed_ball_anchor_px"], dtype=np.float64)
    bx = int(np.clip(round(float(ball_anchor_px[0])), 0, width - 1))
    by = int(np.clip(round(float(ball_anchor_px[1])), 0, height - 1))
    moge_ball_m = float(depth_m[by, bx])
    scale_at_ball = ball_z_cm / (100.0 * moge_ball_m) if np.isfinite(moge_ball_m) and moge_ball_m > 0 else None

    qa = {
        "model": args.pretrained,
        "device": str(device),
        "num_tokens": args.num_tokens,
        "source_resolution": [width, height],
        "known_fov_x_deg": round(float(fov_x), 6),
        "calibrated_fx_px": round(fx, 6),
        "moge_intrinsics_normalized": intr.tolist(),
        "valid_fraction": round(float(finite.mean()), 6),
        "metric_ball_camera_z_cm": round(ball_z_cm, 6),
        "ball_anchor_pixel": [round(float(x), 3) for x in ball_anchor_px],
        "moge_ball_depth_m": round(moge_ball_m, 6) if np.isfinite(moge_ball_m) else None,
        "metric_scale_multiplier_from_ball": round(float(scale_at_ball), 6) if scale_at_ball is not None else None,
        "policy": "MoGe supplies geometry only. It never supplies output appearance pixels.",
    }
    (args.out / "moge_reference_qa_v1.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=2), flush=True)


if __name__ == "__main__":
    main()
