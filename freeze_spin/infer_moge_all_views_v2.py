from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

import build_small_angle_real_pixel_prototype_v2 as v2

WIDTH = 960
HEIGHT = 540


def safe(label: str) -> str:
    return label.replace(" ", "_")


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

    rows, alignment_qa = v2.aligned_camera_rows(args.cameras, args.locked_images, args.ball_report)
    ball = json.loads(args.ball_report.read_text(encoding="utf-8"))
    ball_views = {v["label"]: v for v in ball["views"]}
    Xball = np.asarray(ball["ball_center_world_cm"], dtype=np.float64)

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()

    report = {
        "model": args.pretrained,
        "device": str(device),
        "num_tokens": args.num_tokens,
        "source_resolution": [WIDTH, HEIGHT],
        "policy": "MoGe supplies geometry only. Every output appearance pixel remains official NBA source imagery.",
        "exact_state_alignment_qa": alignment_qa,
        "views": {},
    }

    for row in rows:
        label = row["label"]
        if label not in ball_views:
            raise KeyError(label)
        fx = float(row["K"][0, 0])
        fov_x = math.degrees(2.0 * math.atan(WIDTH / (2.0 * fx)))
        rgb = cv2.cvtColor(row["image"], cv2.COLOR_BGR2RGB)
        image = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.inference_mode():
            output = model.infer(image, num_tokens=args.num_tokens, fov_x=fov_x, use_fp16=False, apply_mask=False)
        depth_m = output["depth"].detach().cpu().numpy().astype(np.float32)
        valid = output["mask"].detach().cpu().numpy().astype(bool) if "mask" in output else np.isfinite(depth_m)
        np.save(args.out / f"{safe(label)}_depth_m.npy", depth_m)
        np.save(args.out / f"{safe(label)}_valid.npy", valid)

        finite = np.isfinite(depth_m) & valid & (depth_m > 0)
        vis = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        if np.any(finite):
            lo, hi = np.percentile(depth_m[finite], [2, 98])
            vis[finite] = np.clip((depth_m[finite] - lo) * 255.0 / max(float(hi - lo), 1e-6), 0, 255).astype(np.uint8)
        cv2.imwrite(str(args.out / f"{safe(label)}_depth_visual.png"), vis)
        cv2.imwrite(str(args.out / f"{safe(label)}_aligned_input.png"), row["image"])

        z_ball = float((row["R"] @ Xball.reshape(3, 1) + row["t"])[2, 0])
        ball_px = np.asarray(ball_views[label]["observed_ball_anchor_px"], dtype=np.float64)
        bx = int(np.clip(round(float(ball_px[0])), 0, WIDTH - 1))
        by = int(np.clip(round(float(ball_px[1])), 0, HEIGHT - 1))
        md = float(depth_m[by, bx])
        scale = z_ball / (100.0 * md) if np.isfinite(md) and md > 0 else None
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"No valid metric MoGe scale for {label}")
        report["views"][label] = {
            "known_fov_x_deg": round(float(fov_x), 6),
            "calibrated_fx_px": round(float(fx), 6),
            "valid_fraction": round(float(finite.mean()), 6),
            "metric_ball_camera_z_cm": round(float(z_ball), 6),
            "ball_anchor_pixel": [round(float(x), 3) for x in ball_px],
            "moge_ball_depth_m": round(float(md), 6),
            "metric_scale_multiplier_from_ball": round(float(scale), 6),
            "depth_file": f"{safe(label)}_depth_m.npy",
            "valid_file": f"{safe(label)}_valid.npy",
        }

    (args.out / "moge_all_views_qa_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
