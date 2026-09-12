from __future__ import annotations

"""Evaluate an exact-freeze held-out camera render.

Numeric metrics are diagnostic only. Visual inspection remains authoritative for
identity, anatomy, blur, merged bodies and missing limbs.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    r = cv2.imread(str(args.render), cv2.IMREAD_COLOR)
    g = cv2.imread(str(args.gt), cv2.IMREAD_COLOR)
    if r is None or g is None or r.shape != g.shape:
        raise RuntimeError(f"render/gt mismatch: {None if r is None else r.shape} vs {None if g is None else g.shape}")

    rf = r.astype(np.float32) / 255.0
    gf = g.astype(np.float32) / 255.0
    mse = float(np.mean((rf - gf) ** 2))
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    ssim = float(structural_similarity(g, r, channel_axis=2, data_range=255))

    # Sharpness is reported separately. A plausible synthesis should not collapse
    # the held-out action into a smooth/blurred approximation.
    gray_r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    gray_g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    sharp_r = float(cv2.Laplacian(gray_r, cv2.CV_64F).var())
    sharp_g = float(cv2.Laplacian(gray_g, cv2.CV_64F).var())
    sharp_ratio = float(sharp_r / max(sharp_g, 1e-9))

    diff = cv2.absdiff(r, g)
    heat = cv2.applyColorMap(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_TURBO)
    panel = np.concatenate([g, r, heat], axis=1)
    cv2.putText(panel, "GROUND TRUTH", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "HELD-OUT RENDER", (g.shape[1]+10, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "ABS DIFF", (2*g.shape[1]+10, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "v32d_holdout_freeze_compare.png"), panel)

    qa = {
        "shape_hw": [int(r.shape[0]), int(r.shape[1])],
        "psnr_db_whole_crop": psnr,
        "ssim_whole_crop": ssim,
        "render_laplacian_variance": sharp_r,
        "ground_truth_laplacian_variance": sharp_g,
        "sharpness_ratio_render_to_gt": sharp_ratio,
        "numeric_screen_only": True,
        "visual_pass_required": True,
        "visual_rejection_conditions": [
            "Adams/action identity not immediately recognizable",
            "missing or doubled limb",
            "merged player bodies",
            "material motion blur at frozen action",
            "ball or rim geometry instability",
        ],
    }
    (args.out / "v32d_holdout_freeze_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
