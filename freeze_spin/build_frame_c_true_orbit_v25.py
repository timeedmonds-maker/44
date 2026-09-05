from __future__ import annotations

"""Frame-C true-orbit v25: instance-level calibration masking.

Timing is immutable: Right Slash chooser C at 8.275733 s, decoded frame 248.
No geometry acceptance threshold is changed from v13 and no secondary dynamic
pixels are admitted to rendering.

v24 showed the right conceptual split (calibration mask != render mask), but the
post-dilation union person mask can merge crowd and foreground players into huge
connected components.  v25 therefore preserves Mask R-CNN PERSON INSTANCES at
detection time.  Only large/tall/low foreground person instances are excluded
from SIFT camera calibration.  Compact seated/distant spectator instances may
supply static arena landmarks, where they still have to survive v23 descriptor
filtering, forward PnP/RANSAC, reverse PnP, reprojection, cheirality, depth-scale
consistency and SE(3) closure.

The render mask is exactly the same conservative union of every detected person
(score>=0.35, mask>=0.38) followed by the same 17x17 dilation used previously.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

import build_frame_c_true_orbit_v22 as v22
import build_frame_c_true_orbit_v23 as v23
import build_portable_moge_pnp_freeview_v13 as reciprocal

H, W = 540, 960
PERSON_CLASS = 1
BALL_CLASS = 37

_OUT: Path | None = None
_CAL_EXCLUSION: dict[int, np.ndarray] = {}
_INSTANCE_STATS: dict[int, dict] = {}
_MASK_STATS_BY_LABEL: dict[str, dict] = {}
_SAVED: set[str] = set()


def output_dir_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--out":
            return Path(sys.argv[i + 1])
    return None


def detect_dynamic_ball_and_calibration_instances(model, image: np.ndarray):
    """Return the unchanged render mask while retaining calibration instances."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        p = model([to_tensor(rgb)])[0]

    render_dynamic = np.zeros((H, W), np.uint8)
    calibration_foreground = np.zeros((H, W), np.uint8)
    balls = []
    person_rows = []

    scores = p["scores"].cpu().numpy()
    labels = p["labels"].cpu().numpy()
    boxes = p["boxes"].cpu().numpy()
    masks = p["masks"].cpu().numpy()[:, 0]

    for sc, lab, box, prob in zip(scores, labels, boxes, masks):
        sc = float(sc)
        lab = int(lab)
        x1, y1, x2, y2 = [float(v) for v in box]
        if lab == PERSON_CLASS and sc >= 0.35:
            raw = prob >= 0.38
            render_dynamic[raw] = 255
            bh = max(0.0, y2 - y1)
            bw = max(0.0, x2 - x1)
            area = int(raw.sum())

            # Calibration-only foreground classification.  NBA players and
            # officials are tall/large and/or extend into the lower court.
            # Compact seated spectators are intentionally permitted as camera
            # landmarks but remain forbidden for rendering.
            foreground = (
                bh >= 118.0
                or area >= 2200
                or (y2 >= 0.78 * H and bh >= 68.0)
                or (bw >= 105.0 and bh >= 92.0)
            )
            if foreground:
                calibration_foreground[raw] = 255
            person_rows.append({
                "score": sc,
                "box": [x1, y1, x2, y2],
                "box_width": bw,
                "box_height": bh,
                "mask_area": area,
                "calibration_foreground_excluded": bool(foreground),
            })
        elif lab == BALL_CLASS and sc >= 0.10:
            balls.append({
                "score": sc,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "box": [x1, y1, x2, y2],
            })

    # EXACT previous render safety policy.
    render_dynamic = cv2.dilate(
        render_dynamic,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    )

    # Calibration foreground gets a slightly wider safety halo because it is
    # protecting player silhouettes, not trying to maximize render coverage.
    calibration_foreground = cv2.dilate(
        calibration_foreground,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        iterations=1,
    )

    balls.sort(key=lambda r: r["score"], reverse=True)
    key = id(image)
    _CAL_EXCLUSION[key] = calibration_foreground > 0
    _INSTANCE_STATS[key] = {
        "detected_person_instances": len(person_rows),
        "calibration_foreground_instances": int(sum(r["calibration_foreground_excluded"] for r in person_rows)),
        "spectator_instances_readmitted_for_calibration": int(sum(not r["calibration_foreground_excluded"] for r in person_rows)),
        "render_dynamic_pixels": int((render_dynamic > 0).sum()),
        "calibration_foreground_pixels_before_ball_exclusion": int((calibration_foreground > 0).sum()),
        "person_instances": person_rows,
    }
    return render_dynamic > 0, balls


def calibration_exclusion(view: dict) -> tuple[np.ndarray, dict]:
    key = id(view["image"])
    # If instance metadata is unavailable for any reason, fail conservatively
    # back to the full render mask rather than accidentally admitting players.
    exclusion = _CAL_EXCLUSION.get(key)
    fallback = exclusion is None
    if fallback:
        exclusion = np.asarray(view["dynamic"], dtype=bool).copy()
    else:
        exclusion = exclusion.copy()

    yy, xx = np.ogrid[:H, :W]
    ball_added = 0
    for b in view.get("balls", []):
        if float(b.get("score", 0.0)) < 0.10:
            continue
        cx, cy = float(b["cx"]), float(b["cy"])
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= 30.0 ** 2
        before = int(exclusion.sum())
        exclusion[disk] = True
        ball_added += int(exclusion.sum()) - before

    valid = np.asarray(view["valid"], dtype=bool)
    render_dyn = np.asarray(view["dynamic"], dtype=bool)
    stats = dict(_INSTANCE_STATS.get(key, {}))
    stats.update({
        "label": view.get("label"),
        "fallback_to_full_render_mask": bool(fallback),
        "render_dynamic_pixels": int(render_dyn.sum()),
        "calibration_excluded_pixels": int(exclusion.sum()),
        "render_static_candidate_pixels": int((valid & ~render_dyn).sum()),
        "calibration_candidate_pixels": int((valid & ~exclusion).sum()),
        "candidate_gain_pixels": int((valid & ~exclusion).sum() - (valid & ~render_dyn).sum()),
        "ball_exclusion_added_pixels": int(ball_added),
        "render_dynamic_mask_changed": False,
    })
    return exclusion, stats


def maybe_save(view: dict, exclusion: np.ndarray, stats: dict) -> None:
    if _OUT is None:
        return
    label = str(view.get("label", "unknown"))
    if label in _SAVED:
        return
    _OUT.mkdir(parents=True, exist_ok=True)
    valid = np.asarray(view["valid"], bool)
    candidate = ((~exclusion) & valid).astype(np.uint8) * 255
    cv2.imwrite(str(_OUT / f"{label.replace(' ', '_')}_calibration_candidate_mask_v25.png"), candidate)
    # A separate image explicitly shows what is protected from calibration.
    cv2.imwrite(str(_OUT / f"{label.replace(' ', '_')}_calibration_foreground_exclusion_v25.png"), exclusion.astype(np.uint8) * 255)
    _MASK_STATS_BY_LABEL[label] = stats
    _SAVED.add(label)


def solve_with_instance_calibration_mask(ref: dict, tgt: dict) -> dict:
    ref_ex, ref_stats = calibration_exclusion(ref)
    tgt_ex, tgt_stats = calibration_exclusion(tgt)
    maybe_save(ref, ref_ex, ref_stats)
    maybe_save(tgt, tgt_ex, tgt_stats)

    r = dict(ref)
    t = dict(tgt)
    r["dynamic"] = ref_ex
    t["dynamic"] = tgt_ex
    s = reciprocal.solve_target_from_reference_reciprocal(r, t)
    s["calibration_mask_policy"] = "instance-level foreground exclusion; compact spectator instances may calibrate"
    s["reference_calibration_mask_stats"] = ref_stats
    s["target_calibration_mask_stats"] = tgt_stats
    s["geometry_acceptance_gate_change"] = "none"
    return s


def main() -> None:
    global _OUT
    _OUT = output_dir_from_argv()

    # Capture per-person instance masks while returning the exact same render
    # union mask interface expected by v22.
    v22.detect_dynamic_and_ball = detect_dynamic_ball_and_calibration_instances
    reciprocal.base.sift_matches = v23.calibration_sift_matches
    v22.solve_target_from_reference_reciprocal = solve_with_instance_calibration_mask

    v22.main()

    if _OUT is None:
        return
    p = _OUT / "frame_c_true_orbit_qa_v22.json"
    if not p.exists():
        return
    q = json.loads(p.read_text())
    q["prototype"] = "frame_c_instance_calibration_mask_true_orbit_v25"
    q["calibration_candidate_matcher"] = {
        "method": "v23 higher-recall SIFT candidates + one-to-one target assignment",
        "nfeatures": 10000,
        "contrast_threshold": 0.015,
        "lowe_ratio": 0.75,
        "acceptance_gate_change": "none",
    }
    q["calibration_mask_policy"] = {
        "method": "Mask R-CNN instance-level foreground/spectator separation",
        "foreground_rule": "height>=118 OR mask area>=2200 OR low-court(height>=68) OR wide-near foreground",
        "foreground_dilation_px": 21,
        "ball_exclusion_radius_px": 30,
        "render_person_mask": "unchanged all-person union + 17x17 dilation",
        "render_dynamic_mask_changed": False,
        "secondary_render_person_pixels_allowed": False,
        "geometry_acceptance_gate_change": "none",
        "geometry_gates": "unchanged v13 forward/reverse PnP, reprojection, cheirality, depth-scale consistency and SE(3) closure",
    }
    q["calibration_mask_stats"] = _MASK_STATS_BY_LABEL
    p.write_text(json.dumps(q, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
