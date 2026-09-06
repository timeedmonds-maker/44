from __future__ import annotations

"""Frame-C true-orbit v24: separate calibration masking from render masking.

Frame timing remains immutable: Right Slash chooser C, 8.275733 s, decoded
frame 248.  v24 does not change any geometric acceptance gate from v13 and does
not alter v23's SIFT candidate matcher.  It changes only which detected-person
pixels are excluded from CAMERA CALIBRATION.

Why: v22/v23 used the render-safety person mask directly as the SIFT mask.  On
this event Mask R-CNN merges large parts of the seated crowd into the dynamic
mask, removing static arena texture that is useful for camera calibration.
Those pixels are still unsafe for cross-camera rendering, so the original full
person mask remains authoritative for compositing.

Calibration policy:
- retain exclusion for large/tall/wide person components (players, officials,
  foreground spectators and merged foreground groups);
- permit only compact detected-person components back into the SIFT candidate
  pool, where robust PnP/RANSAC + reciprocal closure must still accept them;
- exclude a small region around detected basketball candidates;
- preserve all v13 forward/reverse PnP, reprojection, cheirality, depth-scale
  consistency and SE(3) closure thresholds unchanged.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

import build_frame_c_true_orbit_v22 as v22
import build_frame_c_true_orbit_v23 as v23
import build_portable_moge_pnp_freeview_v13 as reciprocal

H, W = 540, 960

_OUT: Path | None = None
_SAVED: set[str] = set()
_MASK_STATS: dict[str, dict] = {}


def output_dir_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--out":
            return Path(sys.argv[i + 1])
    return None


def calibration_exclusion(view: dict) -> tuple[np.ndarray, dict]:
    """Return a calibration-only exclusion mask.

    `view['dynamic']` remains untouched and is still used by v22 for rendering.
    We only re-admit compact detected-person components into the calibration
    candidate pool.  Any false/static assumption must survive the unchanged
    robust geometry gates; this function cannot make a camera pass by itself.
    """
    dyn = np.asarray(view["dynamic"], dtype=bool)
    u8 = dyn.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)

    # Start with the full conservative render exclusion, then re-admit only
    # compact components typical of seated/distant spectators.  Foreground NBA
    # players are tall/large and remain excluded even if their upper bodies sit
    # in the crowd region.
    exclusion = dyn.copy()
    readmitted_components = 0
    readmitted_pixels = 0
    protected_components = 0
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        compact = area <= 3200 and h <= 125 and w <= 175
        # Extra safeguard for compact detections very low in the image: these
        # are more likely players/officials than static seated spectators.
        low_foreground = (y + h) >= int(0.82 * H) and h >= 75
        if compact and not low_foreground:
            m = labels == i
            exclusion[m] = False
            readmitted_components += 1
            readmitted_pixels += int(m.sum())
        else:
            protected_components += 1

    # Basketball appearance must never become a calibration landmark.
    ball_excluded_pixels = 0
    yy, xx = np.ogrid[:H, :W]
    for b in view.get("balls", []):
        if float(b.get("score", 0.0)) < 0.10:
            continue
        cx, cy = float(b["cx"]), float(b["cy"])
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= 28.0 ** 2
        before = int(exclusion.sum())
        exclusion[disk] = True
        ball_excluded_pixels += int(exclusion.sum()) - before

    valid = np.asarray(view["valid"], dtype=bool)
    original_candidates = valid & (~dyn)
    new_candidates = valid & (~exclusion)
    stats_out = {
        "label": view.get("label"),
        "original_dynamic_pixels": int(dyn.sum()),
        "calibration_excluded_pixels": int(exclusion.sum()),
        "original_candidate_pixels": int(original_candidates.sum()),
        "calibration_candidate_pixels": int(new_candidates.sum()),
        "candidate_gain_pixels": int(new_candidates.sum() - original_candidates.sum()),
        "readmitted_components": int(readmitted_components),
        "readmitted_pixels": int(readmitted_pixels),
        "protected_components": int(protected_components),
        "ball_excluded_pixels": int(ball_excluded_pixels),
        "render_dynamic_mask_changed": False,
    }
    return exclusion, stats_out


def maybe_save_mask(view: dict, exclusion: np.ndarray, stats: dict) -> None:
    global _OUT
    if _OUT is None:
        return
    label = str(view.get("label", "unknown"))
    if label in _SAVED:
        return
    _OUT.mkdir(parents=True, exist_ok=True)
    candidate = ((~exclusion) & np.asarray(view["valid"], bool)).astype(np.uint8) * 255
    cv2.imwrite(str(_OUT / f"{label.replace(' ', '_')}_calibration_candidate_mask_v24.png"), candidate)
    _SAVED.add(label)
    _MASK_STATS[label] = stats


def solve_with_separate_calibration_mask(ref: dict, tgt: dict) -> dict:
    ref_ex, ref_stats = calibration_exclusion(ref)
    tgt_ex, tgt_stats = calibration_exclusion(tgt)
    maybe_save_mask(ref, ref_ex, ref_stats)
    maybe_save_mask(tgt, tgt_ex, tgt_stats)

    # Shallow copies keep all MoGe/appearance data identical.  Only the
    # `dynamic` field seen by the calibration solver is replaced.  v22's real
    # view dictionaries retain their original aggressive render masks.
    r = dict(ref)
    t = dict(tgt)
    r["dynamic"] = ref_ex
    t["dynamic"] = tgt_ex

    s = reciprocal.solve_target_from_reference_reciprocal(r, t)
    s["calibration_mask_policy"] = "compact spectator components may calibrate; render dynamic mask unchanged"
    s["reference_calibration_mask_stats"] = ref_stats
    s["target_calibration_mask_stats"] = tgt_stats
    s["geometry_acceptance_gate_change"] = "none"
    return s


def main() -> None:
    global _OUT
    _OUT = output_dir_from_argv()

    # v23 changed only candidate generation.  Keep it exactly, and route the
    # reciprocal solve through the calibration-only masks above.
    reciprocal.base.sift_matches = v23.calibration_sift_matches
    v22.solve_target_from_reference_reciprocal = solve_with_separate_calibration_mask

    v22.main()

    if _OUT is None:
        return
    p = _OUT / "frame_c_true_orbit_qa_v22.json"
    if not p.exists():
        return
    q = json.loads(p.read_text())
    q["prototype"] = "frame_c_separate_calibration_mask_true_orbit_v24"
    q["calibration_candidate_matcher"] = {
        "method": "v23 higher-recall static SIFT candidates",
        "nfeatures": 10000,
        "contrast_threshold": 0.015,
        "lowe_ratio": 0.75,
        "acceptance_gate_change": "none",
    }
    q["calibration_mask_policy"] = {
        "purpose": "recover static spectator/arena texture removed by render-safety person masking",
        "readmit_rule": "connected person-mask component area<=3200 px, height<=125 px, width<=175 px, excluding low foreground components",
        "ball_exclusion_radius_px": 28,
        "render_dynamic_mask_changed": False,
        "secondary_render_person_pixels_allowed": False,
        "geometry_gates": "unchanged v13 forward/reverse PnP, reprojection, cheirality, depth-scale consistency and SE(3) closure",
    }
    q["calibration_mask_stats"] = _MASK_STATS
    p.write_text(json.dumps(q, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
