from __future__ import annotations

"""V32 Stage-A: sharp latent freeze + all-on-court scene package.

This is a deliberate architecture break from the v12-v31 anatomical-mesh path.
It does NOT render a novel view.  It prepares the source-grounded, synchronized,
metric scene package required by a learned sparse-view Gaussian backend.

Key guarantees:
- native 960x540 source frames only; no upscale/UHD;
- freeze timing remains tied to the accepted v11 visual state;
- sharpness is evaluated jointly across the three solved cameras over a small
  synchronized temporal window rather than trusting one raw frame;
- a 13-frame synchronized burst is exported for each camera so the learned model
  can recover sharp appearance/occluded surfaces while geometry stays locked to
  the freeze instant;
- all detected on-court people are inventoried and source masks are exported;
- camera intrinsics/extrinsics and existing three-view ball solution are carried
  forward in one machine-readable scene manifest.

Final novel-view synthesis is intentionally deferred to the GPU Gaussian stage.
"""

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v5 as v5

W, H = 960, 540
CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
CENTERS_V11 = {
    "Left Above Rim": 261,
    "Broadcast": 277,
    "Right Above Rim": 257,
}
# User-validated focal region seeds from the exact-state work.  For sharpness QA
# only; these do not define 3-D geometry or final segmentation.
FOCAL_SEEDS = {
    "Left Above Rim": [438, 160, 512, 326],
    "Broadcast": [472, 100, 570, 312],
    "Right Above Rim": [514, 184, 662, 340],
}


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def find_clip(clips_dir: Path, label: str) -> Path:
    token = safe_label(label)
    rows = sorted(clips_dir.glob(f"*_{token}_SOURCE.mp4"))
    # Guard against Broadcast accidentally matching Other/Mobile Broadcast.
    if label == "Broadcast":
        rows = [p for p in rows if "Other_Broadcast" not in p.name and "Mobile_Broadcast" not in p.name]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one native clip for {label}; got {rows}")
    return rows[0]


def decode_indices(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    need = sorted(set(int(x) for x in indices))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: dict[int, np.ndarray] = {}
    for idx in need:
        if idx < 0 or idx >= n:
            raise RuntimeError(f"frame {idx} outside {path.name} frame_count={n}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"decode failed {path.name} frame={idx}")
        if frame.shape[:2] != (H, W):
            raise RuntimeError(f"source resolution changed for {path.name}: {frame.shape[:2]}")
        out[idx] = frame
    cap.release()
    return out


def expand_box(box, scale=1.65):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    bw *= scale; bh *= scale
    xa = max(0, int(math.floor(cx - bw / 2)))
    xb = min(W - 1, int(math.ceil(cx + bw / 2)))
    ya = max(0, int(math.floor(cy - bh / 2)))
    yb = min(H - 1, int(math.ceil(cy + bh / 2)))
    return [xa, ya, xb, yb]


def sharpness_metrics(image: np.ndarray, box=None):
    if box is None:
        roi = image
    else:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2 + 1, x1:x2 + 1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return {
        "laplacian_variance": float(lap.var()),
        "tenengrad_mean": float(np.mean(gx * gx + gy * gy)),
        "strong_edge_fraction": float(np.mean(grad > 40.0)),
    }


def _rel(v, vmax):
    return float(v / max(vmax, 1e-9))


def choose_freeze(decoded, candidate_offsets):
    rows = []
    per_cam_raw = {lab: {} for lab in CAMERAS}
    for lab in CAMERAS:
        action_box = expand_box(FOCAL_SEEDS[lab], 1.65)
        for d in candidate_offsets:
            idx = CENTERS_V11[lab] + d
            im = decoded[lab][idx]
            action = sharpness_metrics(im, action_box)
            court = sharpness_metrics(im, [0, 75, W - 1, H - 1])
            per_cam_raw[lab][d] = {"frame_index": idx, "action": action, "court": court}

    action_max = {
        lab: max(x["action"]["laplacian_variance"] for x in per_cam_raw[lab].values())
        for lab in CAMERAS
    }
    court_max = {
        lab: max(x["court"]["laplacian_variance"] for x in per_cam_raw[lab].values())
        for lab in CAMERAS
    }

    for d in candidate_offsets:
        cams = {}
        action_rel = []
        court_rel = []
        for lab in CAMERAS:
            x = per_cam_raw[lab][d]
            ar = _rel(x["action"]["laplacian_variance"], action_max[lab])
            cr = _rel(x["court"]["laplacian_variance"], court_max[lab])
            cams[lab] = {**x, "action_relative_to_camera_best": ar, "court_relative_to_camera_best": cr}
            action_rel.append(ar); court_rel.append(cr)
        mean_a = float(np.mean(action_rel)); min_a = float(np.min(action_rel)); mean_c = float(np.mean(court_rel))
        # Keep the actual basketball state tightly tied to v11.  Sharpness can win
        # within the same synchronized instant family, but not by drifting far away.
        proximity_penalty = 0.025 * abs(int(d))
        score = 0.52 * min_a + 0.34 * mean_a + 0.14 * mean_c - proximity_penalty
        rows.append({
            "global_delta_frames": int(d),
            "seconds_from_v11_at_30fps": float(d / 30.0),
            "cameras": cams,
            "mean_action_relative": mean_a,
            "minimum_action_relative": min_a,
            "mean_court_relative": mean_c,
            "state_proximity_penalty": proximity_penalty,
            "joint_sharpness_score": float(score),
        })

    strict = [r for r in rows if abs(r["global_delta_frames"]) <= 2]
    strict.sort(key=lambda r: r["joint_sharpness_score"], reverse=True)
    broader = sorted(rows, key=lambda r: r["joint_sharpness_score"], reverse=True)
    chosen = strict[0]
    # If there is no common sharp instant within ±2 frames, do not silently move
    # the basketball state.  Keep the strict choice and flag the Gaussian backend
    # to use the burst for a sharp latent appearance at the locked instant.
    common_sharp = bool(chosen["minimum_action_relative"] >= 0.68)
    return rows, chosen, broader[0], common_sharp


def camera_json(cam):
    C, R, K = cam
    extr = np.zeros((3, 4), np.float64)
    extr[:, :3] = R
    extr[:, 3] = -(R @ C.reshape(3, 1)).ravel()
    return {
        "C_world_cm": np.asarray(C, float).tolist(),
        "R_world_to_camera": np.asarray(R, float).tolist(),
        "K_px": np.asarray(K, float).tolist(),
        "extrinsic_world_to_camera_3x4": extr.tolist(),
    }


def save_person_assets(out: Path, label: str, image: np.ndarray, instances, balls):
    tag = safe_label(label)
    pdir = out / "players" / tag
    pdir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    rows = []
    for i, inst in enumerate(instances):
        m = inst["mask"].astype(bool)
        ys, xs = np.where(m)
        if not len(xs):
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        full = (m.astype(np.uint8) * 255)
        cv2.imwrite(str(pdir / f"person_{i:02d}_mask_full.png"), full)
        crop = image[y1:y2 + 1, x1:x2 + 1]
        alpha = full[y1:y2 + 1, x1:x2 + 1]
        rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        cv2.imwrite(str(pdir / f"person_{i:02d}_rgba.png"), rgba)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(overlay, f"P{i}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append({
            "camera_person_id": int(i),
            "score": float(inst["score"]),
            "bbox_xyxy": [x1, y1, x2, y2],
            "foot_px": [int(x) for x in inst["foot_px"]],
            "foot_world_cm": [float(x) for x in inst["foot_world_cm"]],
            "mask_pixels": int(m.sum()),
            "mask_full": str((pdir / f"person_{i:02d}_mask_full.png").relative_to(out)),
            "rgba_crop": str((pdir / f"person_{i:02d}_rgba.png").relative_to(out)),
        })
    cv2.imwrite(str(out / f"v32_all_players_{tag}.png"), overlay)
    ball_rows = []
    for i, b in enumerate(balls[:5]):
        ball_rows.append({k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in b.items()})
    return rows, ball_rows


def cluster_world_people(obs, max_dist_cm=175.0):
    """Loose union inventory across cameras, not a claimed identity solve."""
    clusters = []
    # High-confidence detections first stabilise centroids.
    ordered = sorted(obs, key=lambda x: float(x["score"]), reverse=True)
    for row in ordered:
        p = np.asarray(row["foot_world_cm"][:2], np.float64)
        best = None
        for ci, c in enumerate(clusters):
            if row["camera"] in c["cameras"]:
                continue
            cen = np.asarray(c["centroid_xy_cm"], np.float64)
            d = float(np.linalg.norm(p - cen))
            if d <= max_dist_cm and (best is None or d < best[0]):
                best = (d, ci)
        if best is None:
            clusters.append({
                "track_id": len(clusters),
                "centroid_xy_cm": p.tolist(),
                "cameras": [row["camera"]],
                "observations": [row],
            })
        else:
            c = clusters[best[1]]
            c["observations"].append(row)
            c["cameras"].append(row["camera"])
            pts = np.asarray([x["foot_world_cm"][:2] for x in c["observations"]], np.float64)
            c["centroid_xy_cm"] = np.median(pts, axis=0).tolist()
    clusters.sort(key=lambda c: (c["centroid_xy_cm"][0], c["centroid_xy_cm"][1]))
    for i, c in enumerate(clusters):
        c["track_id"] = int(i)
        c["camera_count"] = int(len(set(c["cameras"])))
        c["cameras"] = sorted(set(c["cameras"]))
    return clusters


def read_json_if(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--rar-report", type=Path, required=True)
    ap.add_argument("--broadcast-event-frame", type=Path, required=True)
    ap.add_argument("--v31-qa", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--candidate-radius", type=int, default=4)
    ap.add_argument("--burst-radius", type=int, default=6)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if (W, H) != (960, 540):
        raise RuntimeError("v32 must remain native 960x540")

    clips = {lab: find_clip(args.clips_dir, lab) for lab in CAMERAS}
    candidate_offsets = list(range(-args.candidate_radius, args.candidate_radius + 1))
    decode_need = {
        lab: [CENTERS_V11[lab] + d for d in candidate_offsets]
        for lab in CAMERAS
    }
    decoded = {lab: decode_indices(clips[lab], decode_need[lab]) for lab in CAMERAS}
    candidates, chosen, broad_best, common_sharp = choose_freeze(decoded, candidate_offsets)
    d = int(chosen["global_delta_frames"])
    chosen_indices = {lab: int(CENTERS_V11[lab] + d) for lab in CAMERAS}

    # Export chosen native frames.
    chosen_images = {}
    for lab in CAMERAS:
        im = decoded[lab][chosen_indices[lab]]
        chosen_images[lab] = im
        cv2.imwrite(str(args.out / f"v32_chosen_{safe_label(lab)}_frame{chosen_indices[lab]:04d}.png"), im)

    # Export a wider synchronized burst around the chosen strict-state instant.
    burst = {}
    for lab in CAMERAS:
        idxs = list(range(chosen_indices[lab] - args.burst_radius, chosen_indices[lab] + args.burst_radius + 1))
        frames = decode_indices(clips[lab], idxs)
        bdir = args.out / "burst" / safe_label(lab)
        bdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for rel, idx in enumerate(idxs, start=-args.burst_radius):
            p = bdir / f"rel{rel:+03d}_frame{idx:04d}.png"
            cv2.imwrite(str(p), frames[idx])
            rows.append({"relative_frame": int(rel), "source_frame_index": int(idx), "file": str(p.relative_to(args.out))})
        burst[lab] = rows

    cams = base.load_cameras(args.registry, args.rar_report, args.broadcast_event_frame)

    # Inventory all on-court people at the locked freeze.  This is deliberately
    # independent from the focal-player mesh path: nobody is allowed to disappear
    # merely because they are not the focal subject.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    person_model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    people_by_camera = {}
    balls_by_camera = {}
    all_obs = []
    for lab in CAMERAS:
        C, R, K = cams[lab]
        _dyn, instances, balls = v5.detect_oncourt(person_model, chosen_images[lab], K, R, C)
        rows, ball_rows = save_person_assets(args.out, lab, chosen_images[lab], instances, balls)
        people_by_camera[lab] = rows
        balls_by_camera[lab] = ball_rows
        for r in rows:
            all_obs.append({"camera": lab, **r})

    union_tracks = cluster_world_people(all_obs)
    v31 = read_json_if(args.v31_qa)
    carried_ball = None if v31 is None else v31.get("ball")

    # A scene input manifest for the GPU learned representation.  It deliberately
    # includes every on-court observation and the full source bursts rather than
    # an Adams-only crop.
    manifest = {
        "version": "v32_stage_a_sharp_latent_all_court",
        "event": {"game_id": "0022500301", "event_id": 489},
        "resolution": [W, H],
        "source_policy": "official NBA HLS native frames; no upscale/UHD; no generated fill in Stage A",
        "architecture_change": "retire focal anatomical mesh as final player representation; prepare temporally informed layered Gaussian scene",
        "freeze": {
            "v11_centers": CENTERS_V11,
            "chosen_global_delta_frames": d,
            "chosen_seconds_from_v11_at_30fps": float(d / 30.0),
            "chosen_frame_indices": chosen_indices,
            "common_sharp_frame_gate": "PASS" if common_sharp else "TEMPORAL_LATENT_REQUIRED",
            "minimum_action_sharpness_relative_to_camera_best": float(chosen["minimum_action_relative"]),
            "best_broader_window_delta_frames": int(broad_best["global_delta_frames"]),
            "rule": "never move beyond ±2 frames merely to chase sharpness; if no common sharp instant exists, lock geometry to the strict-state instant and use the synchronized burst for sharp latent appearance",
        },
        "cameras": {lab: camera_json(cams[lab]) for lab in CAMERAS},
        "clips": {lab: clips[lab].name for lab in CAMERAS},
        "burst_radius_frames": int(args.burst_radius),
        "burst_frames_per_camera": int(2 * args.burst_radius + 1),
        "burst": burst,
        "all_on_court_people": {
            "per_camera": people_by_camera,
            "loose_world_union_tracks": union_tracks,
            "union_track_count": int(len(union_tracks)),
            "observation_count": int(len(all_obs)),
            "note": "union tracks are a coverage inventory, not final player identity labels; GPU stage must preserve every listed source observation or explain an occlusion",
        },
        "ball": {
            "carried_three_view_solution_from_v31": carried_ball,
            "detector_candidates_at_v32_freeze": balls_by_camera,
        },
        "gpu_stage_contract": {
            "focal_and_interacting_players": "dense learned sparse-view Gaussian representation from synchronized bursts, not marching-cubes anatomy mesh",
            "other_on_court_people": "source-grounded shallow Gaussian/billboard layers with metric court anchors and correct depth ordering",
            "court_rim_backboard": "deterministic metric geometry + registered texture atlas",
            "distant_arena": "static source-grounded layer; lower priority than player/court fidelity",
            "held_out_camera_test_required": True,
            "render_angles_deg": [0, 5, 10, 15, 20, 25],
            "visual_fail_conditions": [
                "focal player not recognisable",
                "any on-court player disappears without geometrically valid occlusion",
                "missing/doubled limbs in focal action",
                "basket or ball geometry unstable",
                "freeze visibly motion-blurred",
                "large near-action unsupported holes",
            ],
        },
    }
    (args.out / "v32_scene_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "v32_freeze_candidates.json").write_text(json.dumps({
        "candidates": candidates,
        "chosen": chosen,
        "best_broader_window": broad_best,
        "common_sharp_frame": common_sharp,
    }, indent=2))

    # Compact visual montage of chosen source state.
    montage = np.hstack([chosen_images[x] for x in CAMERAS])
    cv2.imwrite(str(args.out / "v32_chosen_three_camera_montage.png"), montage)

    summary = {
        "chosen_delta": d,
        "chosen_indices": chosen_indices,
        "common_sharp_frame_gate": manifest["freeze"]["common_sharp_frame_gate"],
        "minimum_action_relative": manifest["freeze"]["minimum_action_sharpness_relative_to_camera_best"],
        "people_per_camera": {k: len(v) for k, v in people_by_camera.items()},
        "world_union_tracks": len(union_tracks),
        "all_person_observations": len(all_obs),
        "burst_frames_per_camera": 2 * args.burst_radius + 1,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
