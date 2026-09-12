from __future__ import annotations

"""V32d: source-native fixed-size action crops for learned 4D Gaussian foreground.

V32c correctly narrows the scene, but black-masking most of a 960x540 frame would
let background zeros dominate the photometric training loss. V32d therefore
crops, never resizes, each solved camera to a common 384x448 action window. The
principal point is translated by the exact crop origin, so the calibrated rays are
identical to the corresponding pixels in the original 960x540 source.

384x448 is deliberate: all three tight action ROIs fit without interpolation,
while the narrower width removes single-view peripheral players that would waste
model capacity. Final replay output remains native 960x540. These are training
crops, not an output rescale or quality reduction.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import export_v32b_quality_cluster_4dgs as v32b
from freeze_spin.export_v32c_quality_cluster_tight import tight_projected_roi

CAMERAS = v32b.CAMERAS
CAM_IDS = v32b.CAM_IDS
SOURCE_W, SOURCE_H = 960, 540
CROP_W, CROP_H = 384, 448


def common_crop_from_roi(roi):
    x1, y1, x2, y2 = map(float, roi)
    if x2 - x1 + 1 > CROP_W or y2 - y1 + 1 > CROP_H:
        raise RuntimeError(f"tight ROI does not fit native {CROP_W}x{CROP_H} crop: {roi}")
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    xa = int(round(cx - CROP_W / 2))
    ya = int(round(cy - CROP_H / 2))
    xa = max(0, min(SOURCE_W - CROP_W, xa))
    ya = max(0, min(SOURCE_H - CROP_H, ya))
    return [xa, ya, xa + CROP_W - 1, ya + CROP_H - 1]


def adjusted_K(K, crop):
    A = np.asarray(K, np.float64).copy()
    A[0, 2] -= float(crop[0])
    A[1, 2] -= float(crop[1])
    return A.tolist()


def copy_native_crops(manifest: dict, stage_a: Path, out: Path, crops: dict):
    ims = out / "ims"
    ims.mkdir(parents=True, exist_ok=True)
    per_cam = {lab: {int(r["relative_frame"]): r for r in manifest["burst"][lab]} for lab in CAMERAS}
    rels = sorted(set.intersection(*[set(x.keys()) for x in per_cam.values()]))
    times = []
    for rel in rels:
        fns, ks, w2cs, ids = [], [], [], []
        for lab in CAMERAS:
            src = stage_a / per_cam[lab][rel]["file"]
            im = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if im is None or im.shape[:2] != (SOURCE_H, SOURCE_W):
                raise RuntimeError(f"bad native source {src}")
            x1, y1, x2, y2 = crops[lab]
            crop = im[y1:y2 + 1, x1:x2 + 1]
            if crop.shape[:2] != (CROP_H, CROP_W):
                raise RuntimeError((lab, crop.shape, crops[lab]))
            fn = f"t{rel:+03d}_{v32b.safe(lab)}.png"
            cv2.imwrite(str(ims / fn), crop)
            fns.append(fn)
            ks.append(adjusted_K(manifest["cameras"][lab]["K_px"], crops[lab]))
            w2cs.append(v32b.w2c_from_manifest(manifest["cameras"][lab]).tolist())
            ids.append(CAM_IDS[lab])
        times.append({"relative_frame": int(rel), "fn": fns, "k": ks, "w2c": w2cs, "cam_id": ids})
    return times


def meta_payload(times, keep_cam_ids=None):
    fn, k, w2c, cam_id = [], [], [], []
    for t in times:
        ii = list(range(len(t["cam_id"]))) if keep_cam_ids is None else [i for i, cid in enumerate(t["cam_id"]) if cid in keep_cam_ids]
        fn.append([t["fn"][i] for i in ii])
        k.append([t["k"][i] for i in ii])
        w2c.append([t["w2c"][i] for i in ii])
        cam_id.append([t["cam_id"][i] for i in ii])
    return {"w": CROP_W, "h": CROP_H, "fn": fn, "k": k, "w2c": w2c, "cam_id": cam_id}


def crop_montage(stage_a: Path, out: Path, crops: dict):
    panels = []
    for lab in CAMERAS:
        src = sorted(stage_a.glob(f"v32_chosen_{v32b.safe(lab)}_frame*.png"))[0]
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        x1, y1, x2, y2 = crops[lab]
        c = im[y1:y2 + 1, x1:x2 + 1].copy()
        cv2.putText(c, lab, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2, cv2.LINE_AA)
        panels.append(c)
        cv2.imwrite(str(out / f"v32d_native_crop_{v32b.safe(lab)}.png"), c)
    cv2.imwrite(str(out / "v32d_native_crop_montage.png"), np.concatenate(panels, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((args.stage_a / "v32_scene_manifest.json").read_text())
    if m["resolution"] != [SOURCE_W, SOURCE_H]:
        raise RuntimeError(f"unexpected native source resolution {m['resolution']}")

    selected, selection = v32b.choose_tracks(m, max_tracks=args.max_tracks)
    ball_xy = np.asarray(selection["ball_xy_cm"], np.float64)
    world_box = v32b.action_world_box_cm(selected, ball_xy)
    tight = {lab: tight_projected_roi(m["cameras"][lab], selected, lab, world_box) for lab in CAMERAS}
    crops = {lab: common_crop_from_roi(tight[lab]) for lab in CAMERAS}

    times = copy_native_crops(m, args.stage_a, args.out, crops)
    all3 = meta_payload(times)
    (args.out / "train_meta.json").write_text(json.dumps(all3, indent=2))
    (args.out / "test_meta.json").write_text(json.dumps(all3, indent=2))
    holdouts = {}
    for lab in CAMERAS:
        hid = CAM_IDS[lab]
        tr = meta_payload(times, {x for x in CAM_IDS.values() if x != hid})
        te = meta_payload(times, {hid})
        stem = v32b.safe(lab).lower()
        trp = args.out / f"train_meta_holdout_{stem}.json"
        tep = args.out / f"test_meta_holdout_{stem}.json"
        trp.write_text(json.dumps(tr, indent=2)); tep.write_text(json.dumps(te, indent=2))
        holdouts[lab] = {"train_meta": trp.name, "test_meta": tep.name, "test_camera_id": hid}

    cloud = v32b.make_init_cloud(m, args.stage_a, args.out, selected, world_box)
    crop_montage(args.stage_a, args.out, crops)
    backend = {
        "version": "v32d_native_action_crops",
        "source_resolution": [SOURCE_W, SOURCE_H],
        "training_crop_resolution": [CROP_W, CROP_H],
        "resized": False,
        "native_pixels_preserved": True,
        "final_output_resolution": [SOURCE_W, SOURCE_H],
        "time_steps": len(times),
        "training_images": len(times) * len(CAMERAS),
        "selected_track_ids": selection["selected_track_ids"],
        "mandatory_anchor_track_id": selection["mandatory_anchor_track_id"],
        "tight_rois_xyxy_source": tight,
        "native_crop_xyxy_source": crops,
        "full_frame_K_px": {lab: m["cameras"][lab]["K_px"] for lab in CAMERAS},
        "crop_K_px": {lab: adjusted_K(m["cameras"][lab]["K_px"], crops[lab]) for lab in CAMERAS},
        "held_out_camera_manifests": holdouts,
        "initial_point_cloud": cloud,
        "sharp_latent_required": m["freeze"]["common_sharp_frame_gate"] == "TEMPORAL_LATENT_REQUIRED",
        "reason": "quality-first action crops remove peripheral single-view players and zero-padded loss area while preserving exact native source rays",
        "upscale": False,
        "uhd": False,
    }
    (args.out / "v32d_quality_selection.json").write_text(json.dumps(selection, indent=2))
    (args.out / "v32d_backend.json").write_text(json.dumps(backend, indent=2))
    print(json.dumps(backend, indent=2), flush=True)


if __name__ == "__main__":
    main()
