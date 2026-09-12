from __future__ import annotations

"""V32b: quality-qualified action-cluster exporter for sparse-view 4D Gaussians.

This intentionally supersedes the whole-court-player requirement for the learned
foreground layer.  The user requirement is quality first: Adams/the focal action
is mandatory; only nearby players that have enough multi-camera source support
are admitted.  Peripheral or weakly observed people are excluded from the learned
foreground rather than forcing low-quality reconstructions.

Important constraints:
- native 960x540 official-source pixels only;
- exact v32 Stage-A synchronized 13-frame bursts are preserved;
- camera K / w2c are unchanged;
- excluded regions are set to black (not inpainted / generated) for the foreground
  training layer, so the model is not asked to spend capacity on peripheral people;
- the deterministic court/background compositor remains a separate downstream
  layer and can restore the visible static environment outside this action ROI.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
CAM_IDS = {"Left Above Rim": 0, "Broadcast": 1, "Right Above Rim": 2}
W, H = 960, 540


def safe(label: str) -> str:
    return label.replace(" ", "_")


def w2c_from_manifest(cam: dict) -> np.ndarray:
    R = np.asarray(cam["R_world_to_camera"], np.float64)
    C_cm = np.asarray(cam["C_world_cm"], np.float64)
    C_m = C_cm / 100.0
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = -(R @ C_m)
    return M


def project_points_cm(points_cm: np.ndarray, cam: dict):
    P = np.asarray(points_cm, np.float64) / 100.0
    K = np.asarray(cam["K_px"], np.float64)
    M = w2c_from_manifest(cam)
    X = np.concatenate([P, np.ones((len(P), 1), np.float64)], axis=1)
    Xc = (M @ X.T).T[:, :3]
    q = (K @ Xc.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = q[:, :2] / q[:, 2:3]
    ok = np.isfinite(uv).all(axis=1) & (Xc[:, 2] > 0.05)
    return uv, ok


def find_ball_xy_cm(manifest: dict) -> np.ndarray:
    b = manifest.get("ball", {}).get("carried_three_view_solution_from_v31", {})
    c = b.get("center_world_cm")
    if isinstance(c, list) and len(c) >= 2:
        a = np.asarray(c[:2], np.float64)
        if np.isfinite(a).all():
            return a
    return np.zeros(2, np.float64)


def track_quality(track: dict, ball_xy: np.ndarray) -> dict:
    obs = track.get("observations", [])
    cam_count = int(track.get("camera_count", len(set(o.get("camera") for o in obs))))
    c = np.asarray(track["centroid_xy_cm"], np.float64)
    d = float(np.linalg.norm(c - ball_xy))
    scores = [float(o.get("score", 0.0)) for o in obs]
    pixels = [int(o.get("mask_pixels", 0)) for o in obs]
    mean_score = float(np.mean(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_pixels = int(max(pixels)) if pixels else 0
    min_pixels = int(min(pixels)) if pixels else 0
    # Score strongly prefers actual multi-camera support and useful pixel area.
    support = min(cam_count, 3) / 3.0
    area = min(1.0, math.sqrt(max_pixels / 4500.0)) if max_pixels > 0 else 0.0
    proximity = max(0.0, 1.0 - d / 650.0)
    q = 0.50 * support + 0.24 * area + 0.16 * mean_score + 0.10 * proximity
    return {
        "camera_count": cam_count,
        "distance_to_ball_xy_cm": d,
        "mean_detector_score": mean_score,
        "minimum_detector_score": min_score,
        "maximum_mask_pixels": max_pixels,
        "minimum_mask_pixels": min_pixels,
        "quality_score": float(q),
    }


def choose_tracks(manifest: dict, max_tracks: int = 5) -> tuple[list[dict], dict]:
    tracks = manifest["all_on_court_people"]["loose_world_union_tracks"]
    ball_xy = find_ball_xy_cm(manifest)
    scored = []
    for tr in tracks:
        q = track_quality(tr, ball_xy)
        scored.append({"track": tr, "qa": q})

    multi = [x for x in scored if x["qa"]["camera_count"] >= 2]
    if not multi:
        raise RuntimeError("no multi-camera player/action track available")
    anchor = min(multi, key=lambda x: x["qa"]["distance_to_ball_xy_cm"])
    anchor_id = int(anchor["track"]["track_id"])

    selected = []
    for x in scored:
        q = x["qa"]
        tid = int(x["track"]["track_id"])
        is_anchor = tid == anchor_id
        # Deliberately strict: peripheral/single-view people are excluded.  The
        # action anchor is mandatory; every optional player must be seen from at
        # least two solved cameras and have meaningful native pixel support.
        passes = is_anchor or (
            q["camera_count"] >= 2
            and q["distance_to_ball_xy_cm"] <= 525.0
            and q["maximum_mask_pixels"] >= 1400
            and q["mean_detector_score"] >= 0.88
        )
        if passes:
            selected.append(x)

    selected.sort(key=lambda x: (0 if int(x["track"]["track_id"]) == anchor_id else 1, -x["qa"]["quality_score"]))
    selected = selected[:max_tracks]
    ids = {int(x["track"]["track_id"]) for x in selected}
    if anchor_id not in ids:
        raise RuntimeError("mandatory focal action anchor was lost")

    decision_rows = []
    for x in sorted(scored, key=lambda x: x["qa"]["distance_to_ball_xy_cm"]):
        tr = x["track"]
        tid = int(tr["track_id"])
        decision_rows.append({
            "track_id": tid,
            "selected": tid in ids,
            "mandatory_action_anchor": tid == anchor_id,
            "centroid_xy_cm": tr["centroid_xy_cm"],
            "cameras": tr.get("cameras", []),
            "observation_refs": [
                {"camera": o["camera"], "camera_person_id": int(o["camera_person_id"]), "mask_pixels": int(o["mask_pixels"]), "score": float(o["score"])}
                for o in tr.get("observations", [])
            ],
            **x["qa"],
        })
    return [x["track"] for x in selected], {
        "ball_xy_cm": ball_xy.tolist(),
        "mandatory_anchor_track_id": anchor_id,
        "selected_track_ids": [int(x["track"]["track_id"]) for x in selected],
        "decisions": decision_rows,
    }


def action_world_box_cm(selected: list[dict], ball_xy: np.ndarray):
    xy = [np.asarray(t["centroid_xy_cm"], np.float64) for t in selected]
    xy.append(np.asarray(ball_xy, np.float64))
    A = np.stack(xy)
    lo = A.min(axis=0) - np.asarray([115.0, 115.0])
    hi = A.max(axis=0) + np.asarray([115.0, 115.0])
    # Bound the action layer so a stray loose match cannot explode the crop.
    lo = np.maximum(lo, np.asarray([-700.0, -800.0]))
    hi = np.minimum(hi, np.asarray([950.0, 850.0]))
    return [float(lo[0]), float(lo[1]), 0.0, float(hi[0]), float(hi[1]), 390.0]


def projected_roi(cam: dict, selected: list[dict], label: str, world_box: list[float]):
    x0, y0, z0, x1, y1, z1 = world_box
    corners = np.asarray([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], np.float64)
    # Add a small basket/ball core so ROI remains useful even when one camera has
    # only one selected human observation due occlusion.
    uv, ok = project_points_cm(corners, cam)
    pts = [p for p, good in zip(uv, ok) if good]
    for tr in selected:
        for o in tr.get("observations", []):
            if o.get("camera") == label:
                bx = o["bbox_xyxy"]
                pts += [np.asarray([bx[0], bx[1]]), np.asarray([bx[2], bx[3]])]
    if not pts:
        raise RuntimeError(f"cannot define action ROI for {label}")
    P = np.asarray(pts, np.float64)
    xa = int(max(0, math.floor(np.min(P[:, 0]) - 28)))
    xb = int(min(W - 1, math.ceil(np.max(P[:, 0]) + 28)))
    ya = int(max(0, math.floor(np.min(P[:, 1]) - 28)))
    yb = int(min(H - 1, math.ceil(np.max(P[:, 1]) + 28)))
    if xb - xa < 80 or yb - ya < 80:
        raise RuntimeError(f"implausibly small ROI for {label}: {(xa,ya,xb,yb)}")
    return [xa, ya, xb, yb]


def copy_masked_bursts(manifest: dict, stage_a: Path, out: Path, rois: dict):
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
            if im is None or im.shape[:2] != (H, W):
                raise RuntimeError(f"bad source frame {src}")
            xa, ya, xb, yb = rois[lab]
            masked = np.zeros_like(im)
            masked[ya:yb + 1, xa:xb + 1] = im[ya:yb + 1, xa:xb + 1]
            fn = f"t{rel:+03d}_{safe(lab)}.png"
            cv2.imwrite(str(ims / fn), masked)
            fns.append(fn)
            ks.append(manifest["cameras"][lab]["K_px"])
            w2cs.append(w2c_from_manifest(manifest["cameras"][lab]).tolist())
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
    return {"w": W, "h": H, "fn": fn, "k": k, "w2c": w2c, "cam_id": cam_id}


def median_track_color(stage_a: Path, tr: dict):
    colors = []
    for o in tr.get("observations", []):
        p = stage_a / o["rgba_crop"]
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        a = im[:, :, 3] > 0
        if int(a.sum()) < 10:
            continue
        pix = im[:, :, :3][a][:, ::-1].astype(np.float32) / 255.0
        colors.append(np.median(pix, axis=0))
    if not colors:
        return np.asarray([0.45, 0.45, 0.45], np.float32)
    return np.median(np.stack(colors), axis=0).astype(np.float32)


def make_init_cloud(manifest: dict, stage_a: Path, out: Path, selected: list[dict], world_box: list[float]):
    x0, y0, _, x1, y1, _ = world_box
    xs = np.arange(x0 / 100.0, x1 / 100.0 + 1e-6, 0.22, dtype=np.float64)
    ys = np.arange(y0 / 100.0, y1 / 100.0 + 1e-6, 0.22, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)]).astype(np.float32)
    floor_rgb = np.tile(np.asarray([[0.58, 0.43, 0.30]], np.float32), (len(floor), 1))

    pts, cols = [floor], [floor_rgb]
    for tr in selected:
        xy = np.asarray(tr["centroid_xy_cm"], np.float64) / 100.0
        col = median_track_color(stage_a, tr)
        body = []
        for z in np.arange(0.08, 2.31, 0.14):
            r = 0.11 if z < 0.24 or z > 1.95 else 0.24
            n = 10 if r > 0.2 else 6
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                body.append([xy[0] + r*np.cos(a), xy[1] + r*np.sin(a), z])
        body = np.asarray(body, np.float32)
        pts.append(body); cols.append(np.tile(col[None, :], (len(body), 1)))

    b = manifest.get("ball", {}).get("carried_three_view_solution_from_v31", {}).get("center_world_cm")
    if isinstance(b, list) and len(b) >= 3:
        c = np.asarray(b[:3], np.float64) / 100.0
        ball = []
        for ph in np.linspace(0.25, np.pi-0.25, 7):
            for th in np.linspace(0, 2*np.pi, 12, endpoint=False):
                ball.append(c + .12*np.asarray([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)]))
        ball = np.asarray(ball, np.float32)
        pts.append(ball); cols.append(np.tile(np.asarray([[.78,.34,.08]], np.float32), (len(ball),1)))

    xyz = np.concatenate(pts, axis=0); rgb = np.concatenate(cols, axis=0)
    data = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
    np.savez_compressed(out / "init_pt_cld.npz", data=data)
    return {"total_points": int(len(data)), "selected_player_seed_tracks": int(len(selected)), "floor_seed_points": int(len(floor))}


def diagnostics(stage_a: Path, out: Path, selected: list[dict], selection: dict, rois: dict):
    selected_ids = {int(t["track_id"]) for t in selected}
    color_sel = (0, 255, 0); color_drop = (0, 0, 255)
    frames = []
    for lab in CAMERAS:
        srcs = sorted(stage_a.glob(f"v32_chosen_{safe(lab)}_frame*.png"))
        im = cv2.imread(str(srcs[0]))
        for d in selection["decisions"]:
            for o in d["observation_refs"]:
                if o["camera"] != lab:
                    continue
                # Resolve bbox from Stage-A per-camera people table.
                row = next(x for x in json.loads((stage_a/'v32_scene_manifest.json').read_text())["all_on_court_people"]["per_camera"][lab] if int(x["camera_person_id"])==int(o["camera_person_id"]))
                x1,y1,x2,y2 = map(int,row["bbox_xyxy"])
                col = color_sel if d["selected"] else color_drop
                cv2.rectangle(im,(x1,y1),(x2,y2),col,2)
                cv2.putText(im,f"T{d['track_id']}",(x1,max(14,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.42,col,1,cv2.LINE_AA)
        xa,ya,xb,yb = rois[lab]
        cv2.rectangle(im,(xa,ya),(xb,yb),(255,255,0),2)
        cv2.putText(im,"QUALITY ACTION ROI",(xa+4,min(H-8,ya+18)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,0),1,cv2.LINE_AA)
        cv2.imwrite(str(out/f"v32b_quality_selection_{safe(lab)}.png"),im)
        frames.append(im)
    montage = np.concatenate(frames,axis=1)
    cv2.imwrite(str(out/"v32b_quality_selection_montage.png"),montage)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--stage-a",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--max-tracks",type=int,default=5)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    m=json.loads((args.stage_a/"v32_scene_manifest.json").read_text())
    if m["resolution"] != [W,H]:
        raise RuntimeError(f"native resolution violation: {m['resolution']}")

    selected, selection = choose_tracks(m, max_tracks=args.max_tracks)
    ball_xy = np.asarray(selection["ball_xy_cm"],np.float64)
    world_box = action_world_box_cm(selected,ball_xy)
    rois={lab:projected_roi(m["cameras"][lab],selected,lab,world_box) for lab in CAMERAS}
    times=copy_masked_bursts(m,args.stage_a,args.out,rois)
    all3=meta_payload(times)
    (args.out/"train_meta.json").write_text(json.dumps(all3,indent=2))
    (args.out/"test_meta.json").write_text(json.dumps(all3,indent=2))
    holdouts={}
    for lab in CAMERAS:
        hid=CAM_IDS[lab]
        tr=meta_payload(times,{x for x in CAM_IDS.values() if x!=hid})
        te=meta_payload(times,{hid})
        stem=safe(lab).lower()
        trp=args.out/f"train_meta_holdout_{stem}.json"; tep=args.out/f"test_meta_holdout_{stem}.json"
        trp.write_text(json.dumps(tr,indent=2)); tep.write_text(json.dumps(te,indent=2))
        holdouts[lab]={"train_meta":trp.name,"test_meta":tep.name,"train_camera_ids":sorted({x for x in CAM_IDS.values() if x!=hid}),"test_camera_id":hid}

    cloud=make_init_cloud(m,args.stage_a,args.out,selected,world_box)
    diagnostics(args.stage_a,args.out,selected,selection,rois)
    backend={
        "version":"v32b_quality_qualified_action_cluster",
        "native_resolution":[W,H],
        "freeze_relative_frame":0,
        "time_steps":len(times),
        "training_images":len(times)*len(CAMERAS),
        "source_cameras":list(CAMERAS),
        "selection_policy":"quality first: focal action mandatory; optional players require >=2 solved cameras, proximity to ball, sufficient mask pixels, and high detection confidence; peripheral/single-view people excluded",
        "selected_track_ids":selection["selected_track_ids"],
        "mandatory_anchor_track_id":selection["mandatory_anchor_track_id"],
        "selected_track_count":len(selected),
        "action_world_box_cm_xyzxyz":world_box,
        "per_camera_action_roi_xyxy":rois,
        "excluded_region_policy":"black outside action ROI in learned foreground training frames; no inpainting or generated fill; deterministic court/background compositor is separate",
        "sharp_latent_required":m["freeze"]["common_sharp_frame_gate"]=="TEMPORAL_LATENT_REQUIRED",
        "held_out_camera_manifests":holdouts,
        "initial_point_cloud":cloud,
        "visual_pass_rule":"every selected visible person must remain recognisable and anatomically coherent in held-out real-camera renders; failing optional tracks are removed rather than lowering the replay quality bar",
        "upscale":False,
        "uhd":False,
    }
    (args.out/"v32b_quality_selection.json").write_text(json.dumps(selection,indent=2))
    (args.out/"v32b_backend.json").write_text(json.dumps(backend,indent=2))
    print(json.dumps(backend,indent=2),flush=True)

if __name__=="__main__":
    main()
