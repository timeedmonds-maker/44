from __future__ import annotations

"""Exact visual-state synchronization v11 for the three-camera Adams/Jazz proof.

Audio selects the centre decoded frame in each physical camera.  This module
searches a small decoded-frame neighbourhood and asks which frame triple best
supports one or more *true three-view* articulated human poses under the
already accepted metric camera matrices.  RF-DETR pose is used only as a
semantic state measurement; it does not alter camera calibration or invent
appearance.

The chosen frames are exported as a minimal three-frame directory ready for the
hardened v10b identity-aware reconstruction.  Full candidate diagnostics are
written whether or not a strict three-view match is found.
"""

import argparse
import itertools
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_rfdetr_v10 as v10
from freeze_spin import build_three_camera_semantic_v9 as v9
from freeze_spin import build_three_camera_volumetric_v8 as v8
from rfdetr import RFDETRKeypointPreview

CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
FILE_LABEL = {
    "Left Above Rim": "K_Left_Above_Rim",
    "Broadcast": "A_Broadcast",
    "Right Above Rim": "L_Right_Above_Rim",
}
SOURCE_RE = re.compile(r"_489_(.+)_SOURCE\.mp4$")
WORLD_BOUNDS = (-320.0, 1250.0, -720.0, 720.0, -40.0, 390.0)


def _project(cam, X):
    uv, _, valid = v8.project_metric(cam, np.asarray(X, np.float64).reshape(1, 3))
    return (uv[0] if valid[0] else None)


def _in_world(X):
    x0, x1, y0, y1, z0, z1 = WORLD_BOUNDS
    return bool(x0 <= X[0] <= x1 and y0 <= X[1] <= y1 and z0 <= X[2] <= z1)


def _decode_frame(clip: Path, frame_index: int):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, image = cap.read()
    actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1))
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode {clip.name} frame {frame_index}")
    if actual != int(frame_index):
        raise RuntimeError(f"decoded frame drift for {clip.name}: wanted {frame_index}, got {actual}")
    if image.shape[:2] != (base.H, base.W):
        raise RuntimeError(f"unexpected shape {image.shape} for {clip.name} frame {frame_index}")
    return image


def _clip_map(clips: Path):
    out = {}
    for p in sorted(clips.glob("*_489_*_SOURCE.mp4")):
        m = SOURCE_RE.search(p.name)
        if not m:
            continue
        label = m.group(1).replace("_", " ")
        if label in CAMERAS:
            out[label] = p
    missing = [c for c in CAMERAS if c not in out]
    if missing:
        raise RuntimeError(f"missing source clips {missing}")
    return out


def _centres(options_path: Path):
    d = json.loads(options_path.read_text())
    rows = {r["camera"]: r for r in d["options"]}
    missing = [c for c in CAMERAS if c not in rows]
    if missing:
        raise RuntimeError(f"options missing cameras {missing}")
    return {c: int(rows[c]["decoded_frame_index"]) for c in CAMERAS}, d


def _pose_predict(model, image, threshold):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kp = model.predict(rgb, threshold=float(threshold))
    try:
        kp = kp.with_nms()
    except Exception:
        pass
    xy = np.asarray(kp.xy, np.float64)
    conf = getattr(kp, "keypoint_confidence", None)
    conf = np.ones(xy.shape[:2], np.float64) if conf is None else np.asarray(conf, np.float64)
    det_conf = getattr(kp, "detection_confidence", None)
    det_conf = np.ones(len(xy), np.float64) if det_conf is None else np.asarray(det_conf, np.float64)
    boxes = np.asarray(kp.data.get("xyxy", np.empty((0, 4))), np.float64)
    return {"xy": xy, "conf": conf, "det_conf": det_conf, "boxes": boxes}


def _near_play_indices(pose, cam, max_rim_distance_px=430.0):
    if len(pose["xy"]) == 0:
        return []
    rim_uv = _project(cam, v9.RIM)
    if rim_uv is None:
        return list(range(len(pose["xy"])))
    keep = []
    for i in range(len(pose["xy"])):
        box = pose["boxes"][i] if i < len(pose["boxes"]) else None
        if box is not None and np.isfinite(box).all():
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        else:
            good = pose["conf"][i] >= 0.18
            if not np.any(good):
                continue
            cx, cy = np.median(pose["xy"][i][good], axis=0)
        if math.hypot(float(cx-rim_uv[0]), float(cy-rim_uv[1])) <= max_rim_distance_px:
            keep.append(i)
    return keep


def _triple_pose(cams, pa, ia, pb, ib, pc, ic, joint_min=0.24):
    labels = CAMERAS
    poses = (pa, pb, pc)
    ids = (ia, ib, ic)
    joints = []
    Xs = {}
    for j in range(17):
        obs = {}
        mean_conf = []
        for label, p, idx in zip(labels, poses, ids):
            if idx >= len(p["xy"]) or j >= p["xy"].shape[1] or p["conf"][idx, j] < joint_min:
                obs = {}
                break
            obs[label] = p["xy"][idx, j]
            mean_conf.append(float(p["conf"][idx, j]))
        if len(obs) != 3:
            continue
        X = v8.dlt_point(cams, obs)
        if X is None or not np.isfinite(X).all() or not _in_world(X):
            continue
        errs = {}
        for label in labels:
            uv = _project(cams[label], X)
            if uv is None:
                errs = {}
                break
            errs[label] = float(np.linalg.norm(uv - obs[label]))
        if not errs:
            continue
        rms = float(np.sqrt(np.mean(np.square(list(errs.values())))))
        joints.append({
            "joint_index": j,
            "joint": v10.COCO_NAMES[j],
            "rms_reprojection_px": rms,
            "per_view_error_px": errs,
            "mean_confidence": float(np.mean(mean_conf)),
            "world_cm": [float(x) for x in X],
        })
        Xs[j] = np.asarray(X, np.float64)

    if not joints:
        return {
            "common_joints": 0, "good15": 0, "good25": 0,
            "median_rms_px": 999.0, "p75_rms_px": 999.0,
            "measured_bones": 0, "plausible_bones": 0,
            "plausible_bone_fraction": 0.0, "strict": False, "joints": [],
        }
    e = np.asarray([r["rms_reprojection_px"] for r in joints], np.float64)
    measured = plausible = 0
    bone_rows = []
    for a, b, _rad, lo, hi in v10.BONES:
        if a not in Xs or b not in Xs:
            continue
        L = float(np.linalg.norm(Xs[a] - Xs[b]))
        ok = bool(lo <= L <= hi)
        measured += 1
        plausible += int(ok)
        bone_rows.append({"bone": f"{v10.COCO_NAMES[a]}-{v10.COCO_NAMES[b]}", "length_cm": L, "plausible": ok})
    frac = float(plausible / measured) if measured else 0.0
    med = float(np.median(e)); p75 = float(np.percentile(e, 75))
    good15 = int(np.sum(e <= 15.0)); good25 = int(np.sum(e <= 25.0))
    strict = bool(len(joints) >= 5 and good15 >= 4 and med <= 16.0 and p75 <= 24.0 and (measured < 3 or frac >= 0.34))
    edge_score = (
        (400.0 if strict else 0.0)
        + 18.0 * good15 + 5.0 * good25 + 1.2 * len(joints)
        - 1.6 * min(med, 120.0) - 0.45 * min(p75, 160.0)
        + 20.0 * frac
    )
    return {
        "common_joints": int(len(joints)), "good15": good15, "good25": good25,
        "median_rms_px": med, "p75_rms_px": p75,
        "measured_bones": int(measured), "plausible_bones": int(plausible),
        "plausible_bone_fraction": frac, "strict": strict,
        "edge_score": float(edge_score), "joints": joints, "bones": bone_rows,
    }


def _assign_triples(cams, poses, near, joint_min):
    A, B, C = CAMERAS
    edges = []
    for ia in near[A]:
        for ib in near[B]:
            for ic in near[C]:
                q = _triple_pose(cams, poses[A], ia, poses[B], ib, poses[C], ic, joint_min)
                if q["common_joints"] < 3:
                    continue
                edges.append({"ids": {A: ia, B: ib, C: ic}, **q})
    edges.sort(key=lambda r: (bool(r["strict"]), r["edge_score"]), reverse=True)
    used_a, used_b, used_c = set(), set(), set()
    best = []
    for e in edges:
        ids = e["ids"]
        if ids[A] in used_a or ids[B] in used_b or ids[C] in used_c:
            continue
        if e["edge_score"] <= -80.0:
            continue
        best.append(e)
        used_a.add(ids[A]); used_b.add(ids[B]); used_c.add(ids[C])
    best_score = float(sum(e["edge_score"] for e in best))
    strict_edges = [e for e in best if e["strict"]]
    all_err = [j["rms_reprojection_px"] for e in best for j in e["joints"]]
    strict_err = [j["rms_reprojection_px"] for e in strict_edges for j in e["joints"]]
    return {
        "assigned_tracks": best,
        "assigned_track_count": int(len(best)),
        "strict_track_count": int(len(strict_edges)),
        "strict_good15_joints": int(sum(e["good15"] for e in strict_edges)),
        "strict_common_joints": int(sum(e["common_joints"] for e in strict_edges)),
        "median_assigned_joint_rms_px": float(np.median(all_err)) if all_err else 999.0,
        "median_strict_joint_rms_px": float(np.median(strict_err)) if strict_err else 999.0,
        "assignment_score": float(best_score if np.isfinite(best_score) else -1e9),
        "candidate_edge_count": int(len(edges)),
    }


def _combo_key(q):
    return (
        int(q["strict_track_count"]),
        int(q["strict_good15_joints"]),
        int(q["strict_common_joints"]),
        -float(q["median_strict_joint_rms_px"] if q["strict_track_count"] else q["median_assigned_joint_rms_px"]),
        float(q["assignment_score"]),
        -int(q["offset_l1"]),
    )


def _draw_pose(image, pose, near, title):
    out = image.copy()
    for i in near:
        if i < len(pose["boxes"]):
            x1, y1, x2, y2 = np.rint(pose["boxes"][i]).astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255,255,255), 1)
        for a, b in v10.DRAW_EDGES:
            if a < pose["xy"].shape[1] and b < pose["xy"].shape[1] and pose["conf"][i,a] >= .20 and pose["conf"][i,b] >= .20:
                cv2.line(out, tuple(np.rint(pose["xy"][i,a]).astype(int)), tuple(np.rint(pose["xy"][i,b]).astype(int)), (255,255,255), 1, cv2.LINE_AA)
        for j in range(min(17, pose["xy"].shape[1])):
            if pose["conf"][i,j] >= .20:
                cv2.circle(out, tuple(np.rint(pose["xy"][i,j]).astype(int)), 2, (255,255,255), -1, cv2.LINE_AA)
        if i < len(pose["boxes"]):
            x1, y1 = np.rint(pose["boxes"][i][:2]).astype(int)
            cv2.putText(out, f"p{i}", (x1, max(15,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0,0), (base.W,31), (0,0,0), -1)
    cv2.putText(out, title, (8,21), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--options", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--rar-report", type=Path, required=True)
    ap.add_argument("--broadcast-event-frame", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--pose-threshold", type=float, default=0.18)
    ap.add_argument("--joint-threshold", type=float, default=0.24)
    ap.add_argument("--require-strict-tracks", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected_dir = args.out / "selected_frames"; selected_dir.mkdir(parents=True, exist_ok=True)

    centres, options = _centres(args.options)
    clips = _clip_map(args.clips)
    cams = base.load_cameras(args.registry, args.rar_report, args.broadcast_event_frame)

    print("Loading RF-DETR Keypoint Preview for exact-state search...", flush=True)
    model = RFDETRKeypointPreview()
    print("RFDETR_EXACT_SYNC_MODEL_READY", flush=True)

    cache = {c: {} for c in CAMERAS}
    detection_qa = {c: [] for c in CAMERAS}
    for label in CAMERAS:
        for off in range(-args.radius, args.radius + 1):
            idx = centres[label] + off
            image = _decode_frame(clips[label], idx)
            pose = _pose_predict(model, image, args.pose_threshold)
            near = _near_play_indices(pose, cams[label])
            balls = v9.roi_ball_candidates(image, cams[label], [])
            cache[label][off] = {"frame": idx, "image": image, "pose": pose, "near": near, "balls": balls}
            detection_qa[label].append({
                "offset": off, "frame": idx, "pose_detections": int(len(pose["xy"])),
                "near_play_pose_indices": [int(x) for x in near],
                "near_play_pose_count": int(len(near)),
                "mean_confident_joints": float(np.mean(np.sum(pose["conf"] >= args.joint_threshold, axis=1))) if len(pose["conf"]) else 0.0,
                "rim_roi_ball_candidates": balls,
            })
            print("CANDIDATE", label, off, idx, "poses", len(pose["xy"]), "near", len(near), flush=True)

    combos = []
    offs = range(-args.radius, args.radius + 1)
    for oa, ob, oc in itertools.product(offs, offs, offs):
        pos = {
            "Left Above Rim": cache["Left Above Rim"][oa]["pose"],
            "Broadcast": cache["Broadcast"][ob]["pose"],
            "Right Above Rim": cache["Right Above Rim"][oc]["pose"],
        }
        near = {
            "Left Above Rim": cache["Left Above Rim"][oa]["near"],
            "Broadcast": cache["Broadcast"][ob]["near"],
            "Right Above Rim": cache["Right Above Rim"][oc]["near"],
        }
        q = _assign_triples(cams, pos, near, args.joint_threshold)
        q.update({
            "offsets": {"Left Above Rim": oa, "Broadcast": ob, "Right Above Rim": oc},
            "frames": {
                "Left Above Rim": cache["Left Above Rim"][oa]["frame"],
                "Broadcast": cache["Broadcast"][ob]["frame"],
                "Right Above Rim": cache["Right Above Rim"][oc]["frame"],
            },
            "offset_l1": abs(oa)+abs(ob)+abs(oc),
        })
        combos.append(q)

    combos.sort(key=_combo_key, reverse=True)
    best = combos[0]
    baseline = next(x for x in combos if all(v == 0 for v in x["offsets"].values()))
    print("BASELINE", json.dumps({k: baseline[k] for k in ("offsets","frames","strict_track_count","strict_good15_joints","strict_common_joints","median_strict_joint_rms_px","median_assigned_joint_rms_px","assignment_score")}, indent=2), flush=True)
    print("SELECTED", json.dumps({k: best[k] for k in ("offsets","frames","strict_track_count","strict_good15_joints","strict_common_joints","median_strict_joint_rms_px","median_assigned_joint_rms_px","assignment_score")}, indent=2), flush=True)

    for label in CAMERAS:
        off = int(best["offsets"][label]); row = cache[label][off]
        fn = f"{FILE_LABEL[label]}_v11_off{off:+d}_frame{row['frame']:04d}.png"
        cv2.imwrite(str(selected_dir / fn), row["image"])
        ann = _draw_pose(row["image"], row["pose"], row["near"], f"v11 selected | {label} | audio offset {off:+d} | frame {row['frame']}")
        cv2.imwrite(str(args.out / f"selected_pose_{label.replace(' ','_')}.png"), ann)

    ims = [cv2.imread(str(args.out / f"selected_pose_{c.replace(' ','_')}.png")) for c in CAMERAS]
    montage = np.concatenate(ims, axis=1)
    cv2.imwrite(str(args.out / "v11_selected_pose_montage.png"), montage)

    improvement = {
        "strict_tracks_delta": int(best["strict_track_count"] - baseline["strict_track_count"]),
        "strict_good15_delta": int(best["strict_good15_joints"] - baseline["strict_good15_joints"]),
        "strict_common_joints_delta": int(best["strict_common_joints"] - baseline["strict_common_joints"]),
        "selected_is_audio_centre": bool(all(v == 0 for v in best["offsets"].values())),
    }
    qa = {
        "schema_version": 11,
        "status": "EXACT_VISUAL_STATE_SELECTED" if best["strict_track_count"] >= args.require_strict_tracks else "EXACT_VISUAL_STATE_UNSOLVED",
        "game_id": "0022500301", "event_id": 489,
        "policy": "audio prediction is centre only; selected frame triple maximizes true three-view RF-DETR joint reprojection consistency under accepted metric cameras",
        "search_radius_frames": args.radius,
        "audio_centre_frames": centres,
        "audio_sync_source_policy": options.get("policy"),
        "detection_audit": detection_qa,
        "baseline_audio_centre": baseline,
        "selected": best,
        "improvement_vs_audio_centre": improvement,
        "top20": combos[:20],
        "ranking": "lexicographic strict 3-view tracks -> <=15px joints -> common joints -> reprojection -> assignment score -> smallest audio offset",
        "appearance_policy": "selection only; no generated pixels or temporal interpolation",
    }
    (args.out / "exact_visual_state_v11_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (args.out / "selected_offsets.json").write_text(json.dumps({"offsets": best["offsets"], "frames": best["frames"], "status": qa["status"]}, indent=2), encoding="utf-8")
    if best["strict_track_count"] < args.require_strict_tracks:
        raise RuntimeError(f"no exact-state solution meeting strict track gate; best={best['offsets']} strict={best['strict_track_count']}")


if __name__ == "__main__":
    main()
