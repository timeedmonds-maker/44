from __future__ import annotations

"""v33f: source-grounded body + ball fit on v33e exact three-view states.

Consumes the accepted v33e evidence and searches only its top exact states.
The three physical cameras remain locked. Player observations come from the same
identity/flow policy as v33e. The basketball must be detected as a COCO
"sports ball" in at least two real source views; a third view may be explicitly
classified as occluded only when the projected center is hidden by the locked
Adams box or rim structure. No source pixels are generated and no novel view is
rendered here.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview, RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

from freeze_spin import run_v33e_locked_three_camera_wide_flow_state_search as v33e
from freeze_spin import run_v33d_locked_three_camera_joint_state_search as v33d
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
CAMS = (LAR, RAR, BCAST)
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF
W, H = v32j.W, v32j.H
RIM_WORLD_CM = np.array([38.1, 0.0, 304.8], float)
BALL_MATCH_MAX_PX = 18.0
BALL_RIM_DISTANCE_MAX_CM = 200.0


def coco_name(cid: int) -> str:
    try:
        x = COCO_CLASSES[int(cid)]
    except Exception:
        try:
            x = COCO_CLASSES.get(int(cid), "")
        except Exception:
            x = ""
    return str(x).strip().lower()


def detect_sports_balls(model, image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    d = model.predict(rgb, threshold=0.05)
    boxes = np.asarray(getattr(d, "xyxy", np.empty((0, 4))), float)
    conf = np.asarray(getattr(d, "confidence", np.empty((0,))), float)
    cls = np.asarray(getattr(d, "class_id", np.empty((0,))), int)
    out = []
    for i in range(len(boxes)):
        if coco_name(int(cls[i])) != "sports ball":
            continue
        b = boxes[i]
        if not np.all(np.isfinite(b)):
            continue
        x1, y1, x2, y2 = map(float, b)
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 3 or w > 120 or h > 120:
            continue
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], float)
        xi1, yi1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
        xi2, yi2 = min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
        orange = 0.0
        if xi2 > xi1 and yi2 > yi1:
            hsv = cv2.cvtColor(image[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV)
            m = cv2.inRange(hsv, (0, 65, 45), (32, 255, 255))
            orange = float(np.mean(m > 0))
        out.append({"bbox": [x1, y1, x2, y2], "center": center,
                    "confidence": float(conf[i]), "orange_fraction": orange})
    out.sort(key=lambda x: -(x["confidence"] + 0.15 * x["orange_fraction"]))
    return out[:6]


def inside_image(uv):
    return uv is not None and 0 <= float(uv[0]) < W and 0 <= float(uv[1]) < H


def box_contains(box, uv, pad=18.0):
    x1, y1, x2, y2 = map(float, box)
    return (x1 - pad) <= uv[0] <= (x2 + pad) and (y1 - pad) <= uv[1] <= (y2 + pad)


def camera_for_state(scene, cam_states, rels):
    s = copy.deepcopy(scene)
    for c in CAMS:
        s["cameras"][c] = cam_states[c][int(rels[c])]
    return s, {c: v32j.cam(s, c) for c in CAMS}


def body_fit(scene_state, obs_rows):
    cams = {c: v32j.cam(scene_state, c) for c in CAMS}
    joints, residuals, observations = {}, [], {}
    for j in BODY:
        o = {c: np.asarray(obs_rows[c]["xy"][j], float) for c in CAMS
             if np.asarray(obs_rows[c]["conf"], float)[j] >= MIN_CONF}
        observations[j] = o
        if len(o) < 2:
            continue
        X = v32j.triangulate_rays(cams, o)
        if X is None or not (-350 <= X[0] <= 1250 and -750 <= X[1] <= 750 and -60 <= X[2] <= 450):
            continue
        joints[j] = X
        for c, uv0 in o.items():
            uv = v32j.project(cams[c], X)
            if uv is not None:
                residuals.append(float(np.linalg.norm(uv - uv0)))
    a = np.asarray(residuals, float)
    repro = {"joints": len(joints), "observations": len(residuals),
             "median_px": float(np.median(a)) if len(a) else 999.0,
             "p90_px": float(np.percentile(a, 90)) if len(a) else 999.0}
    repro["gate"] = bool(repro["joints"] >= 7 and repro["median_px"] <= 12.0 and repro["p90_px"] <= 24.0)
    bones, nb, nok, bfrac = v32j.bone_stats(joints)
    anatomy_gate = bool(nb >= 5 and bfrac >= 0.70)
    loo = []
    for j, o in observations.items():
        if len(o) != 3:
            continue
        for held in CAMS:
            train = {c: uv for c, uv in o.items() if c != held}
            X = v32j.triangulate_rays(cams, train)
            if X is None:
                continue
            uv = v32j.project(cams[held], X)
            if uv is not None:
                loo.append({"joint": int(j), "heldout": held,
                            "error_px": float(np.linalg.norm(uv - o[held]))})
    le = np.asarray([x["error_px"] for x in loo], float)
    looq = {"observations": len(loo),
            "median_px": float(np.median(le)) if len(le) else 999.0,
            "p90_px": float(np.percentile(le, 90)) if len(le) else 999.0,
            "per_observation": loo}
    looq["gate"] = bool(looq["observations"] >= 12 and looq["median_px"] <= 18.0 and looq["p90_px"] <= 35.0)
    gate = bool(repro["gate"] and anatomy_gate and looq["gate"])
    return {"gate": gate, "reprojection": repro,
            "anatomy": {"measured_bones": int(nb), "plausible_bones": int(nok),
                        "plausible_fraction": float(bfrac), "gate": anatomy_gate, "details": bones},
            "leave_one_camera_out": looq,
            "joints_world_cm": {v32j.NAMES[j]: joints[j].tolist() for j in joints},
            "_joints": joints}


def ball_hypothesis(cams, rows, candidates):
    Fs = {(a, b): v32j.fundamental(cams[a], cams[b]) for a in CAMS for b in CAMS if a != b}
    hyps = []
    for ia in range(len(CAMS)):
        for ib in range(ia + 1, len(CAMS)):
            a, b = CAMS[ia], CAMS[ib]
            if not candidates[a] or not candidates[b]:
                continue
            for ca in candidates[a]:
                for cb in candidates[b]:
                    epi = float(v32j.epi(Fs[(a, b)], ca["center"], cb["center"]))
                    if epi > 22.0:
                        continue
                    X = v32j.triangulate_rays(cams, {a: ca["center"], b: cb["center"]})
                    if X is None or not np.all(np.isfinite(X)):
                        continue
                    rim_dist = float(np.linalg.norm(X - RIM_WORLD_CM))
                    if rim_dist > BALL_RIM_DISTANCE_MAX_CM or not (120.0 <= X[2] <= 430.0):
                        continue
                    projected = {c: v32j.project(cams[c], X) for c in CAMS}
                    if not all(inside_image(projected[c]) for c in CAMS):
                        continue
                    matched = {}
                    for c in CAMS:
                        if candidates[c]:
                            nearest = min(candidates[c], key=lambda z: float(np.linalg.norm(z["center"] - projected[c])))
                            dist = float(np.linalg.norm(nearest["center"] - projected[c]))
                            if dist <= BALL_MATCH_MAX_PX:
                                matched[c] = (nearest, dist)
                    if len(matched) == 3:
                        X2 = v32j.triangulate_rays(cams, {c: matched[c][0]["center"] for c in CAMS})
                        if X2 is not None:
                            X = X2
                            rim_dist = float(np.linalg.norm(X - RIM_WORLD_CM))
                            projected = {c: v32j.project(cams[c], X) for c in CAMS}
                            matched2 = {}
                            for c in CAMS:
                                nearest = min(candidates[c], key=lambda z: float(np.linalg.norm(z["center"] - projected[c])))
                                dist = float(np.linalg.norm(nearest["center"] - projected[c]))
                                if dist <= BALL_MATCH_MAX_PX:
                                    matched2[c] = (nearest, dist)
                            matched = matched2
                    occluded = {}
                    for c in CAMS:
                        if c in matched:
                            occluded[c] = False
                            continue
                        uv = projected[c]
                        rim_uv = v32j.project(cams[c], RIM_WORLD_CM)
                        near_rim = rim_uv is not None and float(np.linalg.norm(uv - rim_uv)) <= 28.0
                        player_occ = box_contains(rows[c]["box"], uv, 16.0)
                        occluded[c] = bool(near_rim or player_occ)
                    support = len(matched)
                    hidden = [c for c in CAMS if c not in matched and occluded[c]]
                    pass_gate = bool(support >= 2 and rim_dist <= BALL_RIM_DISTANCE_MAX_CM and
                                     (support == 3 or (support == 2 and len(hidden) == 1)))
                    residuals = [v[1] for v in matched.values()]
                    med = float(np.median(residuals)) if residuals else 999.0
                    mx = float(max(residuals)) if residuals else 999.0
                    confsum = float(sum(v[0]["confidence"] for v in matched.values()))
                    score = med + 0.20 * mx + 0.03 * rim_dist + 12.0 * (3 - support) - 3.0 * confsum
                    hyps.append({"gate": pass_gate, "score": float(score),
                                 "triangulation_pair": [a, b], "pair_epipolar_px": epi,
                                 "world_cm": X.tolist(), "distance_to_rim_center_cm": rim_dist,
                                 "support_views": list(matched.keys()), "support_count": support,
                                 "occluded_views": hidden,
                                 "matched": {c: {"bbox": matched[c][0]["bbox"],
                                                 "center": matched[c][0]["center"].tolist(),
                                                 "confidence": matched[c][0]["confidence"],
                                                 "orange_fraction": matched[c][0]["orange_fraction"],
                                                 "reprojection_error_px": matched[c][1]} for c in matched},
                                 "projected_centers": {c: projected[c].tolist() for c in CAMS},
                                 "median_supported_reprojection_px": med,
                                 "max_supported_reprojection_px": mx})
    hyps.sort(key=lambda x: (not x["gate"], x["score"]))
    return (hyps[0] if hyps else None), hyps[:20]


def draw_overlay(image, row, joints, cams, label, ball_candidates, ball_best):
    out = image.copy()
    valid = np.asarray(row["conf"], float) >= MIN_CONF
    for a, b in v32j.DRAW:
        if valid[a] and valid[b]:
            cv2.line(out, tuple(np.rint(row["xy"][a]).astype(int)), tuple(np.rint(row["xy"][b]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    for j in BODY:
        if valid[j]:
            cv2.circle(out, tuple(np.rint(row["xy"][j]).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
    for a, b in v32j.DRAW:
        if a in joints and b in joints:
            ua, ub = v32j.project(cams[label], joints[a]), v32j.project(cams[label], joints[b])
            if ua is not None and ub is not None:
                cv2.line(out, tuple(np.rint(ua).astype(int)), tuple(np.rint(ub).astype(int)), (255, 0, 255), 1, cv2.LINE_AA)
    for X in joints.values():
        uv = v32j.project(cams[label], X)
        if uv is not None:
            cv2.circle(out, tuple(np.rint(uv).astype(int)), 3, (255, 0, 255), 1, cv2.LINE_AA)
    for c in ball_candidates:
        x1, y1, x2, y2 = np.rint(c["bbox"]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 1, cv2.LINE_AA)
    if ball_best is not None:
        uv = np.asarray(ball_best["projected_centers"][label], float)
        cv2.circle(out, tuple(np.rint(uv).astype(int)), 8, (255, 255, 0), 2, cv2.LINE_AA)
        state = "OBSERVED" if label in ball_best["support_views"] else ("OCCLUDED" if label in ball_best["occluded_views"] else "UNVERIFIED")
        cv2.putText(out, f"ball {state}", (max(0, int(uv[0]) + 10), max(18, int(uv[1]) - 8)), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 30), (0, 0, 0), -1)
    cv2.putText(out, f"v33f {label} | yellow source pose | magenta 3D | green ball detections | cyan ball reproj", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--v73-frame0257", type=Path, required=True)
    ap.add_argument("--v33e-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    accepted = json.loads(args.v33e_json.read_text())
    if accepted.get("status") != "PASS_V33E_EXACT_THREE_VIEW_STATE" or accepted.get("camera_lock") != [LAR, RAR, BCAST]:
        raise RuntimeError("v33f requires accepted locked-three-camera v33e evidence")
    exact_rows = [x for x in accepted.get("top_triplets", []) if x.get("exact_state")]
    if not exact_rows:
        raise RuntimeError("v33f v33e artifact contains no exact top states")

    rels = tuple(range(-20, 21))
    v33d.RELS = rels
    stage = args.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    if scene.get("resolution") != [W, H] or set(scene.get("cameras", {})) != {LAR, BCAST, RAR}:
        raise RuntimeError("v33f locked camera/source mismatch")
    centers = {k: int(v) for k, v in scene["freeze"]["chosen_frame_indices"].items()}
    wide_stage = args.out / "wide_stage"
    wide_stage.mkdir(parents=True, exist_ok=True)
    source_audit = v33e.export_wide_burst(args.clips_dir, centers, wide_stage, rels)
    fs = {c: v33d.frames(wide_stage, c) for c in CAMS}

    cert = cv2.imread(str(args.v73_frame0257), cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape != (H, W):
        raise RuntimeError("v33f missing v73 accepted RAR certificate frame")
    cam_states, transfer_audit = {c: {} for c in CAMS}, {c: {} for c in CAMS}
    for c in (LAR, BCAST):
        for rel in rels:
            cc, qa = v33d.transfer(fs[c][0][2], fs[c][rel][2], scene["cameras"][c])
            transfer_audit[c][str(rel)] = qa
            if cc is not None:
                cam_states[c][rel] = cc
    for rel in rels:
        cc, qa = v33b.transfer_rar_camera(cert, fs[RAR][rel][2], scene["cameras"][RAR])
        transfer_audit[RAR][str(rel)] = qa
        if cc is not None:
            cam_states[RAR][rel] = cc

    keypoint_model = RFDETRKeypointPreview()
    identity, identity_audit = {}, {}
    for c in CAMS:
        identity[c], identity_audit[c] = v33d.track(keypoint_model, c, fs[c])
    obs, obs_audit = {}, {}
    for c in CAMS:
        obs[c], obs_audit[c] = v33e.flow_observations(fs[c], identity[c], c, rels)

    needed = {c: sorted({int(x["rels"][c]) for x in exact_rows}) for c in CAMS}
    ball_model = RFDETRMedium()
    ball = {c: {} for c in CAMS}
    for c in CAMS:
        for rel in needed[c]:
            ball[c][rel] = detect_sports_balls(ball_model, fs[c][rel][1])

    state_qa, chosen, chosen_body, chosen_ball, chosen_cams, chosen_rows = [], None, None, None, None, None
    for rank, x in enumerate(exact_rows, start=1):
        relmap = {c: int(x["rels"][c]) for c in CAMS}
        if any(relmap[c] not in cam_states[c] for c in CAMS):
            state_qa.append({"rank": rank, "rels": relmap, "status": "STATIC_TRANSFER_NOT_AVAILABLE"})
            continue
        s, cams = camera_for_state(scene, cam_states, relmap)
        rows = {c: obs[c][relmap[c]] for c in CAMS}
        candidates = {c: ball[c].get(relmap[c], []) for c in CAMS}
        bb, _ = ball_hypothesis(cams, rows, candidates)
        bfit = body_fit(s, rows)
        ball_gate = bool(bb and bb["gate"])
        gate = bool(ball_gate and bfit["gate"])
        state_qa.append({"rank": rank, "rels": relmap, "frames": x["frames"], "v33e_score": float(x["score"]),
                         "source_ball_detection_counts": {c: len(candidates[c]) for c in CAMS},
                         "body_gate": bool(bfit["gate"]), "body_reprojection": bfit["reprojection"],
                         "body_anatomy": {k: v for k, v in bfit["anatomy"].items() if k != "details"},
                         "body_leave_one_camera_out": {k: v for k, v in bfit["leave_one_camera_out"].items() if k != "per_observation"},
                         "ball_gate": ball_gate, "best_ball": bb, "gate": gate})
        if gate:
            chosen, chosen_body, chosen_ball, chosen_cams, chosen_rows = x, bfit, bb, cams, rows
            break

    passed = chosen is not None
    if passed:
        relmap = {c: int(chosen["rels"][c]) for c in CAMS}
        overlays = []
        for c in CAMS:
            ov = draw_overlay(fs[c][relmap[c]][1], chosen_rows[c], chosen_body["_joints"], chosen_cams, c, ball[c].get(relmap[c], []), chosen_ball)
            cv2.imwrite(str(args.out / f"v33f_{c.replace(' ', '_')}_body_ball_qa.png"), ov)
            overlays.append(ov)
            cv2.imwrite(str(args.out / f"v33f_source_{c.replace(' ', '_')}.png"), fs[c][relmap[c]][1])
        cv2.imwrite(str(args.out / "v33f_three_camera_body_ball_montage.png"), np.hstack(overlays))

    body_json = None if chosen_body is None else {k: v for k, v in chosen_body.items() if not k.startswith("_")}
    qa = {"version": "v33f_source_grounded_three_view_body_ball_qa",
          "status": "PASS_V33F_BODY_BALL_VISUAL_QA" if passed else "FAIL_CLOSED_V33F_NO_SOURCE_GROUNDED_BODY_BALL_STATE",
          "upstream_v33e_status": accepted.get("status"), "camera_lock": [LAR, RAR, BCAST], "camera_count": 3,
          "camera_addition_or_substitution_used": False, "physical_center_refit_from_player_or_ball": False,
          "native_resolution": [W, H], "generated_rgb": False, "upscaled": False, "novel_view_rendered": False,
          "source_audit": source_audit, "top_exact_states_considered": len(exact_rows),
          "ball_model_role": "RF-DETR COCO object detector; source observation only; only class 'sports ball' accepted",
          "player_model_role": "RF-DETR Keypoint Preview; same source-local identity/flow observation policy as v33e",
          "ball_gate_policy": {"minimum_source_supported_views": 2,
                               "third_view_rule": "third source detection within 18 px OR explicit occlusion by locked Adams bbox/rim projection",
                               "max_supported_center_reprojection_px": BALL_MATCH_MAX_PX,
                               "max_world_distance_from_rim_center_cm": BALL_RIM_DISTANCE_MAX_CM,
                               "no_geometry_created_ball_candidate": True},
          "body_gate_policy": {"same_v33d_reprojection_gate": {"min_joints": 7, "median_px_max": 12.0, "p90_px_max": 24.0},
                               "anatomy_gate": {"min_measured_bones": 5, "plausible_fraction_min": 0.70},
                               "leave_one_camera_out_gate": {"min_observations": 12, "median_px_max": 18.0, "p90_px_max": 35.0}},
          "chosen_state": None if not passed else {"rels": {c: int(chosen["rels"][c]) for c in CAMS},
                                                     "frames": chosen["frames"], "v33e_score": float(chosen["score"]),
                                                     "body": body_json, "ball": chosen_ball},
          "state_diagnostics": state_qa, "freeview_render_unlocked": bool(passed),
          "next_if_pass": "first native 960x540 source-grounded free-view render using this locked three-camera/body/ball state; preserve visual QA and fail closed on artifacts",
          "next_if_fail": "stay on LAR+RAR+Broadcast; inspect source ball observability across additional v33e exact states or widen source-local ball detector support without changing cameras or geometry gates"}
    (args.out / "v33f_body_ball_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({"status": qa["status"], "top_exact_states_considered": qa["top_exact_states_considered"],
                      "chosen_state": qa["chosen_state"], "freeview_render_unlocked": qa["freeview_render_unlocked"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
