from __future__ import annotations

"""v33g: harden v33f ball QA with source-grounded appearance + temporal motion.

The accepted v33e three-camera exact-state/camera solution is immutable here.
No camera centres are refit and no fourth camera is introduced. The body gate
is the v33f/v33e gate. The only change is basketball localization:

* RF-DETR sports-ball observations are retained, but cannot pass by class label
  + epipolar geometry alone.
* Saturated compact orange source components are also admitted so a clearly
  visible basketball missed by RF-DETR can still be used.
* Every accepted exact-frame candidate must be supported by real neighboring
  source frames and move relative to the projected physical rim. This rejects
  static rim/net false positives while leaving the evaluated exact state as one
  real frame per camera.
* At least one of the two (or three) metric support views must still have an
  exact-frame semantic sports-ball detection.

Neighbor frames are localization evidence only. They are never mixed into the
triangulated exact state and are never rendered.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview, RFDETRMedium

from freeze_spin import run_v33e_locked_three_camera_wide_flow_state_search as v33e
from freeze_spin import run_v33d_locked_three_camera_joint_state_search as v33d
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import run_v33f_source_grounded_body_ball_qa as v33f
from freeze_spin.select_jazz_ball_apex_multiview_v3 import orange_components

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
CAMS = (LAR, RAR, BCAST)
W, H = v32j.W, v32j.H
RIM_WORLD_CM = np.array([38.1, 0.0, 304.8], float)
BALL_MATCH_MAX_PX = 18.0
BALL_RIM_DISTANCE_MAX_CM = 200.0
TEMPORAL_OFFSETS = (-3, -2, -1, 1, 2, 3)


def _shape_features(image: np.ndarray, bbox: list[float]) -> dict:
    x1, y1, x2, y2 = map(float, bbox)
    xi1, yi1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    xi2, yi2 = min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return {"orange_fraction": 0.0, "largest_orange_area": 0.0,
                "largest_orange_circularity": 0.0, "largest_orange_fill": 0.0,
                "mean_saturation": 0.0}
    roi = image[yi1:yi2, xi1:xi2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([1, 58, 42], np.uint8), np.array([34, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = (0.0, 0.0, 0.0)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        x, y, w, h = cv2.boundingRect(cnt)
        circ = float(4.0 * math.pi * area / max(peri * peri, 1e-6))
        fill = float(area / max(w * h, 1))
        if area > best[0]:
            best = (area, circ, fill)
    return {"orange_fraction": float(np.mean(mask > 0)),
            "largest_orange_area": best[0],
            "largest_orange_circularity": best[1],
            "largest_orange_fill": best[2],
            "mean_saturation": float(np.mean(hsv[..., 1])) if hsv.size else 0.0}


def _camera_for_rel(scene: dict, cam_states: dict, cam: str, rel: int):
    s = copy.deepcopy(scene)
    s["cameras"][cam] = cam_states[cam][rel]
    return v32j.cam(s, cam)


def build_frame_candidates(model, image: np.ndarray, rim_uv: np.ndarray) -> list[dict]:
    semantic = v33f.detect_sports_balls(model, image)
    rows: list[dict] = []
    for c in semantic:
        q = dict(c)
        q["center"] = np.asarray(q["center"], float)
        q["source"] = "semantic"
        q["shape"] = _shape_features(image, q["bbox"])
        rows.append(q)

    compact, _ = orange_components(image)
    for c in compact:
        if float(c.get("area", 0.0)) < 25.0:
            continue
        center = np.array([float(c["cx"]), float(c["cy"])], float)
        if any(float(np.linalg.norm(center - q["center"])) <= 14.0 for q in rows):
            continue
        x, y, w, h = float(c["x"]), float(c["y"]), float(c["w"]), float(c["h"])
        rows.append({"bbox": [x, y, x + w, y + h], "center": center,
                     "confidence": 0.0, "orange_fraction": float(c.get("fill", 0.0)),
                     "source": "orange_track",
                     "shape": {"orange_fraction": float(c.get("fill", 0.0)),
                               "largest_orange_area": float(c.get("area", 0.0)),
                               "largest_orange_circularity": float(c.get("circularity", 0.0)),
                               "largest_orange_fill": float(c.get("fill", 0.0)),
                               "mean_saturation": float(c.get("mean_saturation", 0.0)),
                               "ball_color_score": float(c.get("ball_color_score", 0.0))}})

    out = []
    for q in rows:
        dx = abs(float(q["center"][0] - rim_uv[0]))
        dy = float(q["center"][1] - rim_uv[1])
        if dx > 285.0 or dy < -305.0 or dy > 185.0:
            continue
        sh = q["shape"]
        if q["source"] == "semantic":
            appearance = bool(sh["orange_fraction"] >= 0.10 and
                              sh["largest_orange_area"] >= 18.0 and
                              sh["largest_orange_circularity"] >= 0.16)
            appearance_strength = (2.2 * sh["orange_fraction"] +
                                   0.55 * sh["largest_orange_circularity"] +
                                   0.35 * min(sh["largest_orange_area"] / 160.0, 2.0) +
                                   0.35 * float(q["confidence"]))
        else:
            appearance = bool(sh["largest_orange_area"] >= 45.0 and
                              sh["largest_orange_circularity"] >= 0.20 and
                              sh["largest_orange_fill"] >= 0.30 and
                              sh["mean_saturation"] >= 85.0)
            appearance_strength = (0.60 * float(sh.get("ball_color_score", 0.0)) +
                                   0.45 * sh["largest_orange_circularity"] +
                                   0.30 * min(sh["largest_orange_area"] / 160.0, 2.0))
        q["basket_relative_xy"] = (q["center"] - rim_uv).tolist()
        q["appearance_gate"] = appearance
        q["appearance_strength"] = float(appearance_strength)
        out.append(q)
    out.sort(key=lambda z: (not z["appearance_gate"], -z["appearance_strength"], -z["confidence"]))
    return out[:18]


def add_temporal_evidence(cache: dict[int, list[dict]], rel: int, candidate: dict) -> dict:
    q = dict(candidate)
    p0 = np.asarray(q["basket_relative_xy"], float)
    matches = []
    for d in TEMPORAL_OFFSETS:
        rr = rel + d
        if rr not in cache:
            continue
        pool = [x for x in cache[rr] if x.get("appearance_gate")]
        if not pool:
            continue
        nearest = min(pool, key=lambda x: float(np.linalg.norm(np.asarray(x["basket_relative_xy"], float) - p0)))
        dist = float(np.linalg.norm(np.asarray(nearest["basket_relative_xy"], float) - p0))
        if dist <= 82.0:
            matches.append({"offset": int(d), "distance_from_exact_px": dist,
                            "source": nearest["source"], "center": nearest["center"].tolist(),
                            "basket_relative_xy": nearest["basket_relative_xy"],
                            "appearance_strength": float(nearest["appearance_strength"])})
    dynamic = [x for x in matches if x["distance_from_exact_px"] >= 5.0]
    static = [x for x in matches if x["distance_from_exact_px"] <= 4.0]
    pts = [p0] + [np.asarray(x["basket_relative_xy"], float) for x in matches]
    span = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            span = max(span, float(np.linalg.norm(pts[i] - pts[j])))
    support_offsets = sorted(x["offset"] for x in matches)
    has_pre = any(x < 0 for x in support_offsets)
    has_post = any(x > 0 for x in support_offsets)
    temporal_gate = bool(q.get("appearance_gate") and len(matches) >= 2 and
                         len(dynamic) >= 2 and span >= 10.0 and
                         len(static) <= max(1, len(matches) // 2) and
                         (has_pre or has_post))
    q["temporal"] = {"matched_neighbor_frames": len(matches),
                     "dynamic_neighbor_frames": len(dynamic),
                     "static_neighbor_frames": len(static),
                     "trajectory_span_rim_compensated_px": float(span),
                     "support_offsets": support_offsets,
                     "has_pre_support": bool(has_pre), "has_post_support": bool(has_post),
                     "matches": matches, "gate": temporal_gate}
    q["temporal_gate"] = temporal_gate
    q["quality"] = float(q["appearance_strength"] + 0.035 * min(span, 80.0) +
                         (0.55 if q["source"] == "semantic" else 0.0))
    return q


def ball_hypothesis(cams: dict, rows: dict, candidates: dict[str, list[dict]]):
    Fs = {(a, b): v32j.fundamental(cams[a], cams[b]) for a in CAMS for b in CAMS if a != b}
    hyps = []
    for ia in range(len(CAMS)):
        for ib in range(ia + 1, len(CAMS)):
            a, b = CAMS[ia], CAMS[ib]
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
                    if not all(v33f.inside_image(projected[c]) for c in CAMS):
                        continue
                    matched = {}
                    for c in CAMS:
                        if not candidates[c]:
                            continue
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
                            rematched = {}
                            for c in CAMS:
                                nearest = min(candidates[c], key=lambda z: float(np.linalg.norm(z["center"] - projected[c])))
                                dist = float(np.linalg.norm(nearest["center"] - projected[c]))
                                if dist <= BALL_MATCH_MAX_PX:
                                    rematched[c] = (nearest, dist)
                            matched = rematched
                    occluded = {}
                    for c in CAMS:
                        if c in matched:
                            occluded[c] = False
                            continue
                        uv = projected[c]
                        rim_uv = v32j.project(cams[c], RIM_WORLD_CM)
                        near_rim = rim_uv is not None and float(np.linalg.norm(uv - rim_uv)) <= 28.0
                        player_occ = v33f.box_contains(rows[c]["box"], uv, 16.0)
                        occluded[c] = bool(near_rim or player_occ)
                    support = len(matched)
                    hidden = [c for c in CAMS if c not in matched and occluded[c]]
                    source_types = [v[0]["source"] for v in matched.values()]
                    semantic_support = sum(x == "semantic" for x in source_types)
                    pass_gate = bool(support >= 2 and semantic_support >= 1 and
                                     rim_dist <= BALL_RIM_DISTANCE_MAX_CM and
                                     (support == 3 or (support == 2 and len(hidden) == 1)))
                    residuals = [v[1] for v in matched.values()]
                    med = float(np.median(residuals)) if residuals else 999.0
                    mx = float(max(residuals)) if residuals else 999.0
                    quality = float(sum(v[0].get("quality", 0.0) for v in matched.values()))
                    score = med + 0.20 * mx + 0.03 * rim_dist + 11.0 * (3 - support) - 2.2 * quality
                    hyps.append({"gate": pass_gate, "score": float(score),
                                 "triangulation_pair": [a, b], "pair_epipolar_px": epi,
                                 "world_cm": X.tolist(), "distance_to_rim_center_cm": rim_dist,
                                 "support_views": list(matched.keys()), "support_count": support,
                                 "semantic_support_count": semantic_support,
                                 "support_sources": {c: matched[c][0]["source"] for c in matched},
                                 "occluded_views": hidden,
                                 "matched": {c: {"bbox": matched[c][0]["bbox"],
                                                 "center": matched[c][0]["center"].tolist(),
                                                 "source": matched[c][0]["source"],
                                                 "confidence": float(matched[c][0]["confidence"]),
                                                 "orange_fraction": float(matched[c][0].get("orange_fraction", 0.0)),
                                                 "appearance_strength": float(matched[c][0].get("appearance_strength", 0.0)),
                                                 "temporal": matched[c][0].get("temporal", {}),
                                                 "reprojection_error_px": matched[c][1]} for c in matched},
                                 "projected_centers": {c: projected[c].tolist() for c in CAMS},
                                 "median_supported_reprojection_px": med,
                                 "max_supported_reprojection_px": mx,
                                 "source_quality_sum": quality})
    hyps.sort(key=lambda x: (not x["gate"], x["score"]))
    return (hyps[0] if hyps else None), hyps[:30]


def draw_overlay(image, row, joints, cams, label, candidates, ball_best):
    out = v33f.draw_overlay(image, row, joints, cams, label, candidates, ball_best)
    cv2.rectangle(out, (0, 0), (W, 31), (0, 0, 0), -1)
    cv2.putText(out, f"v33g {label} | exact source frame | green=temporal ball candidates | cyan=3D reproj",
                (7, 20), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    if ball_best is not None and label in ball_best.get("matched", {}):
        m = ball_best["matched"][label]
        uv = np.rint(np.asarray(m["center"], float)).astype(int)
        cv2.putText(out, f"{m['source']} q={m['appearance_strength']:.2f}",
                    (max(4, int(uv[0]) + 10), min(H - 8, max(45, int(uv[1]) + 24))),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def serial_candidate(c: dict) -> dict:
    out = {k: v for k, v in c.items() if k not in ("center",)}
    out["center"] = np.asarray(c["center"], float).tolist()
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
        raise RuntimeError("v33g requires accepted locked-three-camera v33e evidence")
    exact_rows = [x for x in accepted.get("top_triplets", []) if x.get("exact_state")]
    if not exact_rows:
        raise RuntimeError("v33g v33e artifact contains no exact top states")

    rels = tuple(range(-20, 21))
    v33d.RELS = rels
    stage = args.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    if scene.get("resolution") != [W, H] or set(scene.get("cameras", {})) != {LAR, BCAST, RAR}:
        raise RuntimeError("v33g locked camera/source mismatch")
    centers = {k: int(v) for k, v in scene["freeze"]["chosen_frame_indices"].items()}
    wide_stage = args.out / "wide_stage"
    wide_stage.mkdir(parents=True, exist_ok=True)
    source_audit = v33e.export_wide_burst(args.clips_dir, centers, wide_stage, rels)
    fs = {c: v33d.frames(wide_stage, c) for c in CAMS}

    cert = cv2.imread(str(args.v73_frame0257), cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape != (H, W):
        raise RuntimeError("v33g missing v73 accepted RAR certificate frame")
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

    exact_needed = {c: sorted({int(x["rels"][c]) for x in exact_rows}) for c in CAMS}
    temporal_needed = {}
    for c in CAMS:
        s = set()
        for r in exact_needed[c]:
            for d in (0,) + TEMPORAL_OFFSETS:
                rr = r + d
                if rr in fs[c] and rr in cam_states[c]:
                    s.add(rr)
        temporal_needed[c] = sorted(s)

    ball_model = RFDETRMedium()
    candidate_cache: dict[str, dict[int, list[dict]]] = {c: {} for c in CAMS}
    for c in CAMS:
        for rel in temporal_needed[c]:
            cam = _camera_for_rel(scene, cam_states, c, rel)
            rim_uv = v32j.project(cam, RIM_WORLD_CM)
            if rim_uv is None:
                candidate_cache[c][rel] = []
                continue
            candidate_cache[c][rel] = build_frame_candidates(ball_model, fs[c][rel][1], rim_uv)

    validated: dict[str, dict[int, list[dict]]] = {c: {} for c in CAMS}
    for c in CAMS:
        for rel in exact_needed[c]:
            pool = [add_temporal_evidence(candidate_cache[c], rel, q)
                    for q in candidate_cache[c].get(rel, [])]
            pool = [q for q in pool if q.get("temporal_gate")]
            pool.sort(key=lambda q: (-q.get("quality", 0.0), -q.get("confidence", 0.0)))
            validated[c][rel] = pool[:10]

    state_qa = []
    passing_states = []
    for rank, x in enumerate(exact_rows, start=1):
        relmap = {c: int(x["rels"][c]) for c in CAMS}
        if any(relmap[c] not in cam_states[c] for c in CAMS):
            state_qa.append({"rank": rank, "rels": relmap, "status": "STATIC_TRANSFER_NOT_AVAILABLE"})
            continue
        s, cams = v33f.camera_for_state(scene, cam_states, relmap)
        rows = {c: obs[c][relmap[c]] for c in CAMS}
        candidates = {c: validated[c].get(relmap[c], []) for c in CAMS}
        bb, top_hyps = ball_hypothesis(cams, rows, candidates)
        bfit = v33f.body_fit(s, rows)
        ball_gate = bool(bb and bb["gate"])
        gate = bool(ball_gate and bfit["gate"])
        diag = {"rank": rank, "rels": relmap, "frames": x["frames"], "v33e_score": float(x["score"]),
                "raw_candidate_counts": {c: len(candidate_cache[c].get(relmap[c], [])) for c in CAMS},
                "temporal_ball_candidate_counts": {c: len(candidates[c]) for c in CAMS},
                "temporal_ball_candidates": {c: [serial_candidate(z) for z in candidates[c][:5]] for c in CAMS},
                "body_gate": bool(bfit["gate"]), "body_reprojection": bfit["reprojection"],
                "body_anatomy": {k: v for k, v in bfit["anatomy"].items() if k != "details"},
                "body_leave_one_camera_out": {k: v for k, v in bfit["leave_one_camera_out"].items() if k != "per_observation"},
                "ball_gate": ball_gate, "best_ball": bb, "top_ball_hypotheses": top_hyps[:5], "gate": gate}
        state_qa.append(diag)
        if gate:
            passing_states.append((rank, x, bfit, bb, cams, rows, candidates))

    chosen_tuple = min(passing_states, key=lambda t: (t[0], t[3]["score"])) if passing_states else None
    passed = chosen_tuple is not None
    chosen = chosen_body = chosen_ball = chosen_cams = chosen_rows = chosen_candidates = None
    if passed:
        _, chosen, chosen_body, chosen_ball, chosen_cams, chosen_rows, chosen_candidates = chosen_tuple
        relmap = {c: int(chosen["rels"][c]) for c in CAMS}
        overlays = []
        for c in CAMS:
            ov = draw_overlay(fs[c][relmap[c]][1], chosen_rows[c], chosen_body["_joints"],
                              chosen_cams, c, chosen_candidates[c], chosen_ball)
            cv2.imwrite(str(args.out / f"v33g_{c.replace(' ', '_')}_body_ball_qa.png"), ov)
            overlays.append(ov)
            cv2.imwrite(str(args.out / f"v33g_source_{c.replace(' ', '_')}.png"), fs[c][relmap[c]][1])
        cv2.imwrite(str(args.out / "v33g_three_camera_body_ball_montage.png"), np.hstack(overlays))

    body_json = None if chosen_body is None else {k: v for k, v in chosen_body.items() if not k.startswith("_")}
    qa = {"version": "v33g_temporal_source_grounded_three_view_body_ball_qa",
          "status": ("PASS_V33G_NUMERICAL_AWAITING_EXTERNAL_VISUAL_QA" if passed else
                     "FAIL_CLOSED_V33G_NO_VALIDATED_SOURCE_BALL_STATE"),
          "upstream_v33e_status": accepted.get("status"),
          "camera_lock": [LAR, RAR, BCAST], "camera_count": 3,
          "camera_addition_or_substitution_used": False,
          "physical_center_refit_from_player_or_ball": False, "exact_state_reopened": False,
          "native_resolution": [W, H], "generated_rgb": False, "upscaled": False,
          "novel_view_rendered": False, "source_audit": source_audit,
          "top_exact_states_considered": len(exact_rows),
          "ball_gate_policy": {"exact_frame_sources": ["RF-DETR sports-ball", "compact saturated orange source component"],
                               "temporal_evidence_only": True, "temporal_offsets": list(TEMPORAL_OFFSETS),
                               "motion_reference": "candidate position relative to projected physical rim for the same locked camera state",
                               "minimum_source_supported_views": 2, "minimum_exact_semantic_support_views": 1,
                               "max_supported_center_reprojection_px": BALL_MATCH_MAX_PX,
                               "max_world_distance_from_rim_center_cm": BALL_RIM_DISTANCE_MAX_CM,
                               "no_neighbor_frame_used_as_exact_state_observation": True,
                               "no_geometry_created_ball_candidate": True},
          "body_gate_policy": "unchanged v33f/v33e body reprojection + anatomy + leave-one-camera-out gates",
          "chosen_state": None if not passed else {"rels": {c: int(chosen["rels"][c]) for c in CAMS},
                                                     "frames": chosen["frames"], "v33e_score": float(chosen["score"]),
                                                     "body": body_json, "ball": chosen_ball},
          "state_diagnostics": state_qa,
          "numerical_body_ball_gate_passed": bool(passed), "external_visual_qa_required": True,
          "freeview_render_unlocked": False,
          "render_lock_reason": ("await external visual QA of v33g source/overlay montage" if passed else
                                 "no source-grounded body+ball exact state passed"),
          "next_if_numerical_pass": "inspect source pixels and overlays; unlock render only if selected ball is the real basketball in every observed support view",
          "next_if_fail": "stay on LAR+RAR+Broadcast and accepted camera centres; diagnose ball observability/localization within accepted v33e exact states"}
    (args.out / "v33g_body_ball_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({"status": qa["status"], "top_exact_states_considered": qa["top_exact_states_considered"],
                      "chosen_state": qa["chosen_state"],
                      "numerical_body_ball_gate_passed": qa["numerical_body_ball_gate_passed"],
                      "freeview_render_unlocked": qa["freeview_render_unlocked"],
                      "render_lock_reason": qa["render_lock_reason"]}, indent=2), flush=True)
    if not passed:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
