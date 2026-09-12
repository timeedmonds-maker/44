from __future__ import annotations

"""v33b: identify the exact Right-Above-Rim dynamic state against an
RAR-independent LAR+Broadcast reference pair.

v33a proved that the one-frame RAR camera-state lineage error is real and can be
repaired from static image structure, but the corrected LAR+RAR MHR body still
misses Broadcast.  Pairwise evidence indicates LAR and Broadcast agree much
better with each other than Broadcast and RAR.

This diagnostic therefore removes RAR from reference-pair selection:
1. build coherent Houston-dark LAR and Broadcast temporal candidates from only
   real B32 burst frames; later anchors are dense-flow tracked back to t+00;
2. choose the LAR+Broadcast pair by their own calibrated epipolar agreement,
   with the known Broadcast focal region used only as a weak identity prior;
3. for every real RAR frame rel-06..rel+06, transfer the accepted v73 frame0257
   metric camera to that exact frame using static pixels only;
4. detect coherent Houston-dark RAR bodies directly in that real frame;
5. rank every RAR person/frame candidate simultaneously against the fixed
   LAR+Broadcast reference pair;
6. pass only if one RAR candidate satisfies the existing v32v raw-state gate
   against BOTH independent reference views.

No camera fitting from players, no per-joint temporal mixing, no generated RGB,
no interpolation, no threshold weakening, no upscale, and no free-view render.
All source and QA frames remain native 960x540.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview

from freeze_spin import prepare_v33a_exact_rar_frame_camera as v33a
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF
LEFT_WRIST = 9
W, H = v32j.W, v32j.H
REF_RELS = (0, 2, 3, 4, 5, 6)
RAR_RELS = tuple(range(-6, 7))


def serialize_epi(e):
    return {
        "joint_count": int(e["joint_count"]),
        "median_px": float(e["median_px"]),
        "p75_px": float(e["p75_px"]),
        "p90_px": float(e["p90_px"]),
        "max_px": float(e["max_px"]),
        "per_joint": e.get("per_joint", []),
    }


def static_mask():
    mask = np.full((H, W), 255, np.uint8)
    mask[120:350, 300:680] = 0
    mask[0:180, 0:350] = 0
    mask[150:330, 0:180] = 0
    mask[360:540, 500:720] = 0
    return mask


def transfer_rar_camera(cert257_gray, target_gray, source_cam):
    """Static-only shared-centre transfer from certified frame0257 to target."""
    mask = static_mask()
    orb = cv2.ORB_create(nfeatures=6500, fastThreshold=10)
    kt, dt = orb.detectAndCompute(target_gray, mask)
    kc, dc = orb.detectAndCompute(cert257_gray, mask)
    if dt is None or dc is None:
        return None, {"status": "FAIL_DESCRIPTORS"}
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dt, dc, k=2)
    good = [m for m, n in knn if m.distance < 0.70 * n.distance]
    if len(good) < 300:
        return None, {"status": "FAIL_MATCH_COUNT", "ratio_test_matches": len(good)}
    pt = np.float32([kt[m.queryIdx].pt for m in good])
    pc = np.float32([kc[m.trainIdx].pt for m in good])
    H_t_to_c, inl = cv2.findHomography(pt, pc, cv2.RANSAC, 1.25, maxIters=12000, confidence=0.999)
    if H_t_to_c is None or inl is None:
        return None, {"status": "FAIL_HOMOGRAPHY", "ratio_test_matches": len(good)}
    keep = inl.ravel().astype(bool)
    pred = cv2.perspectiveTransform(pt.reshape(-1, 1, 2), H_t_to_c).reshape(-1, 2)
    err = np.linalg.norm(pred - pc, axis=1)[keep]
    stat = {
        "ratio_test_matches": int(len(good)),
        "ransac_inliers": int(np.sum(keep)),
        "inlier_fraction": float(np.mean(keep)),
        "median_px": float(np.median(err)) if len(err) else 999.0,
        "p90_px": float(np.percentile(err, 90)) if len(err) else 999.0,
        "p95_px": float(np.percentile(err, 95)) if len(err) else 999.0,
        "max_px": float(np.max(err)) if len(err) else 999.0,
    }
    stat_gate = bool(
        stat["ransac_inliers"] >= 500
        and stat["inlier_fraction"] >= 0.40
        and stat["median_px"] <= 0.80
        and stat["p95_px"] <= 1.50
        and stat["max_px"] <= 2.25
    )
    if not stat_gate:
        return None, {"status": "FAIL_STATIC_GATE", "registration": stat}

    H_c_to_t = np.linalg.inv(H_t_to_c)
    H_c_to_t /= H_c_to_t[2, 2]
    Pcert = v33a.camera_projection(source_cam)
    Ptarget = H_c_to_t @ Pcert
    K, R, C = v33a.decompose_projection(Ptarget)
    center_delta = float(np.linalg.norm(C - np.asarray(source_cam["C_world_cm"], float)))
    camera_gate = bool(
        center_delta <= 1e-6
        and 350.0 <= K[0, 0] <= 900.0
        and 350.0 <= K[1, 1] <= 900.0
        and 350.0 <= K[0, 2] <= 610.0
        and 150.0 <= K[1, 2] <= 380.0
    )
    if not camera_gate:
        return None, {
            "status": "FAIL_CAMERA_DECOMPOSITION", "registration": stat,
            "center_delta_cm": center_delta, "K": K.tolist(),
        }
    corrected = copy.deepcopy(source_cam)
    corrected["K_px"] = K.tolist()
    corrected["R_world_to_camera"] = R.tolist()
    corrected["C_world_cm"] = C.tolist()
    corrected["extrinsic_world_to_camera_3x4"] = np.c_[R, -R @ C].tolist()
    audit = {
        "status": "PASS_STATIC_TRANSFER",
        "registration": stat,
        "center_delta_cm": center_delta,
        "H_target_to_cert257": H_t_to_c.tolist(),
        "H_cert257_to_target": H_c_to_t.tolist(),
        "K_px": K.tolist(),
    }
    return corrected, audit


def temporal_identity_candidates(model, stage: Path, label: str, qual: dict, drop_left_wrist=False):
    frames = v32v.load_burst(stage, label, range(0, 7))
    rows = []
    target_bbox = None
    obs = qual.get("focal_player", {}).get("observations", {})
    if label in obs:
        target_bbox = np.asarray(obs[label]["bbox_xyxy"], float)
        target_center = np.array([(target_bbox[0] + target_bbox[2]) / 2.0,
                                  (target_bbox[1] + target_bbox[3]) / 2.0], float)
    else:
        target_center = None

    for rel in REF_RELS:
        dets = v32j.infer(model, frames[rel])
        for i, d in enumerate(dets):
            dark, torso, confident = v32w.appearance(d, frames[rel])
            if confident < 7:
                continue
            if not (torso >= 0.45 or dark >= 0.55):
                continue
            if rel == 0:
                xy0 = np.asarray(d["xy"], float)
                valid0 = np.asarray(d["conf"], float) >= MIN_CONF
                trace = None
                source = "direct_t00"
            else:
                xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
                valid0 = np.asarray(valid0, bool) & (np.asarray(d["conf"], float) >= MIN_CONF)
                source = f"temporal_t+{rel}"
            if drop_left_wrist:
                valid0[LEFT_WRIST] = False
            supported = int(sum(bool(valid0[j]) for j in BODY))
            if supported < 6:
                continue
            box = np.asarray(d["box"], float)
            center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], float)
            dist = float(np.linalg.norm(center - target_center)) if target_center is not None else 0.0
            rows.append({
                "label": label, "source": source, "anchor_rel": int(rel),
                "selected_index": int(i), "xy0": np.asarray(xy0, float),
                "valid0": np.asarray(valid0, bool), "anchor_conf": np.asarray(d["conf"], float),
                "trace": trace, "dark_fraction": float(dark), "torso_dark_score": float(torso),
                "confident_body_joints": int(confident), "supported_t0_body_joints": supported,
                "box_xyxy": box.tolist(), "det_conf": float(d.get("det_conf", 0.0)),
                "target_center_distance_px": dist,
            })
    if not rows:
        raise RuntimeError(f"v33b no Houston-dark coherent candidates for {label}")
    return rows, frames


def serial_ref(c):
    return {
        "label": c["label"], "source": c["source"], "anchor_rel": c["anchor_rel"],
        "selected_index": c["selected_index"], "box_xyxy": c["box_xyxy"],
        "dark_fraction": c["dark_fraction"], "torso_dark_score": c["torso_dark_score"],
        "confident_body_joints": c["confident_body_joints"],
        "supported_t0_body_joints": c["supported_t0_body_joints"],
        "target_center_distance_px": c["target_center_distance_px"],
    }


def choose_lar_broadcast_reference(scene, lar_rows, b_rows):
    pairs = []
    for li, l in enumerate(lar_rows):
        for bi, b in enumerate(b_rows):
            e = v32v.epipolar_stats(scene, LAR, BCAST, l["xy0"], l["valid0"], b["xy0"], b["valid0"])
            if int(e["joint_count"]) < 6:
                continue
            shortage = max(0, 8 - int(e["joint_count"]))
            # Geometry dominates. The Broadcast focal distance is only a weak
            # identity prior preventing a different Houston player winning an
            # otherwise excellent cross-view match.
            score = float(
                e["median_px"] + 0.30 * e["p90_px"] + 10.0 * shortage
                + 0.018 * b["target_center_distance_px"]
                + (0.0 if l["anchor_rel"] == 0 else 0.20)
                + (0.0 if b["anchor_rel"] == 0 else 0.20)
                - 0.35 * l["torso_dark_score"] - 0.35 * b["torso_dark_score"]
            )
            pairs.append({"li": li, "bi": bi, "score": score, "epipolar": e})
    if not pairs:
        raise RuntimeError("v33b no LAR-Broadcast candidate pair has >=6 comparable joints")
    pairs.sort(key=lambda r: (r["score"], r["epipolar"]["median_px"]))
    best = pairs[0]
    l = lar_rows[best["li"]]
    b = b_rows[best["bi"]]
    serial = []
    for rank, p in enumerate(pairs[:40], 1):
        serial.append({
            "rank": rank, "score": float(p["score"]),
            "lar": serial_ref(lar_rows[p["li"]]),
            "broadcast": serial_ref(b_rows[p["bi"]]),
            "epipolar": serialize_epi(p["epipolar"]),
        })
    return l, b, serial


def raw_pair_gate(e):
    return bool(int(e["joint_count"]) >= 7 and float(e["median_px"]) <= 18.0 and float(e["p90_px"]) <= 32.0)


def draw_detection(img, det, title):
    out = img.copy()
    if det is not None:
        for a, b in v32j.DRAW:
            if det["conf"][a] >= MIN_CONF and det["conf"][b] >= MIN_CONF:
                cv2.line(out, tuple(np.rint(det["xy"][a]).astype(int)),
                         tuple(np.rint(det["xy"][b]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        for j in range(17):
            if det["conf"][j] >= MIN_CONF:
                cv2.circle(out, tuple(np.rint(det["xy"][j]).astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(out, title[:135], (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--v73-frame0257", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())
    if scene.get("resolution") != [W, H]:
        raise RuntimeError(f"v33b unexpected resolution {scene.get('resolution')}")
    cert = cv2.imread(str(a.v73_frame0257), cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape != (H, W):
        raise RuntimeError("v33b missing certified v73 RAR frame0257")

    model = RFDETRKeypointPreview()

    # Reference pair is selected without any RAR evidence.
    lar_rows, lar_frames = temporal_identity_candidates(model, stage, LAR, qual, drop_left_wrist=True)
    b_rows, b_frames = temporal_identity_candidates(model, stage, BCAST, qual, drop_left_wrist=False)
    lref, bref, pair_rank = choose_lar_broadcast_reference(scene, lar_rows, b_rows)
    lb_epi = v32v.epipolar_stats(scene, LAR, BCAST, lref["xy0"], lref["valid0"], bref["xy0"], bref["valid0"])
    lb_gate = raw_pair_gate(lb_epi)

    # Weak spatial prior for RAR identity only; dynamic geometry decides.
    r_obs = qual.get("focal_player", {}).get("observations", {}).get(RAR, {})
    rb = np.asarray(r_obs.get("bbox_xyxy", [430, 180, 590, 340]), float)
    r_target = np.array([(rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0], float)
    source_cam = scene["cameras"][RAR]

    sweep = []
    selected_visuals = []
    for rel in RAR_RELS:
        p = v32k.burst_path(stage, RAR, rel)
        img = cv2.imread(str(p))
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None or gray is None:
            sweep.append({"rel": rel, "frame_file": p.name, "status": "FAIL_IMAGE"})
            continue
        corrected, static_audit = transfer_rar_camera(cert, gray, source_cam)
        if corrected is None:
            sweep.append({"rel": rel, "frame_file": p.name, "status": "FAIL_STATIC_TRANSFER", "static": static_audit})
            continue
        scene_r = copy.deepcopy(scene)
        scene_r["cameras"][RAR] = corrected
        dets = v32j.infer(model, img)
        candidates = []
        for i, d in enumerate(dets):
            dark, torso, confident = v32w.appearance(d, img)
            if confident < 7:
                continue
            if not (torso >= 0.45 or dark >= 0.55):
                continue
            valid = np.asarray(d["conf"], float) >= MIN_CONF
            e_l = v32v.epipolar_stats(scene_r, RAR, LAR, np.asarray(d["xy"], float), valid,
                                      lref["xy0"], lref["valid0"])
            e_b = v32v.epipolar_stats(scene_r, RAR, BCAST, np.asarray(d["xy"], float), valid,
                                      bref["xy0"], bref["valid0"])
            if min(int(e_l["joint_count"]), int(e_b["joint_count"])) < 5:
                continue
            box = np.asarray(d["box"], float)
            c = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], float)
            dist = float(np.linalg.norm(c - r_target))
            shortage = max(0, 7 - int(e_l["joint_count"])) + max(0, 7 - int(e_b["joint_count"]))
            score = float(
                e_l["median_px"] + e_b["median_px"]
                + 0.25 * (e_l["p90_px"] + e_b["p90_px"])
                + 10.0 * shortage + 0.010 * dist
                - 0.30 * torso - 0.15 * dark
            )
            gates = {"lar_rar": raw_pair_gate(e_l), "broadcast_rar": raw_pair_gate(e_b)}
            candidates.append({
                "index": int(i), "score": score, "box_xyxy": box.tolist(),
                "dark_fraction": float(dark), "torso_dark_score": float(torso),
                "confident_body_joints": int(confident), "target_center_distance_px": dist,
                "lar_rar": serialize_epi(e_l), "broadcast_rar": serialize_epi(e_b),
                "gates": gates, "exact_state_gate": bool(gates["lar_rar"] and gates["broadcast_rar"]),
                "_det": d,
            })
        candidates.sort(key=lambda r: (not r["exact_state_gate"], r["score"],
                                       r["lar_rar"]["median_px"] + r["broadcast_rar"]["median_px"]))
        serial = [{k: v for k, v in row.items() if k != "_det"} for row in candidates[:12]]
        best = candidates[0] if candidates else None
        row = {
            "rel": int(rel), "frame_file": p.name,
            "status": "CANDIDATE" if best is not None else "NO_HOUSTON_DARK_BODY",
            "static": static_audit, "candidate_count": int(len(candidates)),
            "best": None if best is None else {k: v for k, v in best.items() if k != "_det"},
            "top_candidates": serial,
        }
        sweep.append(row)
        if best is not None:
            title = (f"v33b RAR rel{rel:+d} {p.stem.split('_')[-1]} | "
                     f"L {best['lar_rar']['median_px']:.1f}px B {best['broadcast_rar']['median_px']:.1f}px "
                     f"gate={best['exact_state_gate']}")
            ov = draw_detection(img, best["_det"], title)
            cv2.imwrite(str(a.out / f"v33b_rar_rel{rel:+03d}.png"), ov)
            selected_visuals.append((rel, ov))

    valid_rows = [r for r in sweep if r.get("best") is not None]
    valid_rows.sort(key=lambda r: (not bool(r["best"]["exact_state_gate"]), float(r["best"]["score"])))
    winner = valid_rows[0] if valid_rows else None
    exact_found = bool(lb_gate and winner is not None and winner["best"]["exact_state_gate"])

    # Native 960x540 diagnostic contact sheet (4x4 thumbnails, no upscale).
    if selected_visuals:
        thumb_w, thumb_h = 240, 135
        canvas = np.zeros((H, W, 3), np.uint8)
        for slot, (rel, ov) in enumerate(sorted(selected_visuals, key=lambda x: x[0])[:16]):
            r, c = divmod(slot, 4)
            thumb = cv2.resize(ov, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            canvas[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb
        cv2.imwrite(str(a.out / "v33b_rar_sweep_contact_native_960x540.png"), canvas)

    # Reference-pair visuals at the t+00 images used for comparison.
    l0 = lar_frames[0]
    b0 = b_frames[0]
    ltitle = f"v33b LAR reference {lref['source']} | L-B median {lb_epi['median_px']:.2f}px"
    btitle = f"v33b Broadcast reference {bref['source']} | L-B median {lb_epi['median_px']:.2f}px"
    # Draw the anchor detections only when the reference is direct; otherwise
    # show the exact t+00 source image and report tracked coordinates in JSON.
    cv2.imwrite(str(a.out / "v33b_lar_reference_t00.png"), draw_detection(l0, None, ltitle))
    cv2.imwrite(str(a.out / "v33b_broadcast_reference_t00.png"), draw_detection(b0, None, btitle))

    qa = {
        "version": "v33b_rar_exact_state_dual_reference_sweep",
        "status": "PASS_V33B_RAR_EXACT_STATE_FOUND" if exact_found else "FAIL_CLOSED_V33B_NO_RAR_EXACT_STATE",
        "native_resolution": [W, H],
        "generated_rgb": False, "novel_view_rendered": False, "upscaled": False,
        "policy": "LAR+Broadcast selected without RAR; each real RAR frame receives a static-only exact camera transfer and direct whole-body Adams candidate; no per-joint temporal mixing",
        "existing_raw_state_gate": {"min_joint_count": 7, "median_px_max": 18.0, "p90_px_max": 32.0},
        "lar_broadcast_reference": {
            "lar": serial_ref(lref), "broadcast": serial_ref(bref),
            "epipolar": serialize_epi(lb_epi), "gate": lb_gate,
            "top_pair_candidates": pair_rank,
        },
        "rar_sweep": sweep,
        "winner": winner,
        "gate": {
            "lar_broadcast_reference_consistent": lb_gate,
            "rar_candidate_found": winner is not None,
            "rar_agrees_with_lar": bool(winner and winner["best"]["gates"]["lar_rar"]),
            "rar_agrees_with_broadcast": bool(winner and winner["best"]["gates"]["broadcast_rar"]),
            "exact_three_view_state_found": exact_found,
            "freeview_render_unlocked": False,
        },
        "next_if_pass": "fit one articulated MHR body to all three exact-state observations and validate leave-one-view-out before any static free-view render",
        "next_if_fail": "treat RAR dynamic foreground as unsupported for this freeze and seek an additional independent camera rather than forcing or inventing the RAR body",
    }
    (a.out / "v33b_rar_exact_state_sweep.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({
        "status": qa["status"],
        "lar_broadcast_reference": qa["lar_broadcast_reference"],
        "winner": winner,
        "gate": qa["gate"],
    }, indent=2), flush=True)
    if not exact_found:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
