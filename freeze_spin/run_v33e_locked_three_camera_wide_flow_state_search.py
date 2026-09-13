from __future__ import annotations

"""v33e: wide, source-grounded exact-state search on the LOCKED three cameras.

Changes from v33d:
- reacquires the same official NBA event clips and decodes a wider native window;
- tracks the already identity-locked Adams anchor through real frames with
  forward/backward-checked optical flow, rather than trusting raw limb keypoints
  independently on every frame;
- preserves one real frame per camera per tested state and all accepted physical
  camera centres;
- precomputes pairwise geometry so a wider search remains cheap.

No fourth camera, generated RGB, interpolation, upscale or render.
"""

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRKeypointPreview

from freeze_spin import prepare_v32_layered_scene as v32prep
from freeze_spin import run_v33b_rar_exact_state_sweep as v33b
from freeze_spin import run_v33d_locked_three_camera_joint_state_search as v33d
from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
CAMS = (LAR, RAR, BCAST)
W, H = v32j.W, v32j.H
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF


def safe(label: str) -> str:
    return label.replace(" ", "_")


def export_wide_burst(clips_dir: Path, centers: dict[str, int], out_stage: Path, rels: tuple[int, ...]):
    burst = out_stage / "burst"
    burst.mkdir(parents=True, exist_ok=True)
    audit = {}
    for label in CAMS:
        clip = v32prep.find_clip(clips_dir, label)
        center = int(centers[label])
        indices = [center + r for r in rels]
        decoded = v32prep.decode_indices(clip, indices)
        d = burst / safe(label)
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for rel, idx in zip(rels, indices):
            fn = f"rel{rel:+03d}_frame{idx:04d}.png"
            p = d / fn
            if not cv2.imwrite(str(p), decoded[idx]):
                raise RuntimeError(f"write failed {p}")
            rows.append({"rel": int(rel), "frame_index": int(idx), "file": str(p.relative_to(out_stage))})
        audit[label] = {"clip": clip.name, "center_frame": center, "frames": rows}
    return audit


def flow_step(prev: np.ndarray, nxt: np.ndarray, pts: np.ndarray, valid: np.ndarray):
    g0 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
    p0 = pts.astype(np.float32).reshape(-1, 1, 2)
    lk = dict(
        winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
        minEigThreshold=1e-5,
    )
    p1, s1, e1 = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    back, s2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    if p1 is None or back is None:
        return pts.copy(), np.zeros_like(valid, dtype=bool), {"survivors": 0, "median_fb_px": 999.0}
    fb = np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
    err = np.asarray(e1).reshape(-1)
    q = p1.reshape(-1, 2)
    inside = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    good = np.asarray(valid, bool) & np.asarray(s1).reshape(-1).astype(bool) & np.asarray(s2).reshape(-1).astype(bool)
    good &= inside & (fb <= 2.5) & (err <= 35.0)
    fbg = fb[good]
    return q.astype(float), good, {
        "survivors": int(good.sum()),
        "median_fb_px": float(np.median(fbg)) if len(fbg) else 999.0,
        "p90_fb_px": float(np.percentile(fbg, 90)) if len(fbg) else 999.0,
    }


def flow_observations(fs, identity_track, label: str, rels: tuple[int, ...]):
    anchor_rel = int(v33d.ANCH[label][0])
    anchor = identity_track[anchor_rel]
    xy0 = np.asarray(anchor["xy"], float)
    valid0 = np.asarray(anchor["conf"], float) >= MIN_CONF
    states = {anchor_rel: {"xy": xy0.copy(), "valid": valid0.copy(), "source": "verified_anchor", "flow": None}}

    pts, ok = xy0.copy(), valid0.copy()
    for rel in range(anchor_rel + 1, max(rels) + 1):
        pts, ok, qa = flow_step(fs[rel - 1][1], fs[rel][1], pts, ok)
        states[rel] = {"xy": pts.copy(), "valid": ok.copy(), "source": "anchor_flow", "flow": qa}

    pts, ok = xy0.copy(), valid0.copy()
    for rel in range(anchor_rel - 1, min(rels) - 1, -1):
        pts, ok, qa = flow_step(fs[rel + 1][1], fs[rel][1], pts, ok)
        states[rel] = {"xy": pts.copy(), "valid": ok.copy(), "source": "anchor_flow", "flow": qa}

    out = {}
    audit = {}
    for rel in rels:
        direct = identity_track[rel]
        flow = states[rel]
        flow_body = int(sum(bool(flow["valid"][j]) for j in BODY))
        direct_valid = np.asarray(direct["conf"], float) >= MIN_CONF
        direct_body = int(sum(bool(direct_valid[j]) for j in BODY))
        use_flow = flow_body >= 6
        if use_flow:
            conf = np.where(flow["valid"], 1.0, 0.0)
            row = {**direct, "xy": np.asarray(flow["xy"], float), "conf": conf}
            source = "anchor_flow"
        else:
            row = direct
            source = "direct_identity_locked_detection"
        out[rel] = row
        audit[str(rel)] = {
            "measurement_source": source,
            "flow_body_joint_count": flow_body,
            "direct_body_joint_count": direct_body,
            "flow_qa": flow.get("flow"),
        }
    return out, audit


def make_scene(base_scene: dict, cam_states: dict[str, dict[int, dict]], a: str, ar: int, b: str, br: int):
    s = copy.deepcopy(base_scene)
    s["cameras"][a] = cam_states[a][ar]
    s["cameras"][b] = cam_states[b][br]
    return s


def pair_score(e: dict) -> float:
    return float(e["median_px"] + 0.25 * e["p90_px"] + 12.0 * max(0, 6 - int(e["joint_count"])))


def pair_table(base_scene, cam_states, obs, a, b, rel_a, rel_b):
    out = {}
    for ra in rel_a:
        for rb in rel_b:
            s = make_scene(base_scene, cam_states, a, ra, b, rb)
            ea = v32v.epipolar_stats(
                s, a, b,
                np.asarray(obs[a][ra]["xy"], float), np.asarray(obs[a][ra]["conf"], float) >= MIN_CONF,
                np.asarray(obs[b][rb]["xy"], float), np.asarray(obs[b][rb]["conf"], float) >= MIN_CONF,
            )
            out[(ra, rb)] = {"epi": ea, "gate": bool(v33d.pgate(ea)), "score": pair_score(ea)}
    return out


def serial_pair(x):
    return {"epi": v33b.serialize_epi(x["epi"]), "gate": bool(x["gate"]), "score": float(x["score"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", type=Path, required=True)
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--v73-frame0257", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not (8 <= args.radius <= 30):
        raise RuntimeError("v33e radius must be 8..30 real frames")
    rels = tuple(range(-args.radius, args.radius + 1))
    v33d.RELS = rels

    source_stage = args.b32_root / "stage_a"
    scene = json.loads((source_stage / "v32_scene_manifest.json").read_text())
    if scene.get("resolution") != [W, H] or set(scene.get("cameras", {})) != {LAR, BCAST, RAR}:
        raise RuntimeError("v33e three-camera lock/source mismatch")
    centers = {k: int(v) for k, v in scene["freeze"]["chosen_frame_indices"].items()}
    if set(centers) != {LAR, BCAST, RAR}:
        raise RuntimeError("v33e missing B32 centers")

    wide_stage = args.out / "wide_stage"
    wide_stage.mkdir(parents=True, exist_ok=True)
    source_audit = export_wide_burst(args.clips_dir, centers, wide_stage, rels)

    fs = {c: v33d.frames(wide_stage, c) for c in CAMS}
    cert = cv2.imread(str(args.v73_frame0257), cv2.IMREAD_GRAYSCALE)
    if cert is None or cert.shape != (H, W):
        raise RuntimeError("v33e missing accepted v73 RAR frame0257")

    cam_states = {c: {} for c in CAMS}
    transfer_audit = {c: {} for c in CAMS}
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

    valid_rels = {c: tuple(r for r in rels if r in cam_states[c]) for c in CAMS}
    if any(len(valid_rels[c]) < 8 for c in CAMS):
        raise RuntimeError(f"v33e insufficient fixed-centre static states { {c:len(valid_rels[c]) for c in CAMS} }")

    model = RFDETRKeypointPreview()
    identity = {}
    identity_audit = {}
    for c in CAMS:
        identity[c], identity_audit[c] = v33d.track(model, c, fs[c])
    obs = {}
    observation_audit = {}
    for c in CAMS:
        obs[c], observation_audit[c] = flow_observations(fs[c], identity[c], c, rels)

    lr = pair_table(scene, cam_states, obs, LAR, RAR, valid_rels[LAR], valid_rels[RAR])
    lb = pair_table(scene, cam_states, obs, LAR, BCAST, valid_rels[LAR], valid_rels[BCAST])
    rb = pair_table(scene, cam_states, obs, RAR, BCAST, valid_rels[RAR], valid_rels[BCAST])

    tested = 0
    pairwise_pass = 0
    exact_pass = 0
    candidates = []
    for l in valid_rels[LAR]:
        for r in valid_rels[RAR]:
            xlr = lr[(l, r)]
            for b in valid_rels[BCAST]:
                tested += 1
                xlb = lb[(l, b)]
                xrb = rb[(r, b)]
                gates = (xlr["gate"], xlb["gate"], xrb["gate"])
                span = max(l, r, b) - min(l, r, b)
                score = float(xlr["score"] + xlb["score"] + xrb["score"] + 0.30 * span)
                base = {
                    "rels": {LAR: int(l), RAR: int(r), BCAST: int(b)},
                    "frames": {LAR: fs[LAR][l][0].name, RAR: fs[RAR][r][0].name, BCAST: fs[BCAST][b][0].name},
                    "score": score,
                    "span": int(span),
                    "pair_gates": [bool(x) for x in gates],
                    "all_pairwise": bool(all(gates)),
                }
                if all(gates):
                    pairwise_pass += 1
                    s = copy.deepcopy(scene)
                    s["cameras"][LAR] = cam_states[LAR][l]
                    s["cameras"][RAR] = cam_states[RAR][r]
                    s["cameras"][BCAST] = cam_states[BCAST][b]
                    q = v33d.repro(s, obs[LAR][l], obs[RAR][r], obs[BCAST][b])
                    base["reprojection"] = q
                    base["exact_state"] = bool(q["gate"])
                    if q["gate"]:
                        exact_pass += 1
                else:
                    base["reprojection"] = None
                    base["exact_state"] = False
                candidates.append(base)

    candidates.sort(key=lambda x: (not x["exact_state"], not x["all_pairwise"], x["score"]))
    winner = candidates[0] if candidates else None
    exact = bool(winner and winner["exact_state"])

    def enrich(row):
        if row is None:
            return None
        l, r, b = row["rels"][LAR], row["rels"][RAR], row["rels"][BCAST]
        return {
            **row,
            "epi": {
                "LR": serial_pair(lr[(l, r)]),
                "LB": serial_pair(lb[(l, b)]),
                "RB": serial_pair(rb[(r, b)]),
            },
            "observation_sources": {
                LAR: observation_audit[LAR][str(l)],
                RAR: observation_audit[RAR][str(r)],
                BCAST: observation_audit[BCAST][str(b)],
            },
        }

    winner_full = enrich(winner)
    top = [enrich(x) for x in candidates[:50]]

    if winner:
        for c in CAMS:
            rel = winner["rels"][c]
            shutil.copy2(fs[c][rel][0], args.out / f"v33e_best_{safe(c)}_{fs[c][rel][0].name}")

    qa = {
        "version": "v33e_locked_three_camera_wide_flow_state_search",
        "status": "PASS_V33E_EXACT_THREE_VIEW_STATE" if exact else "FAIL_CLOSED_V33E_NO_EXACT_THREE_VIEW_STATE",
        "camera_lock": [LAR, RAR, BCAST],
        "camera_count": 3,
        "camera_addition_or_substitution_used": False,
        "physical_center_refit_from_players": False,
        "native_resolution": [W, H],
        "generated_rgb": False,
        "upscaled": False,
        "novel_view_rendered": False,
        "search_rels": [int(min(rels)), int(max(rels))],
        "source_window_frames_per_camera": len(rels),
        "source_audit": source_audit,
        "valid_static_camera_states": {c: len(valid_rels[c]) for c in CAMS},
        "identity_tracks": identity_audit,
        "observation_policy": "Verified/source-local Adams identity anchors are propagated only through real adjacent native frames with forward/backward-checked optical flow. Whole-pose measurement source is selected source-locally from flow support, never from cross-view residuals.",
        "observation_audit": observation_audit,
        "camera_transfers": transfer_audit,
        "tested_triplets": int(tested),
        "pairwise_pass_triplets": int(pairwise_pass),
        "exact_pass_triplets": int(exact_pass),
        "winner": winner_full,
        "top_triplets": top,
        "freeview_render_unlocked": False,
        "gates": {
            "pair_min_joint_count": 6,
            "pair_median_epipolar_px_max": 18.0,
            "pair_p90_epipolar_px_max": 35.0,
            "reprojection_min_joint_count": 7,
            "reprojection_median_px_max": 12.0,
            "reprojection_p90_px_max": 24.0,
            "threshold_policy": "UNCHANGED_FROM_V33D_DO_NOT_WEAKEN"
        },
        "next_if_pass": "source-grounded three-view body/ball fit and held-out visual QA; still no render until that passes",
        "next_if_fail": "keep the same three cameras; inspect v33e per-joint/flow diagnostics and expand to +/-30 only if source support remains valid"
    }
    out_json = args.out / "v33e_locked_three_camera_wide_flow_state_search.json"
    out_json.write_text(json.dumps(qa, indent=2))
    print(json.dumps({
        "status": qa["status"],
        "valid_static_camera_states": qa["valid_static_camera_states"],
        "tested_triplets": tested,
        "pairwise_pass_triplets": pairwise_pass,
        "exact_pass_triplets": exact_pass,
        "winner": winner_full
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
