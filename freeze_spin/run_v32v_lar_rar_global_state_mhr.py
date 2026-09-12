from __future__ import annotations

"""v32v: one coherent LAR+RAR body state, Broadcast held out.

v32u failed because the Broadcast pseudo-observation mixed body joints selected
from several temporal anchors.  v32v changes the diagnostic, not the accepted
NBA camera geometry:

* Right Above Rim (RAR) remains the trusted temporal-deblend observation at the
  exact t+00 freeze, measured from one real t+04 anchor tracked back through
  real frames.
* Left Above Rim (LAR) is measured from ONE globally selected real temporal
  anchor (direct t+00 or one of t+02..t+06 tracked back to t+00).  Candidate
  selection is global for the whole body; joints are never mixed across times.
* One articulated MHR body is fitted only to LAR + RAR image-space evidence.
* Broadcast t+00 is detected independently from the known B32 focal box and is
  strictly held out from optimization.  It is used only for leave-one-view-out
  validation.

No generated RGB, no camera refinement from moving players, no fourth camera,
no frame interpolation and no upscale.  All QA images remain native 960x540.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from rfdetr import RFDETRKeypointPreview

from freeze_spin import fit_v32q_mhr_direct_multiview_2d as base
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR, BCAST, RAR = v32j.LAR, v32j.BCAST, v32j.RAR
CAMS = v32j.CAMS
W, H = v32j.W, v32j.H
COCO_TO_MHR = base.COCO_TO_MHR
BODY = tuple(sorted(COCO_TO_MHR))
MIN_CONF = 0.20


def focal_rank(img: np.ndarray, detections, bbox: np.ndarray):
    """Identity ranking independent of the fitted 3-D body."""
    tc = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], float)
    rows = []
    for i, d in enumerate(detections):
        box = np.asarray(d["box"], float)
        dc = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], float)
        dist = float(np.linalg.norm(dc - tc))
        dark = float(v32j.dark_fraction(img, box))
        td = float(v32k.torso_dark_score(img, d))
        confident = int(np.sum(np.asarray(d["conf"]) >= MIN_CONF))
        score = (
            3.2 * float(v32j.iou(box, bbox))
            + 1.4 * dark + 1.1 * td
            + 0.20 * float(d.get("det_conf", 0.0))
            + 0.025 * confident - 0.0040 * dist
        )
        rows.append({
            "index": int(i), "score": float(score), "iou": float(v32j.iou(box, bbox)),
            "dark_fraction": dark, "torso_dark_score": td,
            "confident_joints": confident, "center_distance_px": dist,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return (rows[0]["index"] if rows else None), rows


def epipolar_stats(scene: dict, a_label: str, b_label: str,
                    a_xy: np.ndarray, a_valid: np.ndarray,
                    b_xy: np.ndarray, b_valid: np.ndarray):
    cams = {c: v32j.cam(scene, c) for c in CAMS}
    F = v32j.fundamental(cams[a_label], cams[b_label])
    vals = []
    rows = []
    for j in BODY:
        if not bool(a_valid[j]) or not bool(b_valid[j]):
            continue
        e = float(v32j.epi(F, np.asarray(a_xy[j], float), np.asarray(b_xy[j], float)))
        vals.append(e)
        rows.append({"joint": int(j), "name": v32j.NAMES[j], "epi_px": e})
    if not vals:
        return {"joint_count": 0, "median_px": 999.0, "p75_px": 999.0,
                "p90_px": 999.0, "max_px": 999.0, "per_joint": rows}
    x = np.asarray(vals, float)
    return {
        "joint_count": int(len(x)), "median_px": float(np.median(x)),
        "p75_px": float(np.percentile(x, 75)), "p90_px": float(np.percentile(x, 90)),
        "max_px": float(np.max(x)), "per_joint": rows,
    }


def load_burst(stage: Path, label: str, rels):
    out = {}
    for rel in rels:
        p = v32k.burst_path(stage, label, rel)
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"cannot read {p}")
        out[int(rel)] = im
    return out


def choose_global_lar_state(model, stage: Path, qual: dict, scene: dict,
                            rar_xy: np.ndarray, rar_valid: np.ndarray):
    """Select one whole-body LAR temporal source; never per-joint temporal mixing."""
    frames = load_burst(stage, LAR, range(0, 7))
    bbox = np.asarray(qual["focal_player"]["observations"][LAR]["bbox_xyxy"], float)
    candidates = []

    # Direct t+00 is a valid candidate and has no optical-flow uncertainty.
    dets0 = v32j.infer(model, frames[0])
    idx0, rank0 = focal_rank(frames[0], dets0, bbox)
    if idx0 is not None:
        d0 = dets0[idx0]
        valid0 = np.asarray(d0["conf"] >= MIN_CONF, bool)
        epi0 = epipolar_stats(scene, LAR, RAR, d0["xy"], valid0, rar_xy, rar_valid)
        candidates.append({
            "source": "direct_t00", "anchor_rel": 0, "selected_index": int(idx0),
            "ranked_detections": rank0, "xy0": np.asarray(d0["xy"], float),
            "valid0": valid0, "anchor_conf": np.asarray(d0["conf"], float),
            "trace": None, "epipolar": epi0,
        })

    # Cleaner later real frames may identify Adams better; each candidate is
    # tracked as one coherent body back to t+00 before comparison with RAR.
    for rel in (2, 3, 4, 5, 6):
        dets = v32j.infer(model, frames[rel])
        idx, ranked = focal_rank(frames[rel], dets, bbox)
        if idx is None:
            continue
        d = dets[idx]
        xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
        valid0 = np.asarray(valid0, bool) & (np.asarray(d["conf"], float) >= MIN_CONF)
        epi = epipolar_stats(scene, LAR, RAR, xy0, valid0, rar_xy, rar_valid)
        candidates.append({
            "source": f"temporal_t+{rel}", "anchor_rel": int(rel),
            "selected_index": int(idx), "ranked_detections": ranked,
            "xy0": np.asarray(xy0, float), "valid0": valid0,
            "anchor_conf": np.asarray(d["conf"], float), "trace": trace,
            "epipolar": epi,
        })

    if not candidates:
        raise RuntimeError("v32v: no LAR whole-body candidate")

    # Global state score: prioritize cross-view median/p90 while penalizing weak
    # body coverage.  Direct t00 gets a tiny tie preference only.
    for c in candidates:
        e = c["epipolar"]
        shortage = max(0, 8 - int(e["joint_count"]))
        c["selection_score"] = float(e["median_px"] + .30 * e["p90_px"] + 12.0 * shortage + (0.0 if c["anchor_rel"] == 0 else .35))
    candidates.sort(key=lambda c: c["selection_score"])
    best = candidates[0]

    serial = []
    for c in candidates:
        serial.append({
            "source": c["source"], "anchor_rel": c["anchor_rel"],
            "selected_index": c["selected_index"], "selection_score": c["selection_score"],
            "epipolar": c["epipolar"], "valid_body_joints": int(sum(bool(c["valid0"][j]) for j in BODY)),
            "ranked_detections": c["ranked_detections"],
        })
    return best, serial, frames


def repro_stats(project_fn, Xw: np.ndarray, camera_label: str, observations: dict, cams: dict, jslot: dict):
    vals = []
    for j, row in observations.items():
        uv = project_fn(cams[camera_label], Xw[jslot[j]].reshape(1, 3))[0]
        if np.all(np.isfinite(uv)):
            vals.append(float(np.linalg.norm(uv - row["xy"])))
    if not vals:
        return {"joint_count": 0, "median_px": 999.0, "p75_px": 999.0, "p90_px": 999.0, "max_px": 999.0}
    a = np.asarray(vals, float)
    return {"joint_count": int(len(a)), "median_px": float(np.median(a)),
            "p75_px": float(np.percentile(a, 75)), "p90_px": float(np.percentile(a, 90)),
            "max_px": float(np.max(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=360)
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())
    assert scene["resolution"] == [W, H] and set(scene["cameras"]) == set(CAMS)
    cams = {c: base.camera(scene, c) for c in CAMS}
    vcams = {c: v32j.cam(scene, c) for c in CAMS}
    model = RFDETRKeypointPreview()

    # Trusted RAR observation: ONE real t+04 Adams pose tracked back to t+00.
    rar_frames = load_burst(stage, RAR, range(0, 5))
    rb = np.asarray(qual["focal_player"]["observations"][RAR]["bbox_xyxy"], float)
    target_center = np.array([(rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0], float)
    r4dets = v32j.infer(model, rar_frames[4])
    ri, rrank = v32k.select_adams_detection(rar_frames[4], r4dets, target_center)
    if ri is None:
        raise RuntimeError("v32v: no RAR t+04 Adams candidate")
    r4 = r4dets[ri]
    rxy, rvalid, rtrace = v32l.dense_track_back(rar_frames, 4, r4["xy"], r4["conf"])
    rvalid = np.asarray(rvalid, bool) & (np.asarray(r4["conf"], float) >= MIN_CONF)

    # LAR is selected globally as one coherent body state.
    lbest, lar_candidates, lar_frames = choose_global_lar_state(model, stage, qual, scene, rxy, rvalid)
    lxy = np.asarray(lbest["xy0"], float)
    lvalid = np.asarray(lbest["valid0"], bool)
    lconf = np.asarray(lbest["anchor_conf"], float)
    raw_epi = lbest["epipolar"]

    # Broadcast is held out from optimization.  Identity is selected only from
    # the known B32 focal box / uniform evidence, never from the fitted body.
    bimg = cv2.imread(str(stage / "v32_chosen_Broadcast_frame0276.png"))
    if bimg is None:
        raise RuntimeError("v32v: missing Broadcast chosen frame")
    bdets = v32j.infer(model, bimg)
    bb = np.asarray(qual["focal_player"]["observations"][BCAST]["bbox_xyxy"], float)
    bi, brank = focal_rank(bimg, bdets, bb)
    if bi is None:
        raise RuntimeError("v32v: no independent Broadcast focal detection")
    bdet = bdets[bi]

    obs = {LAR: {}, RAR: {}, BCAST: {}}
    for j in BODY:
        if bool(lvalid[j]) and float(lconf[j]) >= MIN_CONF:
            obs[LAR][j] = {"xy": lxy[j], "weight": float(np.clip(lconf[j], MIN_CONF, 1.0))}
        if bool(rvalid[j]) and float(r4["conf"][j]) >= MIN_CONF:
            obs[RAR][j] = {"xy": np.asarray(rxy[j], float), "weight": float(np.clip(r4["conf"][j], MIN_CONF, 1.0))}
        if float(bdet["conf"][j]) >= MIN_CONF:
            obs[BCAST][j] = {"xy": np.asarray(bdet["xy"][j], float), "weight": float(np.clip(bdet["conf"][j], MIN_CONF, 1.0))}

    common = sorted(set(obs[LAR]) & set(obs[RAR]))
    if len(obs[LAR]) < 7 or len(obs[RAR]) < 7 or len(common) < 6:
        raise RuntimeError(f"v32v: insufficient LAR/RAR observations L={len(obs[LAR])} R={len(obs[RAR])} common={len(common)}")

    # State-consistency gate is evaluated independently of the MHR fit.  We do
    # not abort early so failed candidates still produce useful diagnostics.
    state_gate = bool(raw_epi["joint_count"] >= 7 and raw_epi["median_px"] <= 18.0 and raw_epi["p90_px"] <= 32.0)

    geo, character, fbx, model_path = base.ensure_assets(a.work)
    jnames = list(character.skeleton.joint_names); ji = {n: i for i, n in enumerate(jnames)}
    pnames = list(character.parameter_transform.names); pi = {n: i for i, n in enumerate(pnames)}
    if any(name not in ji for name in COCO_TO_MHR.values()):
        raise RuntimeError("v32v: required MHR anatomical joint missing")
    joint_ids = {j: ji[name] for j, name in COCO_TO_MHR.items()}
    zero = np.zeros(len(pnames), np.float64)

    # Ray intersections initialize only.  They are never residuals or truth.
    provisional = {}
    for j in common:
        X = v32j.triangulate_rays(vcams, {LAR: obs[LAR][j]["xy"], RAR: obs[RAR][j]["xy"]})
        if X is not None and np.all(np.isfinite(X)) and (-350 <= X[0] <= 1250 and -750 <= X[1] <= 750 and -60 <= X[2] <= 450):
            provisional[j] = np.asarray(X, float)
    R0, s0, t0 = base.similarity_from_provisional(character, geo, joint_ids, zero, provisional)

    all_body_ids = np.array([joint_ids[j] for j in COCO_TO_MHR], np.int32)
    driven = np.asarray(character.parameters_for_joints(all_body_ids.tolist()), bool)
    pose_mask = np.asarray(character.parameter_transform.pose_parameters, bool)
    active = driven & pose_mask
    for n in ("spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible",
              "arm_length_flexible", "hip_width_flexible", "leg_length_flexible"):
        if n in pi:
            active[pi[n]] = True
    for i, n in enumerate(pnames):
        ln = n.lower()
        if i < 6 or any(tok in ln for tok in ("thumb", "index", "middle", "ring", "pinky", "finger")):
            active[i] = False
    lo_full, hi_full = character.model_parameter_limits
    lo_full = np.asarray(lo_full, float); hi_full = np.asarray(hi_full, float)
    active &= np.isfinite(lo_full) & np.isfinite(hi_full) & ((hi_full - lo_full) > 1e-7)
    active_idx = np.flatnonzero(active)
    ilo, ihi = [], []
    for idx in active_idx:
        lo, hi = float(lo_full[idx]), float(hi_full[idx])
        lo, hi = max(lo, -3.2), min(hi, 3.2)
        if "_flexible" in pnames[idx]:
            lo, hi = max(lo, -2.2), min(hi, 2.2)
        if not hi > lo:
            raise RuntimeError(f"v32v invalid active bound {pnames[idx]} {lo} {hi}")
        ilo.append(lo); ihi.append(hi)

    ordered_j = sorted(COCO_TO_MHR)
    ordered_ids = np.array([joint_ids[j] for j in ordered_j], np.int32)
    offsets = np.zeros((len(ordered_ids), 3), float)
    jslot = {j: i for i, j in enumerate(ordered_j)}

    def decode(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        scale = math.exp(float(x[3]))
        trans = np.asarray(x[4:7], float)
        mp = zero.copy(); mp[active_idx] = x[7:]
        mp = np.asarray(character.apply_model_param_limits(mp), float)
        return R, scale, trans, mp

    def world_joints(x):
        R, scale, trans, mp = decode(x)
        local = np.asarray(geo.model_parameters_to_positions(character, mp, ordered_ids, offsets), float)
        return scale * (local @ R.T) + trans

    r0 = Rotation.from_matrix(R0).as_rotvec()
    x0 = np.concatenate([r0, [math.log(s0)], t0, np.zeros(len(active_idx))])
    lower = np.concatenate([r0 - 2.2, [math.log(.68)], t0 - np.array([260., 260., 240.]), np.asarray(ilo)])
    upper = np.concatenate([r0 + 2.2, [math.log(1.85)], t0 + np.array([260., 260., 240.]), np.asarray(ihi)])
    starts = []
    for yaw_deg in (0., 90., -90., 180.):
        Ry = Rotation.from_rotvec(np.array([0., 0., math.radians(yaw_deg)])).as_matrix()
        rr = Rotation.from_matrix(Ry @ R0).as_rotvec()
        xx = x0.copy(); xx[:3] = rr
        ll = lower.copy(); uu = upper.copy(); ll[:3] = rr - 2.2; uu[:3] = rr + 2.2
        starts.append((yaw_deg, xx, ll, uu))

    eval_count = 0
    def residual(x):
        nonlocal eval_count
        eval_count += 1
        Xw = world_joints(x)
        res = []
        for c in (LAR, RAR):
            for j, row in obs[c].items():
                uv = base.project_points(cams[c], Xw[jslot[j]].reshape(1, 3))[0]
                if not np.all(np.isfinite(uv)):
                    res.extend([300., 300.]); continue
                w = math.sqrt(max(MIN_CONF, row["weight"]))
                res.extend(((uv - row["xy"]) * w).tolist())
        for value, idx in zip(x[7:], active_idx):
            lam = .35 if "_flexible" in pnames[idx] else .10
            res.append(lam * float(value))
        return np.asarray(res, float)

    trials = []
    best = None
    for yaw_deg, xx, ll, uu in starts:
        result = least_squares(residual, xx, bounds=(ll, uu), method="trf", loss="soft_l1", f_scale=4.0,
                               x_scale="jac", max_nfev=a.max_nfev, ftol=2e-7, xtol=2e-7, gtol=2e-7, verbose=0)
        Xw = world_joints(result.x)
        ls = repro_stats(base.project_points, Xw, LAR, obs[LAR], cams, jslot)
        rs = repro_stats(base.project_points, Xw, RAR, obs[RAR], cams, jslot)
        score = ls["median_px"] + rs["median_px"] + .20 * (ls["p90_px"] + rs["p90_px"])
        row = {"yaw_start_deg": yaw_deg, "success": bool(result.success), "cost": float(result.cost),
               "nfev": int(result.nfev), "lar": ls, "rar": rs, "selection_score": float(score)}
        trials.append(row)
        if best is None or score < best[0]:
            best = (score, result, Xw)
    assert best is not None
    _, result, Xw = best
    R, scale, trans, mp = decode(result.x)

    lstats = repro_stats(base.project_points, Xw, LAR, obs[LAR], cams, jslot)
    rstats = repro_stats(base.project_points, Xw, RAR, obs[RAR], cams, jslot)
    bstats = repro_stats(base.project_points, Xw, BCAST, obs[BCAST], cams, jslot)

    state = np.asarray(geo.model_parameters_to_skeleton_state(character, mp), float)
    verts_local = np.asarray(character.skin_points(state), float)
    verts_world = scale * (verts_local @ R.T) + trans
    faces = np.asarray(character.mesh.faces, np.int32)
    np.savez_compressed(
        a.out / "v32v_fit.npz", model_parameters=mp.astype(np.float32),
        rotation=R.astype(np.float32), scale=np.array([scale], np.float32),
        translation=trans.astype(np.float32), vertices_world_cm=verts_world.astype(np.float32), faces=faces,
    )

    projected = {c: {j: base.project_points(cams[c], Xw[jslot[j]].reshape(1, 3))[0] for j in ordered_j} for c in CAMS}
    measured = {
        LAR: {j: row["xy"] for j, row in obs[LAR].items()},
        RAR: {j: row["xy"] for j, row in obs[RAR].items()},
        BCAST: {j: row["xy"] for j, row in obs[BCAST].items()},
    }
    limg = lar_frames[0]
    overlays = []
    for c, img in ((LAR, limg), (BCAST, bimg), (RAR, rar_frames[0])):
        mask, _ = base.mesh_mask(cams[c], verts_world, faces)
        role = "FIT" if c in (LAR, RAR) else "HELD-OUT"
        ov = base.draw_pose_overlay(img, measured[c], projected[c],
                                    f"v32v {c} {role} | yellow measured | magenta fitted MHR | cyan mesh", mask)
        cv2.imwrite(str(a.out / f"v32v_{c.replace(' ', '_')}_overlay.png"), ov)
        overlays.append(ov)
    cv2.imwrite(str(a.out / "v32v_three_camera_overlay.png"), np.hstack(overlays))

    # Preserve state-selection provenance without enormous optical-flow traces.
    state_audit = {
        "version": "v32v_global_lar_state_selection",
        "policy": "one whole-body LAR source selected globally; no per-joint temporal source mixing",
        "selected_lar_source": lbest["source"], "selected_lar_anchor_rel": int(lbest["anchor_rel"]),
        "selected_lar_detection_index": int(lbest["selected_index"]),
        "selected_lar_epipolar": raw_epi, "lar_candidates": lar_candidates,
        "rar_source": "real_t+04_dense_tracked_to_t00", "rar_detection_index": int(ri),
        "rar_ranked_candidates": rrank,
        "broadcast_source": "direct_t00_heldout", "broadcast_detection_index": int(bi),
        "broadcast_ranked_candidates": brank,
    }
    (a.out / "v32v_state_audit.json").write_text(json.dumps(state_audit, indent=2))

    lar_gate = bool(lstats["joint_count"] >= 7 and lstats["median_px"] <= 8.0 and lstats["p90_px"] <= 16.0 and lstats["max_px"] <= 26.0)
    rar_gate = bool(rstats["joint_count"] >= 7 and rstats["median_px"] <= 8.0 and rstats["p90_px"] <= 16.0 and rstats["max_px"] <= 26.0)
    broadcast_holdout_gate = bool(bstats["joint_count"] >= 7 and bstats["median_px"] <= 24.0 and bstats["p75_px"] <= 36.0)
    mesh_finite = bool(np.all(np.isfinite(verts_world)))
    two_view_gate = bool(result.success and state_gate and lar_gate and rar_gate and mesh_finite)
    production_gate = bool(two_view_gate and broadcast_holdout_gate)

    qa = {
        "version": "v32v_lar_rar_global_state_mhr",
        "status": ("PASS_V32V_THREE_CAMERA_LEAVE_ONE_OUT" if production_gate else
                   "PASS_V32V_LAR_RAR_GEOMETRY_BROADCAST_HELDOUT_FAIL" if two_view_gate else
                   "FAIL_CLOSED_V32V_LAR_RAR_GLOBAL_STATE"),
        "native_resolution": [W, H], "generated_rgb": False, "novel_view_rendered": False,
        "camera_geometry_modified": False, "cameras_used_for_fit": [LAR, RAR], "heldout_camera": BCAST,
        "temporal_mixing_per_joint": False,
        "selected_lar_source": lbest["source"], "selected_lar_anchor_rel": int(lbest["anchor_rel"]),
        "raw_lar_rar_epipolar": raw_epi,
        "per_camera_reprojection": {LAR: lstats, RAR: rstats, BCAST: bstats},
        "vertex_count": int(len(verts_world)), "face_count": int(len(faces)),
        "active_model_parameter_count": int(len(active_idx)),
        "external_scale": float(scale), "external_rotation": R.tolist(), "external_translation_cm": trans.tolist(),
        "multistart_trials": trials,
        "optimizer": {"success": bool(result.success), "message": str(result.message), "cost": float(result.cost),
                      "nfev": int(result.nfev), "total_residual_calls": int(eval_count)},
        "gate": {
            "raw_state_consistency": state_gate, "lar_fit_gate": lar_gate, "rar_fit_gate": rar_gate,
            "broadcast_heldout_gate": broadcast_holdout_gate, "mesh_finite": mesh_finite,
            "two_view_geometry_gate": two_view_gate, "three_camera_leave_one_out_gate": production_gate,
            "static_0_15_geometry_unlocked": production_gate,
        },
        "next_if_pass": "render native 960x540 source-grounded static 0/5/10/15 degree arc; no animation yet",
        "warning": "Numerical PASS still requires visual inspection that the mesh follows Adams, especially arms/hands at the rim, in all three native overlays.",
    }
    (a.out / "v32v_qa.json").write_text(json.dumps(qa, indent=2))
    print(json.dumps({"status": qa["status"], "selected_lar_source": lbest["source"],
                      "raw_epipolar": raw_epi, "lar": lstats, "rar": rstats, "broadcast_heldout": bstats,
                      "optimizer": qa["optimizer"], "gate": qa["gate"]}, indent=2), flush=True)

    # Fail closed unless the independent Broadcast holdout also supports the body.
    if not production_gate:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
