from __future__ import annotations

"""v32z: identity-locked Broadcast temporal holdout + LAR occlusion rejection.

v32y established two concrete measurement failures:
* LAR left_wrist is the lone 37.3 px articulated-fit outlier.  In the source
  view it is severely overlapped/foreshortened; it is dropped as unsupported,
  not replaced by another time or generated point.
* Broadcast geometry-only identity selection chose a white Utah defender.
  The correct Houston-black Adams detection is a different person, but direct
  t00 is not the best synchronized body state relative to trusted RAR.

v32z therefore:
1) keeps the v32x rank-1 coherent LAR t+03 state;
2) drops only LAR left_wrist (COCO 9), with an explicit evidence record;
3) scans real Broadcast t00 and t+02..t+06 detections; each candidate is one
   whole body tracked to t00, never per-joint temporal mixing;
4) restricts identity candidates to Houston-dark uniforms before geometry
   ranking, then selects the best whole-body Broadcast state by calibrated
   epipolar agreement with trusted RAR;
5) Broadcast remains strictly held out from the MHR optimization;
6) uses the unchanged v32v acceptance gates and native 960x540 inputs only.

No camera refinement, no synthesized RGB, no interpolation, no upscale.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import run_v32x_lar_candidate_sweep as v32x
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR, BCAST, RAR = v32v.LAR, v32v.BCAST, v32v.RAR
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF
LEFT_WRIST = 9
CTX = {}


def choose_lar_rank1_drop_occluded_wrist(model, stage, qual, scene, rar_xy, rar_valid):
    candidates, frames = v32x.all_coherent_candidates(model, stage, scene, rar_xy, rar_valid)
    best = copy.deepcopy(candidates[0])
    if LEFT_WRIST < len(best["valid0"]):
        best["valid0"][LEFT_WRIST] = False
        best["anchor_conf"][LEFT_WRIST] = 0.0
    CTX.update({
        "model": model, "stage": stage, "scene": scene,
        "rar_xy": np.asarray(rar_xy, float), "rar_valid": np.asarray(rar_valid, bool),
        "lar_candidate": v32x.serial_candidate(1, candidates[0]),
        "lar_rejected": [{
            "joint": LEFT_WRIST, "name": v32j.NAMES[LEFT_WRIST],
            "reason": "v32y verified 37.323 px MHR residual; source overlap/foreshortening makes detector wrist unsupported; dropped, not replaced"
        }],
    })
    serial = [v32x.serial_candidate(i + 1, c) for i, c in enumerate(candidates)]
    return best, serial, frames


def broadcast_temporal_candidates():
    model = CTX["model"]; stage = CTX["stage"]; scene = CTX["scene"]
    rar_xy = CTX["rar_xy"]; rar_valid = CTX["rar_valid"]
    frames = v32v.load_burst(stage, BCAST, range(0, 7))
    rows = []
    for rel in (0, 2, 3, 4, 5, 6):
        dets = v32j.infer(model, frames[rel])
        for i, d in enumerate(dets):
            dark, torso, confident = v32w.appearance(d, frames[rel])
            if confident < 7:
                continue
            # Identity first: Adams wears Houston black.  The v32y wrong Utah
            # defender had dark=0.169/torso=0.196; the direct Adams candidate
            # had dark=0.843/torso=0.884.  Keep a conservative Houston-dark set.
            identity_dark = bool(torso >= 0.45 or dark >= 0.55)
            if not identity_dark:
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
            epi = v32v.epipolar_stats(scene, BCAST, RAR, xy0, valid0, rar_xy, rar_valid)
            common = int(epi["joint_count"])
            if common < 7:
                continue
            shortage = max(0, 9 - common)
            # Geometry decides within the identity-locked Houston set.
            score = float(epi["median_px"] + .30 * epi["p90_px"] + 10.0 * shortage - 0.75 * torso - 0.30 * dark)
            pseudo = dict(d)
            pseudo["xy"] = np.asarray(xy0, float)
            # Retain only tracked/supported joints in the held-out measurement.
            conf = np.asarray(d["conf"], float).copy()
            conf[~np.asarray(valid0, bool)] = 0.0
            pseudo["conf"] = conf
            rows.append({
                "source": source, "anchor_rel": int(rel), "selected_index": int(i),
                "score": score, "dark_fraction": float(dark), "torso_dark_score": float(torso),
                "confident_body_joints": confident, "box_xyxy": np.asarray(d["box"], float).tolist(),
                "epipolar": epi, "pseudo": pseudo,
            })
    rows.sort(key=lambda r: (r["score"], r["epipolar"]["median_px"], -r["torso_dark_score"]))
    if not rows:
        raise RuntimeError("v32z: no Houston-dark Broadcast candidate has >=7 RAR-comparable joints")
    return rows


def broadcast_identity_temporal_rank(_img, detections, _bbox):
    rows = broadcast_temporal_candidates()
    best = rows[0]
    # Append the selected coherent tracked body as one held-out observation.
    # v32v subsequently reads bdets[bi]; this does not enter MHR optimization.
    detections.append(best["pseudo"])
    bi = len(detections) - 1
    serial = []
    for rank, r in enumerate(rows, 1):
        serial.append({k: v for k, v in r.items() if k != "pseudo"} | {"rank": rank})
    CTX["broadcast_selected"] = serial[0]
    CTX["broadcast_rank"] = serial[:12]
    return bi, serial


def repro_stats_with_joints(project_fn, Xw, camera_label, observations, cams, jslot):
    vals=[]; rows=[]
    for j,row in observations.items():
        uv=project_fn(cams[camera_label],Xw[jslot[j]].reshape(1,3))[0]
        if not np.all(np.isfinite(uv)): continue
        m=np.asarray(row["xy"],float); e=float(np.linalg.norm(uv-m)); vals.append(e)
        rows.append({"joint":int(j),"name":v32j.NAMES[int(j)],"residual_px":e,
                     "measured_xy":m.tolist(),"projected_xy":np.asarray(uv,float).tolist()})
    rows.sort(key=lambda r:r["residual_px"],reverse=True)
    if not vals:
        return {"joint_count":0,"median_px":999.,"p75_px":999.,"p90_px":999.,"max_px":999.,"per_joint":rows}
    a=np.asarray(vals,float)
    return {"joint_count":int(len(a)),"median_px":float(np.median(a)),"p75_px":float(np.percentile(a,75)),
            "p90_px":float(np.percentile(a,90)),"max_px":float(np.max(a)),"per_joint":rows}


def main():
    orig_sel=v32v.choose_global_lar_state; orig_rank=v32v.focal_rank; orig_repro=v32v.repro_stats
    v32v.choose_global_lar_state=choose_lar_rank1_drop_occluded_wrist
    v32v.focal_rank=broadcast_identity_temporal_rank
    v32v.repro_stats=repro_stats_with_joints
    rc=0
    try:
        v32v.main()
    except SystemExit as exc:
        rc=int(exc.code or 0)
    finally:
        v32v.choose_global_lar_state=orig_sel; v32v.focal_rank=orig_rank; v32v.repro_stats=orig_repro
    out=None
    if "--out" in sys.argv: out=Path(sys.argv[sys.argv.index("--out")+1])
    if out is not None:
        qp=out/"v32v_qa.json"; qa=json.loads(qp.read_text()) if qp.exists() else None
        audit={
            "version":"v32z_identity_locked_broadcast_temporal",
            "lar_rejected":CTX.get("lar_rejected",[]),
            "broadcast_selected":CTX.get("broadcast_selected"),
            "broadcast_rank":CTX.get("broadcast_rank",[]),
            "qa":qa,"solver_exit_code":rc,
        }
        (out/"v32z_policy_and_residuals.json").write_text(json.dumps(audit,indent=2))
    if rc: raise SystemExit(rc)

if __name__ == "__main__":
    main()
