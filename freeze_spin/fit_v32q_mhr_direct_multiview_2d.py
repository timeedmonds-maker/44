from __future__ import annotations

"""v32q: fit one anatomical MHR body directly to real Broadcast + RAR pixels.

v32m is explicitly NOT 3-D truth: a tighter audit showed independent joint
triangulation could split the reprojection error unevenly and drift the body onto
the Utah defender. v32q removes that failure mode. It re-measures the source
images and fits one articulated body jointly in image space.

Optimization evidence:
  * Broadcast t+00: direct RF-DETR Adams keypoints.
  * RAR t+00: clean RF-DETR Adams pose at real t+04, dense-flow tracked backward
    through real source frames; only stable tracked joints are used.
  * Left Above Rim is withheld from optimization and used only for validation.

A provisional ray intersection is used only to initialize global body placement.
It is never a target, residual, gate, or reported 3-D truth.

No generated RGB, no fourth camera, no image interpolation, no capsules, no
Gaussian player body, no upscaling. All source-view diagnostics remain 960x540.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from rfdetr import RFDETRKeypointPreview

from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32k_rar_temporal_deblend as v32k
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR, BCAST, RAR = v32j.LAR, v32j.BCAST, v32j.RAR
CAMS = v32j.CAMS
W, H = v32j.W, v32j.H
NAMES = v32j.NAMES
DRAW = v32j.DRAW

COCO_TO_MHR = {
    5: "l_uparm", 6: "r_uparm",
    7: "l_lowarm", 8: "r_lowarm",
    9: "l_wrist", 10: "r_wrist",
    11: "l_upleg", 12: "r_upleg",
    13: "l_lowleg", 14: "r_lowleg",
    15: "l_foot", 16: "r_foot",
}


def ensure_assets(work: Path):
    import pymomentum.geometry as geo
    from mhr.io import get_mhr_fbx_path, get_mhr_model_path
    assets = work / "assets"
    fbx = Path(get_mhr_fbx_path(assets, 1))
    model_path = Path(get_mhr_model_path(assets))
    if not fbx.exists() or not model_path.exists():
        subprocess.run(["mhr-download-assets", "--dest", str(work)], check=True)
    assert fbx.exists(), fbx
    assert model_path.exists(), model_path
    character = geo.Character.load_fbx(str(fbx), str(model_path), load_blendshapes=True)
    return geo, character, fbx, model_path


def camera(scene: dict, label: str):
    d = scene["cameras"][label]
    return np.asarray(d["K_px"], float), np.asarray(d["R_world_to_camera"], float), np.asarray(d["C_world_cm"], float)


def project_points(cam, X: np.ndarray) -> np.ndarray:
    K, R, C = cam
    X = np.atleast_2d(np.asarray(X, float))
    xc = (R @ (X - C).T).T
    q = (K @ xc.T).T
    uv = np.full((len(X), 2), np.nan, float)
    ok = np.abs(q[:, 2]) > 1e-8
    uv[ok] = q[ok, :2] / q[ok, 2:3]
    return uv


def mesh_mask(cam, verts: np.ndarray, faces: np.ndarray):
    uv = project_points(cam, verts)
    mask = np.zeros((H, W), np.uint8)
    tri = uv[faces]
    valid = np.all(np.isfinite(tri), axis=(1, 2)) & np.all(np.abs(tri) < 6000, axis=(1, 2))
    for t in tri[valid]:
        cv2.fillConvexPoly(mask, np.rint(t).astype(np.int32), 255, lineType=cv2.LINE_8)
    return mask, uv


def draw_pose_overlay(img, measured: dict[int, np.ndarray], projected: dict[int, np.ndarray], title: str, mask=None):
    out = img.copy()
    if mask is not None:
        cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, (255, 255, 0), 2, cv2.LINE_AA)
    for a, b in DRAW:
        if a in measured and b in measured:
            cv2.line(out, tuple(np.rint(measured[a]).astype(int)), tuple(np.rint(measured[b]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
        if a in projected and b in projected:
            cv2.line(out, tuple(np.rint(projected[a]).astype(int)), tuple(np.rint(projected[b]).astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
    for p in measured.values():
        cv2.circle(out, tuple(np.rint(p).astype(int)), 4, (0, 255, 255), -1, cv2.LINE_AA)
    for p in projected.values():
        cv2.circle(out, tuple(np.rint(p).astype(int)), 3, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (W, 29), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def similarity_from_provisional(character, geo, joint_ids, local_zero, provisional):
    """Robust Sim(3) initializer only; provisional rays never enter final loss."""
    if len(provisional) < 4:
        # MHR is Y-up; map to NBA Z-up with a conservative action-region seed.
        R0 = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
        return R0, 1.12, np.array([100., 10., 170.])
    idx = np.array([joint_ids[j] for j in provisional], np.int32)
    offsets = np.zeros((len(idx), 3), float)
    src = np.asarray(geo.model_parameters_to_positions(character, local_zero, idx, offsets), float)
    dst = np.stack([provisional[j] for j in provisional])
    # Umeyama similarity, used only as a starting point.
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    denom = float(np.sum(A * A))
    scale = float(np.sum(S) / max(denom, 1e-8))
    scale = float(np.clip(scale, .72, 1.75))
    trans = md - scale * (R @ ms)
    return R, scale, trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-nfev", type=int, default=260)
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True)

    stage = a.b32_root / "stage_a"
    scene = json.loads((stage / "v32_scene_manifest.json").read_text())
    qual = json.loads((stage / "v32_quality_cluster.json").read_text())
    assert scene["resolution"] == [W, H] and set(scene["cameras"]) == set(CAMS)
    cams = {c: camera(scene, c) for c in CAMS}

    model = RFDETRKeypointPreview()
    bimg = cv2.imread(str(stage / "v32_chosen_Broadcast_frame0276.png"))
    limg = cv2.imread(str(stage / "v32_chosen_Left_Above_Rim_frame0260.png"))
    assert bimg is not None and limg is not None

    # Broadcast: direct exact-freeze observation, identity anchored to B32 focal box.
    bdets = v32j.infer(model, bimg)
    bb = np.asarray(qual["focal_player"]["observations"][BCAST]["bbox_xyxy"], float)
    brows = []
    for i, d in enumerate(bdets):
        score = 4.0 * v32j.iou(d["box"], bb) + .5 * v32j.dark_fraction(bimg, d["box"])
        brows.append((float(score), i))
    brows.sort(reverse=True)
    if not brows:
        raise RuntimeError("no Broadcast detections")
    bi = brows[0][1]; bdet = bdets[bi]

    # RAR: identify Adams on clean real t+04, then dense-track his joints to t+00.
    rar_frames = {rel: cv2.imread(str(v32k.burst_path(stage, RAR, rel))) for rel in range(0, 5)}
    assert all(x is not None for x in rar_frames.values())
    rb = np.asarray(qual["focal_player"]["observations"][RAR]["bbox_xyxy"], float)
    target_center = np.array([(rb[0]+rb[2])/2, (rb[1]+rb[3])/2], float)
    r4dets = v32j.infer(model, rar_frames[4])
    ri, rrank = v32k.select_adams_detection(rar_frames[4], r4dets, target_center)
    if ri is None:
        raise RuntimeError("no RAR t+04 Adams candidate")
    r4 = r4dets[ri]
    rxy, rvalid, rtrace = v32l.dense_track_back(rar_frames, 4, r4["xy"], r4["conf"])

    # Assemble only anatomical body observations. No v32m 3-D positions are read.
    obs = {BCAST: {}, RAR: {}}
    for j in COCO_TO_MHR:
        if bdet["conf"][j] >= .20:
            obs[BCAST][j] = {"xy": np.asarray(bdet["xy"][j], float), "weight": float(np.clip(bdet["conf"][j], .20, 1.0))}
        if bool(rvalid[j]) and r4["conf"][j] >= .20:
            obs[RAR][j] = {"xy": np.asarray(rxy[j], float), "weight": float(np.clip(r4["conf"][j], .20, 1.0))}
    common = sorted(set(obs[BCAST]) & set(obs[RAR]))
    if len(obs[BCAST]) < 8 or len(obs[RAR]) < 7 or len(common) < 6:
        raise RuntimeError(f"insufficient direct observations B={len(obs[BCAST])} R={len(obs[RAR])} common={len(common)}")

    geo, character, fbx, model_path = ensure_assets(a.work)
    jnames = list(character.skeleton.joint_names); ji = {n:i for i,n in enumerate(jnames)}
    pnames = list(character.parameter_transform.names); pi = {n:i for i,n in enumerate(pnames)}
    if any(name not in ji for name in COCO_TO_MHR.values()):
        raise RuntimeError("required MHR anatomical joint missing")
    joint_ids = {j: ji[name] for j, name in COCO_TO_MHR.items()}
    zero = np.zeros(len(pnames), np.float64)

    # Provisional ray intersections ONLY initialize global Sim(3).
    provisional = {}
    vcams = {c: v32j.cam(scene, c) for c in CAMS}
    for j in common:
        X = v32j.triangulate_rays(vcams, {BCAST: obs[BCAST][j]["xy"], RAR: obs[RAR][j]["xy"]})
        if X is not None and np.all(np.isfinite(X)) and (-350 <= X[0] <= 1250 and -750 <= X[1] <= 750 and -60 <= X[2] <= 450):
            provisional[j] = np.asarray(X, float)
    R0, s0, t0 = similarity_from_provisional(character, geo, joint_ids, zero, provisional)

    all_body_ids = np.array([joint_ids[j] for j in COCO_TO_MHR], np.int32)
    driven = np.asarray(character.parameters_for_joints(all_body_ids.tolist()), bool)
    pose_mask = np.asarray(character.parameter_transform.pose_parameters, bool)
    active = driven & pose_mask
    # Add restrained gross body proportions; keep hands/fingers/face/identity inactive.
    for n in ("spine_length_flexible", "neck_length_flexible", "shoulder_width_flexible", "arm_length_flexible", "hip_width_flexible", "leg_length_flexible"):
        if n in pi: active[pi[n]] = True
    for i, n in enumerate(pnames):
        ln = n.lower()
        if i < 6 or any(tok in ln for tok in ("thumb", "index", "middle", "ring", "pinky", "finger")):
            active[i] = False
    lo_full, hi_full = character.model_parameter_limits
    lo_full = np.asarray(lo_full, float); hi_full = np.asarray(hi_full, float)
    # Critical SciPy rule: fixed/equal-bound MHR params are removed, never widened.
    active &= np.isfinite(lo_full) & np.isfinite(hi_full) & ((hi_full - lo_full) > 1e-7)
    active_idx = np.flatnonzero(active)

    ilo, ihi = [], []
    for idx in active_idx:
        lo, hi = float(lo_full[idx]), float(hi_full[idx])
        lo = max(lo, -3.2); hi = min(hi, 3.2)
        if "_flexible" in pnames[idx]:
            lo, hi = max(lo, -2.2), min(hi, 2.2)
        if not hi > lo:
            raise RuntimeError(f"invalid active bound {pnames[idx]} {lo} {hi}")
        ilo.append(lo); ihi.append(hi)

    # Optimize all mapped joint positions in one low-level FK call.
    ordered_j = sorted(COCO_TO_MHR)
    ordered_ids = np.array([joint_ids[j] for j in ordered_j], np.int32)
    offsets = np.zeros((len(ordered_ids), 3), float)
    jslot = {j:i for i,j in enumerate(ordered_j)}

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

    def repro_stats(Xw, camera_label, observations):
        vals = []
        for j, row in observations.items():
            uv = project_points(cams[camera_label], Xw[jslot[j]].reshape(1,3))[0]
            vals.append(float(np.linalg.norm(uv - row["xy"])))
        a = np.asarray(vals, float)
        return {"joint_count": int(len(a)), "median_px": float(np.median(a)), "p75_px": float(np.percentile(a,75)), "p90_px": float(np.percentile(a,90)), "max_px": float(np.max(a))}

    r0 = Rotation.from_matrix(R0).as_rotvec()
    x0 = np.concatenate([r0, [math.log(s0)], t0, np.zeros(len(active_idx))])
    lower = np.concatenate([r0-2.2, [math.log(.68)], t0-np.array([260.,260.,240.]), np.asarray(ilo)])
    upper = np.concatenate([r0+2.2, [math.log(1.85)], t0+np.array([260.,260.,240.]), np.asarray(ihi)])

    # Multiple yaw starts reduce dependence on imperfect provisional triangulation.
    starts = []
    for yaw_deg in (0., 90., -90., 180.):
        Ry = Rotation.from_rotvec(np.array([0., 0., math.radians(yaw_deg)])).as_matrix()
        rr = Rotation.from_matrix(Ry @ R0).as_rotvec()
        xx = x0.copy(); xx[:3] = rr
        # Bounds are centered on each start's rotation to avoid rotvec wrap artifacts.
        ll = lower.copy(); uu = upper.copy(); ll[:3] = rr-2.2; uu[:3] = rr+2.2
        starts.append((yaw_deg, xx, ll, uu))

    eval_count = 0
    def residual(x):
        nonlocal eval_count
        eval_count += 1
        Xw = world_joints(x)
        res = []
        for c in (BCAST, RAR):
            for j, row in obs[c].items():
                uv = project_points(cams[c], Xw[jslot[j]].reshape(1,3))[0]
                if not np.all(np.isfinite(uv)):
                    res.extend([300.,300.]); continue
                # Confidence affects reliability but never lets one camera dominate by focal length.
                w = math.sqrt(max(.20, row["weight"]))
                res.extend(((uv-row["xy"]) * w).tolist())
        # Anatomical prior: MHR limits enforce hard validity; this light L2 avoids
        # unconstrained limbs taking extreme legal values to overfit occlusion.
        for value, idx in zip(x[7:], active_idx):
            lam = .35 if "_flexible" in pnames[idx] else .10
            res.append(lam * float(value))
        return np.asarray(res, float)

    trials = []
    best = None
    for yaw_deg, xx, ll, uu in starts:
        result = least_squares(residual, xx, bounds=(ll,uu), method="trf", loss="soft_l1", f_scale=4.0,
                               x_scale="jac", max_nfev=a.max_nfev, ftol=2e-7, xtol=2e-7, gtol=2e-7, verbose=0)
        Xw = world_joints(result.x)
        bs = repro_stats(Xw, BCAST, obs[BCAST]); rs = repro_stats(Xw, RAR, obs[RAR])
        score = bs["median_px"] + rs["median_px"] + .20*(bs["p90_px"]+rs["p90_px"])
        row = {"yaw_start_deg":yaw_deg,"success":bool(result.success),"cost":float(result.cost),"nfev":int(result.nfev),"broadcast":bs,"rar":rs,"selection_score":float(score)}
        trials.append(row)
        if best is None or score < best[0]: best = (score, result, Xw)
    assert best is not None
    _, result, Xw = best
    R, scale, trans, mp = decode(result.x)

    bstats = repro_stats(Xw, BCAST, obs[BCAST]); rstats = repro_stats(Xw, RAR, obs[RAR])

    # Left is truly withheld: infer every candidate after fitting and score only now.
    ldets = v32j.infer(model, limg)
    left_rows = []
    for i, d in enumerate(ldets):
        es = []
        for j in ordered_j:
            if d["conf"][j] < .20: continue
            uv = project_points(cams[LAR], Xw[jslot[j]].reshape(1,3))[0]
            es.append(float(np.linalg.norm(uv-d["xy"][j])))
        if len(es) >= 4:
            ea = np.asarray(es,float)
            med=float(np.median(ea)); p75=float(np.percentile(ea,75)); p90=float(np.percentile(ea,90)); mx=float(np.max(ea))
        else:
            med=p75=p90=mx=999.
        left_rows.append({"index":i,"joint_count":len(es),"median_px":med,"p75_px":p75,"p90_px":p90,"max_px":mx,"cost":med+.25*p75})
    left_rows.sort(key=lambda z:z["cost"])
    left_best = left_rows[0] if left_rows else {"index":None,"joint_count":0,"median_px":999.,"p75_px":999.,"p90_px":999.,"max_px":999.}
    ldet = None if left_best["index"] is None else ldets[left_best["index"]]

    # Skin the actual MHR triangle surface through the same proven low-level path.
    state = np.asarray(geo.model_parameters_to_skeleton_state(character, mp), float)
    verts_local = np.asarray(character.skin_points(state), float)
    verts_world = scale * (verts_local @ R.T) + trans
    faces = np.asarray(character.mesh.faces, np.int32)
    np.savez_compressed(a.out/"v32q_fit.npz", model_parameters=mp.astype(np.float32), rotation=R.astype(np.float32), scale=np.array([scale],np.float32), translation=trans.astype(np.float32), vertices_world_cm=verts_world.astype(np.float32), faces=faces)

    projected = {c:{j:project_points(cams[c], Xw[jslot[j]].reshape(1,3))[0] for j in ordered_j} for c in CAMS}
    measured_b = {j:r["xy"] for j,r in obs[BCAST].items()}
    measured_r = {j:r["xy"] for j,r in obs[RAR].items()}
    measured_l = {} if ldet is None else {j:np.asarray(ldet["xy"][j],float) for j in ordered_j if ldet["conf"][j]>=.20}
    masks = {}; overlays=[]
    for c, img, meas in ((LAR,limg,measured_l),(BCAST,bimg,measured_b),(RAR,rar_frames[0],measured_r)):
        mask,_ = mesh_mask(cams[c], verts_world, faces); masks[c]=mask
        ov = draw_pose_overlay(img, meas, projected[c], f"v32q {c} | yellow measured | magenta fitted MHR | cyan mesh", mask)
        cv2.imwrite(str(a.out/f"v32q_{c.replace(' ','_')}_overlay.png"),ov); overlays.append(ov)
    cv2.imwrite(str(a.out/"v32q_three_camera_overlay.png"),np.hstack(overlays))

    # Separate per-camera gates. No pooled residual is allowed.
    train_pass = (bstats["joint_count"]>=8 and bstats["median_px"]<=8.0 and bstats["p90_px"]<=16.0 and bstats["max_px"]<=26.0 and
                  rstats["joint_count"]>=7 and rstats["median_px"]<=8.0 and rstats["p90_px"]<=16.0 and rstats["max_px"]<=26.0)
    left_pass = (left_best["joint_count"]>=6 and left_best["median_px"]<=24.0 and left_best["p75_px"]<=36.0)
    passed = bool(result.success and train_pass and left_pass and np.all(np.isfinite(verts_world)))
    qa = {
        "version":"v32q_mhr_direct_multiview_2d",
        "status":"PASS_V32Q_DIRECT_MULTIVIEW_ANATOMICAL_POSITION" if passed else "FAIL_CLOSED_V32Q_DIRECT_MULTIVIEW_ANATOMICAL_POSITION",
        "native_resolution":[W,H],"cameras_used_for_fit":[BCAST,RAR],"heldout_camera":LAR,
        "generated_rgb":False,"novel_view_rendered":False,"camera_geometry_modified":False,
        "v32m_3d_truth_used":False,"provisional_ray_geometry_role":"initialization only; absent from residual and gates",
        "rar_temporal_identity":"RF-DETR real t+04 Adams; DIS dense backward tracking to real t+00",
        "broadcast_selected_index":int(bi),"rar_t04_selected_index":int(ri),"rar_ranked_candidates":rrank,
        "observed_joints":{"Broadcast":[NAMES[j] for j in sorted(obs[BCAST])],"Right Above Rim":[NAMES[j] for j in sorted(obs[RAR])]},
        "provisional_initialization_joint_count":int(len(provisional)),
        "mhr_loader":"pymomentum.geometry.Character.load_fbx","mhr_fbx":str(fbx),"mhr_model":str(model_path),
        "vertex_count":int(len(verts_world)),"face_count":int(len(faces)),"active_model_parameter_count":int(len(active_idx)),
        "external_scale":float(scale),"external_rotation":R.tolist(),"external_translation_cm":trans.tolist(),
        "multistart_trials":trials,"optimizer":{"success":bool(result.success),"message":str(result.message),"cost":float(result.cost),"nfev":int(result.nfev),"total_residual_calls":int(eval_count)},
        "per_camera_reprojection":{"Broadcast":bstats,"Right Above Rim":rstats},
        "left_candidates":left_rows,"left_best_validation":left_best,
        "gate":{"optimizer_success":bool(result.success),"broadcast_separate_gate":bool(train_pass and bstats["median_px"]<=8.0),"rar_separate_gate":bool(train_pass and rstats["median_px"]<=8.0),"left_heldout_gate":bool(left_pass),"mesh_finite":bool(np.all(np.isfinite(verts_world))),"silhouette_refinement_unlocked":passed},
        "warning":"A numerical PASS still requires human visual inspection that the projected mesh remains on Adams rather than the overlapping Utah defender in all three native overlays."
    }
    (a.out/"v32q_qa.json").write_text(json.dumps(qa,indent=2))
    print(json.dumps({"status":qa["status"],"broadcast":bstats,"rar":rstats,"left":left_best,"optimizer":qa["optimizer"],"gate":qa["gate"]},indent=2),flush=True)
    if not passed: raise SystemExit(6)

if __name__ == "__main__":
    main()
