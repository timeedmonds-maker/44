from __future__ import annotations

"""v32i stage 1: source-grounded three-camera articulated position observations.

This stage deliberately stops before human-surface reconstruction.  It uses the
three accepted metric cameras and the real 13-frame B32 burst to recover a robust
3D joint skeleton for the focal player (Steven Adams) at the exact freeze.

Important provenance rules:
- official native 960x540 RGB only;
- cameras are read from the accepted v32 scene manifest and never moved;
- torchvision Keypoint R-CNN is an observation generator only;
- B32 focal masks/bboxes are identity anchors only, not final geometry;
- Broadcast + Right-Above-Rim establish the initial focal identity;
- the resulting 3D skeleton is projected into Left-Above-Rim to find/validate a
  real measured Left pose.  A Left match is never forced through an occlusion;
- no mesh, body capsules, generated texture, inpainting, warping or upscaling.

The output is diagnostic JSON plus native-resolution overlays.  A later v32i
surface stage may proceed only if this observation stage passes its fail-closed
three-view geometry gate.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

CAMERAS = ("Left Above Rim", "Broadcast", "Right Above Rim")
LAR, BCAST, RAR = CAMERAS
REL = tuple(range(-6, 7))
W, H = 960, 540

JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
BONES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a); bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def center(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], np.float64)


def camera(scene, label):
    d = scene["cameras"][label]
    K = np.asarray(d["K_px"], np.float64)
    R = np.asarray(d["R_world_to_camera"], np.float64)
    C = np.asarray(d["C_world_cm"], np.float64)
    t = -R @ C
    P = K @ np.column_stack([R, t])
    return {"K": K, "R": R, "C": C, "t": t, "P": P}


def fundamental(c1, c2):
    # F = K2^-T [t21]_x R21 K1^-1, using world-to-camera poses.
    R1, C1, K1 = c1["R"], c1["C"], c1["K"]
    R2, C2, K2 = c2["R"], c2["C"], c2["K"]
    R21 = R2 @ R1.T
    t21 = R2 @ (C1 - C2)
    tx = np.array([[0, -t21[2], t21[1]], [t21[2], 0, -t21[0]], [-t21[1], t21[0], 0]], np.float64)
    F = np.linalg.inv(K2).T @ tx @ R21 @ np.linalg.inv(K1)
    n = np.linalg.norm(F)
    return F / max(n, 1e-12)


def point_line_dist(p, line):
    a, b, c = map(float, line)
    return abs(a * p[0] + b * p[1] + c) / max(1e-9, math.hypot(a, b))


def symmetric_epi(F, p1, p2):
    x1 = np.array([p1[0], p1[1], 1.0], np.float64)
    x2 = np.array([p2[0], p2[1], 1.0], np.float64)
    return 0.5 * (point_line_dist(p2, F @ x1) + point_line_dist(p1, F.T @ x2))


def triangulate(cams, obs):
    rows = []
    for label, uv in obs.items():
        P = cams[label]["P"]
        u, v = map(float, uv)
        rows += [u * P[2] - P[0], v * P[2] - P[1]]
    if len(rows) < 4:
        return None
    A = np.asarray(rows, np.float64)
    _, _, vh = np.linalg.svd(A)
    Xh = vh[-1]
    if abs(Xh[3]) < 1e-10:
        return None
    X = Xh[:3] / Xh[3]
    return X if np.all(np.isfinite(X)) else None


def project(cam, X):
    xc = cam["R"] @ (np.asarray(X, np.float64) - cam["C"])
    if xc[2] <= 1e-6:
        return None, float(xc[2])
    q = cam["K"] @ xc
    return q[:2] / q[2], float(xc[2])


def torso_dark_fraction(image, box):
    x1, y1, x2, y2 = map(int, np.round(box))
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    # Upper/middle body only; court/floor should not dominate jersey colour.
    yy1 = y1 + int(0.15 * (y2 - y1)); yy2 = y1 + int(0.62 * (y2 - y1))
    xx1 = x1 + int(0.18 * (x2 - x1)); xx2 = x1 + int(0.82 * (x2 - x1))
    patch = image[max(y1,yy1):max(y1+1,yy2), max(x1,xx1):max(x1+1,xx2)]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[...,2] < 105))


def mask_joint_fraction(mask, det, threshold=0.25):
    kp, ks = det["kp"], det["ks"]
    good = np.where(ks >= threshold)[0]
    if len(good) == 0:
        return 0.0
    inside = 0
    for j in good:
        x, y = np.round(kp[j]).astype(int)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
            inside += 1
    return inside / len(good)


def detect_all(model, image, score_threshold):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        o = model([to_tensor(rgb)])[0]
    scores = o["scores"].detach().cpu().numpy()
    boxes = o["boxes"].detach().cpu().numpy()
    kps = o["keypoints"].detach().cpu().numpy()
    raw_ks = o.get("keypoints_scores")
    raw_ks = None if raw_ks is None else raw_ks.detach().cpu().numpy()
    out = []
    for idx in np.where(scores >= score_threshold)[0].tolist():
        ks = raw_ks[idx] if raw_ks is not None else kps[idx,:,2]
        out.append({
            "source_index": int(idx),
            "score": float(scores[idx]),
            "box": boxes[idx].astype(np.float64),
            "kp": kps[idx,:,:2].astype(np.float64),
            "ks": ks.astype(np.float64),
            "dark_fraction": torso_dark_fraction(image, boxes[idx]),
        })
    return out


def anchor_score(det, anchor_box, mask):
    b_iou = iou(det["box"], anchor_box)
    mask_frac = mask_joint_fraction(mask, det)
    cdist = float(np.linalg.norm(center(det["box"]) - center(anchor_box)))
    scale = max(20.0, math.sqrt(max(1.0, (anchor_box[2]-anchor_box[0])*(anchor_box[3]-anchor_box[1]))))
    return 3.0*b_iou + 1.5*mask_frac + 0.35*det["dark_fraction"] - 0.35*(cdist/scale)


def epi_pose_cost(F, a, b, conf=0.35):
    good = np.where((a["ks"] >= conf) & (b["ks"] >= conf))[0]
    if len(good) < 4:
        return 1e6, {"joint_count": int(len(good)), "median_epi_px": 999.0}
    es = np.array([symmetric_epi(F, a["kp"][j], b["kp"][j]) for j in good], np.float64)
    return float(np.median(es)), {
        "joint_count": int(len(good)),
        "median_epi_px": float(np.median(es)),
        "p75_epi_px": float(np.percentile(es, 75)),
    }


def choose_anchor_broadcast(dets, anchor_box, mask):
    if not dets:
        return None, []
    rows = []
    for i, d in enumerate(dets):
        s = anchor_score(d, anchor_box, mask)
        rows.append({"index": i, "score": s, "box": d["box"].tolist(), "dark_fraction": d["dark_fraction"]})
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[0]["index"], rows


def choose_rar(dets, bdet, Fbr, anchor_box, mask, prev=None):
    rows = []
    for i, d in enumerate(dets):
        epi, eq = epi_pose_cost(Fbr, bdet, d)
        if epi >= 1e5:
            continue
        anchor = anchor_score(d, anchor_box, mask)
        motion = 0.0
        if prev is not None:
            motion = float(np.linalg.norm(center(d["box"]) - center(prev["box"]))) / 60.0
        # Epipolar geometry dominates. Anchor/mask/darkness only break crowded-paint ties.
        cost = epi + 3.0*motion - 5.0*anchor
        rows.append({"index": i, "cost": float(cost), "epi": eq, "anchor_score": float(anchor), "motion_norm": float(motion)})
    rows.sort(key=lambda x: x["cost"])
    return (rows[0]["index"] if rows else None), rows


def track_nearest(dets, prev):
    if prev is None or not dets:
        return None
    pc = center(prev["box"])
    pa = max(1.0, (prev["box"][2]-prev["box"][0])*(prev["box"][3]-prev["box"][1]))
    rows = []
    for i,d in enumerate(dets):
        dc = float(np.linalg.norm(center(d["box"]) - pc))
        da = max(1.0, (d["box"][2]-d["box"][0])*(d["box"][3]-d["box"][1]))
        sc = dc/80.0 + 0.8*abs(math.log(da/pa)) - 0.25*d["dark_fraction"]
        rows.append((sc,i))
    rows.sort()
    return rows[0][1]


def triangulated_joints(cams, obs_by_cam, conf=0.30):
    out = {}
    for j,name in enumerate(JOINTS):
        obs = {}
        confs = {}
        for label,d in obs_by_cam.items():
            if d is None or d["ks"][j] < conf:
                continue
            obs[label] = d["kp"][j]
            confs[label] = float(d["ks"][j])
        if len(obs) < 2:
            continue
        X = triangulate(cams, obs)
        if X is None or not (-350 <= X[0] <= 1250 and -750 <= X[1] <= 750 and -60 <= X[2] <= 450):
            continue
        residuals = {}
        valid = True
        for label,uv in obs.items():
            pr,z = project(cams[label], X)
            if pr is None:
                valid = False; break
            residuals[label] = float(np.linalg.norm(pr-uv))
        if not valid:
            continue
        out[j] = {"name": name, "world_cm": X.tolist(), "observed_cameras": list(obs), "confidence": confs, "residual_px": residuals}
    return out


def lar_match_cost(cams, joints, det, conf=0.25):
    errs=[]; used=[]
    for j,row in joints.items():
        if det["ks"][j] < conf:
            continue
        uv,_=project(cams[LAR], np.asarray(row["world_cm"]))
        if uv is None:
            continue
        errs.append(float(np.linalg.norm(uv-det["kp"][j]))); used.append(j)
    if len(errs)<4:
        return 1e6,{"joint_count":len(errs),"median_px":999.0,"p75_px":999.0}
    e=np.asarray(errs)
    return float(np.median(e)+0.20*np.percentile(e,75)), {"joint_count":len(errs),"median_px":float(np.median(e)),"p75_px":float(np.percentile(e,75)),"joint_ids":used}


def draw(image, det, projected, title, measured_colour=(0,255,255), projected_colour=(255,0,255)):
    out=image.copy()
    if det is not None:
        for a,b in BONES:
            if det["ks"][a]>=0.25 and det["ks"][b]>=0.25:
                p1=tuple(np.round(det["kp"][a]).astype(int)); p2=tuple(np.round(det["kp"][b]).astype(int))
                cv2.line(out,p1,p2,measured_colour,2,cv2.LINE_AA)
        for j in range(17):
            if det["ks"][j]>=0.25:
                cv2.circle(out,tuple(np.round(det["kp"][j]).astype(int)),3,measured_colour,-1,cv2.LINE_AA)
    pp={}
    for j,uv in projected.items():
        pp[int(j)]=np.asarray(uv)
    for a,b in BONES:
        if a in pp and b in pp:
            cv2.line(out,tuple(np.round(pp[a]).astype(int)),tuple(np.round(pp[b]).astype(int)),projected_colour,1,cv2.LINE_AA)
    for j,uv in pp.items():
        cv2.circle(out,tuple(np.round(uv).astype(int)),3,projected_colour,1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1)
    cv2.putText(out,title,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),1,cv2.LINE_AA)
    return out


def serial_det(d):
    if d is None: return None
    return {"source_index":d["source_index"],"score":d["score"],"box_xyxy":d["box"].tolist(),"dark_fraction":d["dark_fraction"],"keypoints":{JOINTS[j]:{"xy":d["kp"][j].tolist(),"score":float(d["ks"][j])} for j in range(17)}}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--b32-root",type=Path,required=True,help="extracted B32 artifact root")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--score-threshold",type=float,default=0.35)
    ap.add_argument("--freeze-only",action="store_true",help="debug: infer t+00 only")
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    stage=args.b32_root/"stage_a"
    scene=json.loads((stage/"v32_scene_manifest.json").read_text())
    quality=json.loads((stage/"v32_quality_cluster.json").read_text())
    if tuple(scene.get("resolution",[]))!=(W,H): raise RuntimeError("v32i requires native 960x540")
    if set(scene["cameras"])!=set(CAMERAS): raise RuntimeError("v32i is locked to the three accepted cameras")
    cams={c:camera(scene,c) for c in CAMERAS}
    Fbr=fundamental(cams[BCAST],cams[RAR])

    focal=quality["focal_player"]["observations"]
    b_anchor=np.asarray(focal[BCAST]["bbox_xyxy"],np.float64)
    r_anchor=np.asarray(focal[RAR]["bbox_xyxy"],np.float64)
    bmask=cv2.imread(str(stage/focal[BCAST]["mask_full"]),cv2.IMREAD_GRAYSCALE)
    rmask=cv2.imread(str(stage/focal[RAR]["mask_full"]),cv2.IMREAD_GRAYSCALE)
    if bmask is None or rmask is None: raise RuntimeError("missing B32 focal identity masks")

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    model=keypointrcnn_resnet50_fpn(weights=weights,progress=True).eval()

    rels=(0,) if args.freeze_only else REL
    detections={c:{} for c in CAMERAS}; images={c:{} for c in CAMERAS}
    burst_lookup={c:{int(r["relative_frame"]):r for r in scene["burst"][c]} for c in CAMERAS}
    for rel in rels:
        for c in CAMERAS:
            row=burst_lookup[c][rel]; p=stage/row["file"]
            im=cv2.imread(str(p));
            if im is None or im.shape[:2]!=(H,W): raise RuntimeError(f"bad source frame {p}")
            images[c][rel]=im; detections[c][rel]=detect_all(model,im,args.score_threshold)
        print(f"detected rel {rel:+d}: "+", ".join(f"{c}={len(detections[c][rel])}" for c in CAMERAS),flush=True)

    # Exact freeze identity.
    bi,brows=choose_anchor_broadcast(detections[BCAST][0],b_anchor,bmask)
    if bi is None: raise RuntimeError("no Broadcast focal pose candidate")
    b0=detections[BCAST][0][bi]
    ri,rrows=choose_rar(detections[RAR][0],b0,Fbr,r_anchor,rmask)
    if ri is None: raise RuntimeError("no RAR focal pose candidate passes epipolar pairing")
    r0=detections[RAR][0][ri]
    br_joints=triangulated_joints(cams,{BCAST:b0,RAR:r0})

    lrows=[]
    for i,d in enumerate(detections[LAR][0]):
        cost,q=lar_match_cost(cams,br_joints,d)
        lrows.append({"index":i,"cost":cost,**q})
    lrows.sort(key=lambda x:x["cost"])
    li=None
    if lrows and lrows[0]["joint_count"]>=4 and lrows[0]["median_px"]<=32.0 and lrows[0]["p75_px"]<=50.0:
        li=int(lrows[0]["index"])
    l0=None if li is None else detections[LAR][0][li]

    final_joints=triangulated_joints(cams,{LAR:l0,BCAST:b0,RAR:r0})
    # For joints not re-triangulated from >=2 measurements, preserve valid BR estimate.
    for j,row in br_joints.items(): final_joints.setdefault(j,row)

    # Project solved skeleton back into all three physical cameras.
    projections={c:{} for c in CAMERAS}
    all_res=[]
    for j,row in final_joints.items():
        X=np.asarray(row["world_cm"])
        for c in CAMERAS:
            uv,_=project(cams[c],X)
            if uv is not None: projections[c][j]=uv.tolist()
        all_res.extend(row.get("residual_px",{}).values())

    # Temporal tracking provides support/diagnostic only; it does not alter t+00.
    temporal={0:{BCAST:b0,RAR:r0,LAR:l0}}
    if not args.freeze_only:
        for direction in (-1,1):
            prev_b,prev_r,prev_l=b0,r0,l0
            for rel in range(direction,7*direction,direction):
                bs=detections[BCAST][rel]; bix=track_nearest(bs,prev_b); bd=None if bix is None else bs[bix]
                if bd is None: temporal[rel]={BCAST:None,RAR:None,LAR:None}; continue
                rs=detections[RAR][rel]; rix,rq=choose_rar(rs,bd,Fbr,r_anchor,rmask,prev=prev_r); rd=None if rix is None else rs[rix]
                js=triangulated_joints(cams,{BCAST:bd,RAR:rd}) if rd is not None else {}
                best_l=None; bestq=None
                if js:
                    cand=[]
                    for i,d in enumerate(detections[LAR][rel]):
                        cc,qq=lar_match_cost(cams,js,d); cand.append((cc,i,qq))
                    cand.sort(key=lambda x:x[0])
                    if cand and cand[0][2]["joint_count"]>=4 and cand[0][2]["median_px"]<=40:
                        best_l=detections[LAR][rel][cand[0][1]]; bestq=cand[0][2]
                temporal[rel]={BCAST:bd,RAR:rd,LAR:best_l,"triangulated_joint_count":len(js),"left_match":bestq}
                prev_b,prev_r,prev_l=bd,rd,best_l or prev_l

    tsummary={}
    for rel,row in sorted(temporal.items()):
        tsummary[str(rel)]={
            "broadcast":serial_det(row.get(BCAST)),"right_above_rim":serial_det(row.get(RAR)),"left_above_rim":serial_det(row.get(LAR)),
            "triangulated_joint_count":int(row.get("triangulated_joint_count",len(final_joints) if rel==0 else 0)),
            "left_match":row.get("left_match"),
        }

    residual_median=float(np.median(all_res)) if all_res else 999.0
    left_q=lrows[0] if lrows else {"joint_count":0,"median_px":999.0,"p75_px":999.0}
    three_view_ok=bool(l0 is not None and left_q["joint_count"]>=4 and left_q["median_px"]<=32.0)
    joint_ok=len(final_joints)>=8
    reproj_ok=residual_median<=18.0
    status="PASS_V32I_THREE_CAMERA_ARTICULATED_POSITION" if (three_view_ok and joint_ok and reproj_ok) else "FAIL_CLOSED_V32I_POSITION_NOT_YET_THREE_VIEW_SUPPORTED"

    qa={
        "version":"v32i_stage1_three_camera_pose_observations",
        "status":status,
        "native_resolution":[W,H],
        "source_cameras":list(CAMERAS),
        "source_frame_count":3 if args.freeze_only else 39,
        "generated_rgb":False,"upscaled":False,"warped":False,"mesh_rendered":False,
        "camera_geometry_modified":False,
        "model_role":"torchvision Keypoint R-CNN COCO_V1; OBSERVATION ONLY",
        "focal_identity":"Steven Adams / B32 accepted dark-uniform #12 anchor",
        "broadcast_anchor_candidates":brows,
        "rar_pair_candidates":rrows,
        "left_projection_candidates":lrows,
        "selected_detection_indices":{BCAST:bi,RAR:ri,LAR:li},
        "triangulated_joint_count":len(final_joints),
        "median_observation_reprojection_px":residual_median,
        "left_measured_pose_found":l0 is not None,
        "left_best_projected_pose_median_px":float(left_q.get("median_px",999.0)),
        "left_best_projected_pose_p75_px":float(left_q.get("p75_px",999.0)),
        "joints":{JOINTS[j]:row for j,row in final_joints.items()},
        "temporal_support":tsummary,
        "gate":{
            "at_least_8_world_joints":joint_ok,
            "median_reprojection_le_18px":reproj_ok,
            "real_left_pose_matches_projection":three_view_ok,
            "surface_reconstruction_unlocked":bool(status.startswith("PASS_")),
        },
    }
    (args.out/"v32i_pose_observations_qa.json").write_text(json.dumps(qa,indent=2))

    ovs=[]
    for c,det in ((LAR,l0),(BCAST,b0),(RAR,r0)):
        title=f"v32i t+00 {c} | yellow=measured pose magenta=3D reprojection"
        ov=draw(images[c][0],det,projections[c],title)
        fn=f"v32i_t00_{c.replace(' ','_')}.png"; cv2.imwrite(str(args.out/fn),ov); ovs.append(ov)
    cv2.imwrite(str(args.out/"v32i_t00_three_camera_montage.png"),np.hstack(ovs))
    print(json.dumps({"status":status,"triangulated_joint_count":len(final_joints),"median_reprojection_px":residual_median,"left_match":left_q},indent=2),flush=True)
    if status.startswith("FAIL_"):
        raise SystemExit(3)


if __name__=="__main__":
    main()
