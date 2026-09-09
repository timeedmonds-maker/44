from __future__ import annotations

"""Jazz event 489: three-camera reference-surface perspective diagnostic v10.

v8 failed because global silhouettes created cross-person ghost volumes.
v9 fixed identity mixing, but dense voxels still produced carved/thin players.

v10 changes representation. Left Above Rim (LAR) is the immutable appearance
anchor for the requested small 0->25 degree orbit. Each real LAR person pixel
receives exactly one metric 3-D depth. Coarse depth comes from the compact
same-person LAR<->Broadcast volume selected by v9; every missing/unsupported
reference pixel is filled from aligned MoGe depth and then adjusted along its
LAR camera ray until it lands inside the matched Broadcast silhouette wherever
possible. Thus geometry is multi-view constrained without flattening the player
into a card or exposing the entire voxel volume.

The ball is anchored from the two unambiguous views for this event: Broadcast
and Right Above Rim. Court/backboard remain exact metric planes. All appearance
pixels are copied from real source frames. No generative fill, crossfade or
optical-flow morphing is used.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4
from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_volumetric_v8 as v8
from freeze_spin import build_three_camera_instance_volume_v9 as v9
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W, H = base.W, base.H
RIM = base.RIM.astype(np.float64)
REF, MATCH, OVERHEAD = "Left Above Rim", "Broadcast", "Right Above Rim"
CAMERA_ORDER = (REF, MATCH, OVERHEAD)
BALL_RADIUS_CM = 12.0


def camera_z_from_points(cam, pts):
    C, R, _ = cam
    s = float(v3.forward_sign(R, C))
    Xc = (R @ (pts.astype(np.float64) - C).T).T
    return s * Xc[:, 2]


def backproject_pixels(cam, xs, ys, z):
    C, R, K = cam
    s = float(v3.forward_sign(R, C))
    xn = (xs.astype(np.float64) - K[0, 2]) / K[0, 0]
    yn = (ys.astype(np.float64) - K[1, 2]) / K[1, 1]
    Xc = s * np.column_stack([xn * z, yn * z, z])
    return (R.T @ Xc.T).T + C


def component_front_depth(cam, pts, mask):
    uv, depth, valid = v8.project_metric(cam, pts)
    u = np.rint(uv[:, 0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:, 1]).astype(np.int32, copy=False)
    ok = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    ids = np.where(ok)[0]
    d = np.full(H * W, np.inf, np.float32)
    if len(ids):
        pix = vv[ids] * W + u[ids]
        np.minimum.at(d, pix, depth[ids].astype(np.float32))
    d = d.reshape(H, W)
    d[~mask] = np.inf
    return d


def nearest_fill(values, valid, region):
    if not np.any(valid & region):
        return values.copy()
    invalid = ~(valid & region)
    _, inds = ndimage.distance_transform_edt(invalid, return_indices=True)
    out = values.copy()
    yy, xx = np.where(region & ~valid)
    out[yy, xx] = values[inds[0, yy, xx], inds[1, yy, xx]]
    return out


def smooth_depth(depth, mask, raw_anchor):
    ys, xs = np.where(mask)
    if not len(xs):
        return depth
    x0, x1 = max(0, xs.min()-3), min(W, xs.max()+4)
    y0, y1 = max(0, ys.min()-3), min(H, ys.max()+4)
    sub = depth[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1]
    a = raw_anchor[y0:y1, x0:x1]
    med = float(np.median(sub[m]))
    tmp = sub.copy(); tmp[~m] = med
    for _ in range(2):
        tmp = cv2.bilateralFilter(tmp, 7, 38.0, 5.0)
        # Keep measured component-front depths strong; smooth only the fill.
        tmp[m & a] = 0.78 * sub[m & a] + 0.22 * tmp[m & a]
    out = depth.copy(); out[y0:y1, x0:x1][m] = tmp[m]
    return out


def projected_inside(cam, pts, mask):
    uv, _, valid = v8.project_metric(cam, pts)
    u = np.rint(uv[:,0]).astype(np.int32, copy=False)
    vv = np.rint(uv[:,1]).astype(np.int32, copy=False)
    ok = valid & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
    hit = np.zeros(len(pts), bool)
    ids = np.where(ok)[0]
    if len(ids): hit[ids] = mask[vv[ids], u[ids]]
    return hit


def ray_adjust_to_mask(ref_cam, target_cam, xs, ys, z0, target_mask, max_delta=220.0, step=8.0):
    z = z0.astype(np.float64).copy()
    pts = backproject_pixels(ref_cam, xs, ys, z)
    hit = projected_inside(target_cam, pts, target_mask)
    initial = int(hit.sum())
    unresolved = ~hit
    offsets = []
    for k in range(1, int(max_delta // step) + 1):
        offsets.extend((-k*step, k*step))
    for off in offsets:
        ids = np.where(unresolved)[0]
        if not len(ids): break
        cand_z = z0[ids].astype(np.float64) + off
        physical = (cand_z > 80.0) & (cand_z < 9000.0)
        if not np.any(physical): continue
        sub_ids = ids[physical]
        cand = backproject_pixels(ref_cam, xs[sub_ids], ys[sub_ids], cand_z[physical])
        good = projected_inside(target_cam, cand, target_mask)
        accepted = sub_ids[good]
        if len(accepted):
            z[accepted] = z0[accepted] + off
            unresolved[accepted] = False
    return z, {
        "initial_supported": initial,
        "final_supported": int((~unresolved).sum()),
        "total": int(len(z)),
        "final_support_fraction": float((~unresolved).mean()) if len(z) else 1.0,
        "fallback_pixels": int(unresolved.sum()),
    }


def build_reference_surface(inst, component_pts, ref_cam, match_cam, match_mask, aligned_z, ref_image):
    mask = inst["mask"].astype(bool)
    ys, xs = np.where(mask)
    if not len(xs): return None
    front = component_front_depth(ref_cam, component_pts, mask)
    anchor = np.isfinite(front) & mask
    z = front.copy()

    # Metric-aligned monocular depth is a fill prior, never the sole cross-view proof.
    prior = aligned_z.astype(np.float32)
    observed = z[anchor]
    if len(observed):
        med = float(np.median(observed)); lo, hi = med - 180.0, med + 180.0
    else:
        vals = prior[mask & np.isfinite(prior) & (prior > 50.0)]
        med = float(np.median(vals)) if len(vals) else 1800.0; lo, hi = med - 180.0, med + 180.0
    fill_prior = np.clip(prior, lo, hi)
    use_prior = mask & ~anchor & np.isfinite(fill_prior) & (fill_prior > 50.0)
    z[use_prior] = fill_prior[use_prior]
    valid = mask & np.isfinite(z) & (z > 50.0)
    z = nearest_fill(z, valid, mask)
    z = smooth_depth(z, mask, anchor)

    zv = z[ys, xs].astype(np.float64)
    zv_adj, aqa = ray_adjust_to_mask(ref_cam, match_cam, xs, ys, zv, match_mask)
    pts = backproject_pixels(ref_cam, xs, ys, zv_adj).astype(np.float32)
    cols = ref_image[ys, xs].astype(np.uint8)
    # Keep only physically plausible NBA action-region points.
    good = np.isfinite(pts).all(axis=1) & (pts[:,2] > -40.0) & (pts[:,2] < 420.0) & (pts[:,0] > -450.0) & (pts[:,0] < 1050.0) & (np.abs(pts[:,1]) < 800.0)
    pts, cols = pts[good], cols[good]
    aqa.update({"mask_pixels": int(mask.sum()), "surface_points": int(len(pts)), "component_anchor_pixels": int(anchor.sum())})
    return pts, cols, aqa


def choose_overhead_instance(surface_pts, instances, cam):
    rows = []
    for i, inst in enumerate(instances):
        hit = projected_inside(cam, surface_pts, inst["support_mask"])
        rows.append((float(hit.mean()) if len(hit) else 0.0, i))
    rows.sort(reverse=True)
    if not rows: return None, 0.0, 0.0
    best = rows[0]; second = rows[1][0] if len(rows)>1 else 0.0
    if best[0] >= 0.18 and best[0] - second >= 0.035:
        return best[1], best[0], second
    return None, best[0], second


def refine_with_overhead(ref_cam, overhead_cam, xs, ys, z, overhead_mask, match_cam, match_mask):
    # Conservative: only change points that miss overhead, and only to a depth
    # that remains inside the already-required Broadcast silhouette.
    pts = backproject_pixels(ref_cam, xs, ys, z)
    oh = projected_inside(overhead_cam, pts, overhead_mask)
    miss = ~oh
    changed = 0
    offsets = []
    for k in range(1, 21): offsets.extend((-k*6.0, k*6.0))
    for off in offsets:
        ids = np.where(miss)[0]
        if not len(ids): break
        cand_z = z[ids] + off
        cand = backproject_pixels(ref_cam, xs[ids], ys[ids], cand_z)
        good = projected_inside(overhead_cam, cand, overhead_mask) & projected_inside(match_cam, cand, match_mask)
        acc = ids[good]
        if len(acc):
            z[acc] += off; miss[acc] = False; changed += len(acc)
    return z, {"overhead_initial_fraction": float(oh.mean()) if len(oh) else 0.0, "overhead_final_fraction": float((~miss).mean()) if len(miss) else 0.0, "overhead_adjusted_pixels": int(changed)}


def triangulate_event_ball(cams, ball_candidates):
    # This Jazz frame has an unmistakable ball in Broadcast and Right Above Rim.
    if not ball_candidates[MATCH] or not ball_candidates[OVERHEAD]:
        return None, {"status":"BALL_ANCHOR_MISSING"}
    b = ball_candidates[MATCH][0]; r = ball_candidates[OVERHEAD][0]
    obs = {MATCH: np.asarray([b["cx"],b["cy"]],np.float64), OVERHEAD: np.asarray([r["cx"],r["cy"]],np.float64)}
    X = v8.dlt_point(cams, obs)
    if X is None or not np.isfinite(X).all(): return None, {"status":"BALL_DLT_FAILED"}
    errs = {}
    for k,p in obs.items():
        uv,_,ok=v8.project_metric(cams[k],X.reshape(1,3)); errs[k]=float(np.linalg.norm(uv[0]-p)) if ok[0] else 999.0
    rim_dist=float(np.linalg.norm(X-RIM))
    physical=(-120.0<X[0]<380.0 and abs(X[1])<260.0 and 180.0<X[2]<390.0 and rim_dist<130.0 and max(errs.values())<10.0)
    return (X.astype(np.float64) if physical else None), {
        "status":"BALL_TRIANGULATED" if physical else "BALL_GEOMETRY_REJECTED",
        "center_world_cm":[float(x) for x in X],"views":[MATCH,OVERHEAD],
        "detections":{MATCH:b,OVERHEAD:r},"reprojection_errors_px":errs,"distance_from_rim_center_cm":rim_dist,
    }


def dense_raster(pts, cols, K, R, C, radius=1):
    return v9.dense_raster(pts, cols, K, R, C, radius=radius)


def annotate(img, angle, qa):
    out=img.copy(); cv2.rectangle(out,(0,0),(900,70),(0,0,0),-1)
    cv2.putText(out,"JAZZ EVENT 489 | 3 SOLVED CAMERAS | v10 REFERENCE-SURFACE 3D",(12,21),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f"virtual viewpoint {angle:04.1f} deg | ref IoU {qa['reference_zero_degree_body_iou']:.3f} | Broadcast support {qa['mean_broadcast_surface_support']:.3f} | {qa['ball']['status']}",(12,45),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,"one depth per real LAR body pixel; same-player Broadcast ray constraint; RAR validator; source RGB only",(12,64),cv2.FONT_HERSHEY_SIMPLEX,.33,(210,210,210),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frames-dir",type=Path,required=True); ap.add_argument("--registry",type=Path,required=True)
    ap.add_argument("--rar-report",type=Path,required=True); ap.add_argument("--broadcast-event-frame",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--tokens",type=int,default=1200); ap.add_argument("--voxel-cm",type=float,default=4.0)
    ap.add_argument("--full-arc",action="store_true"); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    lar=base.find_one(args.frames_dir,"Left_Above_Rim"); rar=base.find_one(args.frames_dir,"Right_Above_Rim")
    br=[p for p in sorted(args.frames_dir.rglob("*Broadcast*.png")) if "Mobile" not in p.name and "Other" not in p.name]
    if len(br)!=1: raise RuntimeError(f"Broadcast ambiguity {br}")
    paths={REF:lar,MATCH:br[0],OVERHEAD:rar}; images={k:cv2.imread(str(p)) for k,p in paths.items()}
    cams=base.load_cameras(args.registry,args.rar_report,args.broadcast_event_frame)

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    depth_model=MoGeModel.from_pretrained("Ruicheng/moge-2-vits-normal").eval()
    seg_model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()

    sources={}; instances={}; priors={}; descs={}; aligned_z={}; ball_obs={}
    qa={"schema_version":10,"game_id":"0022500301","event_id":489,"solved_physical_cameras":list(CAMERA_ORDER),"sources":{},"players":[],"frames":[]}
    for label in CAMERA_ORDER:
        C,R,K=cams[label]; im=images[label]
        depth,_,valid,_,_=moge_infer(depth_model,im,args.tokens); align,dqa=v3.robust_depth_align(depth,valid,K,R,C)
        aligned_z[label]=(align[0]*depth.astype(np.float64)+align[1]).astype(np.float32)
        dyn,inst,coco=v6.detect_near_play(seg_model,im,K,R,C); instances[label]=inst
        priors[label]=[]; descs[label]=[]
        for row in inst:
            row["support_mask"]=v9.support_mask(row["mask"]); priors[label].append(v9.instance_metric_prior(depth,valid,align,K,R,C,row["mask"])); descs[label].append(v9.descriptor(im,row["mask"]))
        fv,bv=v5.source_visibility_v5(im,depth,valid,dyn,K,R,C,align); sources[label]={"image":im,"C":C,"R":R,"K":K,"floor_vis":fv,"board_vis":bv,"dynamic":dyn}
        orange,_=v9.orange_ball_candidates(im,cams[label]); merged=list(coco[:3])+orange; merged.sort(key=lambda r:float(r["score"]),reverse=True); ball_obs[label]=merged[:6]
        qa["sources"][label]={"file":paths[label].name,"person_instances":len(inst),"depth_alignment":dqa,"ball_candidates":merged[:6],"camera_center_cm":[float(x) for x in C]}

    # Reuse v9's successful identity-safe matching/component solve as a coarse 3-D prior only.
    xs_g,ys_g,zs_g,shape,grid=v9.make_grid(args.voxel_cm)
    ref_hits=[v8.mask_membership(cams[REF],grid,r["support_mask"]) for r in instances[REF]]
    br_hits=[v8.mask_membership(cams[MATCH],grid,r["support_mask"]) for r in instances[MATCH]]
    rhits=[v8.mask_membership(cams[OVERHEAD],grid,r["support_mask"]) for r in instances[OVERHEAD]]
    assignment,scores,pair_metrics=v9.match_reference_to_broadcast(instances[REF],instances[MATCH],ref_hits,br_hits,grid,priors[REF],descs[REF],descs[MATCH])
    qa["matching"]={"assignment":{str(k):int(v) for k,v in assignment.items()},"score_matrix":scores.tolist()}

    surf_pts=[]; surf_cols=[]; ref_union=np.zeros((H,W),bool); support_fracs=[]
    for i,inst in enumerate(instances[REF]):
        if i not in assignment: qa["players"].append({"reference_instance":i,"status":"UNMATCHED"}); continue
        j=assignment[i]; pair_occ=ref_hits[i]&br_hits[j]
        gflat,cqa=v9.component_from_pair(pair_occ,shape,grid,priors[REF][i],rhits,descs[OVERHEAD],descs[REF][i])
        if gflat is None or len(gflat)<80: qa["players"].append({"reference_instance":i,"broadcast_instance":j,"status":"NO_COARSE_COMPONENT"}); continue
        component_pts=grid[gflat]
        result=build_reference_surface(inst,component_pts,cams[REF],cams[MATCH],instances[MATCH][j]["support_mask"],aligned_z[REF],images[REF])
        if result is None: continue
        pp,cc,sqa=result
        # RAR is a conservative validator/refiner only when one same-player mask is clearly dominant.
        oi,best,second=choose_overhead_instance(pp,instances[OVERHEAD],cams[OVERHEAD])
        overhead_qa={"selected_instance":oi,"best_surface_support":best,"second_surface_support":second,"applied":False}
        if oi is not None:
            ys,xs=np.where(inst["mask"].astype(bool)); n=min(len(xs),len(pp))
            if n==len(pp):
                z=camera_z_from_points(cams[REF],pp)
                z2,oqa=refine_with_overhead(cams[REF],cams[OVERHEAD],xs,ys,z,instances[OVERHEAD][oi]["support_mask"],cams[MATCH],instances[MATCH][j]["support_mask"])
                pp=backproject_pixels(cams[REF],xs,ys,z2).astype(np.float32); overhead_qa.update(oqa); overhead_qa["applied"]=True
        surf_pts.append(pp); surf_cols.append(cc); ref_union|=inst["mask"].astype(bool); support_fracs.append(sqa["final_support_fraction"])
        qa["players"].append({"reference_instance":i,"broadcast_instance":j,"status":"RETAINED","coarse_component":cqa,"surface":sqa,"overhead":overhead_qa})

    if not surf_pts: raise RuntimeError("No reference surfaces retained")
    body_pts=np.concatenate(surf_pts).astype(np.float32); body_cols=np.concatenate(surf_cols).astype(np.uint8)
    qa["retained_player_count"]=len(surf_pts); qa["surface_points"]=int(len(body_pts)); qa["mean_broadcast_surface_support"]=float(np.mean(support_fracs))

    ball_center,ball_qa=triangulate_event_ball(cams,ball_obs); qa["ball"]=ball_qa
    ball_pts=np.empty((0,3),np.float32); ball_cols=np.empty((0,3),np.uint8)
    if ball_center is not None:
        ball_pts=v8.sphere_points(ball_center,n_lat=20,n_lon=40)
        b=ball_obs[OVERHEAD][0]; x1,y1,x2,y2=[int(round(x)) for x in b["box"]]; crop=images[OVERHEAD][max(0,y1):min(H,y2+1),max(0,x1):min(W,x2+1)]
        med=np.median(crop.reshape(-1,3),axis=0).astype(np.uint8) if crop.size else np.asarray([45,110,190],np.uint8); ball_cols=np.repeat(med.reshape(1,3),len(ball_pts),axis=0)

    rp=v8.rim_points(); ruv,_,rv=v8.project_metric(cams[REF],rp); rcols=v8.bilinear_sample(images[REF],ruv); rp=rp[rv]; rcols=rcols[rv]
    for label in CAMERA_ORDER: v8.subtract_metric_rim_from_plane_visibility(sources[label])

    # Geometry-only QA at the exact real 0-degree camera.
    ref_body_img,ref_body_mask=dense_raster(body_pts,body_cols,cams[REF][2],cams[REF][1],cams[REF][0],radius=1)
    qa["reference_zero_degree_body_iou"]=v9.mask_iou(ref_body_mask>0,ref_union)
    qa["reference_zero_degree_body_recall"]=float(np.sum((ref_body_mask>0)&ref_union)/max(1,np.sum(ref_union)))

    C0,R0,K0=cams[REF]
    angles=np.r_[np.zeros(12),np.linspace(0.0,25.0,76),np.full(18,25.0)] if args.full_arc else np.asarray([0,5,10,15,20,25],np.float64)
    for fi,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang))
        Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,"floor"); floor,fm=v4.sample_plane_from_sources(Pf,sf,sources,"floor")
        Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,"board"); board,bm=v4.sample_plane_from_sources(Pb,sb,sources,"board")
        img=floor.copy(); img[bm]=board[bm]
        pts_parts=[body_pts,rp]; col_parts=[body_cols,rcols]
        if len(ball_pts): pts_parts.append(ball_pts); col_parts.append(ball_cols)
        fg,fgm=dense_raster(np.concatenate(pts_parts),np.concatenate(col_parts),K0,Rt,Ct,radius=1); m=fgm>0; img[m]=fg[m]
        # At the exact anchor camera, use the exact real reference body pixels.
        # This is not a transition trick; it is the physically observed endpoint.
        if abs(float(ang))<1e-9:
            img[ref_union]=images[REF][ref_union]
        out=annotate(img,float(ang),qa); cv2.imwrite(str(args.out/f"frame_{fi:03d}.png"),out)
        qa["frames"].append({"frame":fi,"virtual_viewpoint_deg":float(ang),"foreground_pixels":int(m.sum())})

    qa["status"]="STATIC_REFERENCE_SURFACE_RENDERED" if not args.full_arc else "FULL_REFERENCE_SURFACE_ARC_RENDERED"
    qa["method"]="one-depth-per-real-LAR-player-pixel surface; v9 identity-safe pair volume as coarse prior; Broadcast silhouette ray constraint; conservative RAR validation; exact metric court/backboard/rim; Broadcast+RAR ball anchor; real source RGB only"
    qa["gates"]={"zero_degree_iou_target":0.90,"broadcast_support_target":0.85,"ball_must_be_valid":True,"zero_degree_iou_pass":qa["reference_zero_degree_body_iou"]>=0.90,"broadcast_support_pass":qa["mean_broadcast_surface_support"]>=0.85,"ball_pass":ball_center is not None}
    (args.out/"three_camera_reference_surface_v10_qa.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    print(json.dumps({"status":qa["status"],"retained_players":qa["retained_player_count"],"surface_points":qa["surface_points"],"zero_degree_iou":qa["reference_zero_degree_body_iou"],"zero_degree_recall":qa["reference_zero_degree_body_recall"],"mean_broadcast_support":qa["mean_broadcast_surface_support"],"ball":qa["ball"],"gates":qa["gates"]},indent=2),flush=True)

if __name__=="__main__": main()
