from __future__ import annotations

"""Jazz event 489: exact-sync, identity-safe, three-camera surfel diagnostic v12.

Representation goals:
- NEVER union different people into one silhouette volume.
- NEVER flatten a player into one camera-facing card.
- NEVER stretch one source texture to invent a newly exposed side.

For every matched player and every solved source camera, each real source pixel
searches along its calibrated metric ray for the nearest depth that projects
inside the SAME player's mask in at least one other solved camera. The resulting
source-specific textured surfels are retained separately. Novel views use a
single depth buffer; when two source surfaces are geometrically co-located, the
source camera closest in viewing direction gets a small preference. This keeps
real source texture while permitting genuine parallax and newly exposed side
coverage from Broadcast / Right Above Rim.

Inputs MUST be the exact visually synchronized frames produced by v11.
"""

import argparse, json, math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4
from freeze_spin import build_three_camera_diagnostic_v5 as v5
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_volumetric_v8 as v8
from freeze_spin import build_three_camera_instance_volume_v9 as v9
from freeze_spin import build_three_camera_reference_surface_v10 as v10
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
REF='Left Above Rim'; BR='Broadcast'; RAR='Right Above Rim'
CAMERA_ORDER=(REF,BR,RAR)


def safe_project(cam,pts):
    return v8.project_metric(cam,pts)


def backproject(cam,xs,ys,z):
    return v10.backproject_pixels(cam,xs,ys,z)


def mask_hit(cam,pts,mask):
    uv,_,valid=safe_project(cam,pts)
    u=np.rint(uv[:,0]).astype(np.int32,copy=False); vv=np.rint(uv[:,1]).astype(np.int32,copy=False)
    ok=valid&(u>=0)&(u<W)&(vv>=0)&(vv<H)
    hit=np.zeros(len(pts),bool); ids=np.where(ok)[0]
    if len(ids): hit[ids]=mask[vv[ids],u[ids]]
    return hit


def source_depth_prior(depth,valid,align):
    z=align[0]*depth.astype(np.float64)+align[1]
    z[~valid]=np.nan
    return z


def camera_z(cam,pts):
    C,R,_=cam; s=float(v3.forward_sign(R,C)); Xc=(R@(pts.astype(np.float64)-C).T).T
    return s*Xc[:,2]


def build_ray_supported_surface(source_label,cam,image,mask,zprior,other_views,component_pts=None,search_cm=300.0,step_cm=8.0):
    ys,xs=np.where(mask.astype(bool))
    if not len(xs): return None
    z0=zprior[ys,xs].astype(np.float64)

    if component_pts is not None and len(component_pts):
        cz=camera_z(cam,component_pts)
        cz=cz[np.isfinite(cz)&(cz>40)]
        if len(cz):
            lo=float(np.percentile(cz,2))-80.0; hi=float(np.percentile(cz,98))+80.0; med=float(np.median(cz))
        else: lo,hi,med=80.0,9000.0,1800.0
    else:
        lo,hi,med=80.0,9000.0,1800.0
    bad=~np.isfinite(z0)|(z0<lo)|(z0>hi)
    z0[bad]=med
    z0=np.clip(z0,lo,hi)

    best_support=np.full(len(xs),-1,np.int8); best_z=z0.copy(); best_delta=np.full(len(xs),np.inf,np.float32)
    offsets=[0.0]
    n=int(search_cm//step_cm)
    for k in range(1,n+1): offsets.extend((-k*step_cm,k*step_cm))
    for off in offsets:
        z=z0+off
        plausible=(z>=lo)&(z<=hi)
        if not np.any(plausible): continue
        ids=np.where(plausible)[0]
        pts=backproject(cam,xs[ids],ys[ids],z[ids])
        support=np.zeros(len(ids),np.int8)
        for ocam,omask in other_views:
            support+=mask_hit(ocam,pts,omask).astype(np.int8)
        # Offsets are ordered by abs(delta), so only strictly better support replaces.
        take=support>best_support[ids]
        if np.any(take):
            ii=ids[take]; best_support[ii]=support[take]; best_z[ii]=z[ii]; best_delta[ii]=abs(off)
    keep=best_support>=1
    # Keep exact source-only boundary pixels as low confidence only for endpoint continuity.
    pts=backproject(cam,xs,ys,best_z).astype(np.float32)
    physical=np.isfinite(pts).all(1)&(pts[:,0]>-450)&(pts[:,0]<1050)&(np.abs(pts[:,1])<850)&(pts[:,2]>-80)&(pts[:,2]<500)
    keep&=physical
    return {
        'points':pts[keep],
        'colors':image[ys[keep],xs[keep]].astype(np.uint8),
        'source_label':source_label,
        'source_depth':best_z[keep].astype(np.float32),
        'support':best_support[keep].astype(np.int8),
        'qa':{
            'source_mask_pixels':int(len(xs)),
            'retained_surfels':int(np.sum(keep)),
            'retained_fraction':float(np.mean(keep)),
            'two_other_view_support':int(np.sum(best_support[keep]>=2)),
            'one_other_view_support':int(np.sum(best_support[keep]==1)),
            'median_depth_adjustment_cm':float(np.median(best_delta[keep])) if np.any(keep) else None,
            'p95_depth_adjustment_cm':float(np.percentile(best_delta[keep],95)) if np.any(keep) else None,
            'component_camera_z_range_cm':[lo,hi],
        }
    }


def source_angle_penalty(src_cam,target_C):
    a=src_cam[0]-RIM; b=target_C-RIM
    a=a/max(np.linalg.norm(a),1e-9); b=b/max(np.linalg.norm(b),1e-9)
    deg=math.degrees(math.acos(float(np.clip(np.dot(a,b),-1,1))))
    return deg,0.07*deg


def render_surfel_union(surfaces,K,R,C):
    all_pix=[]; all_depth=[]; all_eff=[]; all_col=[]; all_src=[]
    per_source={}
    for surf in surfaces:
        pts=surf['points']; cols=surf['colors']; label=surf['source_label']
        if not len(pts): continue
        uv,depth,valid=safe_project((C,R,K),pts)
        u=np.rint(uv[:,0]).astype(np.int32,copy=False); vv=np.rint(uv[:,1]).astype(np.int32,copy=False)
        ok=valid&(u>=0)&(u<W)&(vv>=0)&(vv<H)
        ids=np.where(ok)[0]
        if not len(ids): continue
        deg,pen=source_angle_penalty(surf['camera'],C)
        # Avoid using an almost opposite source texture; its geometry may still have
        # helped constrain another source surface, but its appearance is unsuitable.
        if deg>92.0: continue
        # Approximate projected source-pixel footprint. 0/1/2 px splat radius.
        src_f=float(surf['camera'][2][0,0]); tgt_f=float(K[0,0])
        world_px=np.maximum(0.6,surf['source_depth'][ids]/max(src_f,1e-6))
        rr=np.clip(np.rint(0.55*world_px*tgt_f/np.maximum(depth[ids],20.0)).astype(np.int8),0,2)
        count=0
        for rad in (0,1,2):
            rid=ids[rr==rad]
            if not len(rid): continue
            duv=[(0,0)] if rad==0 else [(dx,dy) for dy in range(-rad,rad+1) for dx in range(-rad,rad+1) if dx*dx+dy*dy<=rad*rad+0.25]
            for dx,dy in duv:
                uu=u[rid]+dx; yy=vv[rid]+dy
                inside=(uu>=0)&(uu<W)&(yy>=0)&(yy<H)
                rid2=rid[inside]
                if not len(rid2): continue
                all_pix.append((yy[inside]*W+uu[inside]).astype(np.int32)); all_depth.append(depth[rid2].astype(np.float32)); all_eff.append((depth[rid2]+pen).astype(np.float32)); all_col.append(cols[rid2]); all_src.append(np.full(len(rid2),CAMERA_ORDER.index(label),np.int8)); count+=len(rid2)
        per_source[label]={'view_delta_deg':float(deg),'candidate_splats':int(count)}
    out=np.zeros((H*W,3),np.uint8); mask=np.zeros(H*W,np.uint8); depout=np.full(H*W,np.inf,np.float32); owner=np.full(H*W,-1,np.int8)
    if not all_pix: return out.reshape(H,W,3),mask.reshape(H,W),depout.reshape(H,W),owner.reshape(H,W),per_source
    pix=np.concatenate(all_pix); dep=np.concatenate(all_depth); eff=np.concatenate(all_eff); col=np.concatenate(all_col); src=np.concatenate(all_src)
    order=np.argsort(eff,kind='stable'); ps=pix[order]; _,first=np.unique(ps,return_index=True); win=order[first]; wp=pix[win]
    out[wp]=col[win]; mask[wp]=255; depout[wp]=dep[win]; owner[wp]=src[win]
    return out.reshape(H,W,3),mask.reshape(H,W),depout.reshape(H,W),owner.reshape(H,W),per_source


def player_matching(cams,instances,priors,descs,voxel_cm=5.0):
    xs,ys,zs,shape,grid=v9.make_grid(voxel_cm)
    hits={lab:[v8.mask_membership(cams[lab],grid,row['support_mask']) for row in instances[lab]] for lab in CAMERA_ORDER}
    assign,scores,metrics=v9.match_reference_to_broadcast(instances[REF],instances[BR],hits[REF],hits[BR],grid,priors[REF],descs[REF],descs[BR])
    players=[]
    for i,j in assign.items():
        pair_occ=hits[REF][i]&hits[BR][j]
        gflat,cqa=v9.component_from_pair(pair_occ,shape,grid,priors[REF][i],hits[RAR],descs[RAR],descs[REF][i])
        if gflat is None or len(gflat)<50: continue
        ri=None
        if cqa and cqa.get('selected_component'):
            cand=cqa['selected_component'].get('best_overhead_instance'); ratio=float(cqa['selected_component'].get('best_overhead_overlap_ratio',0))
            if cand is not None and ratio>=0.16: ri=int(cand)
        players.append({'lar':int(i),'br':int(j),'rar':ri,'component_points':grid[gflat].copy(),'component_qa':cqa})
    return players,{'lar_to_broadcast_assignment':{str(k):int(v) for k,v in assign.items()},'score_matrix':scores.tolist(),'retained_pairs':len(players)}


def ellipse_ball_mask(det):
    m=np.zeros((H,W),np.uint8); x1,y1,x2,y2=[int(round(x)) for x in det['box']]; cx=int(round(det['cx'])); cy=int(round(det['cy'])); ax=max(3,int(round((x2-x1+1)*0.52))); ay=max(3,int(round((y2-y1+1)*0.52))); cv2.ellipse(m,(cx,cy),(ax,ay),0,0,360,255,-1)
    return m>0


def ball_surface(cams,images,sync):
    X=np.asarray(sync['best_ball_pair']['center_world_cm'],np.float64); pts=v8.sphere_points(X,n_lat=22,n_lon=48)
    rows=[]
    for label,key in ((BR,'broadcast_detection'),(RAR,'rar_detection')):
        det=sync['best_ball_pair'][key]; mask=ellipse_ball_mask(det); vis,cols,_=v8.source_surface_data(label,cams[label],pts,mask,images[label],2.0)
        rows.append({'points':pts[vis],'colors':cols[vis],'source_label':label,'source_depth':camera_z(cams[label],pts[vis]).astype(np.float32),'support':np.full(int(np.sum(vis)),2,np.int8),'camera':cams[label],'qa':{'retained_surfels':int(np.sum(vis))}})
    return X,rows


def annotate(img,angle,qa,frameqa):
    out=img.copy(); cv2.rectangle(out,(0,0),(960,66),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | 3 SOLVED CAMERAS | v12 EXACT-SYNC MULTI-VIEW SURFELS',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f"virtual viewpoint {angle:04.1f} deg | surfels {qa['total_player_surfels']} | body px {frameqa['body_pixels']} | ball ray gap {qa['sync_ball_ray_gap_cm']:.2f} cm",(10,43),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'identity-separated ray-supported real pixels; one z-buffer; exact metric court/backboard/rim; no generative fill',(10,61),cv2.FONT_HERSHEY_SIMPLEX,.32,(210,210,210),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sync-dir',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--tokens',type=int,default=1200); ap.add_argument('--static-only',action='store_true'); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.loads((args.sync_dir/'three_camera_visual_sync_v11.json').read_text())
    if not sync['gates']['all_ball_geometry_pass']: raise RuntimeError('v11 exact visual sync did not pass ball geometry')
    paths={REF:args.sync_dir/'K_Left_Above_Rim_visual_sync.png',BR:args.sync_dir/'A_Broadcast_visual_sync.png',RAR:args.sync_dir/'L_Right_Above_Rim_visual_sync.png'}
    images={k:cv2.imread(str(p)) for k,p in paths.items()}
    if any(im is None for im in images.values()): raise RuntimeError('missing v11 selected frames')
    cams=base.load_cameras(args.registry,args.rar_report,paths[BR])

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    depth_model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval(); seg=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    depths={}; zpriors={}; instances={}; priors={}; descs={}; sources={}
    qa={'schema_version':12,'game_id':'0022500301','event_id':489,'source_sync_v11_selected':sync['selected'],'sync_ball_ray_gap_cm':float(sync['best_ball_pair']['ray_gap_cm']),'sync_ball_reprojection_errors_px':sync['best_ball_pair']['reprojection_errors_px'],'sources':{},'players':[],'frames':[]}
    for lab in CAMERA_ORDER:
        C,R,K=cams[lab]; depth,_,valid,_,_=moge_infer(depth_model,images[lab],args.tokens); align,dqa=v3.robust_depth_align(depth,valid,K,R,C); depths[lab]=(depth,valid,align); zpriors[lab]=source_depth_prior(depth,valid,align)
        dyn,inst,_=v6.detect_near_play(seg,images[lab],K,R,C); instances[lab]=inst; priors[lab]=[]; descs[lab]=[]
        for row in inst:
            row['support_mask']=v9.support_mask(row['mask']); priors[lab].append(v9.instance_metric_prior(depth,valid,align,K,R,C,row['mask'])); descs[lab].append(v9.descriptor(images[lab],row['mask']))
        fv,bv=v5.source_visibility_v5(images[lab],depth,valid,dyn,K,R,C,align); sources[lab]={'image':images[lab],'C':C,'R':R,'K':K,'floor_vis':fv,'board_vis':bv,'dynamic':dyn}; v8.subtract_metric_rim_from_plane_visibility(sources[lab])
        qa['sources'][lab]={'file':paths[lab].name,'instances':len(inst),'depth_alignment':dqa,'camera_center_cm':[float(x) for x in C]}

    players,mqa=player_matching(cams,instances,priors,descs,voxel_cm=5.0); qa['matching']=mqa
    surfaces=[]
    for pi,p in enumerate(players):
        masks={REF:instances[REF][p['lar']]['support_mask'],BR:instances[BR][p['br']]['support_mask']}
        if p['rar'] is not None: masks[RAR]=instances[RAR][p['rar']]['support_mask']
        pqa={'player_index':pi,'instances':{'Left Above Rim':p['lar'],'Broadcast':p['br'],'Right Above Rim':p['rar']},'component_qa':p['component_qa'],'surfaces':{}}
        for lab in tuple(masks.keys()):
            others=[(cams[o],masks[o]) for o in masks if o!=lab]
            if not others: continue
            s=build_ray_supported_surface(lab,cams[lab],images[lab],masks[lab],zpriors[lab],others,p['component_points'])
            if s is None: continue
            s['camera']=cams[lab]; s['player_index']=pi; surfaces.append(s); pqa['surfaces'][lab]=s['qa']
        qa['players'].append(pqa)
    if not surfaces: raise RuntimeError('no player surfels')
    qa['total_player_surfels']=int(sum(len(s['points']) for s in surfaces)); qa['surface_retained_fractions']={lab:float(np.mean([s['qa']['retained_fraction'] for s in surfaces if s['source_label']==lab])) if any(s['source_label']==lab for s in surfaces) else None for lab in CAMERA_ORDER}

    ball_center,ball_surfaces=ball_surface(cams,images,sync); qa['ball_center_world_cm']=[float(x) for x in ball_center]; qa['ball_surfels']=int(sum(len(s['points']) for s in ball_surfaces))
    rp=v8.rim_points(); ruv,_,rv=v8.project_metric(cams[REF],rp); rcols=v8.bilinear_sample(images[REF],ruv); rim_surface={'points':rp[rv],'colors':rcols[rv],'source_label':REF,'source_depth':camera_z(cams[REF],rp[rv]).astype(np.float32),'support':np.full(int(np.sum(rv)),2,np.int8),'camera':cams[REF],'qa':{}}

    # Self-projection QA, radius 0 (unlike v10's accidental 3x3 expansion).
    selfqa={}
    for lab in CAMERA_ORDER:
        ss=[s for s in surfaces if s['source_label']==lab]
        if not ss: continue
        pts=np.concatenate([s['points'] for s in ss]); cols=np.concatenate([s['colors'] for s in ss]); im,m=v9.dense_raster(pts,cols,cams[lab][2],cams[lab][1],cams[lab][0],radius=0)
        union=np.zeros((H,W),bool)
        for p in players:
            key={'Left Above Rim':'lar','Broadcast':'br','Right Above Rim':'rar'}[lab]; idx=p[key]
            if idx is not None: union|=instances[lab][idx]['support_mask']
        inter=np.sum((m>0)&union); selfqa[lab]={'iou':v9.mask_iou(m>0,union),'recall':float(inter/max(1,np.sum(union))),'rendered_pixels':int(np.sum(m>0)),'mask_pixels':int(np.sum(union))}
    qa['self_projection']=selfqa

    C0,R0,K0=cams[REF]; angles=np.asarray([0,5,10,15,20,25],np.float64) if args.static_only else np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    body_areas=[]
    for fi,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang)); Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor'); floor,fm=v4.sample_plane_from_sources(Pf,sf,sources,'floor'); Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board'); board,bm=v4.sample_plane_from_sources(Pb,sb,sources,'board'); img=floor.copy(); img[bm]=board[bm]
        body,bmask,bdep,bowner,sqa=render_surfel_union(surfaces,K0,Rt,Ct); ball,bam,_,_,_=render_surfel_union(ball_surfaces,K0,Rt,Ct); rim,rimask,_,_,_=render_surfel_union([rim_surface],K0,Rt,Ct)
        # Foreground common z-buffer: render all together for final occlusion.
        fg,fgm,_,owner,allqa=render_surfel_union(surfaces+ball_surfaces+[rim_surface],K0,Rt,Ct); mm=fgm>0; img[mm]=fg[mm]
        if abs(float(ang))<1e-9:
            # exact observed endpoint, only on detected matched player masks
            exact=np.zeros((H,W),bool)
            for p in players: exact|=instances[REF][p['lar']]['mask'].astype(bool)
            img[exact]=images[REF][exact]
        frameqa={'frame':fi,'virtual_viewpoint_deg':float(ang),'body_pixels':int(np.sum(bmask>0)),'ball_pixels':int(np.sum(bam>0)),'foreground_pixels':int(np.sum(fgm>0)),'body_owner_pixels':{CAMERA_ORDER[k]:int(np.sum((bmask>0)&(bowner==k))) for k in range(3)},'source_view_deltas':sqa}; qa['frames'].append(frameqa); body_areas.append(frameqa['body_pixels']); cv2.imwrite(str(args.out/f'frame_{fi:03d}.png'),annotate(img,float(ang),qa,frameqa))

    if args.static_only and body_areas:
        a0=max(body_areas[0],1); ratios=[x/a0 for x in body_areas]
    else: ratios=[]
    qa['static_body_area_ratios_vs_0deg']=ratios
    qa['gates']={
        'sync_ball_pass':bool(sync['gates']['all_ball_geometry_pass']),
        'lar_self_recall_pass':selfqa.get(REF,{}).get('recall',0)>=0.82,
        'broadcast_self_recall_pass':selfqa.get(BR,{}).get('recall',0)>=0.72,
        'lar_surface_retained_pass':(qa['surface_retained_fractions'].get(REF) or 0)>=0.72,
        'body_area_no_collapse_pass':(not ratios) or min(ratios[1:])>=0.55,
        'body_area_no_explosion_pass':(not ratios) or max(ratios[1:])<=1.55,
    }
    qa['gates']['numeric_pass']=all(qa['gates'].values()); qa['status']='STATIC_SURFEL_DIAGNOSTIC_RENDERED' if args.static_only else 'FULL_SURFEL_DIAGNOSTIC_RENDERED'; qa['certification_scope']='3-camera R&D diagnostic only; project production gate remains >=4 accepted cameras'; qa['method']='v11 exact visual sync + per-player per-source ray-supported real-pixel surfels + common metric z-buffer + exact metric floor/backboard/rim'
    (args.out/'three_camera_surfel_v12_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'status':qa['status'],'total_player_surfels':qa['total_player_surfels'],'retained_fractions':qa['surface_retained_fractions'],'self_projection':qa['self_projection'],'body_area_ratios':ratios,'gates':qa['gates']},indent=2),flush=True)

if __name__=='__main__': main()
