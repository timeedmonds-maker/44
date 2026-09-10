from __future__ import annotations

"""Jazz event 489: regularized continuous three-camera textured mesh v14.

v13 eliminated point-cloud fragmentation but exposed a new failure: independent
ray-supported depths from LAR and Broadcast can cross by a few centimetres and
cause pixel-level source ping-pong (horizontal texture bands) in the common
z-buffer. v14 regularizes geometry inside each identity while preserving
cross-camera support, then applies LAR-owned appearance for the small 0->25 deg
arc whenever candidate source surfaces are within plausible body thickness.
Broadcast fills genuinely newly exposed surfaces; RAR is one-to-one identity
constrained and remains an independent geometric/ball constraint.
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
from freeze_spin import build_three_camera_surfel_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
REF,BR,RAR=v12.REF,v12.BR,v12.RAR
CAMERA_ORDER=(REF,BR,RAR)


def regularize_surface(surf, other_views, iterations=10, edge_gate_cm=55.0, max_change_cm=70.0):
    pts=surf['points'].astype(np.float64)
    uv,depth,valid=v12.safe_project(surf['camera'],pts)
    u=np.rint(uv[:,0]).astype(np.int32); y=np.rint(uv[:,1]).astype(np.int32)
    good=valid&(u>=0)&(u<W)&(y>=0)&(y<H)&np.isfinite(depth)
    if int(good.sum())<20:
        surf['regularization_qa']={'applied':False,'reason':'too_few_valid_points'}
        return surf
    ids=np.where(good)[0]
    pix=y[ids]*W+u[ids]
    order=np.argsort(depth[ids],kind='stable'); po=pix[order]; _,first=np.unique(po,return_index=True); sel=ids[order[first]]
    uu=u[sel]; vv=y[sel]; z0=depth[sel].astype(np.float32); support=surf['support'][sel].astype(np.int8)
    z=np.full((H,W),np.nan,np.float32); orig=np.full((H,W),np.nan,np.float32); sup=np.full((H,W),-1,np.int8); m=np.zeros((H,W),bool)
    z[vv,uu]=z0; orig[vv,uu]=z0; sup[vv,uu]=support; m[vv,uu]=True

    for _ in range(iterations):
        acc=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            zn=np.roll(np.roll(z,dy,axis=0),dx,axis=1)
            mn=np.roll(np.roll(m,dy,axis=0),dx,axis=1)
            on=np.roll(np.roll(orig,dy,axis=0),dx,axis=1)
            ok=m&mn&np.isfinite(zn)&(np.abs(orig-on)<=edge_gate_cm)
            acc[ok]+=zn[ok]; cnt[ok]+=1.0
        has=m&(cnt>0)
        avg=np.zeros_like(z); avg[has]=acc[has]/cnt[has]
        anchor=np.where(sup>=2,0.58,0.30).astype(np.float32)
        cand=z.copy(); cand[has]=anchor[has]*orig[has]+(1.0-anchor[has])*avg[has]
        cand=np.where(m,np.clip(cand,orig-max_change_cm,orig+max_change_cm),cand)
        z=cand

    znew=z[vv,uu].astype(np.float64)
    pnew=v12.backproject(surf['camera'],uu,vv,znew)
    cross=np.zeros(len(pnew),np.int8)
    for ocam,omask in other_views:
        cross+=v12.mask_hit(ocam,pnew,omask).astype(np.int8)
    accepted=cross>=1
    pold=pts[sel]
    pfinal=pold.copy(); pfinal[accepted]=pnew[accepted]
    zfinal=depth[sel].astype(np.float64); zfinal[accepted]=znew[accepted]
    delta=np.abs(zfinal-depth[sel])

    out=dict(surf)
    out['points']=pfinal.astype(np.float32)
    out['colors']=surf['colors'][sel].copy()
    out['source_depth']=zfinal.astype(np.float32)
    out['support']=surf['support'][sel].copy()
    out['regularization_qa']={
        'applied':True,
        'input_points':int(len(pts)),
        'unique_regularized_points':int(len(sel)),
        'cross_view_support_after_fraction':float(np.mean(accepted)),
        'reverted_vertices':int(np.sum(~accepted)),
        'median_depth_change_cm':float(np.median(delta)),
        'p95_depth_change_cm':float(np.percentile(delta,95)),
        'max_depth_change_cm':float(np.max(delta)),
        'iterations':int(iterations),
        'edge_gate_cm':float(edge_gate_cm),
    }
    return out


def render_mesh_union(meshes,K,R,C,coincident_cm=45.0):
    by={lab:[m for m in meshes if m['source_label']==lab] for lab in CAMERA_ORDER}
    ras={}; qa={}
    for lab in CAMERA_ORDER:
        if not by[lab]: continue
        deg=v13.view_delta_deg(by[lab][0]['camera'],C)
        if deg>96.0:
            qa[lab]={'view_delta_deg':float(deg),'appearance_skipped':True}; continue
        im,mask,dep,rqa=v13.raster_one_source(by[lab],K,R,C)
        ras[lab]=(im,mask,dep); qa[lab]={'view_delta_deg':float(deg),'appearance_skipped':False,**rqa,'pixels':int(mask.sum())}
    out=np.zeros((H,W,3),np.uint8); mask=np.zeros((H,W),bool); depout=np.full((H,W),np.inf,np.float32); owner=np.full((H,W),-1,np.int8)
    if not ras: return out,mask,depout,owner,qa
    labels=[lab for lab in CAMERA_ORDER if lab in ras]
    D=np.stack([np.where(ras[lab][1],ras[lab][2],np.inf) for lab in labels],axis=0)
    dmin=np.min(D,axis=0); anym=np.isfinite(dmin)
    deltas=np.asarray([v13.view_delta_deg(by[lab][0]['camera'],C) for lab in labels],np.float64)
    eligible=D<=dmin[None,:,:]+coincident_cm
    costs=np.where(eligible,deltas[:,None,None],1e9)
    choose=np.argmin(costs,axis=0)
    for k,lab in enumerate(labels):
        take=anym&(choose==k); out[take]=ras[lab][0][take]; depout[take]=ras[lab][2][take]; owner[take]=CAMERA_ORDER.index(lab)
    return out,anym,depout,owner,qa


def owner_switch_fraction(mask,owner):
    num=0; den=0
    for dy,dx in ((0,1),(1,0)):
        m2=np.roll(np.roll(mask,dy,axis=0),dx,axis=1); o2=np.roll(np.roll(owner,dy,axis=0),dx,axis=1)
        valid=mask&m2
        den+=int(valid.sum()); num+=int(np.sum(valid&(owner!=o2)))
    return float(num/max(1,den))


def owner_image(mask,owner):
    out=np.zeros((H,W),np.uint8)
    out[mask&(owner==0)]=85; out[mask&(owner==1)]=170; out[mask&(owner==2)]=255
    return out


def annotate(img,angle,qa,fqa):
    out=img.copy(); cv2.rectangle(out,(0,0),(960,68),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | v14 REGULARIZED CONTINUOUS 3-CAMERA MESH',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f"virtual {angle:04.1f} deg | triangles {qa['total_triangles']} | body px {fqa['body']['pixels']} | owner-switch {fqa['owner_switch_fraction']:.3f}",(10,43),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'regularized metric depth + cross-view support lock + coherent source ownership; no image blur/generative fill',(10,62),cv2.FONT_HERSHEY_SIMPLEX,.32,(210,210,210),1,cv2.LINE_AA)
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
    zpriors={}; instances={}; priors={}; descs={}; sources={}
    qa={'schema_version':14,'game_id':'0022500301','event_id':489,'source_sync_v11_selected':sync['selected'],'sync_ball_ray_gap_cm':float(sync['best_ball_pair']['ray_gap_cm']),'sync_ball_reprojection_errors_px':sync['best_ball_pair']['reprojection_errors_px'],'sources':{},'players':[],'frames':[]}
    for lab in CAMERA_ORDER:
        C,R,K=cams[lab]; depth,_,valid,_,_=moge_infer(depth_model,images[lab],args.tokens); align,dqa=v3.robust_depth_align(depth,valid,K,R,C); zpriors[lab]=v12.source_depth_prior(depth,valid,align)
        dyn,inst,_=v6.detect_near_play(seg,images[lab],K,R,C); instances[lab]=inst; priors[lab]=[]; descs[lab]=[]
        for row in inst:
            row['support_mask']=v9.support_mask(row['mask']); priors[lab].append(v9.instance_metric_prior(depth,valid,align,K,R,C,row['mask'])); descs[lab].append(v9.descriptor(images[lab],row['mask']))
        fv,bv=v5.source_visibility_v5(images[lab],depth,valid,dyn,K,R,C,align); sources[lab]={'image':images[lab],'C':C,'R':R,'K':K,'floor_vis':fv,'board_vis':bv,'dynamic':dyn}; v8.subtract_metric_rim_from_plane_visibility(sources[lab])
        qa['sources'][lab]={'file':paths[lab].name,'instances':len(inst),'depth_alignment':dqa,'camera_center_cm':[float(x) for x in C],'forward_sign':float(v3.forward_sign(R,C))}

    players,mqa=v12.player_matching(cams,instances,priors,descs,voxel_cm=5.0); players,rarqa=v13.enforce_unique_rar(players); qa['matching']=mqa; qa['rar_identity_assignment']=rarqa
    meshes=[]; support_fracs=[]
    for pi,p in enumerate(players):
        masks={REF:instances[REF][p['lar']]['support_mask'],BR:instances[BR][p['br']]['support_mask']}
        if p.get('rar') is not None: masks[RAR]=instances[RAR][p['rar']]['support_mask']
        pqa={'player_index':pi,'instances':{REF:p['lar'],BR:p['br'],RAR:p.get('rar')},'surfaces':{},'meshes':{}}
        for lab in tuple(masks.keys()):
            others=[(cams[o],masks[o]) for o in masks if o!=lab]
            surf=v12.build_ray_supported_surface(lab,cams[lab],images[lab],masks[lab],zpriors[lab],others,p['component_points'])
            if surf is None: continue
            surf['camera']=cams[lab]; surf['player_index']=pi
            surf=regularize_surface(surf,others)
            pqa['surfaces'][lab]={**surf.get('qa',{}),'regularization':surf.get('regularization_qa',{})}
            if surf.get('regularization_qa',{}).get('applied'):
                support_fracs.append(float(surf['regularization_qa']['cross_view_support_after_fraction']))
            grid=1 if lab==REF else 2
            mesh=v13.mesh_from_surface(surf,images[lab],grid_step=grid,max_edge_cm=52.0 if lab!=RAR else 65.0,max_depth_span_cm=38.0 if lab!=RAR else 60.0)
            if mesh is not None: meshes.append(mesh); pqa['meshes'][lab]=mesh['qa']
        qa['players'].append(pqa)
    if not meshes: raise RuntimeError('no player meshes')
    qa['total_vertices']=int(sum(len(m['points']) for m in meshes)); qa['total_triangles']=int(sum(len(m['triangles']) for m in meshes)); qa['mesh_counts_by_source']={lab:{'vertices':int(sum(len(m['points']) for m in meshes if m['source_label']==lab)),'triangles':int(sum(len(m['triangles']) for m in meshes if m['source_label']==lab))} for lab in CAMERA_ORDER}
    qa['self_projection']=v13.self_projection_qa(meshes,players,instances,cams); qa['minimum_regularized_cross_view_support_fraction']=float(min(support_fracs)) if support_fracs else None

    ball_center,ball_surfaces=v12.ball_surface(cams,images,sync); qa['ball_center_world_cm']=[float(x) for x in ball_center]
    rp=v8.rim_points(); ruv,_,rv=v8.project_metric(cams[REF],rp); rcols=v8.bilinear_sample(images[REF],ruv); rim_surface={'points':rp[rv],'colors':rcols[rv],'source_label':REF,'source_depth':v12.camera_z(cams[REF],rp[rv]).astype(np.float32),'support':np.full(int(np.sum(rv)),2,np.int8),'camera':cams[REF],'qa':{}}

    C0,R0,K0=cams[REF]; angles=np.asarray([0,5,10,15,20,25],np.float64) if args.static_only else np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    body_areas=[]; fragmentation=[]; switch_fracs=[]
    for fi,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang)); Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor'); floor,fm=v4.sample_plane_from_sources(Pf,sf,sources,'floor'); Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board'); board,bm=v4.sample_plane_from_sources(Pb,sb,sources,'board'); img=floor.copy(); img[bm]=board[bm]
        body,bmask,bdep,bowner,sqa=render_mesh_union(meshes,K0,Rt,Ct,coincident_cm=45.0)
        pr,pm,pdep,_,_=v12.render_surfel_union(ball_surfaces+[rim_surface],K0,Rt,Ct)
        fg=body.copy(); fgmask=bmask.copy(); fgdep=bdep.copy(); takep=(pm>0)&((~fgmask)|(pdep<fgdep)); fg[takep]=pr[takep]; fgdep[takep]=pdep[takep]; fgmask[takep]=True; img[fgmask]=fg[fgmask]
        if abs(float(ang))<1e-9:
            cv2.imwrite(str(args.out/'raw_mesh_00deg.png'),img)
            exact=np.zeros((H,W),bool)
            for p in players: exact|=instances[REF][p['lar']]['mask'].astype(bool)
            img[exact]=images[REF][exact]
        bstats=v13.mask_stats(bmask); sw=owner_switch_fraction(bmask,bowner); fragmentation.append(bstats['substantial_pixel_fraction']); switch_fracs.append(sw); body_areas.append(bstats['pixels'])
        cv2.imwrite(str(args.out/f'owner_{fi:03d}.png'),owner_image(bmask,bowner))
        fqa={'frame':fi,'virtual_viewpoint_deg':float(ang),'body':bstats,'owner_switch_fraction':sw,'body_owner_pixels':{CAMERA_ORDER[k]:int(np.sum(bmask&(bowner==k))) for k in range(3)},'source_raster':sqa,'foreground_pixels':int(fgmask.sum())}; qa['frames'].append(fqa); cv2.imwrite(str(args.out/f'frame_{fi:03d}.png'),annotate(img,float(ang),qa,fqa))

    ratios=[]
    if args.static_only and body_areas:
        a0=max(body_areas[0],1); ratios=[x/a0 for x in body_areas]
    qa['static_body_area_ratios_vs_0deg']=ratios; qa['minimum_substantial_pixel_fraction']=float(min(fragmentation)) if fragmentation else None; qa['maximum_owner_switch_fraction']=float(max(switch_fracs)) if switch_fracs else None
    rar_present=qa['mesh_counts_by_source'][RAR]['triangles']>0
    qa['gates']={
        'sync_ball_pass':bool(sync['gates']['all_ball_geometry_pass']),
        'rar_identity_unique_pass':bool(rarqa['final_unique']),
        'mesh_triangle_count_pass':qa['total_triangles']>=5000,
        'lar_mesh_self_recall_pass':qa['self_projection'].get(REF,{}).get('recall',0)>=0.86,
        'broadcast_mesh_self_recall_pass':qa['self_projection'].get(BR,{}).get('recall',0)>=0.76,
        'rar_mesh_self_recall_pass':(not rar_present) or qa['self_projection'].get(RAR,{}).get('recall',0)>=0.60,
        'regularized_cross_view_support_pass':(not support_fracs) or min(support_fracs)>=0.88,
        'fragmentation_pass':(not fragmentation) or min(fragmentation)>=0.94,
        'source_owner_stability_pass':(not switch_fracs) or max(switch_fracs)<=0.12,
        'body_area_no_collapse_pass':(not ratios) or min(ratios[1:])>=0.58,
        'body_area_no_explosion_pass':(not ratios) or max(ratios[1:])<=1.50,
    }
    qa['gates']['numeric_pass']=all(qa['gates'].values()); qa['status']='STATIC_REGULARIZED_MESH_DIAGNOSTIC_RENDERED' if args.static_only else 'FULL_REGULARIZED_MESH_DIAGNOSTIC_RENDERED'; qa['certification_scope']='3-camera R&D diagnostic only; visual QA authoritative; production gate remains >=4 accepted cameras'; qa['method']='v11 exact visual sync + one-to-one RAR identity + cross-view-locked spatial depth regularization + continuous textured triangles + coherent source ownership + metric z-buffer'
    (args.out/'three_camera_mesh_v14_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'status':qa['status'],'vertices':qa['total_vertices'],'triangles':qa['total_triangles'],'mesh_counts_by_source':qa['mesh_counts_by_source'],'self_projection':qa['self_projection'],'min_cross_view_support':qa['minimum_regularized_cross_view_support_fraction'],'min_substantial_fraction':qa['minimum_substantial_pixel_fraction'],'max_owner_switch_fraction':qa['maximum_owner_switch_fraction'],'body_area_ratios':ratios,'gates':qa['gates']},indent=2),flush=True)

if __name__=='__main__': main()
