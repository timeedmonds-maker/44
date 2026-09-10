from __future__ import annotations

"""Jazz event 489: v15 static/free-view renderer.

v14 fixed exact visual sync, RAR identity uniqueness, depth regularity and source
ownership, but semantic QA exposed a separate background failure: a player pixel
could survive the plane-visibility test and then be reprojected as if it were
hardwood. v15 therefore separates the two problems explicitly:

1. matched players retain v14's cross-view-locked continuous textured meshes;
2. an unmatched but confidently segmented LAR player is retained as a clearly
   labelled LAR-only metric-depth mesh rather than disappearing;
3. floor/board texture is accepted only under a much stricter metric-depth test
   and a substantially dilated person/ball exclusion mask;
4. background-only diagnostic frames are saved so semantic QA can distinguish a
   plane-texture failure from a foreground-geometry failure.

All texture comes from real synchronized NBA source pixels. No generative fill,
image inpainting, optical-flow morph, or synthetic arena/player texture is used.
"""

import argparse, json
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
from freeze_spin import build_three_camera_reference_surface_v10 as v10
from freeze_spin import build_three_camera_surfel_v12 as v12
from freeze_spin import build_three_camera_mesh_v13 as v13
from freeze_spin import build_three_camera_mesh_v14 as v14
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
REF,BR,RAR=v12.REF,v12.BR,v12.RAR
CAMERA_ORDER=(REF,BR,RAR)


def strict_plane_visibility(depth, valid, dynamic, instances, K, R, C, align):
    z_est=align[0]*depth.astype(np.float64)+align[1]
    exclusion=dynamic.astype(np.uint8)
    for inst in instances:
        exclusion=np.maximum(exclusion,inst['mask'].astype(np.uint8))
    # Deliberately generous: false background is worse than a source-grounded hole.
    exclusion=cv2.dilate(exclusion,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),iterations=1)>0

    tf,Pf=v4.ray_plane_map(K,R,C,'floor')
    fg=(np.isfinite(tf)&(tf>20)&(tf<12000)&
        (Pf[:,:,0]>=v4.COURT_X0)&(Pf[:,:,0]<=v4.COURT_X1)&
        (Pf[:,:,1]>=v4.COURT_Y0)&(Pf[:,:,1]<=v4.COURT_Y1))
    # v5 allowed max(42 cm, 6% range), which can admit lower-body pixels at long range.
    # The exact court is known, so use a deliberately strict range-consistency gate.
    ftol=np.maximum(18.0,0.018*tf)
    floor_vis=fg&valid&np.isfinite(z_est)&(np.abs(z_est-tf)<=ftol)&(~exclusion)
    floor_vis=cv2.morphologyEx(floor_vis.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)>0

    tb,Pb=v4.ray_plane_map(K,R,C,'board')
    bg=(np.isfinite(tb)&(tb>20)&(tb<12000)&
        (np.abs(Pb[:,:,1])<=v4.BOARD_Y)&
        (Pb[:,:,2]>=v4.BOARD_Z0)&(Pb[:,:,2]<=v4.BOARD_Z1))
    bex=cv2.dilate(exclusion.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)),iterations=1)>0
    # Glass depth is unreliable; identity exclusion is the important protection here.
    board_vis=bg&valid&(~bex)
    return floor_vis,board_vis,exclusion


def monocular_lar_surface(inst, cam, image, zprior):
    """Fallback for a real LAR person absent from the other solved views.

    It is intentionally labelled single-view. Depth is regularized only inside the
    source silhouette and the visible foot contact is gently anchored to z=0 when
    the detector's floor intersection is consistent.
    """
    mask=inst['mask'].astype(bool)
    ys,xs=np.where(mask)
    if len(xs)<80:
        return None
    z=zprior.astype(np.float32).copy()
    valid=mask&np.isfinite(z)&(z>40)&(z<12000)
    vals=z[valid]
    if len(vals)<20:
        return None
    med=float(np.median(vals))
    z=np.where(np.isfinite(z),z,med).astype(np.float32)
    # Restrict to a human-scale depth slab around this detected instance.
    t=float(inst.get('floor_camera_depth_cm',med))
    lo=max(60.0,min(med,t)-190.0); hi=max(lo+80.0,max(med,t)+190.0)
    z=np.clip(z,lo,hi)

    x0=max(0,int(xs.min())-5); x1=min(W,int(xs.max())+6)
    y0=max(0,int(ys.min())-5); y1=min(H,int(ys.max())+6)
    sub=z[y0:y1,x0:x1].copy(); sm=mask[y0:y1,x0:x1]
    smed=float(np.median(sub[sm])); tmp=sub.copy(); tmp[~sm]=smed
    for _ in range(3):
        tmp=cv2.bilateralFilter(tmp,7,28.0,4.0)
    sub[sm]=0.55*sub[sm]+0.45*tmp[sm]
    z[y0:y1,x0:x1]=sub

    # Ground-contact correction identical in spirit to v5, but conservative.
    fx,fy=[int(x) for x in inst['foot_px']]
    yy0=max(0,fy-7); yy1=min(H,fy+1); xx0=max(0,fx-10); xx1=min(W,fx+11)
    local=mask[yy0:yy1,xx0:xx1]&np.isfinite(z[yy0:yy1,xx0:xx1])
    delta=0.0
    if int(local.sum())>=3:
        raw=float(np.median(z[yy0:yy1,xx0:xx1][local])); d=t-raw
        if abs(d)<=75.0:
            z[mask]+=float(d); delta=float(d)

    zv=z[ys,xs].astype(np.float64)
    pts=v10.backproject_pixels(cam,xs,ys,zv).astype(np.float32)
    physical=(np.isfinite(pts).all(1)&(pts[:,0]>-450)&(pts[:,0]<1100)&
              (np.abs(pts[:,1])<850)&(pts[:,2]>-80)&(pts[:,2]<500))
    return {
        'points':pts[physical],
        'colors':image[ys[physical],xs[physical]].astype(np.uint8),
        'source_label':REF,
        'source_depth':zv[physical].astype(np.float32),
        'support':np.zeros(int(np.sum(physical)),np.int8),
        'camera':cam,
        'qa':{
            'single_view_fallback':True,
            'source_mask_pixels':int(mask.sum()),
            'surface_points':int(np.sum(physical)),
            'median_depth_cm':med,
            'depth_slab_cm':[float(lo),float(hi)],
            'ground_contact_delta_cm':delta,
        }
    }


def self_projection_all(meshes, used_indices, instances, cams):
    out={}
    for lab in CAMERA_ORDER:
        ms=[m for m in meshes if m['source_label']==lab]
        if not ms: continue
        _,rm,_,rqa=v13.raster_one_source(ms,cams[lab][2],cams[lab][1],cams[lab][0])
        union=np.zeros((H,W),bool)
        for idx in sorted(used_indices.get(lab,set())):
            union|=instances[lab][idx]['support_mask']
        inter=int(np.sum(rm&union)); uni=int(np.sum(rm|union))
        out[lab]={
            'iou':float(inter/max(1,uni)),
            'recall':float(inter/max(1,int(union.sum()))),
            'rendered_pixels':int(rm.sum()),'mask_pixels':int(union.sum()),**rqa
        }
    return out


def background_sample(P,support,sources,plane):
    """LAR owns known plane texture; stricter other views may fill only real holes."""
    out=np.zeros((H*W,3),np.uint8); owned=np.zeros(H*W,bool); owner=np.full(H*W,-1,np.int8)
    ids=np.where(support)[0]
    if not len(ids): return out.reshape(H,W,3),owned.reshape(H,W),owner.reshape(H,W)
    Pw=P[ids]
    for li,label in enumerate(CAMERA_ORDER):
        src=sources[label]; C,R,K=src['C'],src['R'],src['K']; s=v3.forward_sign(R,C)
        Xc=(R@(Pw-C).T).T; q=(K@Xc.T).T
        with np.errstate(divide='ignore',invalid='ignore'): uv=q[:,:2]/q[:,2:3]
        u=np.rint(uv[:,0]).astype(np.int32); vv=np.rint(uv[:,1]).astype(np.int32)
        ok=np.isfinite(uv).all(1)&(s*Xc[:,2]>20)&(u>=0)&(u<W)&(vv>=0)&(vv<H)
        loc=np.where(ok)[0]
        if not len(loc): continue
        vis=src['floor_vis'] if plane=='floor' else src['board_vis']
        loc=loc[vis[vv[loc],u[loc]]]
        if not len(loc): continue
        tgt=ids[loc]; take=~owned[tgt]; tgt=tgt[take]; loc=loc[take]
        out[tgt]=src['image'][vv[loc],u[loc]]; owned[tgt]=True; owner[tgt]=li
    return out.reshape(H,W,3),owned.reshape(H,W),owner.reshape(H,W)


def annotate(img,angle,qa,fqa):
    out=img.copy(); cv2.rectangle(out,(0,0),(960,70),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | v15 STRICT PLANE + COMPLETE LAR FOREGROUND',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f"virtual {angle:04.1f} deg | triangles {qa['total_triangles']} | body px {fqa['body']['pixels']} | background {fqa['background_pixels']}",(10,43),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'matched players multi-view; unmatched LAR person explicitly single-view; strict metric floor exclusion; no generated fill',(10,63),cv2.FONT_HERSHEY_SIMPLEX,.31,(210,210,210),1,cv2.LINE_AA)
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
    zpriors={}; instances={}; priors={}; descs={}; sources={}; qa={'schema_version':15,'game_id':'0022500301','event_id':489,'source_sync_v11_selected':sync['selected'],'sync_ball_ray_gap_cm':float(sync['best_ball_pair']['ray_gap_cm']),'sources':{},'players':[],'frames':[]}
    for lab in CAMERA_ORDER:
        C,R,K=cams[lab]; depth,_,valid,_,_=moge_infer(depth_model,images[lab],args.tokens); align,dqa=v3.robust_depth_align(depth,valid,K,R,C); zpriors[lab]=v12.source_depth_prior(depth,valid,align)
        dyn,inst,_=v6.detect_near_play(seg,images[lab],K,R,C); instances[lab]=inst; priors[lab]=[]; descs[lab]=[]
        for row in inst:
            row['support_mask']=v9.support_mask(row['mask']); priors[lab].append(v9.instance_metric_prior(depth,valid,align,K,R,C,row['mask'])); descs[lab].append(v9.descriptor(images[lab],row['mask']))
        fv,bv,ex=strict_plane_visibility(depth,valid,dyn,inst,K,R,C,align)
        sources[lab]={'image':images[lab],'C':C,'R':R,'K':K,'floor_vis':fv,'board_vis':bv,'dynamic':ex}
        qa['sources'][lab]={'file':paths[lab].name,'instances':len(inst),'depth_alignment':dqa,'floor_visible_pixels_strict':int(fv.sum()),'board_visible_pixels_strict':int(bv.sum()),'excluded_dynamic_pixels':int(ex.sum()),'camera_center_cm':[float(x) for x in C]}
        cv2.imwrite(str(args.out/f"{lab.replace(' ','_')}_floor_visibility_strict.png"),fv.astype(np.uint8)*255); cv2.imwrite(str(args.out/f"{lab.replace(' ','_')}_dynamic_exclusion_strict.png"),ex.astype(np.uint8)*255)

    players,mqa=v12.player_matching(cams,instances,priors,descs,voxel_cm=5.0); players,rarqa=v13.enforce_unique_rar(players); qa['matching']=mqa; qa['rar_identity_assignment']=rarqa
    meshes=[]; used={lab:set() for lab in CAMERA_ORDER}; support_fracs=[]
    for pi,p in enumerate(players):
        masks={REF:instances[REF][p['lar']]['support_mask'],BR:instances[BR][p['br']]['support_mask']}; used[REF].add(p['lar']); used[BR].add(p['br'])
        if p.get('rar') is not None: masks[RAR]=instances[RAR][p['rar']]['support_mask']; used[RAR].add(p['rar'])
        pqa={'player_index':pi,'mode':'MULTI_VIEW','instances':{REF:p['lar'],BR:p['br'],RAR:p.get('rar')},'surfaces':{},'meshes':{}}
        for lab in tuple(masks.keys()):
            others=[(cams[o],masks[o]) for o in masks if o!=lab]
            surf=v12.build_ray_supported_surface(lab,cams[lab],images[lab],masks[lab],zpriors[lab],others,p['component_points'])
            if surf is None: continue
            surf['camera']=cams[lab]; surf['player_index']=pi; surf=v14.regularize_surface(surf,others)
            if surf.get('regularization_qa',{}).get('applied'): support_fracs.append(float(surf['regularization_qa']['cross_view_support_after_fraction']))
            grid=1 if lab==REF else 2
            mesh=v13.mesh_from_surface(surf,images[lab],grid_step=grid,max_edge_cm=52.0 if lab!=RAR else 65.0,max_depth_span_cm=38.0 if lab!=RAR else 60.0)
            pqa['surfaces'][lab]={**surf.get('qa',{}),'regularization':surf.get('regularization_qa',{})}
            if mesh is not None: meshes.append(mesh); pqa['meshes'][lab]=mesh['qa']
        qa['players'].append(pqa)

    matched_lar={p['lar'] for p in players}; unmatched_lar=sorted(set(range(len(instances[REF])))-matched_lar); qa['unmatched_lar_instances']=unmatched_lar
    for idx in unmatched_lar:
        surf=monocular_lar_surface(instances[REF][idx],cams[REF],images[REF],zpriors[REF])
        if surf is None: continue
        surf['player_index']=len(qa['players']); mesh=v13.mesh_from_surface(surf,images[REF],grid_step=1,max_edge_cm=65.0,max_depth_span_cm=55.0)
        if mesh is None: continue
        mesh['single_view_fallback']=True; meshes.append(mesh); used[REF].add(idx)
        qa['players'].append({'player_index':surf['player_index'],'mode':'LAR_ONLY_UNMATCHED','instances':{REF:idx,BR:None,RAR:None},'surface':surf['qa'],'mesh':mesh['qa']})

    if not meshes: raise RuntimeError('no player meshes')
    qa['total_vertices']=int(sum(len(m['points']) for m in meshes)); qa['total_triangles']=int(sum(len(m['triangles']) for m in meshes)); qa['mesh_counts_by_source']={lab:{'vertices':int(sum(len(m['points']) for m in meshes if m['source_label']==lab)),'triangles':int(sum(len(m['triangles']) for m in meshes if m['source_label']==lab))} for lab in CAMERA_ORDER}; qa['self_projection']=self_projection_all(meshes,used,instances,cams); qa['minimum_regularized_cross_view_support_fraction']=float(min(support_fracs)) if support_fracs else None

    ball_center,ball_surfaces=v12.ball_surface(cams,images,sync); qa['ball_center_world_cm']=[float(x) for x in ball_center]
    rp=v8.rim_points(); ruv,_,rv=v8.project_metric(cams[REF],rp); rcols=v8.bilinear_sample(images[REF],ruv); rim_surface={'points':rp[rv],'colors':rcols[rv],'source_label':REF,'source_depth':v12.camera_z(cams[REF],rp[rv]).astype(np.float32),'support':np.full(int(np.sum(rv)),2,np.int8),'camera':cams[REF],'qa':{}}

    C0,R0,K0=cams[REF]; angles=np.asarray([0,5,10,15,20,25],np.float64) if args.static_only else np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    body_areas=[]; frag=[]; switch=[]
    for fi,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang)); Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor'); floor,fm,fown=background_sample(Pf,sf,sources,'floor'); Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board'); board,bm,bown=background_sample(Pb,sb,sources,'board'); bg=floor.copy(); bg[bm]=board[bm]; bgm=fm|bm; cv2.imwrite(str(args.out/f'background_{fi:03d}.png'),bg)
        img=bg.copy(); body,bmask,bdep,bowner,sqa=v14.render_mesh_union(meshes,K0,Rt,Ct,coincident_cm=45.0); pr,pm,pdep,_,_=v12.render_surfel_union(ball_surfaces+[rim_surface],K0,Rt,Ct); fg=body.copy(); fgmask=bmask.copy(); fgdep=bdep.copy(); takep=(pm>0)&((~fgmask)|(pdep<fgdep)); fg[takep]=pr[takep]; fgdep[takep]=pdep[takep]; fgmask[takep]=True; img[fgmask]=fg[fgmask]
        if abs(float(ang))<1e-9:
            exact=np.zeros((H,W),bool)
            for idx in used[REF]: exact|=instances[REF][idx]['mask'].astype(bool)
            img[exact]=images[REF][exact]
        bstats=v13.mask_stats(bmask); sw=v14.owner_switch_fraction(bmask,bowner); body_areas.append(bstats['pixels']); frag.append(bstats['substantial_pixel_fraction']); switch.append(sw); cv2.imwrite(str(args.out/f'owner_{fi:03d}.png'),v14.owner_image(bmask,bowner)); fqa={'frame':fi,'virtual_viewpoint_deg':float(ang),'body':bstats,'owner_switch_fraction':sw,'background_pixels':int(bgm.sum()),'body_owner_pixels':{CAMERA_ORDER[k]:int(np.sum(bmask&(bowner==k))) for k in range(3)}}; qa['frames'].append(fqa); cv2.imwrite(str(args.out/f'frame_{fi:03d}.png'),annotate(img,float(ang),qa,fqa))

    ratios=[]
    if args.static_only and body_areas:
        a0=max(body_areas[0],1); ratios=[x/a0 for x in body_areas]
    qa['static_body_area_ratios_vs_0deg']=ratios; qa['minimum_substantial_pixel_fraction']=float(min(frag)); qa['maximum_owner_switch_fraction']=float(max(switch)); qa['gates']={
        'sync_ball_pass':bool(sync['gates']['all_ball_geometry_pass']),
        'rar_identity_unique_pass':bool(rarqa['final_unique']),
        'all_lar_people_represented_pass':used[REF]==set(range(len(instances[REF]))),
        'mesh_triangle_count_pass':qa['total_triangles']>=7000,
        'lar_mesh_self_recall_pass':qa['self_projection'].get(REF,{}).get('recall',0)>=0.88,
        'broadcast_mesh_self_recall_pass':qa['self_projection'].get(BR,{}).get('recall',0)>=0.76,
        'rar_mesh_self_recall_pass':qa['self_projection'].get(RAR,{}).get('recall',0)>=0.60,
        'regularized_cross_view_support_pass':(not support_fracs) or min(support_fracs)>=0.88,
        'fragmentation_pass':min(frag)>=0.94,
        'source_owner_stability_pass':max(switch)<=0.12,
        'body_area_no_collapse_pass':(not ratios) or min(ratios[1:])>=0.58,
        'body_area_no_explosion_pass':(not ratios) or max(ratios[1:])<=1.50,
        'strict_floor_available_pass':qa['sources'][REF]['floor_visible_pixels_strict']>=50000,
    }; qa['gates']['numeric_pass']=all(qa['gates'].values()); qa['status']='STATIC_STRICT_PLANE_COMPLETE_FOREGROUND_RENDERED' if args.static_only else 'FULL_STRICT_PLANE_COMPLETE_FOREGROUND_RENDERED'; qa['certification_scope']='3-camera R&D diagnostic; unmatched LAR person explicitly single-view; semantic visual QA authoritative'; qa['method']='v11 exact sync + v14 regularized multi-view mesh + single-view LAR fallback for unmatched person + strict exact-plane source visibility'
    (args.out/'three_camera_mesh_v15_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'status':qa['status'],'unmatched_lar':unmatched_lar,'used_lar':sorted(used[REF]),'vertices':qa['total_vertices'],'triangles':qa['total_triangles'],'self_projection':qa['self_projection'],'strict_floor_pixels':{k:v['floor_visible_pixels_strict'] for k,v in qa['sources'].items()},'body_area_ratios':ratios,'gates':qa['gates']},indent=2),flush=True)

if __name__=='__main__': main()
