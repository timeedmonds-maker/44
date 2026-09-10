from __future__ import annotations

"""Jazz event 489: exact-sync continuous per-player textured mesh diagnostic v13.

v12 established exact visual-state synchronization and useful ray-supported 3-D
surface points, but point splatting fragmented player appearance. v13 keeps the
same source-grounded 3-D evidence and changes only the representation/rendering:

* each player remains identity separated;
* Right Above Rim assignments are globally one-to-one;
* source pixels become vertices of a depth-discontinuity-aware triangle mesh;
* connected source pixels are joined only when their recovered 3-D positions are
  physically continuous;
* texture coordinates point back to the real source frame;
* novel views use perspective-correct triangle rasterization and a metric z-buffer;
* LAR is preferred where source surfaces are co-located, while Broadcast can fill
  genuinely newly exposed surfaces; RAR remains a third-view geometric constraint
  and may supply appearance only when its view direction is suitable.

No generated fill, crossfade, optical-flow morph, or invented player texture.
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
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
REF,BR,RAR=v12.REF,v12.BR,v12.RAR
CAMERA_ORDER=(REF,BR,RAR)


def enforce_unique_rar(players):
    """Never permit two reconstructed identities to consume one RAR instance."""
    claims={}
    for pi,p in enumerate(players):
        ri=p.get('rar')
        if ri is None:
            continue
        sel=(p.get('component_qa') or {}).get('selected_component') or {}
        ratio=float(sel.get('best_overhead_overlap_ratio',0.0))
        claims.setdefault(int(ri),[]).append((ratio,pi))
    dropped=[]
    for ri,rows in claims.items():
        rows.sort(reverse=True)
        for ratio,pi in rows[1:]:
            players[pi]['rar']=None
            dropped.append({'rar_instance':int(ri),'player_index':int(pi),'overlap_ratio':float(ratio)})
    final=[p.get('rar') for p in players if p.get('rar') is not None]
    return players,{'claims':{str(k):[{'overlap_ratio':float(r),'player_index':int(i)} for r,i in sorted(v,reverse=True)] for k,v in claims.items()},'dropped_duplicate_claims':dropped,'final_unique':len(final)==len(set(final))}


def mesh_from_surface(surf, source_image, grid_step=2, max_edge_cm=55.0, max_depth_span_cm=50.0):
    pts=surf['points'].astype(np.float32)
    if len(pts)<4:
        return None
    cam=surf['camera']
    uv,depth,valid=v12.safe_project(cam,pts)
    u=np.rint(uv[:,0]).astype(np.int32); y=np.rint(uv[:,1]).astype(np.int32)
    keep=valid&(u>=0)&(u<W)&(y>=0)&(y<H)&((u%grid_step)==0)&((y%grid_step)==0)
    ids=np.where(keep)[0]
    if len(ids)<4:
        return None
    pts=pts[ids]; uv_src=uv[ids].astype(np.float32); dep_src=depth[ids].astype(np.float32)
    pix=y[ids]*W+u[ids]
    order=np.argsort(dep_src,kind='stable'); pix_o=pix[order]
    _,first=np.unique(pix_o,return_index=True); take=order[first]
    pts=pts[take]; uv_src=uv_src[take]; dep_src=dep_src[take]; pix=pix[take]
    vid=np.full(H*W,-1,np.int32); vid[pix]=np.arange(len(pix),dtype=np.int32); vid=vid.reshape(H,W)

    ys=np.arange(0,H-grid_step,grid_step,dtype=np.int32)
    xs=np.arange(0,W-grid_step,grid_step,dtype=np.int32)
    a=vid[np.ix_(ys,xs)].ravel()
    b=vid[np.ix_(ys,xs+grid_step)].ravel()
    c=vid[np.ix_(ys+grid_step,xs)].ravel()
    d=vid[np.ix_(ys+grid_step,xs+grid_step)].ravel()
    full=(a>=0)&(b>=0)&(c>=0)&(d>=0)
    if not np.any(full):
        return None
    a,b,c,d=a[full],b[full],c[full],d[full]
    t1=np.column_stack([a,b,d]); t2=np.column_stack([a,d,c]); tris=np.vstack([t1,t2]).astype(np.int32)

    pp=pts[tris].astype(np.float64); dd=dep_src[tris].astype(np.float64)
    e01=np.linalg.norm(pp[:,0]-pp[:,1],axis=1); e12=np.linalg.norm(pp[:,1]-pp[:,2],axis=1); e20=np.linalg.norm(pp[:,2]-pp[:,0],axis=1)
    dep_span=np.ptp(dd,axis=1)
    good=(np.maximum.reduce([e01,e12,e20])<=max_edge_cm)&(dep_span<=max_depth_span_cm)
    area=np.linalg.norm(np.cross(pp[:,1]-pp[:,0],pp[:,2]-pp[:,0]),axis=1)*0.5
    good&=(area>0.02)&np.isfinite(area)
    tris=tris[good]
    if len(tris)<20:
        return None
    return {
        'points':pts,
        'source_uv':uv_src,
        'source_depth':dep_src,
        'triangles':tris,
        'source_label':surf['source_label'],
        'player_index':surf.get('player_index'),
        'camera':cam,
        'image':source_image,
        'qa':{
            'vertices':int(len(pts)),
            'triangles':int(len(tris)),
            'grid_step_px':int(grid_step),
            'candidate_triangles':int(len(t1)+len(t2)),
            'retained_triangle_fraction':float(len(tris)/max(1,len(t1)+len(t2))),
            'max_edge_cm':float(max_edge_cm),
            'max_depth_span_cm':float(max_depth_span_cm),
        }
    }


def bilinear(image,u,v):
    u=np.asarray(u,np.float64); v=np.asarray(v,np.float64)
    u=np.clip(u,0,W-1.001); v=np.clip(v,0,H-1.001)
    x0=np.floor(u).astype(np.int32); y0=np.floor(v).astype(np.int32)
    x1=np.minimum(x0+1,W-1); y1=np.minimum(y0+1,H-1)
    ax=(u-x0)[:,None]; ay=(v-y0)[:,None]
    c00=image[y0,x0].astype(np.float64); c10=image[y0,x1].astype(np.float64)
    c01=image[y1,x0].astype(np.float64); c11=image[y1,x1].astype(np.float64)
    out=(1-ax)*(1-ay)*c00+ax*(1-ay)*c10+(1-ax)*ay*c01+ax*ay*c11
    return np.clip(out,0,255).astype(np.uint8)


def raster_one_source(meshes,K,R,C,max_bbox_px=56):
    out=np.zeros((H,W,3),np.uint8); zbuf=np.full((H,W),np.inf,np.float32); touched=np.zeros((H,W),bool)
    tri_drawn=0; tri_rejected_screen=0
    for mesh in meshes:
        pts=mesh['points']; suv=mesh['source_uv']; tris=mesh['triangles']; src=mesh['image']
        tuv,td,valid=v12.safe_project((C,R,K),pts)
        for tri in tris:
            if not bool(np.all(valid[tri])):
                continue
            q=tuv[tri].astype(np.float64); z=td[tri].astype(np.float64)
            if np.any(z<=20) or not np.isfinite(q).all():
                continue
            xmin=max(0,int(math.floor(float(np.min(q[:,0]))))); xmax=min(W-1,int(math.ceil(float(np.max(q[:,0])))))
            ymin=max(0,int(math.floor(float(np.min(q[:,1]))))); ymax=min(H-1,int(math.ceil(float(np.max(q[:,1])))))
            bw=xmax-xmin+1; bh=ymax-ymin+1
            if bw<=0 or bh<=0 or bw>max_bbox_px or bh>max_bbox_px:
                tri_rejected_screen+=1; continue
            x0,y0=q[0]; x1,y1=q[1]; x2,y2=q[2]
            den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2)
            if abs(den)<1e-7:
                continue
            yy,xx=np.mgrid[ymin:ymax+1,xmin:xmax+1]
            px=xx.astype(np.float64)+0.5; py=yy.astype(np.float64)+0.5
            w0=((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den
            w1=((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den
            w2=1.0-w0-w1
            inside=(w0>=-1e-5)&(w1>=-1e-5)&(w2>=-1e-5)
            if not np.any(inside):
                continue
            iy,ix=np.where(inside); gy=iy+ymin; gx=ix+xmin
            ww=np.column_stack([w0[inside],w1[inside],w2[inside]])
            invz=np.sum(ww/z[None,:],axis=1)
            good=np.isfinite(invz)&(invz>1e-8)
            if not np.any(good):
                continue
            gy=gy[good]; gx=gx[good]; ww=ww[good]; invz=invz[good]
            zz=(1.0/invz).astype(np.float32)
            cur=zbuf[gy,gx]; front=zz<cur
            if not np.any(front):
                continue
            gy=gy[front]; gx=gx[front]; ww=ww[front]; invz=invz[front]; zz=zz[front]
            uvz=suv[tri].astype(np.float64)/z[:,None]
            tex=(ww@uvz)/invz[:,None]
            col=bilinear(src,tex[:,0],tex[:,1])
            out[gy,gx]=col; zbuf[gy,gx]=zz; touched[gy,gx]=True
            tri_drawn+=1
    return out,touched,zbuf,{'triangles_drawn':int(tri_drawn),'triangles_rejected_screen':int(tri_rejected_screen)}


def view_delta_deg(cam,target_C):
    a=cam[0]-RIM; b=target_C-RIM
    a=a/max(np.linalg.norm(a),1e-9); b=b/max(np.linalg.norm(b),1e-9)
    return math.degrees(math.acos(float(np.clip(np.dot(a,b),-1.0,1.0))))


def render_mesh_union(meshes,K,R,C,coincident_cm=14.0):
    by={lab:[m for m in meshes if m['source_label']==lab] for lab in CAMERA_ORDER}
    ras={}; qa={}
    for lab in CAMERA_ORDER:
        if not by[lab]:
            continue
        deg=view_delta_deg(by[lab][0]['camera'],C)
        if deg>96.0:
            qa[lab]={'view_delta_deg':float(deg),'appearance_skipped':True}; continue
        im,mask,dep,rqa=raster_one_source(by[lab],K,R,C)
        ras[lab]=(im,mask,dep); qa[lab]={'view_delta_deg':float(deg),'appearance_skipped':False,**rqa,'pixels':int(mask.sum())}
    out=np.zeros((H,W,3),np.uint8); mask=np.zeros((H,W),bool); depout=np.full((H,W),np.inf,np.float32); owner=np.full((H,W),-1,np.int8)
    if not ras:
        return out,mask,depout,owner,qa
    stack_d=[]; labels=[]
    for lab in CAMERA_ORDER:
        if lab in ras:
            d=ras[lab][2].copy(); d[~ras[lab][1]]=np.inf; stack_d.append(d); labels.append(lab)
    D=np.stack(stack_d,axis=0); dmin=np.min(D,axis=0); anym=np.isfinite(dmin)
    deltas=np.asarray([view_delta_deg(by[lab][0]['camera'],C) for lab in labels],np.float64)
    eligible=D<=dmin[None,:,:]+coincident_cm
    costs=np.where(eligible,deltas[:,None,None],1e9)
    choose=np.argmin(costs,axis=0)
    for k,lab in enumerate(labels):
        take=anym&(choose==k); out[take]=ras[lab][0][take]; depout[take]=ras[lab][2][take]; owner[take]=CAMERA_ORDER.index(lab)
    mask=anym
    return out,mask,depout,owner,qa


def mask_stats(mask,min_component=45):
    m=mask.astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
    areas=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.asarray([],np.int32)
    substantial=areas[areas>=min_component]
    retained=float(substantial.sum()/max(1,int(mask.sum())))
    return {'pixels':int(mask.sum()),'components_total':int(max(0,n-1)),'substantial_components':int(len(substantial)),'substantial_pixel_fraction':retained,'largest_components_px':[int(x) for x in sorted(substantial.tolist(),reverse=True)[:12]]}


def self_projection_qa(meshes,players,instances,cams):
    out={}
    for lab in CAMERA_ORDER:
        ms=[m for m in meshes if m['source_label']==lab]
        if not ms:
            continue
        im,rm,_,rqa=raster_one_source(ms,cams[lab][2],cams[lab][1],cams[lab][0])
        union=np.zeros((H,W),bool); key={REF:'lar',BR:'br',RAR:'rar'}[lab]
        for p in players:
            idx=p.get(key)
            if idx is not None:
                union|=instances[lab][idx]['support_mask']
        inter=int(np.sum(rm&union)); uni=int(np.sum(rm|union))
        out[lab]={'iou':float(inter/max(1,uni)),'recall':float(inter/max(1,int(union.sum()))),'rendered_pixels':int(rm.sum()),'mask_pixels':int(union.sum()),**rqa}
    return out


def annotate(img,angle,qa,fqa):
    out=img.copy(); cv2.rectangle(out,(0,0),(960,68),(0,0,0),-1)
    cv2.putText(out,'JAZZ EVENT 489 | v13 EXACT-SYNC CONTINUOUS TEXTURED MESH',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f"virtual viewpoint {angle:04.1f} deg | triangles {qa['total_triangles']} | body px {fqa['body']['pixels']} | ball ray gap {qa['sync_ball_ray_gap_cm']:.2f} cm",(10,43),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,'per-player depth-continuous triangles; perspective-correct real-source texture; one metric z-buffer; no generated fill',(10,62),cv2.FONT_HERSHEY_SIMPLEX,.32,(210,210,210),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sync-dir',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--tokens',type=int,default=1200); ap.add_argument('--static-only',action='store_true'); ap.add_argument('--grid-step',type=int,default=2); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.loads((args.sync_dir/'three_camera_visual_sync_v11.json').read_text())
    if not sync['gates']['all_ball_geometry_pass']:
        raise RuntimeError('v11 exact visual sync did not pass ball geometry')
    paths={REF:args.sync_dir/'K_Left_Above_Rim_visual_sync.png',BR:args.sync_dir/'A_Broadcast_visual_sync.png',RAR:args.sync_dir/'L_Right_Above_Rim_visual_sync.png'}
    images={k:cv2.imread(str(p)) for k,p in paths.items()}
    if any(im is None for im in images.values()):
        raise RuntimeError('missing v11 selected frames')
    cams=base.load_cameras(args.registry,args.rar_report,paths[BR])

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    depth_model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval(); seg=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    zpriors={}; instances={}; priors={}; descs={}; sources={}
    qa={'schema_version':13,'game_id':'0022500301','event_id':489,'source_sync_v11_selected':sync['selected'],'sync_ball_ray_gap_cm':float(sync['best_ball_pair']['ray_gap_cm']),'sync_ball_reprojection_errors_px':sync['best_ball_pair']['reprojection_errors_px'],'sources':{},'players':[],'frames':[]}
    for lab in CAMERA_ORDER:
        C,R,K=cams[lab]; depth,_,valid,_,_=moge_infer(depth_model,images[lab],args.tokens); align,dqa=v3.robust_depth_align(depth,valid,K,R,C); zpriors[lab]=v12.source_depth_prior(depth,valid,align)
        dyn,inst,_=v6.detect_near_play(seg,images[lab],K,R,C); instances[lab]=inst; priors[lab]=[]; descs[lab]=[]
        for row in inst:
            row['support_mask']=v9.support_mask(row['mask']); priors[lab].append(v9.instance_metric_prior(depth,valid,align,K,R,C,row['mask'])); descs[lab].append(v9.descriptor(images[lab],row['mask']))
        fv,bv=v5.source_visibility_v5(images[lab],depth,valid,dyn,K,R,C,align); sources[lab]={'image':images[lab],'C':C,'R':R,'K':K,'floor_vis':fv,'board_vis':bv,'dynamic':dyn}; v8.subtract_metric_rim_from_plane_visibility(sources[lab])
        qa['sources'][lab]={'file':paths[lab].name,'instances':len(inst),'depth_alignment':dqa,'camera_center_cm':[float(x) for x in C],'forward_sign':float(v3.forward_sign(R,C))}

    players,mqa=v12.player_matching(cams,instances,priors,descs,voxel_cm=5.0); players,rarqa=enforce_unique_rar(players); qa['matching']=mqa; qa['rar_identity_assignment']=rarqa
    meshes=[]
    for pi,p in enumerate(players):
        masks={REF:instances[REF][p['lar']]['support_mask'],BR:instances[BR][p['br']]['support_mask']}
        if p.get('rar') is not None:
            masks[RAR]=instances[RAR][p['rar']]['support_mask']
        pqa={'player_index':pi,'instances':{REF:p['lar'],BR:p['br'],RAR:p.get('rar')},'surfaces':{},'meshes':{}}
        for lab in tuple(masks.keys()):
            others=[(cams[o],masks[o]) for o in masks if o!=lab]
            surf=v12.build_ray_supported_surface(lab,cams[lab],images[lab],masks[lab],zpriors[lab],others,p['component_points'])
            if surf is None:
                continue
            surf['camera']=cams[lab]; surf['player_index']=pi; pqa['surfaces'][lab]=surf['qa']
            grid=args.grid_step if lab!=RAR else max(args.grid_step,2)
            mesh=mesh_from_surface(surf,images[lab],grid_step=grid,max_edge_cm=55.0 if lab!=RAR else 65.0,max_depth_span_cm=50.0 if lab!=RAR else 70.0)
            if mesh is not None:
                meshes.append(mesh); pqa['meshes'][lab]=mesh['qa']
        qa['players'].append(pqa)
    if not meshes:
        raise RuntimeError('no player meshes')
    qa['total_vertices']=int(sum(len(m['points']) for m in meshes)); qa['total_triangles']=int(sum(len(m['triangles']) for m in meshes)); qa['mesh_counts_by_source']={lab:{'vertices':int(sum(len(m['points']) for m in meshes if m['source_label']==lab)),'triangles':int(sum(len(m['triangles']) for m in meshes if m['source_label']==lab))} for lab in CAMERA_ORDER}
    qa['self_projection']=self_projection_qa(meshes,players,instances,cams)

    ball_center,ball_surfaces=v12.ball_surface(cams,images,sync); qa['ball_center_world_cm']=[float(x) for x in ball_center]
    rp=v8.rim_points(); ruv,_,rv=v8.project_metric(cams[REF],rp); rcols=v8.bilinear_sample(images[REF],ruv); rim_surface={'points':rp[rv],'colors':rcols[rv],'source_label':REF,'source_depth':v12.camera_z(cams[REF],rp[rv]).astype(np.float32),'support':np.full(int(np.sum(rv)),2,np.int8),'camera':cams[REF],'qa':{}}

    C0,R0,K0=cams[REF]; angles=np.asarray([0,5,10,15,20,25],np.float64) if args.static_only else np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    body_areas=[]; fragmentation=[]
    for fi,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang)); Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor'); floor,fm=v4.sample_plane_from_sources(Pf,sf,sources,'floor'); Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board'); board,bm=v4.sample_plane_from_sources(Pb,sb,sources,'board'); img=floor.copy(); img[bm]=board[bm]
        body,bmask,bdep,bowner,sqa=render_mesh_union(meshes,K0,Rt,Ct)
        pr,pm,pdep,_,_=v12.render_surfel_union(ball_surfaces+[rim_surface],K0,Rt,Ct)
        fg=body.copy(); fgmask=bmask.copy(); fgdep=bdep.copy(); takep=(pm>0)&((~fgmask)|(pdep<fgdep)); fg[takep]=pr[takep]; fgdep[takep]=pdep[takep]; fgmask[takep]=True
        img[fgmask]=fg[fgmask]
        raw=img.copy()
        if abs(float(ang))<1e-9:
            cv2.imwrite(str(args.out/'raw_mesh_00deg.png'),raw)
            exact=np.zeros((H,W),bool)
            for p in players: exact|=instances[REF][p['lar']]['mask'].astype(bool)
            img[exact]=images[REF][exact]
        bstats=mask_stats(bmask); fragmentation.append(bstats['substantial_pixel_fraction']); body_areas.append(bstats['pixels'])
        fqa={'frame':fi,'virtual_viewpoint_deg':float(ang),'body':bstats,'body_owner_pixels':{CAMERA_ORDER[k]:int(np.sum(bmask&(bowner==k))) for k in range(3)},'source_raster':sqa,'foreground_pixels':int(fgmask.sum())}; qa['frames'].append(fqa); cv2.imwrite(str(args.out/f'frame_{fi:03d}.png'),annotate(img,float(ang),qa,fqa))

    ratios=[]
    if args.static_only and body_areas:
        a0=max(body_areas[0],1); ratios=[x/a0 for x in body_areas]
    qa['static_body_area_ratios_vs_0deg']=ratios; qa['minimum_substantial_pixel_fraction']=float(min(fragmentation)) if fragmentation else None
    rar_present=qa['mesh_counts_by_source'][RAR]['triangles']>0
    qa['gates']={
        'sync_ball_pass':bool(sync['gates']['all_ball_geometry_pass']),
        'rar_identity_unique_pass':bool(rarqa['final_unique']),
        'mesh_triangle_count_pass':qa['total_triangles']>=3500,
        'lar_mesh_self_recall_pass':qa['self_projection'].get(REF,{}).get('recall',0)>=0.82,
        'broadcast_mesh_self_recall_pass':qa['self_projection'].get(BR,{}).get('recall',0)>=0.76,
        'rar_mesh_self_recall_pass':(not rar_present) or qa['self_projection'].get(RAR,{}).get('recall',0)>=0.60,
        'fragmentation_pass':(not fragmentation) or min(fragmentation)>=0.92,
        'body_area_no_collapse_pass':(not ratios) or min(ratios[1:])>=0.58,
        'body_area_no_explosion_pass':(not ratios) or max(ratios[1:])<=1.50,
    }
    qa['gates']['numeric_pass']=all(qa['gates'].values()); qa['status']='STATIC_MESH_DIAGNOSTIC_RENDERED' if args.static_only else 'FULL_MESH_DIAGNOSTIC_RENDERED'; qa['certification_scope']='3-camera R&D diagnostic only; visual QA authoritative; production gate remains >=4 accepted cameras'; qa['method']='v11 exact visual sync + identity-unique three-camera ray support + depth-discontinuity-aware per-player textured triangles + perspective-correct metric z-buffer'
    (args.out/'three_camera_mesh_v13_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'status':qa['status'],'vertices':qa['total_vertices'],'triangles':qa['total_triangles'],'mesh_counts_by_source':qa['mesh_counts_by_source'],'self_projection':qa['self_projection'],'minimum_substantial_pixel_fraction':qa['minimum_substantial_pixel_fraction'],'body_area_ratios':ratios,'rar_identity':qa['rar_identity_assignment'],'gates':qa['gates']},indent=2),flush=True)

if __name__=='__main__':
    main()
