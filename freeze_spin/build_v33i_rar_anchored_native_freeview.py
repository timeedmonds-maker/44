from __future__ import annotations

"""v33i: small native free-view arc from the accepted v33h exact state.

Rendering policy:
- exactly LAR + RAR + Broadcast, no fourth camera;
- exact v33h/v33e per-frame cameras and source frames only;
- RAR is the zero-degree appearance anchor because the real basketball and focal
  player are directly visible there;
- MoGe supplies deterministic per-view depth shape, metrically aligned to the
  regulation floor using the already accepted cameras;
- source point clouds are reprojected into a small virtual orbit; RAR owns pixels
  first, Broadcast then LAR fill only unresolved holes;
- the basketball is additionally stabilized with the exact real RAR ball pixels
  at the v33h contact-constrained 3-D center. No generated ball pixels are used;
- native 960x540 only, no inpainting, optical-flow morph, crossfade or upscale.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel

from freeze_spin import build_three_camera_freeview_preview_v1 as p
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=960,540
LAR='Left Above Rim'; RAR='Right Above Rim'; BCAST='Broadcast'
CAMS=(LAR,RAR,BCAST)
RIM=np.array([38.1,0.0,304.8],float)


def load_state(path:Path):
    q=json.loads(path.read_text())
    if q.get('camera_lock')!=[LAR,RAR,BCAST] or q.get('resolution')!=[W,H]:
        raise RuntimeError('v33i requires locked native v33h render state')
    ball=q.get('ball',{})
    if not ball.get('gate') or ball.get('method','').find('source ball')<0:
        raise RuntimeError('v33i requires passed source/contact ball state')
    return q


def load_sources(root:Path):
    out={}
    for c in CAMS:
        path=root/f"v33h_source_{c.replace(' ','_')}.png"
        im=cv2.imread(str(path))
        if im is None or im.shape[:2]!=(H,W): raise RuntimeError(f'missing native exact source {path}')
        out[c]=im
    return out


def cameras_from_state(q):
    out={}
    for c in CAMS:
        d=q['cameras'][c]
        out[c]={'label':c,'center_cm':np.asarray(d['C_world_cm'],float),
                'K':np.asarray(d['K_px'],float),'R_world_to_camera':np.asarray(d['R_world_to_camera'],float)}
    return out


def alpha_ball_patch(rar:np.ndarray, ball:dict):
    cx,cy=map(float,ball['source_ball_center_px'])
    bbox=np.asarray(ball.get('source_ball_bbox',[cx-18,cy-18,cx+18,cy+18]),float)
    pad=7
    x1=max(0,int(math.floor(bbox[0]))-pad); y1=max(0,int(math.floor(bbox[1]))-pad)
    x2=min(W,int(math.ceil(bbox[2]))+pad); y2=min(H,int(math.ceil(bbox[3]))+pad)
    crop=rar[y1:y2,x1:x2].copy()
    if crop.size==0: raise RuntimeError('empty RAR ball crop')
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([1,55,35],np.uint8),np.array([34,255,255],np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # Keep only orange support near the known source-ball centre; this rejects
    # distant skin/jersey regions if present in the crop.
    yy,xx=np.indices(mask.shape); lx=cx-x1; ly=cy-y1
    radial=((xx-lx)**2+(yy-ly)**2)<=max(9.0,0.75*max(crop.shape[:2]))**2
    mask=((mask>0)&radial).astype(np.uint8)*255
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n>1:
        ci=int(np.clip(round(ly),0,mask.shape[0]-1)); cj=int(np.clip(round(lx),0,mask.shape[1]-1))
        keep=int(lab[ci,cj])
        if keep==0:
            ids=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=8]
            if ids: keep=max(ids,key=lambda i:stats[i,cv2.CC_STAT_AREA])
        if keep>0: mask=(lab==keep).astype(np.uint8)*255
    if int(np.sum(mask>0))<25: raise RuntimeError('insufficient source RGB support for basketball patch')
    return crop,mask,[x1,y1,x2,y2]


def target_ball_radius(K,R,C,X):
    # Use two orthogonal 12cm offsets and take the larger projected half-size.
    uv0,_,ok0=p.project_points(K,R,C,np.asarray(X,float)[None,:])
    if not bool(ok0[0]) if np.ndim(ok0)>0 else False: pass
    pts=np.vstack([np.asarray(X,float),np.asarray(X,float)+[12,0,0],np.asarray(X,float)+[0,12,0],np.asarray(X,float)+[0,0,12]])
    uv,z=p.project_points(K,R,C,pts)
    r=max(float(np.linalg.norm(uv[i]-uv[0])) for i in range(1,4) if np.isfinite(uv[i]).all())
    return float(np.clip(r,5.0,30.0)),uv[0]


def overlay_source_ball(frame,crop,alpha,K,R,C,X):
    rad,uv=target_ball_radius(K,R,C,X)
    size=max(10,int(round(2.35*rad)))
    rgb=cv2.resize(crop,(size,size),interpolation=cv2.INTER_LANCZOS4)
    a=cv2.resize(alpha,(size,size),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    if np.max(a)<=0: return frame,False,uv.tolist(),rad
    cx,cy=np.rint(uv).astype(int); x0=cx-size//2; y0=cy-size//2; x1=x0+size; y1=y0+size
    xa=max(0,x0); ya=max(0,y0); xb=min(W,x1); yb=min(H,y1)
    if xb<=xa or yb<=ya: return frame,False,uv.tolist(),rad
    sx0=xa-x0; sy0=ya-y0; sx1=sx0+(xb-xa); sy1=sy0+(yb-ya)
    aa=a[sy0:sy1,sx0:sx1,None]
    dst=frame[ya:yb,xa:xb].astype(np.float32); src=rgb[sy0:sy1,sx0:sx1].astype(np.float32)
    frame[ya:yb,xa:xb]=np.clip(src*aa+dst*(1-aa),0,255).astype(np.uint8)
    return frame,True,uv.tolist(),rad


def montage(paths,out):
    ims=[cv2.imread(str(x)) for x in paths]; ims=[x for x in ims if x is not None]
    if not ims:return
    thumb=[cv2.resize(x,(480,270)) for x in ims]
    cv2.imwrite(str(out),np.hstack(thumb))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--state',type=Path,required=True); ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--frames',type=int,default=49); ap.add_argument('--max-degree',type=float,default=12.0); ap.add_argument('--tokens',type=int,default=900)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    q=load_state(args.state); images=load_sources(args.sources); cams=cameras_from_state(q)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    clouds={}; depthqa={}
    for c in CAMS:
        cam=cams[c]; depth,_,valid,kmoge,_=moge_infer(model,images[c],args.tokens)
        mapping,qa=p.robust_depth_mapping(depth,valid,cam)
        clouds[c]=p.build_cloud(images[c],depth,valid,mapping,cam,stride=1 if c==RAR else 2)
        qa['moge_valid_fraction']=float(np.mean(valid)); qa['cloud_points']=int(len(clouds[c][0])); qa['moge_reported_intrinsics']=np.asarray(kmoge).tolist(); depthqa[c]=qa

    anchor=cams[RAR]; K0=np.asarray(anchor['K'],float); R0=np.asarray(anchor['R_world_to_camera'],float); C0=np.asarray(anchor['center_cm'],float)
    ball=q['ball']; ballX=np.asarray(ball['world_cm'],float); patch,alpha,patch_box=alpha_ball_patch(images[RAR],ball)
    cv2.imwrite(str(args.out/'v33i_ball_source_crop.png'),patch); cv2.imwrite(str(args.out/'v33i_ball_source_alpha.png'),alpha)

    def render(deg):
        if abs(float(deg))<1e-9:
            full=np.ones((H,W),bool); prov=np.full((H,W),2,np.uint8)
            return images[RAR].copy(),full,prov,{RAR:1.0,BCAST:0.0,LAR:0.0},False,None
        Rt,Ct=p.orbit_pose(C0,R0,RIM,float(deg)); renders={}; cov={}
        for c in CAMS:
            im,m,_=p.raster_source_cloud(clouds[c],K0,Rt,Ct,radius=1)
            renders[c]=(im,m); cov[c]=float(np.mean(m>0))
        out=renders[RAR][0].copy(); owned=renders[RAR][1]>0; prov=np.zeros((H,W),np.uint8); prov[owned]=2
        for c,code in ((BCAST,3),(LAR,1)):
            im,m=renders[c]; take=(~owned)&(m>0); out[take]=im[take]; prov[take]=code; owned|=take
        out,bo,buv,br=overlay_source_ball(out,patch,alpha,K0,Rt,Ct,ballX)
        return out,owned,prov,cov,bo,{'center_px':buv,'radius_px':br}

    key=[]; keypaths=[]
    for deg in (0,3,6,9,12):
        if deg>args.max_degree+1e-6: continue
        im,mask,prov,cov,bo,bq=render(float(deg)); path=args.out/f'v33i_{deg:02d}deg.png'; cv2.imwrite(str(path),im); keypaths.append(path)
        Rt,Ct=p.orbit_pose(C0,R0,RIM,float(deg)); pivot,_,_=p.project_points(K0,Rt,Ct,RIM[None,:]); af,roi=p.action_roi_fraction(mask,pivot[0])
        key.append({'degree':deg,'resolved_fraction_full':float(mask.mean()),'resolved_fraction_action_roi':af,'action_roi':roi,'source_coverage':cov,'ball_overlay_used':bool(bo),'ball_projection':bq})
    montage(keypaths,args.out/'v33i_static_arc_montage.png')

    motion=[]
    for i in range(args.frames):
        phase=i/max(1,args.frames-1); deg=float(args.max_degree*math.sin(math.pi*phase)); im,mask,prov,cov,bo,bq=render(deg)
        fp=args.out/f'motion_{i:03d}.png'; cv2.imwrite(str(fp),im)
        motion.append({'i':i,'degree':deg,'resolved_fraction':float(mask.mean()),'ball_overlay_used':bool(bo),'ball_projection':bq})
    report={'version':'v33i_rar_anchored_native_freeview','status':'V33I_STATIC_AND_MOTION_FRAMES_RENDERED_AWAITING_VISUAL_QA',
            'camera_lock':[LAR,RAR,BCAST],'camera_count':3,'physical_centers_refit':False,'source_resolution':[W,H],'render_resolution':[W,H],
            'anchor':RAR,'source_fill_order':[RAR,BCAST,LAR],'max_degree':float(args.max_degree),'motion_frames':int(args.frames),
            'appearance_policy':'real exact-state NBA RGB only; source point reprojection + RAR source-ball patch; no generated RGB/inpainting/crossfade/upscale',
            'ball_method':q['ball'].get('method'),'ball_world_cm':q['ball'].get('world_cm'),'ball_source_patch_box':patch_box,
            'depth_alignment_qa':depthqa,'key_stills':key,'motion':motion,'visual_pass_required':True}
    (args.out/'v33i_render_qa.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'key_stills':key,'depth_alignment_qa':depthqa},indent=2),flush=True)

if __name__=='__main__': main()
