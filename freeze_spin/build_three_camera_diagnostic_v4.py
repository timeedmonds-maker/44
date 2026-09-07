from __future__ import annotations

"""v4: plane-locked three-camera diagnostic.

Visual QA on v3 showed that full-frame monocular-depth clouds bend the hardwood
and smear the crowd under virtual camera travel.  v4 therefore uses solved
metric geometry where geometry is actually known:

* regulation court floor: exact z=0 ray/plane intersections, textured only from
  real source pixels whose learned depth agrees with that metric floor;
* regulation backboard: exact x=0 board plane, source-pixel texture only;
* players/ball: source-only MoGe depth clouds, restricted by Mask R-CNN/COCO;
* all other unsupported pixels: black.

No generated fill, optical-flow morph, crossfade, or invented camera is used.
"""

import argparse, json
from pathlib import Path
import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin.build_portable_moge_pnp_freeview_v12 import detect_dynamic_and_ball, moge_infer

W,H=base.W,base.H
FT=base.FT
RIM=base.RIM
BOARD_X=0.0
BOARD_Y=91.44          # 72 in / 2
BOARD_Z0=274.32        # 9 ft bottom
BOARD_Z1=381.00        # 12.5 ft top
COURT_X0=-180.0
COURT_X1=3000.0
COURT_Y0=-850.0
COURT_Y1=850.0


def masked_cloud(image, depth, valid, dynamic, K, R, C, align, stride=1):
    s=v3.forward_sign(R,C)
    yy,xx=np.indices((H,W)); pick=((xx%stride)==0)&((yy%stride)==0)
    z=align[0]*depth.astype(np.float64)+align[1]
    ok=valid & dynamic & pick & np.isfinite(z) & (z>20.0) & (z<12000.0)
    ys,xs=np.where(ok)
    zz=z[ys,xs]
    xn=(xs.astype(np.float64)-K[0,2])/K[0,0]
    yn=(ys.astype(np.float64)-K[1,2])/K[1,1]
    Xc=s*np.column_stack([xn*zz,yn*zz,zz])
    Xw=(R.T@Xc.T).T+C
    return Xw.astype(np.float32),image[ys,xs].copy()


def raster(cloud,K,R,C,radius=1):
    X,col=cloud
    Xc=(R@(X.astype(np.float64)-C).T).T
    z=Xc[:,2]
    q=(K@Xc.T).T
    uv=q[:,:2]/q[:,2:3]
    u=np.rint(uv[:,0]).astype(int); v=np.rint(uv[:,1]).astype(int)
    ok=np.isfinite(uv).all(1)&(z>20)&(u>=0)&(u<W)&(v>=0)&(v<H)
    ids=np.where(ok)[0]
    img=np.zeros((H,W,3),np.uint8); mask=np.zeros((H,W),np.uint8); zb=np.full(H*W,np.inf,np.float32)
    if len(ids):
        pix=v[ids]*W+u[ids]
        np.minimum.at(zb,pix,z[ids].astype(np.float32))
        win=ids[z[ids]<=zb[pix]+1e-3]
        img[v[win],u[win]]=col[win]; mask[v[win],u[win]]=255
    for _ in range(int(radius)):
        bi=img.copy(); bm=mask.copy(); holes=mask==0
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            si=np.roll(np.roll(bi,dy,0),dx,1); sm=np.roll(np.roll(bm,dy,0),dx,1)
            take=holes&(mask==0)&(sm>0)
            img[take]=si[take]; mask[take]=255
    return img,mask


def ray_plane_map(K,R,C,plane='floor'):
    s=v3.forward_sign(R,C)
    yy,xx=np.indices((H,W))
    xn=(xx.astype(np.float64)-K[0,2])/K[0,0]
    yn=(yy.astype(np.float64)-K[1,2])/K[1,1]
    dc=np.stack([xn,yn,np.ones_like(xn)],axis=-1).reshape(-1,3)
    dw=(s*dc)@R
    if plane=='floor':
        den=dw[:,2]
        with np.errstate(divide='ignore',invalid='ignore'): t=-float(C[2])/den
    elif plane=='board':
        den=dw[:,0]
        with np.errstate(divide='ignore',invalid='ignore'): t=(BOARD_X-float(C[0]))/den
    else: raise ValueError(plane)
    P=C.reshape(1,3)+t[:,None]*dw
    return t.reshape(H,W),P.reshape(H,W,3)


def source_visibility(image,depth,valid,dynamic,K,R,C,align):
    s=v3.forward_sign(R,C)
    z_est=align[0]*depth.astype(np.float64)+align[1]
    tf,Pf=ray_plane_map(K,R,C,'floor')
    floor_geom=np.isfinite(tf)&(tf>20)&(tf<12000)&(Pf[:,:,0]>=COURT_X0)&(Pf[:,:,0]<=COURT_X1)&(Pf[:,:,1]>=COURT_Y0)&(Pf[:,:,1]<=COURT_Y1)
    # Camera-space positive depth in the normalized forward convention equals t
    tol=np.maximum(28.0,0.045*tf)
    floor_vis=floor_geom & valid & np.isfinite(z_est) & (np.abs(z_est-tf)<=tol) & (~dynamic)
    floor_vis=cv2.morphologyEx(floor_vis.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8),iterations=1)>0

    tb,Pb=ray_plane_map(K,R,C,'board')
    board_geom=np.isfinite(tb)&(tb>20)&(tb<12000)&(np.abs(Pb[:,:,1])<=BOARD_Y)&(Pb[:,:,2]>=BOARD_Z0)&(Pb[:,:,2]<=BOARD_Z1)
    btol=np.maximum(35.0,0.06*tb)
    board_vis=board_geom & valid & np.isfinite(z_est) & (np.abs(z_est-tb)<=btol) & (~dynamic)
    board_vis=cv2.morphologyEx(board_vis.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)>0
    return floor_vis,board_vis


def virtual_plane_points(K,R,C,plane='floor'):
    # Virtual camera comes from the LAR lineage and therefore uses +Z forward.
    yy,xx=np.indices((H,W))
    xn=(xx.astype(np.float64)-K[0,2])/K[0,0]
    yn=(yy.astype(np.float64)-K[1,2])/K[1,1]
    dc=np.stack([xn,yn,np.ones_like(xn)],axis=-1).reshape(-1,3)
    dw=dc@R
    if plane=='floor':
        den=dw[:,2]
        with np.errstate(divide='ignore',invalid='ignore'): t=-float(C[2])/den
    else:
        den=dw[:,0]
        with np.errstate(divide='ignore',invalid='ignore'): t=(BOARD_X-float(C[0]))/den
    P=C.reshape(1,3)+t[:,None]*dw
    if plane=='floor':
        support=np.isfinite(t)&(t>20)&(t<12000)&(P[:,0]>=COURT_X0)&(P[:,0]<=COURT_X1)&(P[:,1]>=COURT_Y0)&(P[:,1]<=COURT_Y1)
    else:
        support=np.isfinite(t)&(t>20)&(t<12000)&(np.abs(P[:,1])<=BOARD_Y)&(P[:,2]>=BOARD_Z0)&(P[:,2]<=BOARD_Z1)
    return P,support


def sample_plane_from_sources(P,support,sources,plane):
    out=np.zeros((H*W,3),np.uint8); owned=np.zeros(H*W,bool)
    ids=np.where(support)[0]
    if not len(ids): return out.reshape(H,W,3),owned.reshape(H,W)
    Pw=P[ids]
    for label in ('Left Above Rim','Broadcast','Right Above Rim'):
        src=sources[label]; C,R,K=src['C'],src['R'],src['K']; s=v3.forward_sign(R,C)
        Xc=(R@(Pw-C).T).T
        q=(K@Xc.T).T; uv=q[:,:2]/q[:,2:3]
        u=np.rint(uv[:,0]).astype(int); v=np.rint(uv[:,1]).astype(int)
        ok=np.isfinite(uv).all(1)&(s*Xc[:,2]>20)&(u>=0)&(u<W)&(v>=0)&(v<H)
        loc=np.where(ok)[0]
        if not len(loc): continue
        vis=src['floor_vis'] if plane=='floor' else src['board_vis']
        loc=loc[vis[v[loc],u[loc]]]
        if not len(loc): continue
        tgt=ids[loc]; take=~owned[tgt]; tgt=tgt[take]; loc=loc[take]
        out[tgt]=src['image'][v[loc],u[loc]]; owned[tgt]=True
    return out.reshape(H,W,3),owned.reshape(H,W)


def hard_fill(base,bmask,cand,cmask):
    take=(~bmask)&cmask
    base[take]=cand[take]; bmask[take]=True
    return int(take.sum())


def annotate(img,angle,cov,dyn_cov):
    out=img.copy(); cv2.rectangle(out,(0,0),(520,58),(0,0,0),-1)
    cv2.putText(out,'3-CAMERA DIAGNOSTIC v4 | PLANE-LOCKED',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f'orbit {angle:04.1f} deg  grounded {cov*100:05.1f}%  subject {dyn_cov*100:04.1f}%',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.50,(255,255,255),1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames-dir',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--broadcast-event-frame',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--tokens',type=int,default=1400); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    lar=base.find_one(a.frames_dir,'Left_Above_Rim'); rar=base.find_one(a.frames_dir,'Right_Above_Rim'); br=[p for p in sorted(a.frames_dir.rglob('*Broadcast*.png')) if 'Mobile' not in p.name and 'Other' not in p.name]
    if len(br)!=1: raise RuntimeError(f'Broadcast ambiguity {br}')
    paths={'Left Above Rim':lar,'Right Above Rim':rar,'Broadcast':br[0]}
    ims={k:cv2.imread(str(p)) for k,p in paths.items()}
    cams=base.load_cameras(a.registry,a.rar_report,a.broadcast_event_frame)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    depth_model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    seg_model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    sources={}; qa={'sources':{},'frames':[]}
    for label in ('Left Above Rim','Right Above Rim','Broadcast'):
        C,R,K=cams[label]; im=ims[label]
        depth,_,valid,_,_=moge_infer(depth_model,im,a.tokens)
        align,dqa=v3.robust_depth_align(depth,valid,K,R,C)
        dyn,balls=detect_dynamic_and_ball(seg_model,im)
        # Ball detections are added explicitly because COCO ball masks are boxes only.
        for b in balls[:3]:
            x1,y1,x2,y2=[int(round(v)) for v in b['box']]
            x1=max(0,x1-6); y1=max(0,y1-6); x2=min(W-1,x2+6); y2=min(H-1,y2+6); dyn[y1:y2+1,x1:x2+1]=True
        floor_vis,board_vis=source_visibility(im,depth,valid,dyn,K,R,C,align)
        cloud=masked_cloud(im,depth,valid,dyn,K,R,C,align,1)
        sources[label]={'image':im,'C':C,'R':R,'K':K,'dynamic':dyn,'floor_vis':floor_vis,'board_vis':board_vis,'cloud':cloud}
        qa['sources'][label]={'depth_alignment':dqa,'dynamic_pixels':int(dyn.sum()),'ball_detections':balls[:3],'floor_visible_pixels':int(floor_vis.sum()),'board_visible_pixels':int(board_vis.sum()),'subject_points':int(len(cloud[0]))}
        cv2.imwrite(str(a.out/f"{label.replace(' ','_')}_dynamic.png"),dyn.astype(np.uint8)*255)
        cv2.imwrite(str(a.out/f"{label.replace(' ','_')}_floor_visibility.png"),floor_vis.astype(np.uint8)*255)
    C0,R0,K0=cams['Left Above Rim']; angles=np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    for i,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang))
        Pf,sf=virtual_plane_points(K0,Rt,Ct,'floor'); floor_img,floor_mask=sample_plane_from_sources(Pf,sf,sources,'floor')
        Pb,sb=virtual_plane_points(K0,Rt,Ct,'board'); board_img,board_mask=sample_plane_from_sources(Pb,sb,sources,'board')
        img=floor_img.copy(); grounded=floor_mask.copy(); take=board_mask; img[take]=board_img[take]; grounded[take]=True
        # Subjects stay source-only. LAR owns first; the other solved cameras fill only holes.
        dyn_img,dyn_mask=raster(sources['Left Above Rim']['cloud'],K0,Rt,Ct,1); dm=dyn_mask>0
        b,bm=raster(sources['Broadcast']['cloud'],K0,Rt,Ct,1); fill_b=hard_fill(dyn_img,dm,b,bm>0)
        r,rm=raster(sources['Right Above Rim']['cloud'],K0,Rt,Ct,1); fill_r=hard_fill(dyn_img,dm,r,rm>0)
        img[dm]=dyn_img[dm]
        total=grounded|dm; cov=float(total.mean()); dcov=float(dm.mean())
        qa['frames'].append({'frame':i,'angle_deg':float(ang),'source_grounded_fraction':cov,'subject_fraction':dcov,'broadcast_subject_fill_px':fill_b,'right_above_rim_subject_fill_px':fill_r,'floor_pixels':int(floor_mask.sum()),'board_pixels':int(board_mask.sum())})
        cv2.imwrite(str(a.out/f'frame_{i:03d}.png'),annotate(img,float(ang),cov,dcov))
    qa['minimum_source_grounded_fraction']=min(x['source_grounded_fraction'] for x in qa['frames']); qa['final_source_grounded_fraction']=qa['frames'][-1]['source_grounded_fraction']; qa['method']='exact solved floor/backboard planes + source-only subject clouds; unsupported background black; no generated fill/crossfade/morph'
    (a.out/'three_camera_diagnostic_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'minimum':qa['minimum_source_grounded_fraction'],'final':qa['final_source_grounded_fraction'],'sources':qa['sources']},indent=2),flush=True)

if __name__=='__main__': main()
