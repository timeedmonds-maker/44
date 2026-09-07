from __future__ import annotations

"""v5: plane-locked court plus on-court-only subject clouds.

v4 proved the floor-plane strategy but Mask R-CNN included courtside spectators
as dynamic subjects.  v5 filters person instances by the metric floor location
of their lowest visible point.  Only instances whose inferred support lands on
the regulation court / immediate apron are retained.  Grounded instances may
receive a small depth offset so their lowest visible contact agrees with the
metric floor; large corrections are rejected (airborne/occluded people retain
raw learned depth shape).
"""

import argparse,json
from pathlib import Path
import cv2
import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights,maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4
from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer

W,H=base.W,base.H
RIM=base.RIM
PERSON_CLASS=1
BALL_CLASS=37


def pixel_floor_point(x,y,K,R,C):
    s=v3.forward_sign(R,C)
    dc=np.asarray([(float(x)-K[0,2])/K[0,0],(float(y)-K[1,2])/K[1,1],1.0],np.float64)
    dw=(s*dc)@R
    if not np.isfinite(dw).all() or abs(float(dw[2]))<1e-9: return None,None
    t=-float(C[2])/float(dw[2])
    if not np.isfinite(t) or t<=20 or t>=12000: return None,None
    return C+t*dw,float(t)


def detect_oncourt(model,image,K,R,C):
    rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    with torch.inference_mode(): p=model([to_tensor(rgb)])[0]
    scores=p['scores'].cpu().numpy(); labels=p['labels'].cpu().numpy(); boxes=p['boxes'].cpu().numpy(); masks=p['masks'].cpu().numpy()[:,0]
    instances=[]; balls=[]; dyn=np.zeros((H,W),bool)
    for sc,lab,box,m in zip(scores,labels,boxes,masks):
        sc=float(sc); lab=int(lab)
        if lab==PERSON_CLASS and sc>=0.42:
            mm=m>=0.42; ys,xs=np.where(mm)
            if len(xs)<80: continue
            ycut=max(int(np.percentile(ys,96)),int(ys.max())-4)
            foot_x=int(round(float(np.median(xs[ys>=ycut])))); foot_y=int(ys.max())
            P,t=pixel_floor_point(foot_x,foot_y,K,R,C)
            if P is None: continue
            # Event is at this basket; retain regulation court plus a narrow apron.
            oncourt=(-170.0<=P[0]<=1500.0 and abs(P[1])<=805.0)
            if not oncourt: continue
            instances.append({'score':sc,'mask':mm,'foot_px':[foot_x,foot_y],'foot_world_cm':P.tolist(),'floor_camera_depth_cm':t,'box':[float(v) for v in box]})
            dyn|=mm
        elif lab==BALL_CLASS and sc>=0.08:
            x1,y1,x2,y2=[float(v) for v in box]; balls.append({'score':sc,'box':[x1,y1,x2,y2],'cx':(x1+x2)/2,'cy':(y1+y2)/2})
    # modest edge support only, not the large v4 spectator dilation
    dyn=cv2.dilate(dyn.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)>0
    balls.sort(key=lambda z:z['score'],reverse=True)
    for b in balls[:3]:
        x1,y1,x2,y2=[int(round(v)) for v in b['box']]
        x1=max(0,x1-5);y1=max(0,y1-5);x2=min(W-1,x2+5);y2=min(H-1,y2+5)
        dyn[y1:y2+1,x1:x2+1]=True
    return dyn,instances,balls


def corrected_dynamic_cloud(image,depth,valid,dynamic,instances,K,R,C,align):
    s=v3.forward_sign(R,C)
    z=align[0]*depth.astype(np.float64)+align[1]
    corrections=[]
    for inst in instances:
        mm=inst['mask']; fx,fy=inst['foot_px']; t=inst['floor_camera_depth_cm']
        y0=max(0,fy-8); y1=min(H,fy+1); x0=max(0,fx-10); x1=min(W,fx+11)
        local=mm[y0:y1,x0:x1]&valid[y0:y1,x0:x1]&np.isfinite(z[y0:y1,x0:x1])
        vals=z[y0:y1,x0:x1][local]
        delta=None
        if len(vals)>=3:
            raw=float(np.median(vals)); d=float(t-raw)
            # Small corrections are consistent with a grounded visible contact;
            # larger values may be airborne/occluded and are not forced to floor.
            if abs(d)<=70.0:
                z[mm]=z[mm]+d; delta=d
        corrections.append({'score':inst['score'],'foot_px':inst['foot_px'],'foot_world_cm':inst['foot_world_cm'],'depth_offset_cm':delta})
    ok=valid&dynamic&np.isfinite(z)&(z>20)&(z<12000)
    ys,xs=np.where(ok); zz=z[ys,xs]
    xn=(xs.astype(np.float64)-K[0,2])/K[0,0]; yn=(ys.astype(np.float64)-K[1,2])/K[1,1]
    Xc=s*np.column_stack([xn*zz,yn*zz,zz]); Xw=(R.T@Xc.T).T+C
    return (Xw.astype(np.float32),image[ys,xs].copy()),corrections


def source_visibility_v5(image,depth,valid,dynamic,K,R,C,align):
    z_est=align[0]*depth.astype(np.float64)+align[1]
    tf,Pf=v4.ray_plane_map(K,R,C,'floor')
    fg=np.isfinite(tf)&(tf>20)&(tf<12000)&(Pf[:,:,0]>=v4.COURT_X0)&(Pf[:,:,0]<=v4.COURT_X1)&(Pf[:,:,1]>=v4.COURT_Y0)&(Pf[:,:,1]<=v4.COURT_Y1)
    tol=np.maximum(42.0,0.06*tf)
    floor_vis=fg&valid&np.isfinite(z_est)&(np.abs(z_est-tf)<=tol)&(~dynamic)
    # Board is a known metric plane; depth estimates are unreliable on transparent glass.
    tb,Pb=v4.ray_plane_map(K,R,C,'board')
    bg=np.isfinite(tb)&(tb>20)&(tb<12000)&(np.abs(Pb[:,:,1])<=v4.BOARD_Y)&(Pb[:,:,2]>=v4.BOARD_Z0)&(Pb[:,:,2]<=v4.BOARD_Z1)
    board_vis=bg&valid&(~dynamic)
    return floor_vis,board_vis


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames-dir',type=Path,required=True);ap.add_argument('--registry',type=Path,required=True);ap.add_argument('--rar-report',type=Path,required=True);ap.add_argument('--broadcast-event-frame',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',type=int,default=1400);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lar=base.find_one(a.frames_dir,'Left_Above_Rim');rar=base.find_one(a.frames_dir,'Right_Above_Rim');br=[p for p in sorted(a.frames_dir.rglob('*Broadcast*.png')) if 'Mobile' not in p.name and 'Other' not in p.name]
    if len(br)!=1:raise RuntimeError(f'Broadcast ambiguity {br}')
    paths={'Left Above Rim':lar,'Right Above Rim':rar,'Broadcast':br[0]};ims={k:cv2.imread(str(p)) for k,p in paths.items()};cams=base.load_cameras(a.registry,a.rar_report,a.broadcast_event_frame)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())));dm=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval();sm=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval()
    sources={};qa={'sources':{},'frames':[]}
    for label in ('Left Above Rim','Right Above Rim','Broadcast'):
        C,R,K=cams[label];im=ims[label];depth,_,valid,_,_=moge_infer(dm,im,a.tokens);align,dqa=v3.robust_depth_align(depth,valid,K,R,C)
        dyn,inst,balls=detect_oncourt(sm,im,K,R,C);cloud,corr=corrected_dynamic_cloud(im,depth,valid,dyn,inst,K,R,C,align);fv,bv=source_visibility_v5(im,depth,valid,dyn,K,R,C,align)
        sources[label]={'image':im,'C':C,'R':R,'K':K,'dynamic':dyn,'floor_vis':fv,'board_vis':bv,'cloud':cloud}
        qa['sources'][label]={'depth_alignment':dqa,'on_court_instances':len(inst),'instance_support':[{'score':x['score'],'foot_px':x['foot_px'],'foot_world_cm':x['foot_world_cm'],'box':x['box']} for x in inst],'ground_contact_corrections':corr,'dynamic_pixels':int(dyn.sum()),'ball_detections':balls[:3],'floor_visible_pixels':int(fv.sum()),'board_visible_pixels':int(bv.sum()),'subject_points':int(len(cloud[0]))}
        cv2.imwrite(str(a.out/f"{label.replace(' ','_')}_dynamic.png"),dyn.astype(np.uint8)*255);cv2.imwrite(str(a.out/f"{label.replace(' ','_')}_floor_visibility.png"),fv.astype(np.uint8)*255)
    C0,R0,K0=cams['Left Above Rim'];angles=np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    for i,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang));Pf,sf=v4.virtual_plane_points(K0,Rt,Ct,'floor');floor_img,floor_mask=v4.sample_plane_from_sources(Pf,sf,sources,'floor');Pb,sb=v4.virtual_plane_points(K0,Rt,Ct,'board');board_img,board_mask=v4.sample_plane_from_sources(Pb,sb,sources,'board')
        img=floor_img.copy();grounded=floor_mask.copy();img[board_mask]=board_img[board_mask];grounded[board_mask]=True
        di,dmk=v4.raster(sources['Left Above Rim']['cloud'],K0,Rt,Ct,1);mk=dmk>0;b,bm=v4.raster(sources['Broadcast']['cloud'],K0,Rt,Ct,1);fb=v4.hard_fill(di,mk,b,bm>0);r,rm=v4.raster(sources['Right Above Rim']['cloud'],K0,Rt,Ct,1);fr=v4.hard_fill(di,mk,r,rm>0);img[mk]=di[mk]
        total=grounded|mk;cov=float(total.mean());dcov=float(mk.mean());qa['frames'].append({'frame':i,'angle_deg':float(ang),'source_grounded_fraction':cov,'subject_fraction':dcov,'broadcast_subject_fill_px':fb,'right_above_rim_subject_fill_px':fr,'floor_pixels':int(floor_mask.sum()),'board_pixels':int(board_mask.sum())});cv2.imwrite(str(a.out/f'frame_{i:03d}.png'),v4.annotate(img,float(ang),cov,dcov))
    qa['minimum_source_grounded_fraction']=min(x['source_grounded_fraction'] for x in qa['frames']);qa['final_source_grounded_fraction']=qa['frames'][-1]['source_grounded_fraction'];qa['method']='exact floor/backboard planes + metric-filtered on-court person/ball source clouds; unsupported background black; no generated fill/crossfade/morph';(a.out/'three_camera_diagnostic_qa.json').write_text(json.dumps(qa,indent=2));print(json.dumps({'minimum':qa['minimum_source_grounded_fraction'],'final':qa['final_source_grounded_fraction'],'sources':{k:{'instances':v['on_court_instances'],'dynamic_pixels':v['dynamic_pixels'],'floor_visible_pixels':v['floor_visible_pixels'],'board_visible_pixels':v['board_visible_pixels']} for k,v in qa['sources'].items()}},indent=2),flush=True)

if __name__=='__main__':main()
