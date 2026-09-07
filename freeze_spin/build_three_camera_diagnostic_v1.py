from __future__ import annotations

import argparse, json
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from moge.model.v2 import MoGeModel

from freeze_spin.build_portable_moge_pnp_freeview_v12 import moge_infer
from freeze_spin.build_metric_anchor_depth_orbit_v67 import K_matrix, recover_accepted_rotation, orbit_pose
from freeze_spin import prove_broadcast_shared_center_v90 as v90
from freeze_spin import solve_frame_c_broadcast_floor_v44 as v44
from freeze_spin import solve_broadcast_direct_target_lines_v87 as v87
from freeze_spin import diagnose_broadcast_homography_conditioning_v85 as v85

W,H=960,540
FT=30.48
IN=2.54
RIM=np.asarray([15.0*IN,0.0,10.0*FT],np.float64)


def read_json(p):
    return json.loads(Path(p).read_text())

def find_one(root:Path, token:str)->Path:
    rows=sorted(root.rglob(f'*{token}*.png'))
    if len(rows)!=1:
        raise RuntimeError(f'expected exactly one *{token}*.png, got {len(rows)}: {rows[:8]}')
    return rows[0]

def project_camera(C,K,R,P):
    Xc=(R@(P-C).T).T
    q=(K@Xc.T).T
    uv=q[:,:2]/q[:,2:3]
    return uv,Xc

def floor_world_grid():
    xs=np.linspace(-110.0,1500.0,44)
    ys=np.linspace(-760.0,760.0,51)
    xx,yy=np.meshgrid(xs,ys)
    return np.column_stack([xx.ravel(),yy.ravel(),np.zeros(xx.size)])

def robust_depth_align(depth,valid,K,R,C):
    P=floor_world_grid(); uv,Xc=project_camera(C,K,R,P)
    x=np.rint(uv[:,0]).astype(int); y=np.rint(uv[:,1]).astype(int)
    good=np.isfinite(uv).all(1)&(Xc[:,2]>30)&(x>=3)&(x<W-3)&(y>=3)&(y<H-3)
    ids=np.where(good)[0]
    ids=ids[valid[y[ids],x[ids]] & np.isfinite(depth[y[ids],x[ids]]) & (depth[y[ids],x[ids]]>1e-4)]
    if len(ids)<100: raise RuntimeError(f'only {len(ids)} floor depth anchors')
    d=depth[y[ids],x[ids]].astype(float); z=Xc[ids,2].astype(float)
    scale=float(np.median(z/np.maximum(d,1e-6)))
    p=np.array([scale,0.0])
    for _ in range(3):
        fit=least_squares(lambda q:q[0]*d+q[1]-z,p,loss='soft_l1',f_scale=35.0,max_nfev=4000)
        p=fit.x
        e=np.abs(p[0]*d+p[1]-z)
        cap=np.percentile(e,72)
        keep=e<=max(cap,25.0)
        d,z=d[keep],z[keep]
    pred=p[0]*d+p[1]; e=np.abs(pred-z)
    return p, {'anchors':int(len(d)),'scale':float(p[0]),'offset_cm':float(p[1]),'median_cm':float(np.median(e)),'p95_cm':float(np.percentile(e,95))}

def metric_cloud(image,depth,valid,K,R,C,align,stride=2):
    yy,xx=np.indices((H,W)); pick=((xx%stride)==0)&((yy%stride)==0)
    z=align[0]*depth.astype(np.float64)+align[1]
    ok=valid&pick&np.isfinite(z)&(z>20)&(z<12000)
    ys,xs=np.where(ok); zz=z[ys,xs]
    xn=(xs-K[0,2])/K[0,0]; yn=(ys-K[1,2])/K[1,1]
    Xc=np.column_stack([xn*zz,yn*zz,zz])
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
        winners=ids[z[ids]<=zb[pix]+1e-3]
        img[v[winners],u[winners]]=col[winners]; mask[v[winners],u[winners]]=255
    for _ in range(radius):
        basei=img.copy(); basem=mask.copy(); holes=mask==0
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            si=np.roll(np.roll(basei,dy,0),dx,1); sm=np.roll(np.roll(basem,dy,0),dx,1)
            take=holes&(mask==0)&(sm>0)
            img[take]=si[take]; mask[take]=255
    return img,mask
def hard_fill(base,bmask,cand,cmask):
    take=(bmask==0)&(cmask>0)
    base[take]=cand[take]; bmask[take]=255
    return int(take.sum())
def solve_broadcast_pose(event_frame):
    v85.patch_line_aware_geometry()
    floor_spec=read_json('freeze_spin/adams_jazz_frame_c_broadcast_floor_v86.json')
    target_c_spec=read_json('freeze_spin/adams_jazz_broadcast_target_lines_v87.json')
    event_spec=read_json('freeze_spin/adams_jazz_broadcast_event155_b21_state_v90.json')
    H_event=np.asarray(event_spec['court_region_homography_from_frame_c']['frame_c_to_event155_H'],float)
    train_c,_=v44.split_groups(floor_spec['observations_px'],floor_spec['held_out_indices'])
    event_all={k:v90.perspective_points(H_event,np.asarray(v,float)) for k,v in floor_spec['observations_px'].items()}
    train_e,_=v44.split_groups(event_all,floor_spec['held_out_indices'])
    target_c={k:np.asarray(target_c_spec['observed_line_samples_px'][k],float) for k in v90.TARGET_KEYS}
    target_e={k:np.asarray(event_spec['observed_target_line_samples_px'][k],float) for k in v90.TARGET_KEYS}
    pc=v90.find_frame_c_seed(train_c,target_c,target_c_spec)
    pe=v87.solve_warm(pc,train_e,target_e,max_nfev=10000)
    C0=(v87.camera_center(pc)+v87.camera_center(pe))/2.0
    z0=np.r_[C0,pc[:3],pc[6],pe[:3],pe[6],(pc[7:9]+pe[7:9])/2.0]
    z,_=v90.solve_joint(z0,train_c,target_c,train_e,target_e,max_nfev=20000)
    C,p1,_=v90.unpack_joint(z)
    R=cv2.Rodrigues(np.asarray(p1[:3],float))[0]
    K=K_matrix(float(np.exp(p1[6])),p1[7:9])
    return C.astype(float),R.astype(float),K.astype(float)
def load_cameras(registry,rar_report,broadcast_event):
    reg=read_json(registry)
    lar=reg['accepted_cameras']['Left Above Rim']; ev=lar['event_489']
    C_l=np.asarray(lar['physical_camera_center_prior_cm'],float)
    K_l=K_matrix(float(ev['focal_px']),ev['principal_point_px'])
    H_l=np.asarray(read_json('freeze_spin/adams_jazz_frame_c_floor_homography_v35.json')['floor_homography_world_to_image'],float)
    rot,_=recover_accepted_rotation(C_l,K_l,H_l); R_l=np.asarray(rot['R'],float)
    rr=read_json(rar_report); C_r=np.asarray(rr['physical_center_cm'],float); st=rr['target_frame_c']
    K_r=K_matrix(float(st['focal_px']),st['principal_point_px']); R_r=cv2.Rodrigues(np.asarray(st['rvec'],float))[0]
    C_b,R_b,K_b=solve_broadcast_pose(broadcast_event)
    return {'Left Above Rim':(C_l,R_l,K_l),'Right Above Rim':(C_r,R_r,K_r),'Broadcast':(C_b,R_b,K_b)}
def annotate(img,angle,cov):
    out=img.copy()
    cv2.rectangle(out,(0,0),(430,54),(0,0,0),-1)
    cv2.putText(out,'3-CAMERA DIAGNOSTIC | 3D REPROJECTION',(12,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f'virtual orbit {angle:04.1f} deg   resolved {cov*100:05.1f}%',(12,43),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames-dir',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--broadcast-event-frame',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--tokens',type=int,default=1400); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    lar_path=find_one(a.frames_dir,'Left_Above_Rim'); rar_path=find_one(a.frames_dir,'Right_Above_Rim')
    b_rows=sorted(a.frames_dir.rglob('*Broadcast*.png')); b_rows=[p for p in b_rows if 'Mobile' not in p.name and 'Other' not in p.name]
    if len(b_rows)!=1: raise RuntimeError(f'Broadcast frame ambiguity: {b_rows}')
    paths={'Left Above Rim':lar_path,'Right Above Rim':rar_path,'Broadcast':b_rows[0]}
    ims={k:cv2.imread(str(p)) for k,p in paths.items()}
    for k,im in ims.items():
        if im is None or im.shape[:2]!=(H,W): raise RuntimeError(f'bad frame {k} {paths[k]}')
    cams=load_cameras(a.registry,a.rar_report,a.broadcast_event_frame)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    model=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    clouds={}; qa={'sources':{},'camera_centers_cm':{}}
    for label in ('Left Above Rim','Right Above Rim','Broadcast'):
        C,R,K=cams[label]; depth,_,valid,_,_=moge_infer(model,ims[label],a.tokens)
        align,dqa=robust_depth_align(depth,valid,K,R,C)
        clouds[label]=metric_cloud(ims[label],depth,valid,K,R,C,align,stride=2)
        qa['sources'][label]={'file':paths[label].name,'depth_alignment':dqa,'point_count':int(len(clouds[label][0]))}
        qa['camera_centers_cm'][label]=C.tolist()
        vis=np.nan_to_num(depth,nan=0.0); vals=vis[valid]; lo,hi=np.percentile(vals,[2,98]); dv=np.clip((vis-lo)/max(hi-lo,1e-6),0,1); cv2.imwrite(str(a.out/f"{label.replace(' ','_')}_depth.png"),(dv*255).astype(np.uint8))
    C0,R0,K0=cams['Left Above Rim']
    angles=np.r_[np.zeros(12),np.linspace(0,25,76),np.full(18,25.0)]
    frame_qa=[]
    for i,ang in enumerate(angles):
        Rt,Ct=orbit_pose(C0,R0,RIM,float(ang))
        img,mask=raster(clouds['Left Above Rim'],K0,Rt,Ct,1)
        b,bm=raster(clouds['Broadcast'],K0,Rt,Ct,1); fill_b=hard_fill(img,mask,b,bm)
        r,rm=raster(clouds['Right Above Rim'],K0,Rt,Ct,1); fill_r=hard_fill(img,mask,r,rm)
        cov=float(np.mean(mask>0)); frame_qa.append({'frame':i,'angle_deg':float(ang),'resolved_fraction':cov,'broadcast_fill_px':fill_b,'right_above_rim_fill_px':fill_r})
        cv2.imwrite(str(a.out/f'frame_{i:03d}.png'),annotate(img,float(ang),cov))
    qa['frames']=frame_qa; qa['minimum_resolved_fraction']=min(x['resolved_fraction'] for x in frame_qa); qa['final_resolved_fraction']=frame_qa[-1]['resolved_fraction']; qa['method']='three accepted metric cameras; MoGe depth shape independently aligned to each solved metric camera using regulation floor; hard source ownership; no crossfade, no generated fill; unsupported pixels remain black'
    (a.out/'three_camera_diagnostic_qa.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps({'sources':qa['sources'],'minimum_resolved_fraction':qa['minimum_resolved_fraction'],'final_resolved_fraction':qa['final_resolved_fraction']},indent=2),flush=True)
if __name__=='__main__': main()
