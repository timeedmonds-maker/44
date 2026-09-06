from __future__ import annotations

"""v68 diagnostic: two-view metric-anchored static free-view.

Left Above Rim v41/v42 remains the only accepted metric camera and establishes
world scale. Play by Play is solved *diagnostically* into that metric frame from
static source pixels plus its already-passed v65 NBA floor plane. MoGe supplies
per-view depth shape only; it cannot promote either camera.

Unlike v67's forward point splats, v68 builds a dense inverse warp from each
source depth map. Depth discontinuities are explicitly withheld. The second
real synchronized view may fill only pixels unsupported by the primary source.
This tests whether real multiview evidence can remove the single-depth-sheet
tearing before any human-specific learned reconstruction is attempted.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from moge.model.v2 import MoGeModel
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from build_metric_anchor_depth_orbit_v67 import (
    W, H, RIM, K_matrix, recover_accepted_rotation, floor_grid,
    fit_depth_models, orbit_pose, project_points, json_safe,
)
from build_portable_moge_pnp_freeview_v12 import detect_dynamic_and_ball, moge_infer, sift_matches

ANGLES=(0.0,3.0,6.0,9.0,12.0)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def world_map_from_depth(depth,valid,K,R,C):
    yy,xx=np.indices((H,W)); z=np.asarray(depth,np.float64)
    ok=valid & np.isfinite(z) & (z>20.0) & (z<12000.0)
    xn=(xx-K[0,2])/K[0,0]; yn=(yy-K[1,2])/K[1,1]
    Xc=np.stack([xn*z,yn*z,z],axis=2)
    Xw=np.einsum('ij,hwj->hwi',R.T,Xc)+C.reshape(1,1,3)
    Xw[~ok]=np.nan
    return Xw.astype(np.float32),ok


def clean_floor_depth_samples(Hm,C,K,rv,depth,valid,dynamic,max_reproj=2.5):
    P,U,meta=floor_grid(Hm)
    pred,Xc,R=project_camera(C,K,rv,P)
    reproj=np.linalg.norm(pred-U,axis=1)
    x=np.rint(U[:,0]).astype(int);y=np.rint(U[:,1]).astype(int)
    good=(x>=0)&(x<W)&(y>=0)&(y<H)&valid[y,x]&np.isfinite(depth[y,x])&(depth[y,x]>0.05)&(~dynamic[y,x])&(Xc[:,2]>20)&(reproj<=max_reproj)
    d=np.asarray(depth[y[good],x[good]],float);z=Xc[good,2].astype(float);m=[meta[i] for i in np.where(good)[0]]
    hold=np.asarray([((int(r[0])+2*int(r[1]))%5)==0 for r in m],bool)
    if int((~hold).sum())<35 or int(hold.sum())<8: raise RuntimeError(f'insufficient floor depth anchors train={int((~hold).sum())} held={int(hold.sum())}')
    return d,z,hold,reproj[good]


def project_camera(C,K,rv,P):
    R,_=cv2.Rodrigues(np.asarray(rv,float).reshape(3,1));Xc=(R@(P-C).T).T;q=(K@Xc.T).T;uv=q[:,:2]/q[:,2:3]
    return uv,Xc,R


def sample_world_at(Xmap,valid,uv):
    x=np.rint(uv[:,0]).astype(int);y=np.rint(uv[:,1]).astype(int)
    good=(x>=0)&(x<W)&(y>=0)&(y<H)
    ids=np.where(good)[0];keep=[];X=[]
    for i in ids:
        if valid[y[i],x[i]] and np.isfinite(Xmap[y[i],x[i]]).all():
            keep.append(i);X.append(Xmap[y[i],x[i]])
    return np.asarray(keep,int),np.asarray(X,np.float64)


def p_from_pnp(rvec,tvec,K):
    R,_=cv2.Rodrigues(rvec);t=np.asarray(tvec,float).reshape(3);C=-R.T@t
    return np.r_[np.asarray(rvec,float).reshape(3),C,math.log(float((K[0,0]+K[1,1])*0.5)),K[0,2],K[1,2]]


def unpack(p):
    rv=p[:3];C=p[3:6];f=math.exp(float(p[6]));K=K_matrix(f,p[7:9]);return rv,C,K


def solve_secondary_pose(X,uv2,Kinit,Hfloor):
    if len(X)<35: raise RuntimeError(f'only {len(X)} static metric correspondences')
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(X,uv2,Kinit,None,flags=cv2.SOLVEPNP_EPNP,reprojectionError=5.0,confidence=.999,iterationsCount=5000)
    if not ok or inliers is None or len(inliers)<28: raise RuntimeError(f'PnP insufficient inliers {0 if inliers is None else len(inliers)}')
    ids=inliers.reshape(-1);Xi=X[ids];Ui=uv2[ids]
    try:rvec,tvec=cv2.solvePnPRefineLM(Xi,Ui,Kinit,None,rvec,tvec)
    except Exception:pass
    Pf,Uf,_=floor_grid(Hfloor)
    if len(Pf)>140:
        sel=np.linspace(0,len(Pf)-1,140).round().astype(int);Pf=Pf[sel];Uf=Uf[sel]
    p0=p_from_pnp(rvec,tvec,Kinit)
    low=np.r_[[-10,-10,-10],[-3000,-7000,100],math.log(250),[-300,-300]]
    high=np.r_[[10,10,10],[6000,7000,3000],math.log(5000),[1260,840]]
    starts=[p0.copy()]
    for sf in (.8,1.2):
        q=p0.copy();q[6]+=math.log(sf);starts.append(q)
    q=p0.copy();q[7:9]=[480,270];starts.append(q)
    roots=[]
    def res(p):
        rv,C,K=unpack(p)
        a,ca,_=project_camera(C,K,rv,Xi);b,cb,_=project_camera(C,K,rv,Pf)
        depth=np.r_[ca[:,2],cb[:,2]]
        pri=np.asarray([(math.log(K[0,0]/max(Kinit[0,0],1e-6)))/.35,(K[0,2]-Kinit[0,2])/180.,(K[1,2]-Kinit[1,2])/180.])
        return np.r_[(a-Ui).ravel(),.8*(b-Uf).ravel(),np.minimum(depth-20,0)*2.,pri]
    for s in starts:
        fit=least_squares(res,np.clip(s,low+1e-6,high-1e-6),bounds=(low,high),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=12000)
        rv,C,K=unpack(fit.x);ps,cs,R=project_camera(C,K,rv,Xi);pf,cf,_=project_camera(C,K,rv,Pf)
        es=np.linalg.norm(ps-Ui,axis=1);ef=np.linalg.norm(pf-Uf,axis=1)
        roots.append({'p':fit.x,'cost':float(fit.cost),'static_med':float(np.median(es)),'static_p95':float(np.percentile(es,95)),'floor_med':float(np.median(ef)),'floor_p95':float(np.percentile(ef,95)),'min_depth':float(min(cs[:,2].min(),cf[:,2].min()))})
    roots.sort(key=lambda r:(r['cost'],r['static_p95']+r['floor_p95']))
    best=roots[0];rv,C,K=unpack(best['p']);_,_,R=project_camera(C,K,rv,Xi)
    return rv,C,K,R,best,roots,ids


def depth_edges(depth,valid):
    z=np.asarray(depth,np.float32);safe=np.where(valid,z,0)
    dx=np.abs(np.diff(safe,axis=1,prepend=safe[:,:1]));dy=np.abs(np.diff(safe,axis=0,prepend=safe[:1,:]))
    denom=np.maximum(np.abs(safe),20.0);e=((dx/denom)>.035)|((dy/denom)>.035)|(~valid)
    return cv2.dilate(e.astype(np.uint8),np.ones((3,3),np.uint8),iterations=1)>0


def inverse_warp(image,depth,valid,dynamic,Ksrc,Rsrc,Csrc,Kt,Rt,Ct):
    yy,xx=np.indices((H,W));z=np.asarray(depth,np.float64)
    xn=(xx-Ksrc[0,2])/Ksrc[0,0];yn=(yy-Ksrc[1,2])/Ksrc[1,1]
    Xc=np.stack([xn*z,yn*z,z],axis=2);Xw=np.einsum('ij,hwj->hwi',Rsrc.T,Xc)+Csrc.reshape(1,1,3)
    Xct=np.einsum('ij,hwj->hwi',Rt,Xw-Ct.reshape(1,1,3));q=np.einsum('ij,hwj->hwi',Kt,Xct)
    ut=q[:,:,0]/q[:,:,2];vt=q[:,:,1]/q[:,:,2]
    flowx=(ut-xx).astype(np.float32);flowy=(vt-yy).astype(np.float32)
    tx=xx.astype(np.float32);ty=yy.astype(np.float32);mx=tx.copy();my=ty.copy()
    for _ in range(6):
        fx=cv2.remap(flowx,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
        fy=cv2.remap(flowy,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
        mx=tx-fx;my=ty-fy
    fx=cv2.remap(flowx,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan);fy=cv2.remap(flowy,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
    residual=np.sqrt((mx+fx-tx)**2+(my+fy-ty)**2)
    vsrc=cv2.remap(valid.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    esrc=cv2.remap(depth_edges(depth,valid).astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=1)>0
    dyn=cv2.remap(dynamic.astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    mag=np.sqrt(fx*fx+fy*fy)
    inside=(mx>=0)&(mx<W-1)&(my>=0)&(my<H-1)&np.isfinite(mx)&np.isfinite(my)&np.isfinite(residual)
    good=inside&vsrc&(residual<=.75)&(~(esrc&(mag>1.0)))
    warped=cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    warped[~good]=0
    return warped,good,dyn&good,{'mean_inverse_residual_px':float(np.nanmean(residual[good])) if good.any() else float('inf'),'p95_inverse_residual_px':float(np.nanpercentile(residual[good],95)) if good.any() else float('inf'),'resolved_fraction':float(np.mean(good))}


def composite(primary,pvalid,secondary,svalid,sdyn,allow_dynamic):
    out=primary.copy();use=(~pvalid)&svalid
    if not allow_dynamic:use&=(~sdyn)
    out[use]=secondary[use]
    return out,pvalid|use,use


def coverage(mask,pivot):
    cx,cy=np.round(pivot).astype(int);x0,x1=max(0,cx-220),min(W,cx+221);y0,y1=max(0,cy-190),min(H,cy+191);roi=np.zeros((H,W),bool);roi[y0:y1,x0:x1]=True
    return {'full':float(np.mean(mask)),'action':float(np.mean(mask[roi])),'holes_action':int((roi&(~mask)).sum()),'roi':[x0,y0,x1,y1]}


def signed_azimuth_delta(C0,C1,pivot):
    a0=math.atan2(C0[1]-pivot[1],C0[0]-pivot[0]);a1=math.atan2(C1[1]-pivot[1],C1[0]-pivot[0]);d=math.atan2(math.sin(a1-a0),math.cos(a1-a0));return math.degrees(d)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lar-frame',type=Path,required=True);ap.add_argument('--pbp-frame',type=Path,required=True);ap.add_argument('--lar-floor',type=Path,required=True);ap.add_argument('--pbp-floor-proof',type=Path,required=True);ap.add_argument('--camera-registry',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--tokens',type=int,default=1600);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lar=cv2.imread(str(a.lar_frame));pbp=cv2.imread(str(a.pbp_frame))
    if lar is None or pbp is None or lar.shape[:2]!=(H,W) or pbp.shape[:2]!=(H,W):raise RuntimeError('v68 requires native 960x540 exact frames')
    lf=json.load(open(a.lar_floor));pf=json.load(open(a.pbp_floor_proof));reg=json.load(open(a.camera_registry))
    if lf.get('status')!='PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35':raise RuntimeError('LAR v35 not accepted')
    if pf.get('status')!='PASS_PLAY_BY_PLAY_WIDE_COURT_FLOOR_V65':raise RuntimeError('PBP v65 not accepted')
    cam=reg['accepted_cameras']['Left Above Rim'];ev=cam['event_489'];C0=np.asarray(cam['physical_camera_center_prior_cm'],float);K0=K_matrix(float(ev['focal_px']),ev['principal_point_px']);H0=np.asarray(lf['floor_homography_world_to_image'],float)
    rot,_=recover_accepted_rotation(C0,K0,H0);rv0=rot['rv'];R0=rot['R']

    torch.set_num_threads(max(1,min(4,torch.get_num_threads())));det=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval();moge=MoGeModel.from_pretrained('Ruicheng/moge-2-vits-normal').eval()
    dyn0,_=detect_dynamic_and_ball(det,lar);dyn1,_=detect_dynamic_and_ball(det,pbp)
    d0,_,v0,Km0,_=moge_infer(moge,lar,a.tokens);d1,_,v1,Km1,_=moge_infer(moge,pbp,a.tokens)
    cv2.imwrite(str(a.out/'lar_dynamic.png'),dyn0.astype(np.uint8)*255);cv2.imwrite(str(a.out/'pbp_dynamic.png'),dyn1.astype(np.uint8)*255)

    fd0,fz0,hold0,_=clean_floor_depth_samples(H0,C0,K0,rv0,d0,v0,dyn0,1.0);bm0,mods0,apply0=fit_depth_models(fd0,fz0,~hold0);md0=apply0(bm0,d0.astype(np.float64));held0=np.abs(md0[np.rint(project_h_for_samples(H0)[1][:,1]).astype(int)%H,np.rint(project_h_for_samples(H0)[1][:,0]).astype(int)%W]) if False else None
    Xmap0,vm0=world_map_from_depth(md0,v0,K0,R0,C0)

    mask0=(~dyn0)&v0;mask1=(~dyn1)&v1;_,u0,u1=sift_matches(lar,pbp,mask0,mask1)
    keep,X=sample_world_at(Xmap0,vm0,u0);u1v=u1[keep]
    H1=np.asarray(pf['homography_world_to_source_px'],float)
    rv1,C1,K1,R1,best,roots,inlier_ids=solve_secondary_pose(X,u1v,Km1,H1)
    fd1,fz1,hold1,_=clean_floor_depth_samples(H1,C1,K1,rv1,d1,v1,dyn1,2.5);bm1,mods1,apply1=fit_depth_models(fd1,fz1,~hold1);md1=apply1(bm1,d1.astype(np.float64));Xmap1,vm1=world_map_from_depth(md1,v1,K1,R1,C1)

    # held-out depth QA from the actual floor anchors used above
    p0=apply0(bm0,fd0);e0=np.abs(p0[hold0]-fz0[hold0]);p1=apply1(bm1,fd1);e1=np.abs(p1[hold1]-fz1[hold1])
    delta=signed_azimuth_delta(C0,C1,RIM);direction=1.0 if delta>=0 else -1.0
    sep=float(np.linalg.norm(C1-C0))
    frames=[]
    for deg in ANGLES:
        signed=direction*deg;Rt,Ct=orbit_pose(C0,R0,RIM,signed);pivot,_=project_points(K0,Rt,Ct,RIM[None,:])
        w0,g0,dg0,m0=inverse_warp(lar,md0,vm0,dyn0,K0,R0,C0,K0,Rt,Ct)
        w1,g1,dg1,m1=inverse_warp(pbp,md1,vm1,dyn1,K1,R1,C1,K0,Rt,Ct)
        sta,gs,us=composite(w0,g0,w1,g1,dg1,False);full,gf,uf=composite(w0,g0,w1,g1,dg1,True)
        cv2.imwrite(str(a.out/f'v68_{int(deg):02d}deg_lar_dense.png'),w0);cv2.imwrite(str(a.out/f'v68_{int(deg):02d}deg_static_multiview.png'),sta);cv2.imwrite(str(a.out/f'v68_{int(deg):02d}deg_full_multiview.png'),full);cv2.imwrite(str(a.out/f'v68_{int(deg):02d}deg_pbp_contribution.png'),uf.astype(np.uint8)*255)
        frames.append({'degree':deg,'signed_degree':signed,'lar':coverage(g0,pivot[0]),'static_multiview':coverage(gs,pivot[0]),'full_multiview':coverage(gf,pivot[0]),'pbp_static_fill_pixels':int(us.sum()),'pbp_full_fill_pixels':int(uf.sum()),'lar_inverse':m0,'pbp_inverse':m1})
    # raw zero degree identity must be measured on dense inverse warp, not hole splat.
    zimg,zmask,_,_=inverse_warp(lar,md0,vm0,dyn0,K0,R0,C0,K0,R0,C0);ident=np.all(zimg==lar,axis=2)&zmask
    zero_identity=float(np.mean(ident[zmask])) if zmask.any() else 0.0
    gates={'lar_metric_floor_reproduction_p95_le_0_55':rot['p95_px']<=.55,'static_metric_matches_ge_35':len(X)>=35,'pbp_pnp_inliers_ge_28':len(inlier_ids)>=28,'pbp_static_match_p95_le_8px':best['static_p95']<=8.,'pbp_floor_p95_le_3px':best['floor_p95']<=3.,'pbp_positive_depth':best['min_depth']>20.,'pbp_distinct_baseline_gt_50cm':sep>50.,'pbp_azimuth_separation_gt_5deg':abs(delta)>5.,'lar_heldout_depth_p95_le_90cm':float(np.percentile(e0,95))<=90.,'pbp_heldout_depth_p95_le_120cm':float(np.percentile(e1,95))<=120.,'zero_dense_identity_ge_0_995':zero_identity>=.995}
    report=json_safe({'status':'PASS_DIAGNOSTIC_METRIC_ANCHORED_MULTIVIEW_V68' if all(gates.values()) else 'FAIL_DIAGNOSTIC_METRIC_ANCHORED_MULTIVIEW_V68','game_id':'0022500301','event_id':489,'method':'accepted LAR metric anchor + PBP v65 floor + static cross-view metric PnP + independently floor-scaled MoGe depth + dense inverse warp; PBP fills only unsupported LAR pixels','frames_sha256':{'lar':sha256(a.lar_frame),'pbp':sha256(a.pbp_frame)},'lar':{'center_cm':C0,'K':K0,'floor_reproduction_p95_px':rot['p95_px'],'depth_model':bm0,'heldout_depth_median_cm':float(np.median(e0)),'heldout_depth_p95_cm':float(np.percentile(e0,95))},'pbp_diagnostic_pose':{'center_cm':C1,'K':K1,'rvec':rv1,'static_correspondence_count':len(X),'pnp_inliers':len(inlier_ids),'best':best,'roots':[{k:v for k,v in r.items() if k!='p'} for r in roots],'baseline_from_lar_cm':sep,'signed_azimuth_separation_deg':delta,'depth_model':bm1,'heldout_depth_median_cm':float(np.median(e1)),'heldout_depth_p95_cm':float(np.percentile(e1,95))},'zero_dense_roundtrip':{'identity_fraction_of_resolved':zero_identity,'resolved_fraction':float(np.mean(zmask))},'frames':frames,'gates':gates,'permissions':{'pbp_metric_camera_promotion_allowed':False,'product_static_novel_view_allowed':False,'replay_render_allowed':False},'visual_qa_required':True})
    (a.out/'metric_anchor_multiview_v68.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'static_matches':len(X),'pnp_inliers':len(inlier_ids),'pbp_static_p95':best['static_p95'],'pbp_floor_p95':best['floor_p95'],'pbp_baseline_cm':sep,'pbp_azimuth_deg':delta,'lar_depth_p95_cm':float(np.percentile(e0,95)),'pbp_depth_p95_cm':float(np.percentile(e1,95)),'zero_identity':zero_identity,'coverage':{str(f['degree']):f['full_multiview']['action'] for f in frames},'gates':gates},indent=2),flush=True)


def project_h_for_samples(Hm):
    P,U,m=floor_grid(Hm);return P,U,m

if __name__=='__main__':main()
