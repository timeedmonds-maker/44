from __future__ import annotations

"""v32o: fit a real MHR anatomical mesh to the accepted v32m Adams pose.

This stage replaces every previous capsule/blob/visual-hull foreground with a
parametric skinned triangle mesh.  The image model is NOT used to place the body:
only v32m's source-grounded metric 3-D joints constrain the fit.  MHR supplies
anatomical articulation and a continuous surface.  The fit is projected back into
all three accepted NBA cameras for visual QA; no novel-view render is produced.

Important: this is a geometry diagnostic, not a photorealistic replay.  No NBA
pixels are synthesized, no learned texture is generated, and no missing joint is
silently copied from another player.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mhr.mhr import MHR
from mhr.io import get_default_asset_folder

W,H=960,540
CAMS=("Left Above Rim","Broadcast","Right Above Rim")
FRAME_NAMES={
 "Left Above Rim":"v32_chosen_Left_Above_Rim_frame0260.png",
 "Broadcast":"v32_chosen_Broadcast_frame0276.png",
 "Right Above Rim":"v32_chosen_Right_Above_Rim_frame0256.png",
}

# MHR joint origins corresponding to COCO-style articulation landmarks.
# Shoulder=upper-arm origin; elbow=lower-arm origin; wrist=wrist; hip=upper-leg
# origin; knee=lower-leg origin; ankle=foot origin. v32n2 separately audits
# these choices against the neutral rig before downstream silhouette fitting.
MHR_MAP={
 "left_shoulder":"l_uparm",
 "right_shoulder":"r_uparm",
 "left_elbow":"l_lowarm",
 "right_elbow":"r_lowarm",
 "left_wrist":"l_wrist",
 "right_wrist":"r_wrist",
 "left_hip":"l_upleg",
 "right_hip":"r_upleg",
 "left_knee":"l_lowleg",
 "right_knee":"r_lowleg",
 "left_ankle":"l_foot",
 "right_ankle":"r_foot",
}

FIT_NAMES=("left_shoulder","left_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","left_ankle","right_ankle")
SEGMENTS=(("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("left_hip","left_knee"),("left_knee","left_ankle"),("left_hip","right_hip"))


def ensure_assets():
    assets=get_default_asset_folder()
    if not (assets/'lod1.fbx').exists():
        subprocess.run(['mhr-download-assets'],check=True)
    return assets


def skew(v):
    z=torch.zeros((),dtype=v.dtype,device=v.device)
    return torch.stack([z,-v[2],v[1], v[2],z,-v[0], -v[1],v[0],z]).reshape(3,3)


def exp_so3(v):
    # matrix_exp is stable around zero and differentiable.
    return torch.matrix_exp(skew(v))


def unit(v):
    n=np.linalg.norm(v)
    return v/max(n,1e-9)


def make_frame(left,right,up_point):
    x=unit(np.asarray(right)-np.asarray(left))
    mid=.5*(np.asarray(left)+np.asarray(right))
    yr=np.asarray(up_point)-mid
    y=unit(yr-x*np.dot(x,yr))
    z=unit(np.cross(x,y))
    # re-orthogonalize y to eliminate accumulated numerical error
    y=unit(np.cross(z,x))
    return np.column_stack([x,y,z])


def camera(scene,label):
    d=scene['cameras'][label]
    return np.asarray(d['K_px'],float),np.asarray(d['R_world_to_camera'],float),np.asarray(d['C_world_cm'],float)


def project(K,R,C,X):
    X=np.asarray(X,float)
    xc=(R@(X-C).T).T
    z=xc[:,2]
    q=(K@xc.T).T
    uv=np.full((len(X),2),np.nan,float)
    ok=np.abs(z)>1e-7
    uv[ok]=q[ok,:2]/q[ok,2:3]
    return uv,z


def mesh_mask(uv,faces):
    mask=np.zeros((H,W),np.uint8)
    tri=uv[faces]
    valid=np.all(np.isfinite(tri),axis=(1,2)) & np.all(np.abs(tri)<6000,axis=(1,2))
    for t in tri[valid]:
        p=np.rint(t).astype(np.int32)
        cv2.fillConvexPoly(mask,p,255,lineType=cv2.LINE_8)
    return mask


def draw_overlay(img,mask,joint_uv,label):
    out=img.copy()
    cnt,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out,cnt,-1,(255,255,0),2,cv2.LINE_AA)
    for name,p in joint_uv.items():
        if p is None or not np.all(np.isfinite(p)): continue
        x,y=np.rint(p).astype(int)
        if -20<=x<W+20 and -20<=y<H+20:
            cv2.circle(out,(x,y),3,(255,0,255),-1,cv2.LINE_AA)
    cv2.rectangle(out,(0,0),(W,28),(0,0,0),-1)
    cv2.putText(out,f'v32o MHR mesh projection | {label} | cyan silhouette, magenta fitted joints',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1,cv2.LINE_AA)
    return out


def smooth_l1_distance(d,beta=5.0):
    return torch.where(d<beta,0.5*d*d/beta,d-0.5*beta)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--v32m-root',type=Path,required=True)
    ap.add_argument('--b32-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--iters',type=int,default=900)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))

    q=json.loads((a.v32m_root/'v32m_single_anchor_qa.json').read_text())
    assert q['status']=='PASS_V32M_SINGLE_ANCHOR_ARTICULATED_POSITION'
    assert q['surface_stage_unlocked'] is True
    target_all={k:np.asarray(v['world_cm'],np.float32) for k,v in q['joints'].items()}
    fit_names=[n for n in FIT_NAMES if n in target_all]
    if len(fit_names)<8: raise RuntimeError(f'insufficient trusted body joints: {fit_names}')

    stage=a.b32_root/'stage_a'
    scene=json.loads((stage/'v32_scene_manifest.json').read_text())
    assert scene['resolution']==[W,H] and set(scene['cameras'])==set(CAMS)
    images={c:cv2.imread(str(stage/FRAME_NAMES[c])) for c in CAMS}
    assert all(v is not None for v in images.values())

    ensure_assets()
    model=MHR.from_files(device=torch.device('cpu'),lod=1,wants_pose_correctives=False)
    char=model.character
    names=list(char.skeleton.joint_names); ji={n:i for i,n in enumerate(names)}
    missing=[MHR_MAP[n] for n in fit_names if MHR_MAP[n] not in ji]
    if missing: raise RuntimeError(f'MHR mapping names missing: {missing}')

    zeros204=torch.zeros((1,204),dtype=torch.float32)
    zeros117=torch.zeros((1,117),dtype=torch.float32)
    def skel_from(mp):
        full=torch.cat([mp,zeros117],dim=1)
        jp=model.character_torch.model_parameters_to_joint_parameters(full)
        return model.character_torch.joint_parameters_to_skeleton_state(jp)
    with torch.no_grad(): neutral=skel_from(zeros204)[0,:,:3].cpu().numpy()

    # Geometric similarity initialization using hip lateral axis + body-up axis.
    sh=np.asarray(neutral[ji['c_head']],float)
    sl=np.asarray(neutral[ji['l_upleg']],float); sr=np.asarray(neutral[ji['r_upleg']],float)
    th=.5*(target_all['left_hip']+target_all['right_hip'])
    # Nose is observation-derived and only used to initialize global orientation,
    # never as a direct MHR joint constraint.
    tup=target_all.get('nose',target_all['left_shoulder'])
    Fs=make_frame(sl,sr,sh); Ft=make_frame(target_all['left_hip'],target_all['right_hip'],tup)
    R0=Ft@Fs.T

    ratios=[]
    for aa,bb in SEGMENTS:
        if aa in target_all and bb in target_all and MHR_MAP[aa] in ji and MHR_MAP[bb] in ji:
            ts=float(np.linalg.norm(target_all[aa]-target_all[bb])); ss=float(np.linalg.norm(neutral[ji[MHR_MAP[aa]]]-neutral[ji[MHR_MAP[bb]]]))
            if ss>1e-5: ratios.append(ts/ss)
    s0=float(np.median(ratios)) if ratios else 1.2
    s0=float(np.clip(s0,.85,1.55))
    sm=.5*(sl+sr); t0=th-s0*(R0@sm)

    R0t=torch.tensor(R0,dtype=torch.float32)
    mp=torch.nn.Parameter(torch.zeros((1,204),dtype=torch.float32))
    drot=torch.nn.Parameter(torch.zeros(3,dtype=torch.float32))
    log_s=torch.nn.Parameter(torch.tensor(math.log(s0),dtype=torch.float32))
    trans=torch.nn.Parameter(torch.tensor(t0,dtype=torch.float32))

    # Body pose only plus six interpretable flexible body-dimension parameters.
    active=list(range(6,68))+list(range(130,136))
    active_mask=torch.zeros(204,dtype=torch.bool); active_mask[active]=True
    lo,hi=char.model_parameter_limits
    lo=np.asarray(lo[:204],np.float32); hi=np.asarray(hi[:204],np.float32)
    # Never optimize the built-in root: external similarity owns world placement.
    weights={
      'left_shoulder':1.15,'left_elbow':1.2,'left_wrist':1.05,'right_wrist':.9,
      'left_hip':1.35,'right_hip':1.35,'left_knee':1.25,'left_ankle':1.15,'right_ankle':1.0,
    }
    tidx=torch.tensor([ji[MHR_MAP[n]] for n in fit_names],dtype=torch.long)
    tgt=torch.tensor(np.stack([target_all[n] for n in fit_names]),dtype=torch.float32)
    wt=torch.tensor([weights.get(n,1.) for n in fit_names],dtype=torch.float32)

    opt=torch.optim.Adam([mp,drot,log_s,trans],lr=.035)
    history=[]
    best=None
    for it in range(a.iters):
        opt.zero_grad(set_to_none=True)
        st=skel_from(mp)[0,:,:3]
        R=exp_so3(drot)@R0t
        ss=torch.exp(log_s)
        pred=ss*(st[tidx]@R.T)+trans
        d=torch.linalg.norm(pred-tgt,dim=1)
        data=(smooth_l1_distance(d,5.)*wt).sum()/wt.sum()
        # Pose regularization is intentionally light; MHR anatomy and parameter
        # limits do most of the plausibility work. Flexible dimensions are more
        # strongly regularized because only one player instance is available.
        reg_pose=.006*torch.mean(mp[0,6:68]**2)
        reg_dim=.035*torch.mean(mp[0,130:136]**2)
        reg_rot=.003*torch.sum(drot**2)
        loss=data+reg_pose+reg_dim+reg_rot
        loss.backward()
        if mp.grad is not None: mp.grad[:,~active_mask]=0
        opt.step()
        with torch.no_grad():
            # Enforce MHR's declared bounds where finite, plus a conservative
            # generic cap for otherwise-unbounded articulated parameters.
            x=mp[0].cpu().numpy()
            finite=np.isfinite(lo)&np.isfinite(hi)&(np.abs(lo)<1e20)&(np.abs(hi)<1e20)
            x[finite]=np.minimum(np.maximum(x[finite],lo[finite]),hi[finite])
            x[6:68]=np.clip(x[6:68],-3.2,3.2)
            x[130:136]=np.clip(x[130:136],-2.5,2.5)
            x[~active_mask.cpu().numpy()]=0
            mp[0].copy_(torch.from_numpy(x))
            log_s.clamp_(math.log(.75),math.log(1.65))
        if it%25==0 or it==a.iters-1:
            rec={'iter':it,'loss':float(loss.detach()),'data':float(data.detach()),'median_cm':float(torch.median(d).detach()),'max_cm':float(torch.max(d).detach()),'scale':float(torch.exp(log_s).detach())}
            history.append(rec)
            score=rec['median_cm']+.25*rec['max_cm']
            if best is None or score<best[0]:
                best=(score,mp.detach().clone(),drot.detach().clone(),log_s.detach().clone(),trans.detach().clone())

    assert best is not None
    _,bmp,br,bs,bt=best
    with torch.no_grad():
        st=skel_from(bmp)[0,:,:3]
        R=(exp_so3(br)@R0t); scale=torch.exp(bs)
        world=scale*(st@R.T)+bt
        pred=world[tidx]
        err=torch.linalg.norm(pred-tgt,dim=1).cpu().numpy()
        # Generate the actual skinned MHR surface with pose correctives enabled.
        # Re-load with correctives now that optimization is done.
        full_model=MHR.from_files(device=torch.device('cpu'),lod=1,wants_pose_correctives=True)
        verts,_=full_model(torch.zeros((1,45)),bmp,torch.zeros((1,72)),apply_correctives=True)
        vworld=(scale*(verts[0]@R.T)+bt).cpu().numpy()
    faces=np.asarray(full_model.character.mesh.faces,np.int32)

    # Native PLY in NBA world centimeters.
    ply=a.out/'v32o_adams_mhr_world_cm.ply'
    with ply.open('w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(vworld)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write(f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n')
        for v in vworld: f.write(f'{v[0]} {v[1]} {v[2]}\n')
        for tri in faces: f.write(f'3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n')

    overlays=[]; camera_qa={}
    joint_world={n:world[ji[MHR_MAP[n]]].cpu().numpy() for n in fit_names}
    for c in CAMS:
        K,Rc,Cc=camera(scene,c); uv,z=project(K,Rc,Cc,vworld); mask=mesh_mask(uv,faces)
        juv={n:project(K,Rc,Cc,p.reshape(1,3))[0][0] for n,p in joint_world.items()}
        ov=draw_overlay(images[c],mask,juv,c); cv2.imwrite(str(a.out/f'v32o_{c.replace(" ","_")}_mesh_overlay.png'),ov); overlays.append(ov)
        camera_qa[c]={'mesh_projected_pixels':int(np.sum(mask>0)),'mesh_vertex_finite_fraction':float(np.mean(np.all(np.isfinite(uv),axis=1))),'signed_depth_median_cm':float(np.nanmedian(z))}
    cv2.imwrite(str(a.out/'v32o_three_camera_mesh_overlay.png'),np.hstack(overlays))

    per={n:float(e) for n,e in zip(fit_names,err)}
    med=float(np.median(err)); p90=float(np.percentile(err,90)); mx=float(np.max(err))
    # This gate only establishes an anatomically coherent metric surface seed.
    # Silhouette/source-pixel QA is a separate downstream gate.
    fit_pass=(len(fit_names)>=8 and med<=6.0 and p90<=10.0 and mx<=15.0 and np.all(np.isfinite(vworld)))
    qa={
      'version':'v32o_mhr_to_v32m_pose','status':'PASS_V32O_MHR_ANATOMICAL_SEED' if fit_pass else 'FAIL_CLOSED_V32O_MHR_ANATOMICAL_SEED',
      'source_pose':'v32m PASS pose only','native_resolution':[W,H],'generated_rgb':False,'novel_view_rendered':False,
      'surface_type':'MHR LOD1 skinned triangle mesh with pose correctives','vertex_count':int(len(vworld)),'face_count':int(len(faces)),
      'fit_joint_names':fit_names,'mhr_joint_mapping':{n:MHR_MAP[n] for n in fit_names},'per_joint_error_cm':per,
      'median_joint_error_cm':med,'p90_joint_error_cm':p90,'max_joint_error_cm':mx,'external_similarity_scale':float(scale),
      'external_rotation_matrix':R.cpu().numpy().tolist(),'external_translation_cm':bt.cpu().numpy().tolist(),
      'optimized_model_parameters':{full_model.character.parameter_transform.names[i]:float(bmp[0,i]) for i in active if abs(float(bmp[0,i]))>1e-6},
      'optimizer_history':history,'camera_projection_qa':camera_qa,
      'gate':{'trusted_body_joints_ge_8':len(fit_names)>=8,'median_joint_error_le_6cm':med<=6.,'p90_joint_error_le_10cm':p90<=10.,'max_joint_error_le_15cm':mx<=15.,'mesh_finite':bool(np.all(np.isfinite(vworld))),'silhouette_stage_unlocked':bool(fit_pass)},
      'warning':'Passing v32o does not mean photorealistic replay is solved; three-camera silhouette/source-pixel validation is still mandatory.'
    }
    (a.out/'v32o_mhr_fit_qa.json').write_text(json.dumps(qa,indent=2))
    np.savez_compressed(a.out/'v32o_mhr_fit.npz',model_parameters=bmp.cpu().numpy(),rotation=R.cpu().numpy(),scale=np.array([float(scale)],np.float32),translation=bt.cpu().numpy(),vertices_world_cm=vworld,faces=faces)
    print(json.dumps({k:qa[k] for k in ['status','median_joint_error_cm','p90_joint_error_cm','max_joint_error_cm','external_similarity_scale','gate']},indent=2))
    if not fit_pass: raise SystemExit(5)

if __name__=='__main__': main()
