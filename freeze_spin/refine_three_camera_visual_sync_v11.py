from __future__ import annotations

"""Jazz event 489: exact visual-state synchronization for the three solved cameras.

Audio provides only the coarse local-time map. This stage searches nearby real
frames and selects a common basketball state using calibrated metric geometry.
Broadcast <-> Right Above Rim is solved first from the observed basketball ray
intersection. Left Above Rim is then selected from nearby frames by player
silhouette-cone consistency with the selected Broadcast state, with the common
3-D ball projection used as an additional cue when visible.

No camera is re-promoted or re-solved here. This is exact-frame refinement after
camera calibration, as required by the project architecture.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v6 as v6
from freeze_spin import build_three_camera_volumetric_v8 as v8
from freeze_spin import build_three_camera_instance_volume_v9 as v9

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
FPS=29.97003
REF="Left Above Rim"; BR="Broadcast"; RAR="Right Above Rim"


def source_clip(clips:Path,label:str)->Path:
    token=label.replace(' ','_')
    rows=sorted(clips.glob(f"*_489_{token}_SOURCE.mp4"))
    if len(rows)!=1: raise RuntimeError(f"clip ambiguity {label}: {rows}")
    return rows[0]


def decode_frame(path:Path,t:float):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000.0); ok,im=cap.read(); idx=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)-1)); cap.release()
    if not ok or im is None: raise RuntimeError(f"decode failed {path} {t}")
    if im.shape[:2]!=(H,W): raise RuntimeError(f"unexpected frame shape {im.shape}")
    return im,idx


def pixel_ray(cam,uv):
    C,R,K=cam; s=float(v3.forward_sign(R,C)); u,v=[float(x) for x in uv]
    dcam=s*np.asarray([(u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],1.0],np.float64)
    d=R.T@dcam; d/=max(np.linalg.norm(d),1e-12)
    return C.astype(np.float64),d


def closest_rays(cam1,uv1,cam2,uv2):
    C1,d1=pixel_ray(cam1,uv1); C2,d2=pixel_ray(cam2,uv2)
    w=C1-C2; a=float(d1@d1); b=float(d1@d2); c=float(d2@d2); d=float(d1@w); e=float(d2@w)
    den=a*c-b*b
    if abs(den)<1e-9: return None
    t=(b*e-c*d)/den; s=(a*e-b*d)/den
    p1=C1+t*d1; p2=C2+s*d2; X=(p1+p2)/2.0
    return X,float(np.linalg.norm(p1-p2)),float(t),float(s)


def ball_pair_metric(cams,b,r,boff,roff):
    uvb=np.asarray([b['cx'],b['cy']],np.float64); uvr=np.asarray([r['cx'],r['cy']],np.float64)
    cr=closest_rays(cams[BR],uvb,cams[RAR],uvr)
    if cr is None: return None
    X,gap,t1,t2=cr
    errs={}
    for k,uv in ((BR,uvb),(RAR,uvr)):
        p,_,ok=v8.project_metric(cams[k],X.reshape(1,3)); errs[k]=float(np.linalg.norm(p[0]-uv)) if ok[0] else 999.0
    rim_dist=float(np.linalg.norm(X-RIM))
    physical=(t1>0 and t2>0 and -140<X[0]<420 and abs(X[1])<300 and 150<X[2]<420)
    physical_pen=0.0 if physical else 600.0
    physical_pen+=max(0.0,rim_dist-150.0)*2.0
    temporal_pen=1.5*(abs(boff)+abs(roff))
    score=gap + 1.4*math.sqrt((errs[BR]**2+errs[RAR]**2)/2.0) + 0.10*rim_dist + temporal_pen + physical_pen
    return {"score":float(score),"center_world_cm":X.tolist(),"ray_gap_cm":gap,"reprojection_errors_px":errs,"rim_distance_cm":rim_dist,"physical":bool(physical),"broadcast_offset_frames":int(boff),"rar_offset_frames":int(roff),"broadcast_detection":b,"rar_detection":r}


def draw_ball_overlay(im,cands,title):
    out=im.copy(); cv2.rectangle(out,(0,0),(700,42),(0,0,0),-1); cv2.putText(out,title,(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    for i,b in enumerate(cands[:3]):
        c=(int(round(b['cx'])),int(round(b['cy']))); cv2.circle(out,c,8,(0,255,255),2); cv2.putText(out,str(i),(c[0]+9,c[1]),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1,cv2.LINE_AA)
    return out


def coarse_grid(step=8.0):
    xs=np.arange(-260.0,721.0,step,np.float32); ys=np.arange(-560.0,561.0,step,np.float32); zs=np.arange(-8.0,373.0,step,np.float32)
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij'); return np.column_stack([X.ravel(),Y.ravel(),Z.ravel()]).astype(np.float32)


def player_match_score(ref_inst,br_inst,ref_desc,br_desc,cams,grid):
    rh=[v8.mask_membership(cams[REF],grid,r['support_mask']) for r in ref_inst]
    bh=[v8.mask_membership(cams[BR],grid,r['support_mask']) for r in br_inst]
    nr,nb=len(rh),len(bh); S=np.full((nr,nb),-1e3,np.float64); details=[]
    for i in range(nr):
        row=[]
        for j in range(nb):
            ov=int(np.sum(rh[i]&bh[j])); norm=ov/max(1.0,math.sqrt(float(np.sum(rh[i]))*float(np.sum(bh[j])))); app=v9.appearance_similarity(ref_desc[i],br_desc[j]); s=26.0*norm+1.0*app
            if ov<22: s=-8.0
            S[i,j]=s; row.append({"overlap_voxels":ov,"normalized_overlap":float(norm),"appearance":float(app),"score":float(s)})
        details.append(row)
    if not nr or not nb: return -1e3,0,{},details
    rr,cc=linear_sum_assignment(-S); pairs=[]; total=0.0
    for i,j in zip(rr,cc):
        if details[int(i)][int(j)]['overlap_voxels']>=22:
            pairs.append([int(i),int(j)]); total+=float(S[i,j])
    return float(total),len(pairs),{"pairs":pairs,"score_matrix":S.tolist()},details


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips-dir',type=Path,required=True); ap.add_argument('--options',type=Path,required=True); ap.add_argument('--frames-dir',type=Path,required=True)
    ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--radius-frames',type=int,default=6)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    opts=json.loads(args.options.read_text()); t0={x['camera']:float(x['requested_local_time']) for x in opts['options']}
    central_br=base.find_one(args.frames_dir,'A_Broadcast')
    cams=base.load_cameras(args.registry,args.rar_report,central_br)
    clips={k:source_clip(args.clips_dir,k) for k in (REF,BR,RAR)}

    candidates={}; contact=[]
    for label in (BR,RAR):
        candidates[label]=[]
        for off in range(-args.radius_frames,args.radius_frames+1):
            t=t0[label]+off/FPS; im,idx=decode_frame(clips[label],t); balls,_=v9.orange_ball_candidates(im,cams[label])
            candidates[label].append({'offset_frames':off,'time_seconds':t,'frame_index':idx,'image':im,'balls':balls})
            contact.append((label,off,draw_ball_overlay(im,balls,f"{label} offset {off:+d} frame {idx}")))

    pair_rows=[]
    for bfr in candidates[BR]:
        for rfr in candidates[RAR]:
            for b in bfr['balls'][:3]:
                for r in rfr['balls'][:3]:
                    m=ball_pair_metric(cams,b,r,bfr['offset_frames'],rfr['offset_frames'])
                    if m is not None:
                        m['broadcast_time_seconds']=bfr['time_seconds']; m['rar_time_seconds']=rfr['time_seconds']; m['broadcast_frame_index']=bfr['frame_index']; m['rar_frame_index']=rfr['frame_index']; pair_rows.append(m)
    if not pair_rows: raise RuntimeError('No Broadcast/RAR ball-pair candidates')
    pair_rows.sort(key=lambda x:x['score']); best=pair_rows[0]; X=np.asarray(best['center_world_cm'],np.float64)

    # Selected Broadcast state for player matching.
    bsel=next(x for x in candidates[BR] if x['offset_frames']==best['broadcast_offset_frames'])
    rsel=next(x for x in candidates[RAR] if x['offset_frames']==best['rar_offset_frames'])
    seg=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval(); torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    _,binst,_=v6.detect_near_play(seg,bsel['image'],*cams[BR][2::-1]) if False else v6.detect_near_play(seg,bsel['image'],cams[BR][2],cams[BR][1],cams[BR][0])
    # detect_near_play signature is image,K,R,C.
    bdesc=[]
    for row in binst: row['support_mask']=v9.support_mask(row['mask']); bdesc.append(v9.descriptor(bsel['image'],row['mask']))
    grid=coarse_grid(8.0); lar_rows=[]; expected_uv,_,expected_ok=v8.project_metric(cams[REF],X.reshape(1,3)); expected_uv=expected_uv[0] if expected_ok[0] else None
    for off in range(-args.radius_frames,args.radius_frames+1):
        t=t0[REF]+off/FPS; im,idx=decode_frame(clips[REF],t); _,inst,_=v6.detect_near_play(seg,im,cams[REF][2],cams[REF][1],cams[REF][0])
        desc=[]
        for row in inst: row['support_mask']=v9.support_mask(row['mask']); desc.append(v9.descriptor(im,row['mask']))
        ps,nmatch,assign,details=player_match_score(inst,binst,desc,bdesc,cams,grid)
        balls,_=v9.orange_ball_candidates(im,cams[REF]); bdist=999.0
        if expected_uv is not None and balls: bdist=min(float(np.linalg.norm(np.asarray([x['cx'],x['cy']])-expected_uv)) for x in balls[:5])
        ball_bonus=max(0.0,45.0-bdist)/12.0
        final=ps+3.0*nmatch+ball_bonus-0.12*abs(off)
        lar_rows.append({'offset_frames':off,'time_seconds':t,'frame_index':idx,'player_score':ps,'matched_players':nmatch,'ball_expected_uv':None if expected_uv is None else expected_uv.tolist(),'nearest_ball_candidate_px':bdist,'ball_bonus':ball_bonus,'final_score':final,'assignment':assign,'image':im,'balls':balls})
    lar_rows.sort(key=lambda x:x['final_score'],reverse=True); lsel=lar_rows[0]

    selected={
        REF:{'offset_frames':lsel['offset_frames'],'time_seconds':lsel['time_seconds'],'frame_index':lsel['frame_index']},
        BR:{'offset_frames':best['broadcast_offset_frames'],'time_seconds':best['broadcast_time_seconds'],'frame_index':best['broadcast_frame_index']},
        RAR:{'offset_frames':best['rar_offset_frames'],'time_seconds':best['rar_time_seconds'],'frame_index':best['rar_frame_index']},
    }
    # Save exact selected images using stable labels consumed by the next renderer.
    cv2.imwrite(str(args.out/'A_Broadcast_visual_sync.png'),bsel['image']); cv2.imwrite(str(args.out/'K_Left_Above_Rim_visual_sync.png'),lsel['image']); cv2.imwrite(str(args.out/'L_Right_Above_Rim_visual_sync.png'),rsel['image'])
    cv2.imwrite(str(args.out/'best_broadcast_ball.png'),draw_ball_overlay(bsel['image'],bsel['balls'],f"SELECTED Broadcast offset {best['broadcast_offset_frames']:+d}"))
    cv2.imwrite(str(args.out/'best_rar_ball.png'),draw_ball_overlay(rsel['image'],rsel['balls'],f"SELECTED RAR offset {best['rar_offset_frames']:+d}"))
    cv2.imwrite(str(args.out/'best_lar_state.png'),draw_ball_overlay(lsel['image'],lsel['balls'],f"SELECTED LAR offset {lsel['offset_frames']:+d}; player score {lsel['final_score']:.2f}"))

    # Contact sheets for visual audit.
    for label in (BR,RAR):
        ims=[x[2] for x in contact if x[0]==label]; thumb=[cv2.resize(x,(480,270),interpolation=cv2.INTER_AREA) for x in ims]
        rows=[]
        for i in range(0,len(thumb),4):
            r=thumb[i:i+4]
            while len(r)<4:r.append(np.zeros_like(thumb[0]))
            rows.append(np.hstack(r))
        cv2.imwrite(str(args.out/f"{label.replace(' ','_')}_candidate_contact.png"),np.vstack(rows))

    report={
        'schema_version':11,'game_id':'0022500301','event_id':489,'method':'audio-coarse -> nearby real-frame search -> Broadcast/RAR ball-ray metric consistency -> LAR player-silhouette metric consistency',
        'initial_audio_times_seconds':t0,'search_radius_frames':args.radius_frames,'selected':selected,'best_ball_pair':best,'top_ball_pairs':pair_rows[:20],
        'lar_candidates':[{k:v for k,v in x.items() if k not in ('image','balls')} for x in lar_rows],
        'gates':{
            'ball_ray_gap_target_cm':20.0,'max_ball_reprojection_target_px':8.0,'ball_rim_distance_target_cm':150.0,
            'ball_ray_gap_pass':best['ray_gap_cm']<=20.0,'ball_reprojection_pass':max(best['reprojection_errors_px'].values())<=8.0,'ball_rim_distance_pass':best['rim_distance_cm']<=150.0,
        }
    }
    report['gates']['all_ball_geometry_pass']=all(report['gates'][k] for k in ('ball_ray_gap_pass','ball_reprojection_pass','ball_rim_distance_pass'))
    (args.out/'three_camera_visual_sync_v11.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'selected':selected,'best_ball_pair':best,'lar_best':{k:v for k,v in lsel.items() if k not in ('image','balls')},'gates':report['gates']},indent=2),flush=True)

if __name__=='__main__': main()
