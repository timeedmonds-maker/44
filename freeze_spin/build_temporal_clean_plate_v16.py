from __future__ import annotations

"""Jazz event 489 v16: native same-camera temporal clean-plate floor proof.

Repairs the static floor holes revealed by v15 without weakening foreground
exclusion and without generative inpainting. Nearby real frames from the fixed
Left Above Rim official HLS clip are aligned to the exact v11 selected frame.
For every recovered floor pixel, v16 writes ONE actually observed source pixel
whose colour is nearest the robust temporal median (a medoid). Exact v15-approved
floor pixels remain authoritative. Output stays native 960x540.
"""

import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from freeze_spin import build_three_camera_diagnostic_v1 as base
from freeze_spin import build_three_camera_diagnostic_v3 as v3
from freeze_spin import build_three_camera_diagnostic_v4 as v4

W,H=base.W,base.H
RIM=base.RIM.astype(np.float64)
REF,BR,RAR='Left Above Rim','Broadcast','Right Above Rim'
ORDER=(REF,BR,RAR)


def norm(s): return ''.join(ch.lower() for ch in s if ch.isalnum())


def find_clip(root,label):
    rows=list(root.rglob('*.mp4'))+list(root.rglob('*.mov'))+list(root.rglob('*.mkv'))
    key=norm(label); got=[]
    for p in rows:
        n=norm(p.stem)
        if label==BR:
            if 'broadcast' in n and 'otherbroadcast' not in n and 'mobilebroadcast' not in n and 'playbyplay' not in n: got.append(p)
        elif key in n: got.append(p)
    if len(got)!=1: raise RuntimeError(f'clip ambiguity for {label}: {got}')
    return got[0]


def read_frame(cap,idx):
    if idx<0: return None
    cap.set(cv2.CAP_PROP_POS_FRAMES,float(idx)); ok,f=cap.read()
    if not ok or f is None: return None
    if f.shape[1]!=W or f.shape[0]!=H: raise RuntimeError(f'native dimensions changed: {f.shape[1]}x{f.shape[0]}')
    return f


def floor_geometry(cam):
    C,R,K=cam; tf,Pf=v4.ray_plane_map(K,R,C,'floor')
    return (np.isfinite(tf)&(tf>20)&(tf<12000)&(Pf[:,:,0]>=v4.COURT_X0)&(Pf[:,:,0]<=v4.COURT_X1)&(Pf[:,:,1]>=v4.COURT_Y0)&(Pf[:,:,1]<=v4.COURT_Y1))


def gray(im): return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


def align_euclidean(ref,cand,mask):
    rg=gray(ref).astype(np.float32)/255.0; cg=gray(cand).astype(np.float32)/255.0
    M=np.eye(2,3,dtype=np.float32); criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-6)
    try: cc,M=cv2.findTransformECC(rg,cg,M,cv2.MOTION_EUCLIDEAN,criteria,inputMask=mask.astype(np.uint8)*255,gaussFiltSize=5)
    except cv2.error: cc=-1.0; M=np.eye(2,3,dtype=np.float32)
    wr=cv2.warpAffine(cand,M,(W,H),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)
    va=cv2.warpAffine(np.ones((H,W),np.uint8)*255,M,(W,H),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT)>0
    a,b=float(M[0,0]),float(M[0,1]); rot=math.degrees(math.atan2(b,a)); tx,ty=float(M[0,2]),float(M[1,2])
    sample=mask&va
    if np.any(sample):
        d=cv2.absdiff(gray(ref),gray(wr))[sample]; med=float(np.median(d)); p80=float(np.percentile(d,80))
    else: med=p80=999.0
    return wr,va,{'ecc':float(cc),'rotation_deg':rot,'tx_px':tx,'ty_px':ty,'median_gray_absdiff':med,'p80_gray_absdiff':p80}


def choose_anchor(cap,expected,exact,stable):
    rows=[]
    for idx in range(max(0,expected-3),expected+4):
        f=read_frame(cap,idx)
        if f is None: continue
        d=cv2.absdiff(gray(exact),gray(f))[stable]; score=float(np.median(d)) if len(d) else 999.0; rows.append((score,idx))
    if not rows: raise RuntimeError('could not decode anchor neighborhood')
    rows.sort(); return rows[0][1],[{'frame_index':int(i),'median_gray_absdiff':float(s)} for s,i in rows]


def temporal_medoid(frames,valids,frame_indices,floor_geom,min_obs=5,min_consensus=3,colour_tol=38.0):
    stack=np.stack(frames,axis=0).astype(np.uint8); vst=np.stack(valids,axis=0).astype(bool)
    atlas=np.zeros((H,W,3),np.uint8); valid_out=np.zeros((H,W),bool); consensus_out=np.zeros((H,W),np.uint8); obs_out=np.zeros((H,W),np.uint8); provenance=np.full((H,W),-1,np.int32)
    fi=np.asarray(frame_indices,np.int32)
    for y0 in range(0,H,45):
        y1=min(H,y0+45); arr=stack[:,y0:y1].astype(np.float32); vv=vst[:,y0:y1]; obs=vv.sum(axis=0)
        masked=np.where(vv[...,None],arr,np.nan)
        with np.errstate(all='ignore'): med=np.nanmedian(masked,axis=0)
        dist=np.sum(np.abs(arr-med[None,...]),axis=3); dist[~vv]=1e9; best=np.argmin(dist,axis=0); cons=np.sum((dist<=colour_tol)&vv,axis=0)
        yy,xx=np.indices((y1-y0,W)); chosen=stack[best,yy+y0,xx]
        good=(obs>=min_obs)&(cons>=min_consensus)&floor_geom[y0:y1]&np.isfinite(med).all(axis=2)
        atlas[y0:y1][good]=chosen[good]; valid_out[y0:y1][good]=True; consensus_out[y0:y1]=np.minimum(cons,255).astype(np.uint8); obs_out[y0:y1]=np.minimum(obs,255).astype(np.uint8); provenance[y0:y1][good]=fi[best[good]]
    return atlas,valid_out,consensus_out,obs_out,provenance


def background_render(cams,sources,angles,out,prefix):
    C0,R0,K0=cams[REF]; rows=[]
    for i,ang in enumerate(angles):
        Rt,Ct=base.orbit_pose(C0,R0,RIM,float(ang)); P,support=v4.virtual_plane_points(K0,Rt,Ct,'floor'); image,owned=v4.sample_plane_from_sources(P,support,sources,'floor')
        sm=support.reshape(H,W); hole=sm&(~owned); nlab,labels,stats,_=cv2.connectedComponentsWithStats(hole.astype(np.uint8),8); largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if nlab>1 else 0; coverage=float(np.sum(owned&sm)/max(1,np.sum(sm)))
        cv2.imwrite(str(out/f'{prefix}_{i:03d}.png'),image); cv2.imwrite(str(out/f'{prefix}_holes_{i:03d}.png'),hole.astype(np.uint8)*255)
        rows.append({'virtual_viewpoint_deg':float(ang),'floor_support_pixels':int(np.sum(sm)),'owned_floor_pixels':int(np.sum(owned&sm)),'coverage':coverage,'largest_hole_pixels':largest})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips-dir',type=Path,required=True); ap.add_argument('--sync-dir',type=Path,required=True); ap.add_argument('--strict-dir',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--rar-report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.loads((args.sync_dir/'three_camera_visual_sync_v11.json').read_text())
    exact_paths={REF:args.sync_dir/'K_Left_Above_Rim_visual_sync.png',BR:args.sync_dir/'A_Broadcast_visual_sync.png',RAR:args.sync_dir/'L_Right_Above_Rim_visual_sync.png'}; exact={k:cv2.imread(str(p)) for k,p in exact_paths.items()}
    if any(v is None for v in exact.values()): raise RuntimeError('missing exact v11 images')
    cams=base.load_cameras(args.registry,args.rar_report,exact_paths[BR])
    strict_floor={}; strict_dyn={}
    for lab in ORDER:
        stem=lab.replace(' ','_'); f=cv2.imread(str(args.strict_dir/f'{stem}_floor_visibility_strict.png'),cv2.IMREAD_GRAYSCALE); d=cv2.imread(str(args.strict_dir/f'{stem}_dynamic_exclusion_strict.png'),cv2.IMREAD_GRAYSCALE)
        if f is None or d is None: raise RuntimeError(f'missing v15 strict masks for {lab}')
        strict_floor[lab]=f>0; strict_dyn[lab]=d>0
    clip=find_clip(args.clips_dir,REF); cap=cv2.VideoCapture(str(clip))
    if not cap.isOpened(): raise RuntimeError(f'cannot open {clip}')
    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS)); expected=int(sync['selected'][REF]['frame_index']); geom=floor_geometry(cams[REF])
    stable=geom&(~cv2.dilate(strict_dyn[REF].astype(np.uint8),np.ones((21,21),np.uint8),iterations=1).astype(bool)); anchor,anchor_rows=choose_anchor(cap,expected,exact[REF],stable)
    offsets=[x for x in range(-112,113,8) if abs(x)>=16]; aligned=[]; valids=[]; indices=[]; align_rows=[]
    for off in offsets:
        idx=anchor+off
        if idx<0 or idx>=frame_count: continue
        fr=read_frame(cap,idx)
        if fr is None: continue
        wr,va,aq=align_euclidean(exact[REF],fr,stable); aq.update({'frame_index':int(idx),'offset_from_anchor':int(off)})
        accept=(aq['ecc']>=0.58 and abs(aq['rotation_deg'])<=1.2 and abs(aq['tx_px'])<=12.0 and abs(aq['ty_px'])<=12.0 and aq['median_gray_absdiff']<=24.0); aq['accepted']=bool(accept); align_rows.append(aq)
        if accept: aligned.append(wr); valids.append(va); indices.append(idx)
    cap.release()
    if len(aligned)<7: raise RuntimeError(f'insufficient aligned temporal LAR samples: {len(aligned)}')
    atlas,atlas_valid,consensus,obs,provenance=temporal_medoid(aligned,valids,indices,geom)
    new_fill=atlas_valid&geom&(~strict_floor[REF]); clean=exact[REF].copy(); clean[new_fill]=atlas[new_fill]; clean_vis=strict_floor[REF]|new_fill
    cv2.imwrite(str(args.out/'lar_temporal_clean_plate.png'),clean); cv2.imwrite(str(args.out/'lar_temporal_new_floor_pixels.png'),new_fill.astype(np.uint8)*255); cv2.imwrite(str(args.out/'lar_temporal_consensus.png'),np.clip(consensus*20,0,255).astype(np.uint8))
    pv=np.zeros((H,W),np.uint8); ok=provenance>=0
    if np.any(ok):
        pmin,pmax=int(provenance[ok].min()),int(provenance[ok].max()); pv[ok]=np.clip(20+220*(provenance[ok]-pmin)/max(1,pmax-pmin),0,255).astype(np.uint8)
    cv2.imwrite(str(args.out/'lar_temporal_provenance.png'),pv)
    base_sources={}; clean_sources={}
    for lab in ORDER:
        C,R,K=cams[lab]; base_sources[lab]={'image':exact[lab],'C':C,'R':R,'K':K,'floor_vis':strict_floor[lab],'board_vis':np.zeros((H,W),bool)}; clean_sources[lab]=dict(base_sources[lab])
    clean_sources[REF]={**clean_sources[REF],'image':clean,'floor_vis':clean_vis}
    angles=[0,5,10,15,20,25]; baseline_rows=background_render(cams,base_sources,angles,args.out,'baseline_floor'); clean_rows=background_render(cams,clean_sources,angles,args.out,'clean_floor')
    exact_hole_target=geom&strict_dyn[REF]&(~strict_floor[REF]); target_n=int(exact_hole_target.sum()); target_filled=int(np.sum(exact_hole_target&new_fill)); target_frac=float(target_filled/max(1,target_n)); mean_base=float(np.mean([r['coverage'] for r in baseline_rows])); mean_clean=float(np.mean([r['coverage'] for r in clean_rows])); hole_base=max(r['largest_hole_pixels'] for r in baseline_rows); hole_clean=max(r['largest_hole_pixels'] for r in clean_rows); hole_ratio=float(hole_clean/max(1,hole_base))
    qa={'schema_version':16,'game_id':'0022500301','event_id':489,'native_dimensions':[W,H],'native_only':True,'method':'same-camera temporal real-pixel medoid clean plate on fixed Left Above Rim camera; exact v15 floor pixels preserved; no generated RGB/inpainting/upscale','lar_clip':str(clip),'clip_fps':fps,'clip_frame_count':frame_count,'v11_expected_anchor':expected,'decoded_anchor':int(anchor),'anchor_candidates':anchor_rows,'temporal_alignment':align_rows,'accepted_temporal_frames':[int(x) for x in indices],'accepted_temporal_count':len(indices),'floor_geometry_pixels':int(geom.sum()),'v15_lar_strict_floor_pixels':int(strict_floor[REF].sum()),'new_temporal_floor_pixels':int(new_fill.sum()),'exact_dynamic_floor_hole_pixels':target_n,'exact_dynamic_floor_hole_filled_pixels':target_filled,'exact_dynamic_floor_hole_fill_fraction':target_frac,'baseline_virtual_floor':baseline_rows,'clean_virtual_floor':clean_rows,'mean_virtual_floor_coverage_baseline':mean_base,'mean_virtual_floor_coverage_clean':mean_clean,'mean_virtual_floor_coverage_gain':mean_clean-mean_base,'largest_virtual_floor_hole_baseline':int(hole_base),'largest_virtual_floor_hole_clean':int(hole_clean),'largest_hole_ratio':hole_ratio}
    qa['gates']={'native_dimensions_pass':bool(W==960 and H==540),'temporal_sample_count_pass':len(indices)>=7,'dynamic_floor_hole_fill_pass':target_frac>=0.55,'new_real_floor_support_pass':int(new_fill.sum())>=3000,'virtual_coverage_improves_pass':(mean_clean-mean_base)>=0.008,'largest_hole_not_worse_pass':hole_clean<=hole_base}; qa['gates']['numeric_pass']=all(qa['gates'].values()); (args.out/'temporal_clean_plate_v16_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps({'status':'V16_TEMPORAL_CLEAN_PLATE_RENDERED','accepted_temporal_count':len(indices),'new_temporal_floor_pixels':int(new_fill.sum()),'dynamic_hole_fill_fraction':target_frac,'coverage_gain':mean_clean-mean_base,'largest_hole_ratio':hole_ratio,'gates':qa['gates']},indent=2),flush=True)

if __name__=='__main__': main()
