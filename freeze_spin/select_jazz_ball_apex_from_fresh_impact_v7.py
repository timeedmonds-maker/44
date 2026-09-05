from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from select_jazz_predunk_ball_apex_v4 import FPS, label_from_name, read_frames, detect_ball_batches, add_rim_and_hybrid_candidates, best_track
from select_jazz_ball_apex_multiview_v3 import orange_components, nearest_rim


def choose_apex(track:list[dict], impact_local:float):
    if len(track)<12: return None,{"reason":f"track too short: {len(track)}"}
    sem=[q for q in track if q['source']=='maskrcnn']
    if len(sem)<5: return None,{"reason":f"only {len(sem)} semantic observations"}
    track=sorted(track,key=lambda q:q['time'])
    t=np.asarray([float(q['time']) for q in track]); h=np.asarray([float(q['height']) for q in track])
    grid=np.arange(t.min(),t.max()+0.25/FPS,1.0/FPS); hg=np.interp(grid,t,h)
    win=min(9,len(grid) if len(grid)%2 else len(grid)-1)
    if win<5: return None,{"reason":"insufficient grid"}
    sm=savgol_filter(hg,win,2,mode='interp')

    # Physical rim crossing should occur close to the fresh 12-camera dunk-impact transient.
    crossings=[]
    for j in range(1,len(grid)):
        if sm[j]<=4.0 and sm[j-1]>4.0:
            premax=float(np.max(sm[max(0,j-16):j]))
            if premax>=18.0:
                dt=float(grid[j]-impact_local)
                if -0.35<=dt<=0.16:
                    crossings.append((abs(dt),j,dt,premax))
    if not crossings:
        return None,{"reason":"no descending rim crossing close to fresh audio impact","impact_local":impact_local,
                    "height_min_px":float(np.min(sm)),"height_max_px":float(np.max(sm))}
    crossings.sort(key=lambda x:x[0]); _,cross_idx,cross_vs_impact,_=crossings[0]
    crossing_t=float(grid[cross_idx])

    candidates=[]
    for i in range(5,cross_idx-2):
        if not (sm[i]>=sm[i-1] and sm[i]>=sm[i+1]): continue
        before=sm[max(0,i-8):i]; after=sm[i+1:min(cross_idx+1,i+9)]
        if len(before)<4 or len(after)<3: continue
        rise=float(sm[i]-np.percentile(before,20)); fall=float(sm[i]-np.percentile(after,20))
        apex_t=float(grid[i]); apex_h=float(sm[i]); lead=crossing_t-apex_t
        if apex_h<12 or rise<3 or fall<4 or not (0.025<=lead<=0.55): continue
        sem_near=sum(1 for q in sem if abs(float(q['time'])-apex_t)<=0.105)
        pre_obs=sum(1 for q in track if apex_t-0.24<=float(q['time'])<apex_t)
        post_obs=sum(1 for q in track if apex_t<float(q['time'])<=min(crossing_t,apex_t+0.24))
        if sem_near<1 or pre_obs<3 or post_obs<2: continue
        candidates.append({'apex_local_time':apex_t,'apex_height_px':apex_h,'rise_px':rise,'fall_px':fall,
                           'rim_crossing_local_time':crossing_t,'rim_crossing_vs_impact_s':cross_vs_impact,
                           'apex_to_crossing_s':lead,'semantic_near_apex':sem_near,
                           'pre_observations':pre_obs,'post_observations':post_obs})
    if not candidates:
        pre=sm[:cross_idx]; im=int(np.argmax(pre))
        return None,{"reason":"no interior rise-top-fall maximum before fresh-impact rim crossing",
                    "impact_local":impact_local,"rim_crossing_local_time":crossing_t,
                    "highest_pre_crossing_time":float(grid[im]),"highest_pre_crossing_height_px":float(pre[im]),
                    "height_min_px":float(np.min(sm)),"height_max_px":float(np.max(sm))}
    candidates.sort(key=lambda r:(r['apex_height_px'],r['rise_px']+r['fall_px']),reverse=True)
    return candidates[0],{'candidate_count':len(candidates)}


def primary_scan(model,path:Path,offset:float,impact_ref:float,out:Path):
    impact_local=float(impact_ref+offset)
    times=list(np.arange(impact_local-1.35,impact_local+0.20,1.0/FPS))
    rows=read_frames(path,times); detect_ball_batches(model,rows); add_rim_and_hybrid_candidates(rows); track=best_track(rows)
    apex,extra=choose_apex(track,impact_local)
    diag={'label':label_from_name(path),'offset_seconds_vs_reference':float(offset),'impact_local_time':impact_local,
          'sampled_frames':len(rows),'semantic_detection_frames':int(sum(r['semantic_count']>0 for r in rows)),
          'track_observations':len(track),'track_semantic_observations':int(sum(q['source']=='maskrcnn' for q in track)),
          'track':[ {k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in q.items()} for q in track]}
    if apex is None: return {**diag,'passed':False,'failure':extra}
    apex_ref=float(apex['apex_local_time']-offset); crossing_ref=float(apex['rim_crossing_local_time']-offset)
    if not (apex_ref<crossing_ref<=impact_ref+0.16):
        return {**diag,'passed':False,'failure':{'reason':'physical ordering vs fresh impact failed','apex_ref':apex_ref,'crossing_ref':crossing_ref,'impact_ref':impact_ref}}
    strip=[]
    for k in range(-5,6):
        tt=apex['apex_local_time']+k/FPS; row=min(rows,key=lambda r:abs(r['time']-tt)); im=row['image'].copy()
        cv2.putText(im,f"Right Slash apex {k:+d}f",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA); strip.append(im)
    cv2.imwrite(str(out/'Right_Slash_fresh_impact_apex_11frame_strip.png'),np.hstack(strip))
    return {**diag,'passed':True,**extra,**apex,'apex_reference_time':apex_ref,'rim_crossing_reference_time':crossing_ref}


def confirm(model,path:Path,offset:float,apex_ref:float,out:Path):
    local=float(apex_ref+offset); times=[local+k/FPS for k in (-2,-1,0,1,2)]
    rows=read_frames(path,times); detect_ball_batches(model,rows,batch=5); obs=[]; strip=[]
    for k,r in zip((-2,-1,0,1,2),rows):
        _,rims=orange_components(r['image']); rim=nearest_rim(rims,r['image'].shape[1],r['image'].shape[0]); im=r['image'].copy(); hits=[]
        if rim is not None:
            for b in r.get('semantic_balls',[]):
                dx=abs(float(b['cx'])-float(rim['cx'])); dy=float(b['cy'])-float(rim['cy'])
                if dx<=240 and -260<=dy<=80:
                    hits.append({'score':float(b['score']),'cx':float(b['cx']),'cy':float(b['cy']),'dx_rim':dx,'dy_rim':dy})
                    cv2.circle(im,(int(round(b['cx'])),int(round(b['cy']))),13,(255,255,255),2)
        if hits: obs.append({'frame_offset':k,'time':float(r['time']),'hits':hits})
        cv2.putText(im,f"{label_from_name(path)} {k:+d}f",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),2,cv2.LINE_AA); strip.append(im)
    if strip: cv2.imwrite(str(out/f"{label_from_name(path).replace(' ','_')}_fresh_apex_confirm.png"),np.hstack(strip))
    best_dy=min((h['dy_rim'] for o in obs for h in o['hits']),default=999.0)
    return {'label':label_from_name(path),'mapped_local_time':local,'observation_frames':len(obs),'best_dy_rim':float(best_dy),'observations':obs,'passed':bool(obs and best_dy<=60)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--clips',type=Path,required=True); ap.add_argument('--sync',type=Path,required=True); ap.add_argument('--impact-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.load(open(args.sync)); impact=json.load(open(args.impact_json)); impact_ref=float(impact['estimated_dunk_impact_reference_time'])
    if impact.get('confidence')=='low': raise RuntimeError(f"Fresh impact confidence is low: {impact}")
    offsets={r['label']:float(r['offset_seconds_vs_reference']) for r in sync['angles']}; files={label_from_name(p):p for p in args.clips.glob('*_489_*_SOURCE.mp4')}
    primary='Right Slash'; confirms=['Right HandHeld','Left Slash','High Tight','Right Above Rim','Left Above Rim']; missing=[x for x in [primary]+confirms if x not in files]
    if missing: raise RuntimeError(f'Missing views {missing}')
    torch.set_num_threads(max(1,min(4,torch.get_num_threads()))); model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    p=primary_scan(model,files[primary],offsets[primary],impact_ref,args.out); print(json.dumps({k:v for k,v in p.items() if k!='track'},indent=2),flush=True)
    if not p.get('passed'): result={'passed':False,'reason':'fresh-impact Right Slash physical apex failed','fresh_impact':impact,'primary':p}
    else:
        apex_ref=float(p['apex_reference_time']); crossing_ref=float(p['rim_crossing_reference_time']); conf=[confirm(model,files[x],offsets[x],apex_ref,args.out) for x in confirms]; good=[r for r in conf if r['passed']]
        result={'passed':bool(len(good)>=2 and apex_ref<crossing_ref<=impact_ref+0.16),
                'method':'fresh 12-camera audio impact anchor + semantic Right Slash basketball trajectory + highest pre-crossing point + synchronized semantic confirmations',
                'fresh_impact_reference_time':impact_ref,'fresh_impact_confidence':impact.get('confidence'),
                'fresh_impact_supporting_views':impact.get('supporting_views',[]),
                'apex_reference_time':apex_ref,'rim_crossing_reference_time':crossing_ref,
                'apex_to_crossing_s':crossing_ref-apex_ref,'apex_to_audio_impact_s':impact_ref-apex_ref,
                'primary':p,'confirmations':conf,'confirmation_count':len(good),'confirmation_labels':[r['label'] for r in good],
                'gate':{'min_confirmation_views':2,'ordering':'apex < descending rim crossing close to fresh current-run audio impact'},
                'policy':'No stale cross-run timestamp is used to select the freeze. Post-contact frames are evidence only.'}
    (args.out/'ball_apex_from_fresh_impact_v7.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in result.items() if k not in ('primary','confirmations')},indent=2),flush=True)
    if not result.get('passed'): raise SystemExit('Fresh-impact physical apex gate failed')

if __name__=='__main__': main()
