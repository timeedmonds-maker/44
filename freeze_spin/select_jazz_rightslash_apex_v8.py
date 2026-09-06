from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

from select_jazz_predunk_ball_apex_v4 import (
    FPS, label_from_name, read_frames, detect_ball_batches,
    add_rim_and_hybrid_candidates, best_track,
)


def choose_apex(track: list[dict], impact_local: float):
    if len(track) < 9:
        return None, {"reason": f"track too short: {len(track)}"}
    sem = [q for q in track if q["source"] == "maskrcnn"]
    if len(sem) < 4:
        return None, {"reason": f"only {len(sem)} semantic ball observations"}

    track = sorted(track, key=lambda q: q["time"])
    # The ball path must span the actual dunk phase, not an unrelated early orange object.
    if track[-1]["time"] < impact_local - 0.18:
        return None, {"reason": "track ends too early relative to fresh dunk impact",
                      "track_end": float(track[-1]["time"]), "impact_local": impact_local}
    if track[0]["time"] > impact_local - 0.30:
        return None, {"reason": "track starts too late to prove rise into apex",
                      "track_start": float(track[0]["time"]), "impact_local": impact_local}

    t = np.asarray([float(q["time"]) for q in track])
    h = np.asarray([float(q["height"]) for q in track])
    grid = np.arange(t.min(), t.max() + 0.25/FPS, 1.0/FPS)
    sm = savgol_filter(np.interp(grid, t, h), min(7, len(grid) if len(grid)%2 else len(grid)-1), 2, mode="interp")

    # Descending rim-plane crossing must be close to the fresh 12-camera impact transient.
    crossings=[]
    for j in range(1, len(grid)):
        if sm[j] <= 4.0 and sm[j-1] > 4.0:
            dt=float(grid[j]-impact_local)
            if -0.25 <= dt <= 0.15 and float(np.max(sm[max(0,j-12):j])) >= 15.0:
                crossings.append((abs(dt), j, dt))
    if not crossings:
        return None, {"reason": "no descending rim crossing in fresh dunk phase",
                      "impact_local": impact_local, "track_start": float(t.min()), "track_end": float(t.max()),
                      "height_min_px": float(np.min(sm)), "height_max_px": float(np.max(sm))}
    crossings.sort(key=lambda x:x[0]); _, cross_i, cross_dt = crossings[0]
    crossing=float(grid[cross_i])

    candidates=[]
    for i in range(3, cross_i-2):
        if not (sm[i] >= sm[i-1] and sm[i] >= sm[i+1]):
            continue
        apex=float(grid[i]); lead=crossing-apex
        if not (0.025 <= lead <= 0.48):
            continue
        # Strong temporal prior from the current impact, not a stale cross-run timestamp.
        if not (impact_local-0.55 <= apex <= impact_local-0.025):
            continue
        before=sm[max(0,i-5):i]; after=sm[i+1:min(cross_i+1,i+6)]
        if len(before)<3 or len(after)<2: continue
        rise=float(sm[i]-np.percentile(before,20)); fall=float(sm[i]-np.percentile(after,20)); height=float(sm[i])
        if height < 10 or rise < 2.5 or fall < 3.5:
            continue
        sem_near=sum(1 for q in sem if abs(float(q["time"])-apex)<=0.105)
        pre_obs=sum(1 for q in track if apex-0.18<=float(q["time"])<apex)
        post_obs=sum(1 for q in track if apex<float(q["time"])<=min(crossing,apex+0.18))
        if sem_near<1 or pre_obs<2 or post_obs<2:
            continue
        candidates.append({"apex_local_time":apex,"apex_height_px":height,"rise_px":rise,"fall_px":fall,
                           "rim_crossing_local_time":crossing,"rim_crossing_vs_impact_s":cross_dt,
                           "apex_to_crossing_s":lead,"apex_to_impact_s":impact_local-apex,
                           "semantic_near_apex":sem_near,"pre_observations":pre_obs,"post_observations":post_obs})
    if not candidates:
        pre=sm[:cross_i]; im=int(np.argmax(pre))
        return None,{"reason":"no semantically anchored interior apex before fresh-phase rim crossing",
                    "highest_pre_crossing_time":float(grid[im]),"highest_pre_crossing_height_px":float(pre[im]),
                    "crossing_local_time":crossing,"impact_local":impact_local}
    candidates.sort(key=lambda r:(r["apex_height_px"], r["rise_px"]+r["fall_px"]), reverse=True)
    return candidates[0], {"candidate_count":len(candidates)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",type=Path,required=True); ap.add_argument("--sync",type=Path,required=True)
    ap.add_argument("--impact-json",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.load(open(args.sync)); impact=json.load(open(args.impact_json))
    if impact.get("confidence")=="low": raise RuntimeError(f"Fresh impact confidence low: {impact}")
    impact_ref=float(impact["estimated_dunk_impact_reference_time"])
    offsets={r["label"]:float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files={label_from_name(p):p for p in args.clips.glob("*_489_*_SOURCE.mp4")}
    label="Right Slash"
    if label not in files: raise RuntimeError("Right Slash clip missing")
    impact_local=impact_ref+offsets[label]

    # Narrow current-run dunk phase: prevents unrelated long-lived orange objects from winning the track.
    times=list(np.arange(impact_local-0.72, impact_local+0.10, 1.0/FPS))
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    rows=read_frames(files[label],times); detect_ball_batches(model,rows); add_rim_and_hybrid_candidates(rows); track=best_track(rows)
    apex,extra=choose_apex(track,impact_local)
    diag={"label":label,"fresh_impact_reference_time":impact_ref,"offset_seconds_vs_reference":offsets[label],
          "fresh_impact_local_time":impact_local,"sampled_frames":len(rows),
          "semantic_detection_frames":int(sum(r["semantic_count"]>0 for r in rows)),
          "track_observations":len(track),"track_semantic_observations":int(sum(q["source"]=="maskrcnn" for q in track)),
          "track":[{k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in q.items()} for q in track]}
    if apex is None:
        result={"passed":False,"reason":"narrow fresh-impact Right Slash apex failed","diagnostics":diag,"failure":extra}
    else:
        apex_ref=float(apex["apex_local_time"]-offsets[label]); crossing_ref=float(apex["rim_crossing_local_time"]-offsets[label])
        result={"passed":True,"method":"narrow fresh-impact Right Slash semantic basketball track + first descending rim crossing",
                "apex_reference_time":apex_ref,"apex_right_slash_local_time":float(apex["apex_local_time"]),
                "rim_crossing_reference_time":crossing_ref,"fresh_impact_reference_time":impact_ref,
                **apex,"diagnostics":diag,
                "policy":"Right Slash is the visually height-sensitive freeze reference. No audio-mapped In Arena frame is allowed to define the frozen player state."}
        strip=[]
        for k in range(-5,6):
            tt=apex["apex_local_time"]+k/FPS; row=min(rows,key=lambda r:abs(r["time"]-tt)); im=row["image"].copy()
            cv2.putText(im,f"Right Slash apex {k:+d}f t={row['time']:.3f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA); strip.append(im)
        cv2.imwrite(str(args.out/"Right_Slash_verified_apex_11frame_strip.png"),np.hstack(strip))
    (args.out/"rightslash_apex_v8.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="diagnostics"},indent=2),flush=True)
    if not result.get("passed"): raise SystemExit("Narrow fresh-impact Right Slash apex gate failed")

if __name__=="__main__": main()
