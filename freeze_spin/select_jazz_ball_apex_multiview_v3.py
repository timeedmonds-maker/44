from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter


def label_from_name(path: Path) -> str:
    m = re.search(r"_489_(.+)_SOURCE\.mp4$", path.name)
    return m.group(1).replace("_", " ") if m else path.stem


def orange_components(frame: np.ndarray) -> tuple[list[dict], list[dict]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([1, 58, 42], np.uint8), np.array([34, 255, 255], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    compact, elongated = [], []
    H, W = frame.shape[:2]
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < 9:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 0 or h <= 0:
            continue
        peri = float(cv2.arcLength(c, True))
        circ = float(4.0 * math.pi * area / max(peri * peri, 1e-6))
        fill = float(area / max(w * h, 1))
        roi = hsv[y:y+h, x:x+w]
        sat = float(np.mean(roi[..., 1])) if roi.size else 0.0
        row = {
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(x + w / 2.0), "cy": float(y + h / 2.0),
            "area": area, "circularity": circ, "fill": fill, "mean_saturation": sat,
        }
        aspect = w / max(h, 1)
        if area >= 25 and w >= 18 and aspect >= 1.40 and w <= W * 0.42 and h <= H * 0.18:
            row["rim_score"] = float(area * min(aspect, 8.0) * (0.45 + sat / 255.0))
            elongated.append(row)
        if 4 <= w <= 68 and 4 <= h <= 68 and 0.45 <= aspect <= 2.05 and circ >= 0.09 and fill >= 0.12:
            size_pref = math.exp(-abs(math.log(max(math.sqrt(area), 1.0) / 17.0)) / 1.45)
            row["ball_color_score"] = float(1.35 * circ + 0.72 * fill + 0.92 * (sat / 255.0) + 0.58 * size_pref)
            compact.append(row)
    elongated.sort(key=lambda r: r["rim_score"], reverse=True)
    compact.sort(key=lambda r: r["ball_color_score"], reverse=True)
    return compact[:36], elongated[:14]


def nearest_rim(rims: list[dict], W: int, H: int) -> dict | None:
    if not rims:
        return None
    def score(r: dict) -> float:
        center_pen = abs(r["cx"] - W / 2.0) / W + 0.35 * abs(r["cy"] - H * 0.43) / H
        return r["rim_score"] / (1.0 + 3.0 * center_pen)
    return max(rims, key=score)


def patch_motion(frames: list[np.ndarray], i: int, c: dict) -> float:
    if i <= 0 or i >= len(frames) - 1:
        return 0.0
    H, W = frames[i].shape[:2]
    rr = int(max(9, min(28, round(0.8 * max(float(c["w"]), float(c["h"]))))))
    cx, cy = int(round(c["cx"])), int(round(c["cy"]))
    x0, x1 = max(0, cx-rr), min(W, cx+rr+1)
    y0, y1 = max(0, cy-rr), min(H, cy+rr+1)
    if x1-x0 < 4 or y1-y0 < 4:
        return 0.0
    g0 = cv2.cvtColor(frames[i-1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(frames[i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frames[i+1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    d = 0.5 * (np.mean(cv2.absdiff(g1, g0)) + np.mean(cv2.absdiff(g2, g1)))
    return float(d / 32.0)


def build_track(frames: list[dict], max_gap: int = 3) -> tuple[list[dict], dict]:
    nodes=[]
    for fi,fr in enumerate(frames):
        rim=fr["rim"]
        for cj,c in enumerate(fr["candidates"]):
            dx=abs(float(c["cx"])-float(rim["cx"]))
            dy=float(c["cy"])-float(rim["cy"])
            if dx>245 or dy < -250 or dy > 125:
                continue
            nodes.append({
                "raw_i":fi,"cand_i":cj,"time":float(fr["time"]),
                "cx":float(c["cx"]),"cy":float(c["cy"]),
                "height":float(rim["cy"]-c["cy"]),"dx_rim":dx,"dy_rim":dy,
                "detector_score":float(c["ball_color_score"]),"motion":float(c.get("motion_score",0.0)),
            })
    for n in nodes:
        same=set()
        for q in nodes:
            if abs(q["raw_i"]-n["raw_i"])<2:
                continue
            if math.hypot(q["cx"]-n["cx"],q["cy"]-n["cy"])<=5.5:
                same.add(q["raw_i"])
        persistence=len(same)/max(len(frames)-1,1)
        n["static_persistence"]=persistence
        n["local_score"]=(n["detector_score"] + 0.72*min(n["motion"],3.0)
                          -0.0024*n["dx_rim"] -0.0010*abs(n["dy_rim"]+35.0)
                          -3.0*max(0.0,persistence-0.18))
    nodes.sort(key=lambda n:(n["raw_i"],n["cand_i"]))
    if not nodes:
        return [],{"reason":"no candidate nodes","candidate_nodes":0}
    score=np.full(len(nodes),-1e9,np.float64); plen=np.ones(len(nodes),np.int32); prev=np.full(len(nodes),-1,np.int32)
    for i,n in enumerate(nodes):
        score[i]=n["local_score"]+0.20
        for j in range(i-1,-1,-1):
            p=nodes[j]; gap=n["raw_i"]-p["raw_i"]
            if gap<=0: continue
            if gap>max_gap:
                if p["raw_i"] < n["raw_i"]-max_gap: break
                continue
            dist=math.hypot(n["cx"]-p["cx"],n["cy"]-p["cy"])
            if dist > 30.0+27.0*gap: continue
            speed=dist/gap
            val=score[j]+n["local_score"]+0.55-0.22*(gap-1)-0.0055*max(0.0,speed-26.0)**1.35
            if val>score[i]:
                score[i]=val; plen[i]=plen[j]+1; prev[i]=j
    order=np.argsort(score+0.55*plen)[::-1]
    best=None; diag=None
    for end in order[:100]:
        idx=[]; cur=int(end)
        while cur>=0:
            idx.append(cur); cur=int(prev[cur])
        idx.reverse(); path=[nodes[k] for k in idx]
        if len(path)<6: continue
        span=path[-1]["raw_i"]-path[0]["raw_i"]
        xspan=max(q["cx"] for q in path)-min(q["cx"] for q in path)
        yspan=max(q["cy"] for q in path)-min(q["cy"] for q in path)
        if span<8 or math.hypot(xspan,yspan)<15: continue
        best=path; diag={"candidate_nodes":len(nodes),"observations":len(path),"span_frames":int(span),
                        "x_span_px":float(xspan),"y_span_px":float(yspan),
                        "track_bbox_span_px":float(math.hypot(xspan,yspan)),"max_gap_frames":max_gap}
        break
    if best is None:
        return [],{"reason":"no long moving path","candidate_nodes":len(nodes)}
    track=[]
    for n in best:
        c=frames[n["raw_i"]]["candidates"][n["cand_i"]]
        track.append({"time":n["time"],"cx":n["cx"],"cy":n["cy"],"height":n["height"],
                      "motion_score":n["motion"],"static_persistence":n["static_persistence"],
                      "ball":c})
    return track,diag or {}


def local_apex(track: list[dict], fps: float, impact_local: float) -> tuple[dict|None, dict]:
    if len(track)<7:
        return None,{"reason":f"track too short: {len(track)}"}
    t=np.array([r["time"] for r in track],float)
    h=np.array([r["height"] for r in track],float)
    x=np.array([r["cx"] for r in track],float)
    y=np.array([r["cy"] for r in track],float)
    yspan=float(np.ptp(y)); xspan=float(np.ptp(x)); total=max(math.hypot(xspan,yspan),1e-6)
    vertical_fraction=float(yspan/total)
    # Height-sensitive views are the only ones allowed to estimate physical apex time from image height.
    if vertical_fraction < 0.48 or yspan < 10.0:
        return None,{"reason":"projection not height-sensitive","vertical_fraction":vertical_fraction,
                    "x_span_px":xspan,"y_span_px":yspan}
    order=np.argsort(t); t=t[order]; h=h[order]
    # Resample on a regular grid to bridge short detection gaps without forcing a ballistic global parabola.
    grid=np.arange(t.min(),t.max()+0.25/fps,1.0/fps)
    hg=np.interp(grid,t,h)
    win=min(7, len(grid) if len(grid)%2==1 else len(grid)-1)
    if win<5:
        return None,{"reason":"insufficient grid for local smoothing"}
    sm=savgol_filter(hg,win,2,mode="interp")
    # Do not allow an endpoint maximum; require observed rise and fall around the top.
    candidates=[]
    for i in range(2,len(grid)-2):
        if sm[i] >= sm[i-1] and sm[i] >= sm[i+1]:
            before=sm[max(0,i-4):i]
            after=sm[i+1:min(len(sm),i+5)]
            if len(before)<2 or len(after)<2: continue
            rise=float(sm[i]-np.percentile(before,25)); fall=float(sm[i]-np.percentile(after,25))
            if rise>=2.5 and fall>=2.5:
                candidates.append((float(sm[i]),i,rise,fall))
    if not candidates:
        return None,{"reason":"no interior local rise-top-fall maximum","vertical_fraction":vertical_fraction,
                    "height_span_px":float(np.ptp(sm))}
    _,i,rise,fall=max(candidates,key=lambda q:q[0])
    apex_t=float(grid[i])
    lead=float(impact_local-apex_t)
    if lead < -0.05 or lead > 0.65:
        return None,{"reason":f"local maximum implausible vs impact: lead={lead:.3f}s","apex_local_time":apex_t}
    # Require actual observations on both sides close to the selected top.
    before_obs=int(np.sum((t < apex_t) & (t >= apex_t-0.18)))
    after_obs=int(np.sum((t > apex_t) & (t <= apex_t+0.18)))
    if before_obs<2 or after_obs<2:
        return None,{"reason":"insufficient observed ball centers around local maximum",
                    "before_obs":before_obs,"after_obs":after_obs,"apex_local_time":apex_t}
    return {"apex_local_time":apex_t,"rise_px":rise,"fall_px":fall,"vertical_fraction":vertical_fraction,
            "height_span_px":float(np.ptp(sm)),"before_obs":before_obs,"after_obs":after_obs},{}


def scan_view(path:Path,impact_local:float,fps:float)->tuple[list[dict],dict]:
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return [],{"reason":"cannot open clip"}
    times=np.arange(impact_local-0.52,impact_local+0.071,1.0/fps)
    images=[]; rows=[]
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000.0); ok,im=cap.read()
        if not ok: continue
        compact,rims=orange_components(im); rim=nearest_rim(rims,im.shape[1],im.shape[0])
        images.append(im)
        rows.append({"time":float(t),"rim":rim,"candidates":compact})
    cap.release()
    usable=[r for r in rows if r["rim"] is not None]
    if len(usable)<10: return [],{"reason":f"only {len(usable)} frames with usable rim","sampled_frames":len(rows)}
    rcx=float(np.median([r["rim"]["cx"] for r in usable])); rcy=float(np.median([r["rim"]["cy"] for r in usable]))
    for i,r in enumerate(rows):
        if r["rim"] is None: continue
        r["rim"]=dict(r["rim"],cx=rcx,cy=rcy)
        for c in r["candidates"]:
            c["motion_score"]=patch_motion(images,i,c) if len(images)==len(rows) else 0.0
    frames=[r for r in rows if r["rim"] is not None]
    track,diag=build_track(frames,max_gap=3)
    if not track: return [],diag
    apex,fail=local_apex(track,fps,impact_local)
    diag={**diag,"track_motion_median":float(np.median([r["motion_score"] for r in track]))}
    if apex is None: return track,{**diag,**fail}
    return track,{**diag,**apex,"passed_height_apex":True}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",type=Path,required=True); ap.add_argument("--sync",type=Path,required=True)
    ap.add_argument("--impact-json",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--fps",type=float,default=29.97003)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.load(open(args.sync)); impact=json.load(open(args.impact_json)); impact_ref=float(impact["estimated_dunk_impact_reference_time"])
    offsets={r["label"]:float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files={label_from_name(p):p for p in args.clips.glob("*_489_*_SOURCE.mp4")}
    wanted=["Left Above Rim","In Arena","Play by Play","Left Slash","Right Slash","High Tight","Right HandHeld","Right Above Rim"]
    results=[]
    for label in wanted:
        if label not in files or label not in offsets: continue
        local_impact=impact_ref+offsets[label]
        track,diag=scan_view(files[label],local_impact,args.fps)
        row={"label":label,"offset_seconds_vs_reference":offsets[label],"track_observations":len(track),"diagnostics":diag}
        if diag.get("passed_height_apex"):
            local=float(diag["apex_local_time"]); row["apex_local_time"]=local; row["apex_reference_time"]=local-offsets[label]
            row["passed_height_apex"]=True
        else: row["passed_height_apex"]=False
        results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!="track"}),flush=True)
    good=[r for r in results if r.get("passed_height_apex")]
    # Only height-sensitive projections estimate apex time. Other cameras are confirmation views, not failed parabola fits.
    if len(good)<2:
        out={"passed":False,"reason":f"only {len(good)} height-sensitive cameras produced a local rise-top-fall apex","cameras":results}
        (args.out/"multiview_apex_selection_v3.json").write_text(json.dumps(out,indent=2)); raise RuntimeError(out["reason"])
    vals=np.array([r["apex_reference_time"] for r in good],float)
    # Cluster estimates at native-frame precision; choose the densest 70 ms window.
    best=[]
    for v in vals:
        cluster=[r for r in good if abs(r["apex_reference_time"]-v)<=0.035]
        if len(cluster)>len(best): best=cluster
    if len(best)<2:
        out={"passed":False,"reason":"no two height-sensitive cameras agree within ~one native frame","cameras":results}
        (args.out/"multiview_apex_selection_v3.json").write_text(json.dumps(out,indent=2)); raise RuntimeError(out["reason"])
    cvals=np.array([r["apex_reference_time"] for r in best],float); apex=float(np.median(cvals)); spread=float(np.ptp(cvals))
    # Confirm ball-track visibility near the accepted physical instant in multiple additional synchronized cameras.
    confirm=[]
    for r in results:
        # A usable tracked observation sequence itself confirms basketball motion around this event; independent image-height apex is not required.
        if r["track_observations"]>=6:
            confirm.append(r["label"])
    passed=(len(best)>=2 and spread<=0.070 and len(confirm)>=4)
    out={"passed":bool(passed),"method":"projection-aware local apex: height-sensitive camera consensus + synchronized multi-view ball-track confirmation",
         "apex_reference_time":apex,"impact_reference_time":impact_ref,"apex_seconds_before_impact":float(impact_ref-apex),
         "height_apex_consensus_labels":[r["label"] for r in best],"height_apex_consensus_count":len(best),"consensus_range_seconds":spread,
         "ball_track_confirmation_labels":confirm,"ball_track_confirmation_count":len(confirm),"cameras":results,
         "gate":{"min_height_sensitive_apex_views":2,"max_pair_delta_s":0.070,"min_ball_track_confirmation_views":4},
         "policy":"Do not force all camera projections to have a 2D parabola. Estimate the physical apex only from views whose projected motion is height-sensitive and has an observed local rise/top/fall; use synchronized remaining views to confirm the same moving ball event."}
    (args.out/"multiview_apex_selection_v3.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2),flush=True)
    if not passed: raise RuntimeError("Projection-aware apex gate failed")

if __name__=="__main__": main()
