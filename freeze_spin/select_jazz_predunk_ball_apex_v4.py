from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor

from select_jazz_ball_apex_multiview_v3 import orange_components, nearest_rim

PERSON_CLASS = 1
BALL_CLASS = 37
FPS = 29.97003


def label_from_name(path: Path) -> str:
    m = re.search(r"_489_(.+)_SOURCE\.mp4$", path.name)
    return m.group(1).replace("_", " ") if m else path.stem


def read_frames(path: Path, times: list[float]):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    rows = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ok, im = cap.read()
        if not ok:
            continue
        rows.append({"time": float(t), "image": im})
    cap.release()
    return rows


def detect_ball_batches(model, rows: list[dict], batch: int = 6):
    for s in range(0, len(rows), batch):
        part = rows[s:s+batch]
        tensors = [to_tensor(cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB)) for r in part]
        with torch.inference_mode():
            preds = model(tensors)
        for r, p in zip(part, preds):
            scores = p["scores"].cpu().numpy(); labels = p["labels"].cpu().numpy(); boxes = p["boxes"].cpu().numpy()
            balls = []
            for sc, lab, box in zip(scores, labels, boxes):
                if int(lab) != BALL_CLASS or float(sc) < 0.045:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                w = x2-x1; h = y2-y1
                if w < 4 or h < 4 or w > 100 or h > 100:
                    continue
                balls.append({"score": float(sc), "cx": (x1+x2)/2, "cy": (y1+y2)/2,
                              "box": [x1,y1,x2,y2], "w": w, "h": h})
            r["semantic_balls"] = balls


def add_rim_and_hybrid_candidates(rows: list[dict]):
    usable_rims = []
    for r in rows:
        compact, rims = orange_components(r["image"])
        rim = nearest_rim(rims, r["image"].shape[1], r["image"].shape[0])
        r["rim_raw"] = rim
        r["orange_compact"] = compact
        if rim is not None:
            usable_rims.append((rim["cx"], rim["cy"]))
    if len(usable_rims) < max(5, len(rows)//2):
        raise RuntimeError(f"Insufficient rim detections: {len(usable_rims)}/{len(rows)}")
    rcx = float(np.median([x for x,_ in usable_rims])); rcy = float(np.median([y for _,y in usable_rims]))
    for r in rows:
        rim = r["rim_raw"] or {"cx": rcx, "cy": rcy}
        # Camera motion is small in this short window; robust median avoids rim-colour jitter.
        r["rim"] = {"cx": rcx, "cy": rcy}
        semantic = []
        for b in r.get("semantic_balls", []):
            dx = abs(b["cx"]-rcx); dy = b["cy"]-rcy
            if dx <= 250 and -260 <= dy <= 150:
                q = dict(b)
                q["source"] = "maskrcnn"
                q["base_score"] = 4.0 + 4.0*float(b["score"])
                semantic.append(q)
        hybrid = list(semantic)
        # Orange candidates are allowed only as continuity support, not as an unseeded apex proof.
        for c in r["orange_compact"]:
            dx = abs(float(c["cx"])-rcx); dy = float(c["cy"])-rcy
            if dx > 250 or dy < -260 or dy > 150:
                continue
            q = {"cx": float(c["cx"]), "cy": float(c["cy"]), "w": float(c["w"]), "h": float(c["h"]),
                 "score": float(c.get("ball_color_score", 0.0)), "source": "orange",
                 "base_score": float(c.get("ball_color_score",0.0))}
            hybrid.append(q)
        r["candidates"] = hybrid
        r["semantic_count"] = len(semantic)


def best_track(rows: list[dict]):
    nodes = []
    for fi, r in enumerate(rows):
        rim = r["rim"]
        for ci, c in enumerate(r["candidates"]):
            dx = abs(c["cx"]-rim["cx"]); dy = c["cy"]-rim["cy"]
            score = float(c["base_score"]) - 0.002*dx - 0.001*abs(dy+35.0)
            nodes.append({"fi": fi, "ci": ci, "time": r["time"], "cx": c["cx"], "cy": c["cy"],
                          "height": rim["cy"]-c["cy"], "source": c["source"], "local": score})
    if not nodes:
        return []
    nodes.sort(key=lambda n:(n["fi"], n["ci"]))
    n = len(nodes)
    dp = np.full(n, -1e9, float); prev = np.full(n, -1, int); sem = np.zeros(n, int); length = np.ones(n, int)
    for i,a in enumerate(nodes):
        dp[i] = a["local"] + (2.0 if a["source"]=="maskrcnn" else 0.0)
        sem[i] = 1 if a["source"]=="maskrcnn" else 0
        for j in range(i-1,-1,-1):
            b = nodes[j]; gap = a["fi"]-b["fi"]
            if gap <= 0: continue
            if gap > 3:
                if b["fi"] < a["fi"]-3: break
                continue
            dist = math.hypot(a["cx"]-b["cx"], a["cy"]-b["cy"])
            if dist > 38 + 26*gap: continue
            val = dp[j] + a["local"] + 1.0 - 0.16*dist - 0.7*(gap-1)
            sem2 = sem[j] + (1 if a["source"]=="maskrcnn" else 0)
            len2 = length[j] + 1
            # Prefer semantically anchored paths when scores are close.
            val += 0.45*sem2 + 0.06*len2
            if val > dp[i]:
                dp[i]=val; prev[i]=j; sem[i]=sem2; length[i]=len2
    order = np.argsort(dp + 0.6*sem + 0.08*length)[::-1]
    for end in order:
        ids=[]; cur=int(end)
        while cur>=0:
            ids.append(cur); cur=int(prev[cur])
        ids.reverse(); path=[nodes[k] for k in ids]
        if len(path) < 7: continue
        if sum(1 for q in path if q["source"]=="maskrcnn") < 3: continue
        if path[-1]["fi"]-path[0]["fi"] < 8: continue
        if math.hypot(max(q["cx"] for q in path)-min(q["cx"] for q in path),
                      max(q["cy"] for q in path)-min(q["cy"] for q in path)) < 18: continue
        return path
    return []


def choose_precontact_apex(track: list[dict], old_local: float):
    if not track:
        return None, {"reason":"no semantically anchored ball track"}
    track = sorted(track, key=lambda q:q["time"])
    t=np.array([q["time"] for q in track]); h=np.array([q["height"] for q in track])
    # Hard visual-state correction from user QA: the old accepted frame is post-dunk/hanging.
    # Therefore every candidate at/after that timestamp is excluded from the freeze proof.
    keep=t < old_local - 0.04
    if keep.sum() < 6:
        return None,{"reason":"too few tracked ball observations before rejected hanging frame"}
    t2=t[keep]; h2=h[keep]
    # Find the first descending approach/crossing of the rim after the highest pre-contact observation.
    imax=int(np.argmax(h2))
    if imax < 1 or imax >= len(t2)-2:
        return None,{"reason":"highest pre-rejected observation is not interior","imax":imax,"n":len(t2)}
    apex_t=float(t2[imax]); apex_h=float(h2[imax])
    before=h2[max(0,imax-3):imax]; after=h2[imax+1:min(len(h2),imax+5)]
    rise=apex_h-float(np.median(before)) if len(before) else 0.0
    fall=apex_h-float(np.median(after)) if len(after) else 0.0
    # Dunk apex must be above rim in this height-sensitive projection and then move downward.
    if apex_h < 8.0 or rise < 2.0 or fall < 3.0:
        return None,{"reason":"candidate lacks clear pre-rim rise/top/descent","apex_height_px":apex_h,"rise_px":rise,"fall_px":fall}
    crossing=None
    for i in range(imax+1,len(h2)):
        if h2[i] <= 4.0:
            crossing=float(t2[i]); break
    if crossing is None:
        return None,{"reason":"no subsequent rim-plane approach/crossing observed before rejected hanging frame",
                    "apex_time":apex_t,"apex_height_px":apex_h}
    lead=crossing-apex_t
    if lead < 0.025 or lead > 0.45:
        return None,{"reason":f"implausible apex-to-rim-crossing lead {lead:.3f}s"}
    sem_near=sum(1 for q in track if q["source"]=="maskrcnn" and abs(q["time"]-apex_t)<=0.11)
    if sem_near < 1:
        return None,{"reason":"apex not locally anchored by semantic sports-ball detection"}
    return {"apex_local_time":apex_t,"apex_height_px":apex_h,"rise_px":rise,"fall_px":fall,
            "rim_crossing_local_time":crossing,"apex_to_crossing_s":lead,"semantic_near_apex":sem_near},{}


def scan_camera(model, path: Path, offset: float, old_ref: float, out: Path):
    old_local=old_ref+offset
    # Search only before the user-rejected hanging state.
    times=list(np.arange(old_local-0.90, old_local-0.035, 1.0/FPS))
    rows=read_frames(path,times)
    detect_ball_batches(model,rows)
    add_rim_and_hybrid_candidates(rows)
    track=best_track(rows)
    apex,fail=choose_precontact_apex(track,old_local)
    diag={"label":label_from_name(path),"offset_seconds_vs_reference":offset,"old_rejected_local_time":old_local,
          "sampled_frames":len(rows),"semantic_detection_frames":int(sum(r["semantic_count"]>0 for r in rows)),
          "track_observations":len(track),"track_semantic_observations":int(sum(q["source"]=="maskrcnn" for q in track)),
          "track":[{k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in q.items()} for q in track]}
    if apex is None:
        return {**diag,"passed":False,"failure":fail}
    apex_ref=float(apex["apex_local_time"]-offset)
    crossing_ref=float(apex["rim_crossing_local_time"]-offset)
    # Export a five-frame audit strip centered on the chosen apex.
    chosen=[]
    for dt in (-2/FPS,-1/FPS,0,1/FPS,2/FPS):
        tt=apex["apex_local_time"]+dt
        row=min(rows,key=lambda r:abs(r["time"]-tt))
        im=row["image"].copy(); cv2.putText(im,f"{label_from_name(path)}  t={row['time']:.3f}",(16,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
        chosen.append(im)
    strip=np.hstack(chosen)
    cv2.imwrite(str(out/f"{label_from_name(path).replace(' ','_')}_precontact_apex_strip.png"),strip)
    return {**diag,"passed":True,**apex,"apex_reference_time":apex_ref,"rim_crossing_reference_time":crossing_ref}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",type=Path,required=True); ap.add_argument("--sync",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--old-rejected-apex-ref",type=float,default=9.635413333166447)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    sync=json.load(open(args.sync)); offsets={r["label"]:float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files={label_from_name(p):p for p in args.clips.glob("*_489_*_SOURCE.mp4")}
    wanted=["Right Slash","Right HandHeld"]
    missing=[x for x in wanted if x not in files]
    if missing: raise RuntimeError(f"Missing height-sensitive views: {missing}")
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    model=maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,progress=True).eval()
    rows=[]
    for label in wanted:
        row=scan_camera(model,files[label],offsets[label],float(args.old_rejected_apex_ref),args.out)
        rows.append(row); print(json.dumps({k:v for k,v in row.items() if k!="track"},indent=2),flush=True)
    good=[r for r in rows if r.get("passed")]
    if len(good)<2:
        result={"passed":False,"reason":"fewer than two height-sensitive semantic ball tracks passed","cameras":rows}
    else:
        refs=[r["apex_reference_time"] for r in good]
        spread=max(refs)-min(refs); apex_ref=float(np.median(refs))
        cross_refs=[r["rim_crossing_reference_time"] for r in good]
        # Tight agreement is required; never loosen this to force a render.
        passed=spread<=0.07 and apex_ref < float(args.old_rejected_apex_ref)-0.04
        result={"passed":bool(passed),"method":"semantic sports-ball track + orange continuity support + first pre-rim-crossing apex",
                "apex_reference_time":apex_ref,"consensus_spread_seconds":spread,
                "rim_crossing_reference_time_median":float(np.median(cross_refs)),
                "old_rejected_hanging_reference_time":float(args.old_rejected_apex_ref),
                "cameras":rows,
                "gate":{"required_height_sensitive_views":2,"max_apex_spread_s":0.07,
                        "must_precede_rejected_hanging_frame_s":0.04},
                "policy":"The v15 9.635413s state was visually rejected by the user as post-dunk/hanging. It is forbidden. Freeze must be a semantically anchored basketball apex before the first observed rim-plane crossing."}
    (args.out/"predunk_ball_apex_v4.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="cameras"},indent=2),flush=True)
    if not result.get("passed"):
        raise SystemExit("Pre-contact semantic apex gate failed")


if __name__=="__main__":
    main()
