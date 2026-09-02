from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from select_jazz_ball_apex_v1 import (
    orange_components,
    nearest_rim,
    dedupe_candidates,
    build_gap_tolerant_track,
    robust_quadratic_apex,
)


def label_from_name(path: Path) -> str:
    m = re.search(r"_489_(.+)_SOURCE\.mp4$", path.name)
    return m.group(1).replace("_", " ") if m else path.stem


def scan_view(path: Path, impact_local: float, fps: float) -> tuple[list[dict], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    rows=[]
    # Wide enough to observe the rise and fall, still constrained to the dunk action.
    times=np.arange(impact_local-0.52, impact_local+0.071, 1.0/fps)
    for i,t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t)*1000.0)
        ok,frame=cap.read()
        if not ok:
            continue
        compact,rims=orange_components(frame)
        rim=nearest_rim(rims,frame.shape[1],frame.shape[0])
        if rim is None:
            continue
        # build_gap_tolerant_track expects the normalized candidate schema used by
        # the v1 detector fusion path, including detector_score/source. For this
        # lightweight multiview pass we intentionally use color-only candidates,
        # but normalize them through the same helper rather than weakening gates.
        candidates=dedupe_candidates(compact, [])
        rows.append({
            "frame_index":i,
            "time":float(t),
            "rim":rim,
            "candidates":candidates,
        })
    cap.release()
    if len(rows)<10:
        return [],{"reason":f"only {len(rows)} usable frames"}
    track,diag=build_gap_tolerant_track(rows,max_gap=4)
    if len(track)<5:
        return [],diag
    try:
        fit,inliers=robust_quadratic_apex(track,fps)
    except Exception as e:
        return [],{**diag,"reason":str(e)}
    return track,{**diag,**fit}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clips",type=Path,required=True)
    ap.add_argument("--sync",type=Path,required=True)
    ap.add_argument("--impact-json",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--fps",type=float,default=29.97003)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    sync=json.load(open(args.sync))
    impact=json.load(open(args.impact_json))
    impact_ref=float(impact["estimated_dunk_impact_reference_time"])
    offsets={r["label"]:float(r["offset_seconds_vs_reference"]) for r in sync["angles"]}
    files={label_from_name(p):p for p in args.clips.glob("*_489_*_SOURCE.mp4")}

    wanted=["Left Above Rim","In Arena","Play by Play","Left Slash","Right Slash","High Tight","Right HandHeld","Right Above Rim"]
    results=[]
    for label in wanted:
        if label not in files or label not in offsets:
            continue
        local_impact=impact_ref+offsets[label]
        track,diag=scan_view(files[label],local_impact,args.fps)
        row={"label":label,"offset_seconds_vs_reference":offsets[label],"diagnostics":diag}
        if track and "apex_time_fit" in diag:
            local=float(diag["apex_time_fit"])
            row["apex_local_time"]=local
            row["apex_reference_time"]=local-offsets[label]
            row["fit_rmse_px"]=float(diag["fit_rmse_px"])
            row["observations"]=len(track)
            row["track"]=track
            row["passed_single_view"]=True
        else:
            row["passed_single_view"]=False
        results.append(row)

    good=[r for r in results if r.get("passed_single_view")]
    if len(good)<3:
        out={"passed":False,"reason":f"only {len(good)} cameras passed independent apex fitting","cameras":results}
        (args.out/"multiview_apex_selection_v2.json").write_text(json.dumps(out,indent=2))
        raise RuntimeError(out["reason"])

    vals=np.array([r["apex_reference_time"] for r in good],float)
    med=float(np.median(vals)); dev=np.abs(vals-med)
    # Reject outlier cameras before consensus.
    keep=dev<=0.055
    consensus=[r for r,k in zip(good,keep) if k]
    if len(consensus)<3:
        out={"passed":False,"reason":f"only {len(consensus)} cameras agree within 55 ms of median","raw_median_reference_time":med,"cameras":results}
        (args.out/"multiview_apex_selection_v2.json").write_text(json.dumps(out,indent=2))
        raise RuntimeError(out["reason"])

    vals2=np.array([r["apex_reference_time"] for r in consensus],float)
    apex=float(np.median(vals2)); spread=float(vals2.max()-vals2.min()); mad=float(np.median(np.abs(vals2-apex)))
    passed=(spread<=0.080 and mad<=0.035)
    out={
        "passed":bool(passed),
        "method":"independent moving-orange-ball trajectory fits in synchronized cameras, then robust reference-time consensus",
        "apex_reference_time":apex,
        "impact_reference_time":impact_ref,
        "apex_seconds_before_impact":float(impact_ref-apex),
        "consensus_camera_count":len(consensus),
        "consensus_labels":[r["label"] for r in consensus],
        "consensus_range_seconds":spread,
        "consensus_mad_seconds":mad,
        "single_view_pass_count":len(good),
        "cameras":results,
        "gate":{"min_consensus_cameras":3,"per_camera_median_window_s":0.055,"max_consensus_range_s":0.080,"max_consensus_mad_s":0.035},
    }
    (args.out/"multiview_apex_selection_v2.json").write_text(json.dumps(out,indent=2))
    if not passed:
        raise RuntimeError(f"Cross-camera apex consensus failed: spread={spread:.4f}s mad={mad:.4f}s")
    print(f"MULTIVIEW_APEX_PASS t_ref={apex:.5f}s before_impact={impact_ref-apex:.5f}s cameras={len(consensus)} spread={spread:.5f}s")

if __name__=="__main__":
    main()
