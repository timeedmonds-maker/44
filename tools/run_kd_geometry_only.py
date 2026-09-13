#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stop',type=int,default=120)
    ap.add_argument('--imgsz',type=int,default=960)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    from nbacv.util import video_meta, save_json
    from nbacv.detect import run_detection
    from nbacv.track import run_tracking, stitch_tracklets
    from nbacv.court import detect_keypoints, calibrate_video, project_point, court_position_ok
    from nbacv.camera_motion import propagate_calibration
    meta=video_meta(args.video); fps=float(meta['fps'])
    dets=run_detection(args.video,args.out/'detections.json',model_name='yolo11m.pt',imgsz=args.imgsz,stop=args.stop)
    tracks=stitch_tracklets(run_tracking(dets,fps)); save_json(tracks,args.out/'tracks.json')
    kps=detect_keypoints(args.video,stop=args.stop)
    calib=calibrate_video(kps,frame_hw=(meta['height'],meta['width']))
    calib=propagate_calibration(args.video,calib,(meta['height'],meta['width']),stop=args.stop)
    save_json({str(k):v for k,v in calib.items()},args.out/'calib.json')
    rows=[]; frame_counts=defaultdict(int)
    for t in tracks:
        tid=t['track_id']
        for f,box in zip(t['frames'],t['boxes']):
            rec=calib.get(f) or {}
            if not rec.get('H'): continue
            cm=project_point(rec['H'],(box[0]+box[2])/2,box[3])
            if not court_position_ok(cm): continue
            rows.append({'frame_index':int(f),'time_s':float(f)/fps,'track_id':tid,'x_cm':float(cm[0]),'y_cm':float(cm[1]),'x_ft':float(cm[0])/30.48,'y_ft':float(cm[1])/30.48,'calibration_source':rec.get('source','fit'),'calibration_support':rec.get('support')})
            frame_counts[int(f)]+=1
    import csv
    with (args.out/'positions_raw.csv').open('w',newline='') as fh:
        fields=['frame_index','time_s','track_id','x_cm','y_cm','x_ft','y_ft','calibration_source','calibration_support']
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
    valid=sum(1 for v in calib.values() if v.get('H'))
    fit=sum(1 for v in calib.values() if v.get('H') and v.get('source','fit')=='fit')
    prop=sum(1 for v in calib.values() if v.get('H') and v.get('source')=='propagated')
    n=args.stop
    q={'video':meta,'frames_requested':n,'tracks':len(tracks),'position_rows':len(rows),'calibration_coverage':valid/max(len(calib),1),'calibration_fit_frames':fit,'calibration_propagated_frames':prop,'frames_with_positions':len(frame_counts),'frames_with_8plus_positioned_players':sum(v>=8 for v in frame_counts.values()),'frames_with_10plus_positioned_players':sum(v>=10 for v in frame_counts.values()),'max_positioned_tracks_in_frame':max(frame_counts.values()) if frame_counts else 0,'gate':'geometry-only; no team/KD identity and no double-team labels','pass_basic_geometry':bool(valid/max(len(calib),1)>=0.5 and sum(v>=8 for v in frame_counts.values())>=10)}
    (args.out/'geometry_only_qa.json').write_text(json.dumps(q,indent=2,default=str))
    print(json.dumps(q,indent=2,default=str))
    if not q['pass_basic_geometry']: raise SystemExit('geometry gate failed')
if __name__=='__main__': main()
