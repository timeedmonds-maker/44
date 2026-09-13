#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--analysis',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args()
    j=json.loads(args.analysis.read_text())
    args.out_dir.mkdir(parents=True,exist_ok=True)
    fps=float((j.get('video') or {}).get('fps') or 30.0)
    n=int(j.get('frames_processed') or 0)
    rows=[]; per_frame=defaultdict(lambda:{'team_0':0,'team_1':0,'other':0})
    role_counts=Counter(); track_cov=[]
    for t in j.get('tracks') or []:
        role=str(t.get('role') or 'unknown'); role_counts[role]+=1
        sm=t.get('court_trajectory_smoothed_cm') or []
        good=0
        for r in sm:
            if not isinstance(r,list) or len(r)<5: continue
            f,x,y,status,segment=r[:5]
            try: f=int(f); x=float(x); y=float(y); status=int(status)
            except Exception: continue
            if not (math.isfinite(x) and math.isfinite(y)): continue
            good+=1
            if role in ('team_0','team_1'): per_frame[f][role]+=1
            else: per_frame[f]['other']+=1
            rows.append({'frame_index':f,'time_s':f/fps,'track_id':t.get('track_id'),'entity_id':t.get('entity_id'),'role':role,'role_confidence':t.get('role_confidence'),'player_id':t.get('player_id'),'player_name':t.get('player_name'),'player_confidence':t.get('player_confidence'),'x_cm':x,'y_cm':y,'x_ft':x/30.48,'y_ft':y/30.48,'imputed':bool(status),'segment':segment})
        track_cov.append({'track_id':t.get('track_id'),'role':role,'n_track_frames':int(t.get('n_frames') or 0),'n_smoothed_position_rows':good})
    out_csv=args.out_dir/'anonymous_positions.csv'
    fields=['frame_index','time_s','track_id','entity_id','role','role_confidence','player_id','player_name','player_confidence','x_cm','y_cm','x_ft','y_ft','imputed','segment']
    with out_csv.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    visible10=sum(1 for f,v in per_frame.items() if v['team_0']+v['team_1']>=10)
    visible8=sum(1 for f,v in per_frame.items() if v['team_0']+v['team_1']>=8)
    both5=sum(1 for f,v in per_frame.items() if v['team_0']>=5 and v['team_1']>=5)
    q={
      'frames_processed':n,'fps':fps,'runtime_s':j.get('runtime_s'),
      'calibration_coverage':j.get('calibration_coverage'),'calibration_sources':j.get('calibration_sources'),
      'ball_coverage':j.get('ball_coverage'),'track_count':len(j.get('tracks') or []),'role_counts':dict(role_counts),
      'position_rows':len(rows),'frames_with_any_position':len(per_frame),
      'frames_with_8plus_team_players_positioned':visible8,
      'frames_with_10plus_team_players_positioned':visible10,
      'frames_with_5v5_positioned':both5,
      'fraction_frames_8plus_team_players_positioned':visible8/n if n else 0,
      'fraction_frames_10plus_team_players_positioned':visible10/n if n else 0,
      'fraction_frames_5v5_positioned':both5/n if n else 0,
      'possession_runs_count':len(j.get('possession_runs') or []),
      'events_count':len(j.get('events') or []),
      'player_assignment':j.get('player_assignment'),
      'gate':'geometry-only proof; does not claim KD identity or double-team labels',
      'pass_geometry_for_double_team_prototype': bool((j.get('calibration_coverage') or 0)>=0.5 and visible8>=max(15,int(n*0.10))),
      'track_coverage':track_cov,
    }
    (args.out_dir/'geometry_qa.json').write_text(json.dumps(q,indent=2))
    print(json.dumps({k:v for k,v in q.items() if k!='track_coverage'},indent=2))
if __name__=='__main__': main()
