#!/usr/bin/env python3
"""Screen Tracker - Kickout 3 deterministic presentation renderer.

Derivative of Screen Tracker graphics only. The canonical `screen_tracker/`
package and locked renderers are imported read-only and never modified.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'tools'))
import locked_broadcast_screen_v2 as v2
import locked_broadcast_screen_v4 as v4

TOOL_ID='SCREEN_TRACKER_KICKOUT_3'
VERSION='0.1.0-poc'


def draw_metric_label(frame,p1,p2,label):
    """High-visibility label centered over the dotted separation line."""
    x=int(round((p1[0]+p2[0])/2)); y=int(round((p1[1]+p2[1])/2))-11
    # shadowed neutral tile, matching Screen Tracker visual language
    fr=v2.draw_tile(frame,max(8,x-64),max(8,y-29),128,56,'CLOSEST DEF',label,22)
    return fr


def render(source:Path, tracks:Path, config:Path, out:Path):
    cfg=json.loads(config.read_text()); out.mkdir(parents=True,exist_ok=True)
    timing=cfg['timing']; start=float(timing['start_s']); end=float(timing['end_s']); catch=float(timing['catch_s']); freeze=float(timing.get('freeze_s',1.2))
    teams={k:v2.rgb_to_bgr(v['primary_rgb']) for k,v in cfg['teams'].items()}
    players={int(p['track_id']):p for p in cfg['players']}
    assert len(players)==4, f'Kickout 3 renders exactly four named/ringed players; got {len(players)}'
    roles=cfg['roles']; durant=int(roles['durant_track_id']); ddef=int(roles['durant_defender_track_id'])
    adams=int(roles['adams_track_id']); adef=int(roles['adams_defender_track_id'])
    assert {durant,ddef,adams,adef}==set(players)
    shot=cfg['shot']; sep=float(cfg['catch_state']['closest_defender_distance_ft'])
    shot_dist=float(shot['official_shot_distance_ft'])
    xfg=float(shot['official_event_xfg_pct']); sim=shot.get('durant_similar_xfg_pct'); sim_n=int(shot.get('durant_similar_sample_fga') or 0)

    df=pd.read_csv(tracks); interp=v2.build_interp(df,players.keys())
    cap=cv2.VideoCapture(str(source)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sf=int(round(start*fps)); ef=int(round(end*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    ring_policy=v4.resolve_ring_policy(interp,players.keys(),catch,W,H); rx=int(ring_policy['rx_px']); ry=int(ring_policy['ry_px'])
    silent=out/'kickout3_silent.mp4'; wr=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); freeze_n=int(round(freeze*fps)); catch_written=False; frame_idx=sf
    freeze_jpg=out/'catch_freeze.jpg'

    while frame_idx<ef:
        ok,orig=cap.read()
        if not ok: break
        t=frame_idx/fps; fr=orig.copy(); boxes={}
        for tid,p in players.items():
            b=v2.box_at(interp,tid,t); boxes[tid]=b
            if b is None: continue
            cx=(b[0]+b[2])/2; cy=b[3]-4; col=teams[p['team']]
            v4.draw_ring(fr,cx,cy,col,rx,ry)
            fr=v2.draw_name(fr,cx,b[1],p['label'],col,float(p.get('label_dx',0)),float(p.get('label_dy',0)))

        # Shot-quality panels remain visible around the catch-to-shot sequence.
        show_quality=(catch-0.45)<=t<=(catch+3.25)
        if show_quality:
            fr=v2.draw_tile(fr,20,20,194,66,'SHOT xFG',f'{xfg:.1f}%',27)
            if sim is not None:
                kicker=f'DURANT 25-26 SIMILAR  n={sim_n}' if sim_n else 'DURANT 25-26 SIMILAR'
                fr=v2.draw_tile(fr,20,92,250,66,kicker,f'xFG  {float(sim):.1f}%',25)

        is_catch=abs(t-catch)<=.5/fps
        if is_catch and not catch_written:
            bd=boxes.get(durant); bf=boxes.get(ddef)
            if bd is None or bf is None: raise RuntimeError('Durant/closest-defender track missing at catch frame')
            # Dotted line uses foot/floor anchors, matching the metric court-distance definition.
            p1=((bd[0]+bd[2])/2,bd[3]-4); p2=((bf[0]+bf[2])/2,bf[3]-4)
            v2.dotted_line(fr,p1,p2,radius=5,gap=13)
            fr=draw_metric_label(fr,p1,p2,f'{sep:.1f} FT')
            dp=players[durant]
            fr=v2.draw_distance_tag(fr,(bd[0]+bd[2])/2+float(dp.get('distance_dx',0)),bd[1]+float(dp.get('distance_dy',0)),f'{shot_dist:g} FT SHOT')
            cv2.imwrite(str(freeze_jpg),fr,[cv2.IMWRITE_JPEG_QUALITY,97])
            wr.write(fr)
            for _ in range(freeze_n): wr.write(fr)
            catch_written=True
        else:
            wr.write(fr)
        frame_idx+=1
    cap.release(); wr.release()
    if not catch_written: raise RuntimeError('Catch freeze was never rendered')

    native=out/f"{cfg['output_basename']}_native.mp4"
    v2.mux_with_freeze_audio(source,silent,native,start,catch,end,freeze)
    uhd=out/f"{cfg['output_basename']}_UHD.mp4"
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-i',str(native),'-vf','hqdn3d=0.6:0.6:2.0:2.0,scale=3840:2160:flags=lanczos,cas=0.22,fps=30','-c:v','libx264','-profile:v','high','-preset','medium','-crf','16','-maxrate','36M','-bufsize','72M','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(uhd)],check=True)
    qa={
      'tool_id':TOOL_ID,'version':VERSION,'parent_tool_id':'SCREEN_TRACKER','parent_version':'1.1.0',
      'protected_parent_sha':'15cde6374333c1956299c2355183549298a466db',
      'deterministic_only':True,'ai_image_generation':False,'ai_super_resolution':False,
      'rendered_player_count':4,'rendered_track_ids':sorted(players),
      'roles':roles,'catch_s':catch,'freeze_s':freeze,
      'closest_defender_distance_ft':sep,'distance_measurement':cfg['catch_state'].get('distance_measurement',{}),
      'official_shot_distance_ft':shot_dist,'official_event_xfg_pct':xfg,
      'durant_similar_xfg_pct':sim,'durant_similar_sample_fga':sim_n,
      'floor_ring_policy_v2':ring_policy,'native':str(native),'uhd':str(uhd),'catch_freeze':str(freeze_jpg)
    }
    (out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--tracks',type=Path,required=True); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    render(a.source,a.tracks,a.config,a.out)

if __name__=='__main__': main()
