#!/usr/bin/env python3
"""LOCKED_BROADCAST_SCREEN_V4 — universal perspective-normalized floor rings.

V4 preserves LOCKED_BROADCAST_SCREEN_V2 graphics and deterministic rendering,
but resolves one ring size per camera angle from focal-player scale at the
screen action. The resolved ring size is held constant for the entire angle.

This prevents tight replay angles from making floor rings look smaller than the
canonical Broadcast rings. Rings never shrink below the canonical Broadcast
size. No generated imagery. No AI super-resolution.
"""
from __future__ import annotations

from pathlib import Path
import json
import subprocess

import cv2
import numpy as np
import pandas as pd

import locked_broadcast_screen_v2 as v2

TOOL_ID = 'LOCKED_BROADCAST_SCREEN_V4'
BASE_W = 960.0
BASE_H = 540.0
BASE_RX = 35.0
BASE_RY = 13.0
REFERENCE_PLAYER_HEIGHT_FRAC = 0.24
MIN_SCALE = 1.0
MAX_SCALE = 2.30


def resolve_ring_policy(interp, player_ids, reference_t, frame_w, frame_h):
    heights = []
    for tid in player_ids:
        b = v2.box_at(interp, tid, reference_t)
        if b is not None:
            heights.append(max(1.0, float(b[3] - b[1])))
    reference_h = REFERENCE_PLAYER_HEIGHT_FRAC * float(frame_h)
    observed_h = float(np.median(heights)) if heights else reference_h
    raw_scale = observed_h / max(reference_h, 1.0)
    scale = float(np.clip(raw_scale, MIN_SCALE, MAX_SCALE))
    base_rx = BASE_RX * (float(frame_w) / BASE_W)
    base_ry = BASE_RY * (float(frame_h) / BASE_H)
    return {
        'mode': 'camera_scale_at_screen_action_constant_within_angle',
        'reference_time_s': float(reference_t),
        'visible_focal_players': int(len(heights)),
        'focal_player_heights_px': [round(float(x),3) for x in heights],
        'reference_player_height_px': reference_h,
        'observed_median_player_height_px': observed_h,
        'raw_scale': raw_scale,
        'resolved_scale': scale,
        'base_rx_px': base_rx,
        'base_ry_px': base_ry,
        'rx_px': int(round(base_rx * scale)),
        'ry_px': int(round(base_ry * scale)),
        'min_scale': MIN_SCALE,
        'max_scale': MAX_SCALE,
        'never_smaller_than_canonical_broadcast': True,
    }


def draw_ring(frame, cx, cy, bgr, rx, ry):
    cx=int(round(cx)); cy=int(round(cy)); rx=int(rx); ry=int(ry)
    outer=v2.tint(bgr,1.18); rear=v2.tint(bgr,.63)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,188,253,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,287,352,rear,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,outer,7,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,0,180,bgr,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,outer,7,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,165,200,bgr,4,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,outer,7,cv2.LINE_AA)
    cv2.ellipse(frame,(cx,cy),(rx,ry),0,340,375,bgr,4,cv2.LINE_AA)


def render(source,tracks,config,out):
    cfg=json.load(open(config)); out=Path(out); out.mkdir(parents=True,exist_ok=True)
    start=float(cfg['timing']['start_s']); end=float(cfg['timing']['end_s']); release=float(cfg['timing']['release_s']); freeze=float(cfg['timing'].get('freeze_s',1.2))
    shot_desc=cfg['shot']['description']; dist=v2.parse_distance(shot_desc); dist_label=f'{dist:g} FT'
    team_bgr={k:v2.rgb_to_bgr(v['primary_rgb']) for k,v in cfg['teams'].items()}
    players={int(p['track_id']):p for p in cfg['players']}; shooter=int(cfg['roles']['shooter_track_id']); defender=int(cfg['roles']['primary_defender_track_id'])
    if cfg['timing'].get('contact_phases'):
        phases=[[float(s),float(e)] for s,e in cfg['timing']['contact_phases']]
    else:
        phases=[[float(cfg['timing'].get('contact_start_s',release)),float(cfg['timing'].get('contact_end_s',release))]]
    assert phases and all(e>=s for s,e in phases), phases
    phases=sorted(phases); cstart=phases[0][0]; contact_total=sum(e-s for s,e in phases)
    similar=cfg['shot'].get('similar_xfg') or None
    df=pd.read_csv(tracks); interp=v2.build_interp(df,players.keys())
    cap=cv2.VideoCapture(source); fps=float(cap.get(cv2.CAP_PROP_FPS) or 29.97); W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); sf=int(round(start*fps)); ef=int(round(end*fps)); cap.set(cv2.CAP_PROP_POS_FRAMES,sf)
    reference_t=float(cfg.get('graphics',{}).get('ring_reference_s',cfg['timing'].get('preview_screen_s',cstart+.20)))
    ring_policy=resolve_ring_policy(interp,players.keys(),reference_t,W,H); rx=int(ring_policy['rx_px']); ry=int(ring_policy['ry_px'])
    silent=out/'locked_screen_silent.mp4'; wr=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); freeze_n=int(round(freeze*fps)); release_written=False; frame_idx=sf
    screen_value=cfg['screen'].get('value','SCREEN ACTION'); preview_times=[float(cfg['timing'].get('preview_screen_s',cstart+.2)),release]; saved=set()
    while frame_idx<ef:
        ok,orig=cap.read()
        if not ok:break
        t=frame_idx/fps; fr=orig.copy(); boxes={}
        for tid,p in players.items():
            b=v2.box_at(interp,tid,t); boxes[tid]=b
            if b is None: continue
            cx=(b[0]+b[2])/2; cy=b[3]-4; col=team_bgr[p['team']]
            draw_ring(fr,cx,cy,col,rx,ry)
            fr=v2.draw_name(fr,cx,b[1],p['label'],col,float(p.get('label_dx',0)),float(p.get('label_dy',0)))
        fr=v2.draw_tile(fr,W-222,20,202,61,cfg['screen'].get('kicker','SCREEN ACTION'),screen_value,22)
        if t>=cstart: fr=v2.draw_contact(fr,W-222,87,202,v2.cumulative_contact(t,phases))
        show_quality=release-1.0<=t<=release+2.0
        if cfg['shot'].get('league_xfg_pct') is not None and show_quality:
            fr=v2.draw_tile(fr,20,20,194,66,'SHOT xFG',f"{float(cfg['shot']['league_xfg_pct']):.1f}%",27)
        if similar is not None and show_quality:
            n=int(similar.get('n',0)); kicker=f"SHEPPARD 25-26 SIMILAR  n={n}" if n else 'SHEPPARD 25-26 SIMILAR'
            fr=v2.draw_tile(fr,20,92,238,66,kicker,f"xFG  {float(similar['xfg_pct']):.1f}%",25)
        is_release=abs(t-release)<=.5/fps
        if is_release and not release_written:
            bs=boxes.get(shooter); bd=boxes.get(defender)
            if bs and bd: v2.dotted_line(fr,v2.center(bs),v2.center(bd))
            if bs:
                sp=players[shooter]; fr=v2.draw_distance_tag(fr,(bs[0]+bs[2])/2+float(sp.get('distance_dx',0)),bs[1]+float(sp.get('distance_dy',0)),dist_label)
            cv2.imwrite(str(out/'release_freeze.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,97]); wr.write(fr)
            for _ in range(freeze_n): wr.write(fr)
            release_written=True
        else: wr.write(fr)
        for pt in preview_times:
            if pt not in saved and abs(t-pt)<=.5/fps:
                cv2.imwrite(str(out/f'preview_{pt:.2f}.jpg'),fr,[cv2.IMWRITE_JPEG_QUALITY,95]); saved.add(pt)
        frame_idx+=1
    cap.release(); wr.release()
    native=out/f"{cfg['output_basename']}_native.mp4"; v2.mux_with_freeze_audio(Path(source),silent,native,start,release,end,freeze)
    uhd=out/f"{cfg['output_basename']}_UHD.mp4"
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-i',str(native),'-vf','hqdn3d=0.6:0.6:2.0:2.0,scale=3840:2160:flags=lanczos,cas=0.22,fps=30','-c:v','libx264','-profile:v','high','-preset','medium','-crf','16','-maxrate','36M','-bufsize','72M','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart',str(uhd)],check=True)
    qa={'tool_id':TOOL_ID,'parent_tool_id':'LOCKED_BROADCAST_SCREEN_V2','visual_lineage':{'baseline':'locked Screen Tracker horseshoe; V4 camera-perspective ring normalization'},'v4_changes':['camera_perspective_normalized_floor_ring_v2'],'floor_ring_policy_v2':ring_policy,'deterministic_only':True,'ai_image_generation':False,'ai_super_resolution':False,'shot_description':shot_desc,'shot_distance_tag':dist_label,'shot_distance_source':'shot_description_text_regex','event_xfg_pct':cfg['shot'].get('league_xfg_pct'),'similar_xfg':similar,'contact_phases':phases,'screen_contact_total_s':contact_total,'team_primary_rgb':{k:v['primary_rgb'] for k,v in cfg['teams'].items()},'panel_style':'neutral_translucent_grey_v1','native':str(native),'uhd':str(uhd),'release_s':release,'freeze_s':freeze}
    (out/'qa.json').write_text(json.dumps(qa,indent=2)); (out/'LOCKED_TOOL_ID.txt').write_text(TOOL_ID+'\n'); print(json.dumps(qa,indent=2))
