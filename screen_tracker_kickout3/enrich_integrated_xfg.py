#!/usr/bin/env python3
"""Enrich Kickout 3 events from the durable 2025-26 xFG/PBP integration.

No live stats.nba.com dependency. Target shot and comparison pool use the
validated exact-key joined project dataset in long-rebound-era.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

DEFAULT_URL=(
 'https://raw.githubusercontent.com/timeedmonds-maker/long-rebound-era/main/'
 'outputs/integration/nba_xfg_2025_26/nba_xfg_pbp_fga_join_2025_26.csv.gz'
)
DURANT_ID=201142

def gid(s):
    return s.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--data',default=DEFAULT_URL); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--distance-window-ft',type=float,default=1.0); ap.add_argument('--xfg-window-pp',type=float,default=2.5); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    m=json.loads(a.manifest.read_text())
    d=pd.read_csv(a.data,compression='infer',low_memory=False)
    required={'game_id','event_num','player_id','shot_distance','xfg_xfg_pct','xfg_available','xfg_made','description'}
    miss=required-set(d.columns)
    if miss: raise RuntimeError(f'integrated xFG file missing columns: {sorted(miss)}')
    d['game_id_norm']=gid(d['game_id']); d['event_num_num']=pd.to_numeric(d.event_num,errors='coerce'); d['player_id_num']=pd.to_numeric(d.player_id,errors='coerce'); d['shot_distance_num']=pd.to_numeric(d.shot_distance,errors='coerce'); d['xfg_pct_num']=pd.to_numeric(d.xfg_xfg_pct,errors='coerce'); d['made_num']=pd.to_numeric(d.xfg_made,errors='coerce')
    kd=d[(d.player_id_num==DURANT_ID)&d.xfg_available.astype(bool)&d.shot_distance_num.notna()&d.xfg_pct_num.notna()].copy()
    if kd.empty: raise RuntimeError('No Durant tracked xFG rows with shot distance')
    for i,e in enumerate(m['selected_events'],1):
        game=str(e['game_id']).zfill(10); ev=int(e['shot_event_num'])
        q=kd[(kd.game_id_norm==game)&(kd.event_num_num==ev)]
        if len(q)!=1: raise RuntimeError(f'Expected one exact target row {game}/{ev}; got {len(q)}')
        t=q.iloc[0]; td=float(t.shot_distance_num); tx=float(t.xfg_pct_num)
        comp=kd[
            (kd.shot_distance_num.sub(td).abs()<=a.distance_window_ft+1e-9)
            &(kd.xfg_pct_num.sub(tx).abs()<=a.xfg_window_pp+1e-9)
            &~((kd.game_id_norm==game)&(kd.event_num_num==ev))
        ].copy()
        n=len(comp); made=int(comp.made_num.fillna(0).sum()) if n else 0
        player_cond=100.0*made/n if n else None
        e['shot_quality']={
            'official_event_xfg_pct':tx,
            'observed_shot_distance_ft':td,
            'shot_distance_source':'ramirobentes PBP shot_distance in durable nba_xfg_pbp_fga_join_2025_26 exact-key integration',
            'durant_similar_xfg_pct':player_cond,
            'durant_similar_sample_fga':int(n),
            'durant_similar_sample_fgm':made,
            'similar_rule':{
                'same_player':True,'season':'2025-26','target_excluded':True,
                'distance_window_ft':float(a.distance_window_ft),'xfg_window_pp':float(a.xfg_window_pp),
                'distance_field':'shot_distance','difficulty_field':'xfg_xfg_pct',
                'statistic':'empirical FG% on Durant comparison sample'
            },
            'xfg_source':'official shotqualityvideologs durable integration',
            'event_match_method':'exact game_id + event_num + player_id',
        }
        comp[['game_id_norm','event_num','shot_distance','xfg_xfg_pct','xfg_made','description']].to_csv(a.out/f'event_{i}_durant_similar_sample.csv',index=False)
    out=a.out/'kickout3_event_manifest_enriched.json'; out.write_text(json.dumps(m,indent=2)); print(json.dumps(m,indent=2))

if __name__=='__main__': main()
