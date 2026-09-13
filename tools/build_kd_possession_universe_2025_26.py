#!/usr/bin/env python3
from __future__ import annotations
import json, math, re
from pathlib import Path
import pandas as pd
import pyreadr, requests

YEAR=2026; SEASON='2025-26'; TEAM='HOU'; KD_ID='201142'; KD_NAME='Kevin Durant'
BASE='https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main'
OUT=Path('artifacts/kd_double_team_universe'); OUT.mkdir(parents=True,exist_ok=True)
TMP=OUT/'tmp'; TMP.mkdir(exist_ok=True)

def download(url,path):
    with requests.get(url,stream=True,timeout=240) as r:
        r.raise_for_status()
        with open(path,'wb') as f:
            for chunk in r.iter_content(8*1024*1024):
                if chunk: f.write(chunk)

def load_rds(url,name):
    p=TMP/name; download(url,p)
    df=next(iter(pyreadr.read_r(str(p)).values()))
    p.unlink(missing_ok=True)
    return df

def norm_gid(v): return re.sub(r'\.0$','',str(v)).zfill(10)
def has_kd(v):
    if pd.isna(v): return False
    s=str(v); return bool(re.search(r'(?<!\d)201142(?!\d)',s)) or KD_NAME.lower() in s.lower()
def clock_sec(v):
    if v is None or (isinstance(v,float) and math.isnan(v)): return math.nan
    s=str(v).strip(); m=re.match(r'^(\d+):(\d+(?:\.\d+)?)$',s)
    if m: return int(m.group(1))*60+float(m.group(2))
    m=re.match(r'^PT(?:(\d+)M)?([0-9.]+)S$',s)
    return int(m.group(1) or 0)*60+float(m.group(2)) if m else math.nan
def player_has_kd(v):
    if pd.isna(v): return False
    s=str(v); return s.startswith('201142 ') or KD_NAME.lower() in s.lower()

def main():
    poss=load_rds(f'{BASE}/possessions{YEAR}/data.rds','poss.rds')
    pbp=load_rds(f'{BASE}/pbp-final-{YEAR}/data.rds','pbp.rds')
    poss['game_id_norm']=poss.game_id.map(norm_gid); pbp['game_id_norm']=pbp.game_id.map(norm_gid)
    poss['team_poss_norm']=poss.team_poss.fillna('').astype(str).str.strip().str.upper()
    u=poss[poss.game_id_norm.str.startswith('002') & poss.team_poss_norm.eq(TEAM) & poss.lineup_team.map(has_kd)].copy().reset_index(drop=True)
    u['period_i']=pd.to_numeric(u.period,errors='coerce').astype('Int64')
    u['possession_uid']=[
        f"{g}-P{int(per)}-HOU-{int(float(n)):04d}" if pd.notna(n) and pd.notna(per) else f"{g}-P{per}-HOU-row{i:05d}"
        for i,(g,per,n) in enumerate(zip(u.game_id_norm,u.period_i,u.poss_num_team),1)
    ]
    u['start_clock_sec']=u.start_time.map(clock_sec); u['end_clock_sec']=u.end_time.map(clock_sec)

    p=pbp[pbp.game_id_norm.isin(set(u.game_id_norm))].copy()
    p['period_i']=pd.to_numeric(p.period,errors='coerce'); p['event_num_i']=pd.to_numeric(p.event_num,errors='coerce'); p['clock_sec']=p.clock.map(clock_sec)
    p['off_team']=p.get('off_team_abb',pd.Series('',index=p.index)).fillna('').astype(str).str.upper()
    p['team_evt']=p.get('team_abb',pd.Series('',index=p.index)).fillna('').astype(str).str.upper()
    for col in ('player1_name','player2_name','player3_name','description','action_type','sub_type'):
        if col not in p.columns: p[col]=''
    p['kd_p1']=p.player1_name.map(player_has_kd); p['kd_p2']=p.player2_name.map(player_has_kd); p['kd_p3']=p.player3_name.map(player_has_kd)
    p['kd_any_event']=p[['kd_p1','kd_p2','kd_p3']].any(axis=1)
    by_gp={(g,int(per)):df for (g,per),df in p[p.period_i.notna()].groupby(['game_id_norm','period_i'])}

    recs=[]; samples=[]
    for _,r in u.iterrows():
        per=int(r.period_i); df=by_gp.get((r.game_id_norm,per),p.iloc[0:0])
        hi=float(r.start_clock_sec); lo=float(r.end_clock_sec)
        q=df[df.clock_sec.between(min(hi,lo)-.15,max(hi,lo)+.15,inclusive='both')].copy() if math.isfinite(hi) and math.isfinite(lo) else df.iloc[0:0].copy()
        q_hou=q[(q.off_team.eq(TEAM)) | (q.team_evt.eq(TEAM))]
        q_anchor=q_hou if len(q_hou) else q
        anchor=q_anchor.sort_values(['clock_sec','event_num_i'],ascending=[True,False],kind='stable').iloc[0] if len(q_anchor) else None
        kdq=q[q.kd_any_event]; kd_p1=q[q.kd_p1]
        rec={
            'season':SEASON,'game_id':r.game_id_norm,'period':per,'poss_num_team':r.poss_num_team,'possession_uid':r.possession_uid,
            'start_time':r.start_time,'end_time':r.end_time,'score_team_start':r.score_team_start,'score_opp_start':r.score_opp_start,
            'opp':r.opp,'pts_poss':r.pts_poss,'type_end':r.type_end,'events_seq':r.events_seq,'lineup_team':r.lineup_team,'lineup_opp':r.lineup_opp,
            'window_event_count':int(len(q)),'hou_event_count':int(len(q_hou)),'kd_event_count':int(len(kdq)),'kd_player1_event_count':int(len(kd_p1)),
            'kd_event_nums':'|'.join(str(int(x)) for x in kdq.event_num_i.dropna().tolist()),
            'kd_player1_event_nums':'|'.join(str(int(x)) for x in kd_p1.event_num_i.dropna().tolist()),
            'anchor_event_num':int(anchor.event_num_i) if anchor is not None and pd.notna(anchor.event_num_i) else None,
            'anchor_clock':str(anchor.clock) if anchor is not None else None,'anchor_description':str(anchor.description) if anchor is not None else None,
        }
        rec['clip_page_url']=f"https://clips.nba.com/?gameNo={rec['game_id']}&eventNum={rec['anchor_event_num']}&source=grs" if rec['anchor_event_num'] is not None else None
        rec['candidate_priority']='A' if len(kdq)>=2 else ('B' if len(kdq)==1 else 'C')
        rec['candidate_reason']='multiple KD-attributed PBP events' if len(kdq)>=2 else ('one KD-attributed PBP event' if len(kdq)==1 else 'KD on court; no KD-attributed PBP event')
        recs.append(rec)
        if len(samples)<30:
            cols=[c for c in ['event_num','period','clock','off_team_abb','team_abb','description','player1_name','player2_name','player3_name','action_type','sub_type','possession'] if c in q.columns]
            samples.append({'possession':rec,'pbp_rows':q[cols].to_dict('records')})

    out=pd.DataFrame(recs)
    out.to_csv(OUT/'kd_possession_universe_2025_26.csv.gz',index=False,compression='gzip')
    cols=['season','game_id','period','poss_num_team','possession_uid','anchor_event_num','anchor_clock','anchor_description','clip_page_url','candidate_priority','candidate_reason']
    out[cols].to_csv(OUT/'kd_video_anchor_manifest_2025_26.csv',index=False)
    (OUT/'samples.json').write_text(json.dumps(samples,indent=2,default=str))
    qa={
        'season':SEASON,'team':TEAM,'player':KD_NAME,'player_id':int(KD_ID),'source_possessions_rows':int(len(poss)),'source_pbp_rows':int(len(pbp)),
        'kd_on_court_hou_offensive_possessions':int(len(out)),'kd_games':int(out.game_id.nunique()),'expected_kd_games_from_existing_repo_diagnostic':78,
        'kd_games_match_expected_78':bool(out.game_id.nunique()==78),'possessions_with_anchor_event':int(out.anchor_event_num.notna().sum()),
        'anchor_coverage':float(out.anchor_event_num.notna().mean()),'possessions_with_any_kd_pbp_event':int((out.kd_event_count>0).sum()),
        'possessions_with_multiple_kd_pbp_events':int((out.kd_event_count>=2).sum()),'candidate_priority_counts':out.candidate_priority.value_counts().to_dict(),
        'duplicate_possession_uid':int(out.possession_uid.duplicated().sum()),'classification':'exact observed possession universe; candidate_priority is workload triage only and is NOT a double-team label'
    }
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,default=str)); print(json.dumps(qa,indent=2,default=str))
    if out.game_id.nunique()!=78: raise SystemExit(f'QA FAIL: expected 78 KD games, got {out.game_id.nunique()}')
    if out.possession_uid.duplicated().any(): raise SystemExit('QA FAIL: duplicate possession_uid')
    if out.anchor_event_num.notna().mean()<.95: raise SystemExit(f'QA FAIL: anchor coverage only {out.anchor_event_num.notna().mean():.1%}')
if __name__=='__main__': main()
