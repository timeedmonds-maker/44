#!/usr/bin/env python3
from __future__ import annotations
import json, math, re
from pathlib import Path
import pandas as pd
import pyreadr, requests

YEAR=2026
SEASON='2025-26'
TEAM='HOU'
GAME='0022500001'
PLAYER_ID='203500'
PLAYER_NAME='Steven Adams'
BASE='https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main'
OUT=Path('artifacts/adams_okc_opener')
OUT.mkdir(parents=True, exist_ok=True)
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
def has_adams(v):
    if pd.isna(v): return False
    s=str(v)
    return bool(re.search(r'(?<!\d)203500(?!\d)',s)) or PLAYER_NAME.lower() in s.lower()
def clock_sec(v):
    if v is None or (isinstance(v,float) and math.isnan(v)): return math.nan
    s=str(v).strip(); m=re.match(r'^(\d+):(\d+(?:\.\d+)?)$',s)
    if m: return int(m.group(1))*60+float(m.group(2))
    m=re.match(r'^PT(?:(\d+)M)?([0-9.]+)S$',s)
    return int(m.group(1) or 0)*60+float(m.group(2)) if m else math.nan

def main():
    poss=load_rds(f'{BASE}/possessions{YEAR}/data.rds','poss.rds')
    pbp=load_rds(f'{BASE}/pbp-final-{YEAR}/data.rds','pbp.rds')
    poss['game_id_norm']=poss.game_id.map(norm_gid); pbp['game_id_norm']=pbp.game_id.map(norm_gid)
    poss['team_poss_norm']=poss.team_poss.fillna('').astype(str).str.strip().str.upper()
    u=poss[(poss.game_id_norm==GAME) & poss.team_poss_norm.eq(TEAM) & poss.lineup_team.map(has_adams)].copy().reset_index(drop=True)
    u['period_i']=pd.to_numeric(u.period,errors='coerce').astype('Int64')
    u['start_clock_sec']=u.start_time.map(clock_sec); u['end_clock_sec']=u.end_time.map(clock_sec)
    u['possession_uid']=[f'{GAME}-P{int(per)}-HOU-{int(float(n)):04d}' if pd.notna(n) and pd.notna(per) else f'{GAME}-row{i:04d}' for i,(per,n) in enumerate(zip(u.period_i,u.poss_num_team),1)]

    p=pbp[pbp.game_id_norm.eq(GAME)].copy()
    p['period_i']=pd.to_numeric(p.period,errors='coerce'); p['event_num_i']=pd.to_numeric(p.event_num,errors='coerce'); p['clock_sec']=p.clock.map(clock_sec)
    p['off_team']=p.get('off_team_abb',pd.Series('',index=p.index)).fillna('').astype(str).str.upper()
    p['team_evt']=p.get('team_abb',pd.Series('',index=p.index)).fillna('').astype(str).str.upper()
    for c in ('description','player1_name','player2_name','player3_name','action_type','sub_type'):
        if c not in p.columns: p[c]=''
    by_period={int(per):df for per,df in p[p.period_i.notna()].groupby('period_i')}

    recs=[]; windows=[]
    for _,r in u.iterrows():
        per=int(r.period_i); df=by_period.get(per,p.iloc[0:0])
        hi=float(r.start_clock_sec); lo=float(r.end_clock_sec)
        q=df[df.clock_sec.between(min(hi,lo)-.15,max(hi,lo)+.15,inclusive='both')].copy() if math.isfinite(hi) and math.isfinite(lo) else df.iloc[0:0].copy()
        q_hou=q[(q.off_team.eq(TEAM)) | (q.team_evt.eq(TEAM))]
        q_anchor=q_hou if len(q_hou) else q
        anchor=q_anchor.sort_values(['clock_sec','event_num_i'],ascending=[True,False],kind='stable').iloc[0] if len(q_anchor) else None
        nums=[int(x) for x in q.event_num_i.dropna().tolist()]
        rec={
            'season':SEASON,'game_id':GAME,'period':per,'poss_num_team':r.poss_num_team,'possession_uid':r.possession_uid,
            'start_time':r.start_time,'end_time':r.end_time,'duration_s':abs(float(r.start_clock_sec)-float(r.end_clock_sec)) if math.isfinite(float(r.start_clock_sec)) and math.isfinite(float(r.end_clock_sec)) else None,
            'score_team_start':r.score_team_start,'score_opp_start':r.score_opp_start,'opp':r.opp,'pts_poss':r.pts_poss,'type_end':r.type_end,
            'events_seq':r.events_seq,'lineup_team':r.lineup_team,'lineup_opp':r.lineup_opp,
            'window_event_nums':'|'.join(map(str,nums)),
            'anchor_event_num':int(anchor.event_num_i) if anchor is not None and pd.notna(anchor.event_num_i) else None,
            'anchor_clock':str(anchor.clock) if anchor is not None else None,
            'anchor_description':str(anchor.description) if anchor is not None else None,
        }
        rec['clip_page_url']=f"https://clips.nba.com/?gameNo={GAME}&eventNum={rec['anchor_event_num']}&source=grs" if rec['anchor_event_num'] is not None else None
        recs.append(rec)
        windows.append({'possession_uid':r.possession_uid,'pbp_rows':q[[c for c in ['event_num','period','clock','off_team_abb','team_abb','description','player1_name','player2_name','player3_name','action_type','sub_type'] if c in q.columns]].to_dict('records')})

    out=pd.DataFrame(recs)
    out.to_csv(OUT/'adams_okc_opener_possessions.csv',index=False)
    (OUT/'pbp_windows.json').write_text(json.dumps(windows,indent=2,default=str))
    qa={
        'game_id':GAME,'player':PLAYER_NAME,'player_id':int(PLAYER_ID),
        'hou_offensive_possessions_adams_on_court':int(len(out)),
        'period_counts':{str(k):int(v) for k,v in out.period.value_counts().sort_index().items()},
        'anchor_coverage':float(out.anchor_event_num.notna().mean()) if len(out) else 0.0,
        'points_on_these_possessions':float(pd.to_numeric(out.pts_poss,errors='coerce').fillna(0).sum()),
        'mean_possession_duration_s':float(pd.to_numeric(out.duration_s,errors='coerce').mean()) if len(out) else None,
        'duplicate_possession_uid':int(out.possession_uid.duplicated().sum()),
    }
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2,default=str))
    print(json.dumps(qa,indent=2,default=str))
    if out.possession_uid.duplicated().any(): raise SystemExit('duplicate possession uid')
    if len(out) and out.anchor_event_num.notna().mean()<.95: raise SystemExit('anchor coverage below 95%')
if __name__=='__main__': main()
