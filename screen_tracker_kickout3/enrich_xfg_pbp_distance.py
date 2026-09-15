#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd, pyreadr, requests

PBP_URL='https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/pbp-final-2026/data.rds'
XFG_URL='https://stats.gleague.nba.com/stats/shotqualityvideologs'
DURANT_ID=201142
HEADERS={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36','Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json, text/plain, */*'}
TL=threading.local(); PLAYER_RE=re.compile(r'^\s*(\d+)\s+(.+?)\s*$')

def sess():
    s=getattr(TL,'s',None)
    if s is None: s=requests.Session(); s.headers.update(HEADERS); TL.s=s
    return s

def num(v):
    try:
        x=float(v); return x if math.isfinite(x) else None
    except: return None

def evnum(s):
    for k in ('eventNum','event_num','eventNumber','eventId','eventID','actionNumber'):
        try:
            if k in s:return int(float(s[k]))
        except: pass
    return None

def xpct(s):
    q=num(s.get('shotQuality'))
    return None if q is None else (100*q if q<=1.5 else q)

def fetch(game,attempts=5):
    last={}
    for n in range(1,attempts+1):
        try:
            r=sess().get(XFG_URL,params={'GameID':game,'PlayerID':DURANT_ID},timeout=(8,35))
            if r.status_code==200:
                j=r.json()
                if str(j.get('gameId') or '').zfill(10)==game and int(j.get('playerId') or 0)==DURANT_ID:return j,{'status':200,'attempts':n}
                last={'status':200,'error':'unexpected_payload'}
            else:last={'status':r.status_code,'body':r.text[:120]}
        except Exception as e:last={'status':None,'error':repr(e)}
        time.sleep(min(5,.7*(2**(n-1))))
    return None,last

def gid(v):return str(v).replace('.0','').zfill(10)
def pid(v):
    m=PLAYER_RE.match(str(v or '')); return int(m.group(1)) if m else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--distance-window-ft',type=float,default=1.0);ap.add_argument('--xfg-window-pp',type=float,default=2.5);ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m=json.loads(a.manifest.read_text()); raw=requests.get(PBP_URL,timeout=180);raw.raise_for_status();tmp=a.out/'pbp.rds';tmp.write_bytes(raw.content);d=next(iter(pyreadr.read_r(str(tmp)).values())).reset_index(drop=True);tmp.unlink(missing_ok=True)
    if 'shot_distance' not in d.columns:raise RuntimeError('PBP has no shot_distance')
    d['pid']=d.player1_name.map(pid); d['gid']=d.game_id.map(gid); d['ev']=pd.to_numeric(d.event_num,errors='coerce'); d['dist']=pd.to_numeric(d.shot_distance,errors='coerce')
    kd=d[(d.pid==DURANT_ID)&d.ev.notna()&d.dist.notna()&d.msg_type.isin([1,2])].copy(); games=sorted(kd.gid.unique())
    xs=[];errs=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fut={ex.submit(fetch,g):g for g in games}
        for f in as_completed(fut):
            g=fut[f]; payload,meta=f.result()
            if payload is None:errs.append({'game_id':g,**meta});continue
            for s in payload.get('shotList') or []:
                ev=evnum(s);x=xpct(s)
                if ev is not None and x is not None:xs.append({'gid':g,'ev':ev,'xfg':x,'made':int(s.get('success') or 0),'action':s.get('actionType'),'shot_type':s.get('shotType')})
    x=pd.DataFrame(xs)
    if x.empty:raise RuntimeError('No Durant xFG shots returned')
    pool=kd.merge(x,on=['gid','ev'],how='inner',validate='one_to_one')
    pool.to_csv(a.out/'durant_xfg_pbp_distance_pool.csv',index=False)
    pd.DataFrame(errs).to_csv(a.out/'xfg_request_errors.csv',index=False)
    for i,e in enumerate(m['selected_events'],1):
        g=str(e['game_id']).zfill(10);ev=int(e['shot_event_num']);q=pool[(pool.gid==g)&(pool.ev==ev)]
        if len(q)!=1:raise RuntimeError(f'target {g}/{ev} rows={len(q)}')
        t=q.iloc[0];td=float(t.dist);tx=float(t.xfg)
        c=pool[(pool.dist.sub(td).abs()<=a.distance_window_ft+1e-9)&(pool.xfg.sub(tx).abs()<=a.xfg_window_pp+1e-9)&~((pool.gid==g)&(pool.ev==ev))].copy()
        n=len(c);made=int(c.made.sum()) if n else 0; px=100*made/n if n else None
        e['shot_quality']={'official_event_xfg_pct':tx,'observed_shot_distance_ft':td,'shot_distance_source':'ramirobentes/nba_pbp_data shot_distance','durant_similar_xfg_pct':px,'durant_similar_sample_fga':int(n),'durant_similar_sample_fgm':made,'similar_rule':{'same_player':True,'season':'2025-26','target_excluded':True,'distance_window_ft':a.distance_window_ft,'xfg_window_pp':a.xfg_window_pp,'statistic':'empirical FG% on Durant comparison sample'},'xfg_source':'official shotqualityvideologs via stats.gleague.nba.com','event_match_method':'exact game_id + event_num + player_id'}
        c[['gid','ev','dist','xfg','made','description','action','shot_type']].to_csv(a.out/f'event_{i}_similar.csv',index=False)
    (a.out/'kickout3_event_manifest_enriched.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
