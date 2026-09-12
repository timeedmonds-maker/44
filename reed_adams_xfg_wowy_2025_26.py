from __future__ import annotations
import json, math, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

PLAYER_ID=1642263; PLAYER_NAME='Reed Sheppard'; ADAMS_ID=203500; TEAM='HOU'
PBP_URL='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
XFG_URL='https://stats.gleague.nba.com/stats/shotqualityvideologs'
HEADERS={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36','Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json, text/plain, */*'}

def norm_gid(v):
    s=str(v); s=s[:-2] if s.endswith('.0') else s
    return ''.join(ch for ch in s if ch.isdigit()).zfill(10)
def ids(v): return {int(x) for x in re.findall(r'(?<!\d)(\d{6,7})(?!\d)',str(v))}
def fetch_xfg(s,gid):
    last=None
    for a in range(5):
        try:
            r=s.get(XFG_URL,params={'GameID':gid,'PlayerID':PLAYER_ID},timeout=(8,35))
            if r.status_code==200:
                j=r.json()
                if int(j.get('playerId') or 0)==PLAYER_ID:return j
            last=f'{r.status_code} {r.text[:120]}'
        except Exception as e:last=repr(e)
        time.sleep(min(5,0.6*(2**a)))
    raise RuntimeError(f'{gid}: {last}')
def fe_beta(df,y):
    d=df[['trio','adams_on',y]].dropna().copy(); v=d.groupby('trio').adams_on.agg(['min','max']); d=d[d.trio.isin(v[(v['min']==0)&(v['max']==1)].index)]
    if d.empty:return None
    x=d.adams_on.astype(float); yy=d[y].astype(float); xm=x-d.groupby('trio').adams_on.transform('mean'); ym=yy-d.groupby('trio')[y].transform('mean'); den=float((xm*xm).sum())
    return None if den==0 else {'effect':float((xm*ym).sum()/den),'rows':len(d),'trios':int(d.trio.nunique())}
def raw(df,col,scale=1):
    out={}
    for lab,val in [('ON',1),('OFF',0)]:
        q=pd.to_numeric(df.loc[df.adams_on.eq(val),col],errors='coerce').dropna();out[lab]={'n':len(q),'mean':float(q.mean()*scale) if len(q) else None}
    out['delta']=out['ON']['mean']-out['OFF']['mean'] if out['ON']['mean'] is not None and out['OFF']['mean'] is not None else None
    return out

def main():
    use=['game_id','event_num','player1_name','is_field_goal','team_home','team_away','lineup_home','lineup_away','shot_pts','shot_result','shot_distance','area','area_detail']
    p=pd.read_csv(PBP_URL,usecols=use,low_memory=False); shooter=p.player1_name.astype(str).str.match(r'^\s*1642263(?:\.0)?\s+',na=False); fg=pd.to_numeric(p.is_field_goal,errors='coerce').fillna(0).eq(1); r=p[shooter&fg].copy(); r['game_id_norm']=r.game_id.map(norm_gid)
    s=requests.Session();s.headers.update(HEADERS); shots=[]; errors=[]
    gids=sorted(r.game_id_norm.unique())
    for i,gid in enumerate(gids,1):
        try:
            j=fetch_xfg(s,gid)
            for sh in j.get('shotList') or []:
                shots.append({'game_id_norm':norm_gid(sh.get('gameId') or gid),'event_num':int(sh['eventNum']) if sh.get('eventNum') is not None else None,'xfg':pd.to_numeric(sh.get('shotQuality'),errors='coerce'),'success':int(sh.get('success') or 0),'action_type':sh.get('actionType'),'shot_type_xfg':sh.get('shotType')})
        except Exception as e:errors.append((gid,repr(e)))
        if i%10==0 or i==len(gids):print('XFG_PROGRESS',i,len(gids),'shots',len(shots),'errors',len(errors),flush=True)
    if errors:raise RuntimeError(errors[:5])
    x=pd.DataFrame(shots).drop_duplicates(['game_id_norm','event_num']); r['event_num']=pd.to_numeric(r.event_num,errors='coerce').astype('Int64'); x['event_num']=pd.to_numeric(x.event_num,errors='coerce').astype('Int64'); d=r.merge(x,on=['game_id_norm','event_num'],how='left',validate='one_to_one')
    def hou_lineup(z):
        if str(z.team_home).upper()==TEAM:return z.lineup_home
        if str(z.team_away).upper()==TEAM:return z.lineup_away
    d['hou_lineup']=d.apply(hou_lineup,axis=1);d['lineup_ids']=d.hou_lineup.map(ids);d['adams_on']=d.lineup_ids.map(lambda z:int(ADAMS_ID in z));d['shot_value']=pd.to_numeric(d.shot_pts,errors='coerce');d['expected_points']=pd.to_numeric(d.xfg,errors='coerce')*d.shot_value;d['expected_efg']=d.expected_points/2;d['actual_efg']=pd.to_numeric(d.success,errors='coerce')*d.shot_value/2;d['over_expected_efg']=d.actual_efg-d.expected_efg;d['is_three']=d.shot_value.eq(3).astype(float);d['rim']=pd.to_numeric(d.shot_distance,errors='coerce').lt(5).astype(float)
    # exact-other-three-teammate comparison
    ex=[]
    for _,z in d.iterrows():
        mates=set(z.lineup_ids)-{PLAYER_ID}
        if ADAMS_ID in mates:
            zz=z.copy();zz['trio']=tuple(sorted(mates-{ADAMS_ID}));ex.append(zz)
        else:
            for fourth in mates:
                zz=z.copy();zz['trio']=tuple(sorted(mates-{fourth}));ex.append(zz)
    m=pd.DataFrame(ex)
    out={'season':'2025-26','subject':'Reed Sheppard','teammate':'Steven Adams','source':'repo44 official NBA shotqualityvideologs joined exactly to ramirobentes PBP event lineups','classification':'descriptive raw WOWY and exact-other-three-teammate fixed effects; non-causal','qa':{'reed_fga':len(d),'xfg_covered':int(d.xfg.notna().sum()),'adams_on_fga':int(d.adams_on.sum()),'adams_off_fga':int((1-d.adams_on).sum()),'games':len(gids)},'metrics':{}}
    for col,scale in [('expected_efg',100),('actual_efg',100),('over_expected_efg',100),('xfg',100),('expected_points',1),('is_three',100),('rim',100),('shot_distance',1)]:
        z=fe_beta(m,col); out['metrics'][col]={'raw':raw(d,col,scale),'matched_fe':None if z is None else {**z,'effect_scaled':z['effect']*scale}}
    for val,label in [(2,'2PT'),(3,'3PT')]:
        q=d[d.shot_value.eq(val)];out['metrics'][label]={'attempts_on':int(q.adams_on.sum()),'attempts_off':int((1-q.adams_on).sum()),'fg_pct':raw(q,'success',100),'xfg_pct':raw(q,'xfg',100),'over_expected_fg_pp':raw(q,'over_expected_efg',100*(2/val))}
    print('RESULT='+json.dumps(out,sort_keys=True,default=str),flush=True)
if __name__=='__main__':main()
