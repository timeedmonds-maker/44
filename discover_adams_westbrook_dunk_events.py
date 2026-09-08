from __future__ import annotations

import json
import tempfile
from urllib.parse import quote

import pandas as pd
import pyreadr
import requests

SEASONS = {2014:'2013-14',2015:'2014-15',2016:'2015-16',2017:'2016-17',2018:'2017-18',2019:'2018-19'}
OUT_CSV='adams_westbrook_dunk_event_links.csv'
OUT_JSON='adams_westbrook_dunk_event_links.json'


def load_rds(url):
    r=requests.get(url,timeout=180)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.rds') as f:
        f.write(r.content); f.flush()
        obj=pyreadr.read_r(f.name)
    if not obj: raise RuntimeError(f'No table in {url}')
    return next(iter(obj.values()))


def norm(s):
    return s.fillna('').astype(str).str.lower().str.replace(r'[^a-z ]','',regex=True).str.strip()


def gid(v):
    s=str(v).strip()
    if s.endswith('.0'): s=s[:-2]
    return s.zfill(10)

rows=[]; coverage=[]
for end_year,season in SEASONS.items():
    for season_type,folder in [('Regular Season',f'pbp-final-{end_year}'),('Playoffs',f'pbp-final-playoffs{end_year}')]:
        url=f'https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main/{folder}/data.rds'
        df=load_rds(url)
        need={'game_date','game_id','event_num','msg_type','period','clock','description','player1_name','player2_name','team_home','team_away'}
        missing=need-set(df.columns)
        if missing: raise RuntimeError(f'{folder} missing {sorted(missing)}')
        p1=norm(df['player1_name']); p2=norm(df['player2_name']); desc=df['description'].fillna('').astype(str)
        msg=pd.to_numeric(df['msg_type'],errors='coerce')
        mask=(msg.eq(1)
              & p1.str.contains(r'\bsteven adams\b|\badams\b',regex=True)
              & p2.str.contains(r'\brussell westbrook\b|\bwestbrook\b',regex=True)
              & desc.str.contains('DUNK',case=False,na=False))
        hit=df.loc[mask].copy()
        coverage.append({'season':season,'season_type':season_type,'rows':len(df),'matches':len(hit),'source':url})
        for _,r in hit.iterrows():
            game=gid(r['game_id']); event=int(round(float(r['event_num']))); d=str(r['description'])
            link=f'https://www.nba.com/stats/events?GameEventID={event}&GameID={game}&Season={season}&flag=1'
            rows.append({
                'season':season,'season_type':season_type,
                'game_date':pd.to_datetime(r['game_date']).date().isoformat(),
                'game_id':game,'event_num':event,
                'period':int(round(float(r['period']))),'clock':str(r['clock']),
                'team_away':str(r['team_away']),'team_home':str(r['team_home']),
                'scorer':str(r['player1_name']),'assister':str(r['player2_name']),
                'description':d,'event_video_link':link,
                'event_video_link_with_title':link+'&title='+quote(d,safe='')
            })

out=pd.DataFrame(rows).sort_values(['game_date','game_id','event_num'],kind='stable').reset_index(drop=True)
if out.empty: raise RuntimeError('No matching events found')
if out.duplicated(['game_id','event_num']).any(): raise RuntimeError('Duplicate exact game/event keys')
assert out['scorer'].str.contains('Adams',case=False,na=False).all()
assert out['assister'].str.contains('Westbrook',case=False,na=False).all()
assert out['description'].str.contains('DUNK',case=False,na=False).all()
out.to_csv(OUT_CSV,index=False)
payload={'definition':'made field goal; scorer Steven Adams; assister Russell Westbrook; official description contains DUNK',
         'coverage':'2013-14 through 2018-19 regular season and playoffs',
         'source':'ramirobentes/nba_pbp_data exact PBP','total':len(out),
         'coverage_qa':coverage,'events':out.to_dict('records')}
open(OUT_JSON,'w').write(json.dumps(payload,indent=2))
print('TOTAL',len(out))
print(out.groupby(['season','season_type']).size().to_string())
print(out[['game_date','season','season_type','game_id','event_num','period','clock','description','event_video_link']].to_string(index=False))
