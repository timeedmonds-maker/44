#!/usr/bin/env python3
import json
from pathlib import Path
import requests

URLS=['https://stats.nba.com/stats/boxscorematchupsv3','https://stats.gleague.nba.com/stats/boxscorematchupsv3']
GAME='0022500001'
KD=201142
OUT=Path('artifacts/kd_double_team_probe'); OUT.mkdir(parents=True,exist_ok=True)
HEAD={
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36',
 'Referer':'https://www.nba.com/','Origin':'https://www.nba.com',
 'Accept':'application/json, text/plain, */*','x-nba-stats-origin':'stats','x-nba-stats-token':'true'
}

def get(params):
 errs=[]
 for url in URLS:
  try:
   r=requests.get(url,params=params,headers=HEAD,timeout=(5,10))
   meta={'host':url,'status':r.status_code,'url':r.url,'prefix':r.text[:160]}
   if r.status_code==200: return r.json(),meta
   errs.append(meta)
  except Exception as e: errs.append({'host':url,'error':repr(e)})
 return None,{'attempts':errs}

def kd_rows(j):
 out=[]
 if not isinstance(j,dict): return out
 root=j.get('boxScoreMatchups') or j.get('boxscoreMatchups') or j
 if isinstance(root,dict):
  for side in ('homeTeam','awayTeam'):
   for p in ((root.get(side) or {}).get('players') or []):
    pid=p.get('personId') or p.get('personIdOff')
    if int(pid or 0)!=KD: continue
    for m in p.get('matchups') or []:
     st=dict(m.get('statistics') or {})
     st.update({'def_personId':m.get('personId') or m.get('personIdDef'),'def_name':' '.join(x for x in [m.get('firstName'),m.get('familyName')] if x)})
     out.append(st)
 for rs in j.get('resultSets') or []:
  h=rs.get('headers') or []
  for vals in rs.get('rowSet') or []:
   r=dict(zip(h,vals))
   if int(r.get('PERSON_ID_OFF') or r.get('personIdOff') or 0)==KD: out.append(r)
 return out

def sig(rows):
 keep=['def_personId','def_name','personIdDef','nameIDef','matchupMinutes','partialPossessions','switchesOn','playerPoints','teamPoints','matchupAssists','matchupPotentialAssists','matchupTurnovers','matchupFieldGoalsMade','matchupFieldGoalsAttempted','helpFieldGoalsAttempted','helpFieldGoalsMade','matchupFreeThrowsAttempted']
 return [{k:r.get(k) for k in keep if k in r} for r in rows]

calls=[
 ('full',dict(GameID=GAME,LeagueID='00',StartPeriod=0,EndPeriod=14,StartRange=0,EndRange=0,RangeType=0)),
 ('q1_first_min',dict(GameID=GAME,LeagueID='00',StartPeriod=1,EndPeriod=1,StartRange=0,EndRange=600,RangeType=2)),
 ('q1_second_min',dict(GameID=GAME,LeagueID='00',StartPeriod=1,EndPeriod=1,StartRange=600,EndRange=1200,RangeType=2)),
 ('q2_first_min_abs',dict(GameID=GAME,LeagueID='00',StartPeriod=2,EndPeriod=2,StartRange=7200,EndRange=7800,RangeType=2)),
 ('q2_first_min_local',dict(GameID=GAME,LeagueID='00',StartPeriod=2,EndPeriod=2,StartRange=0,EndRange=600,RangeType=2)),
]
report={'game_id':GAME,'player_id':KD,'calls':[]}
for name,p in calls:
 j,m=get(p); rec={'name':name,'params':p,'meta':m}
 if j is not None:
  rows=kd_rows(j); rec['kd_row_count']=len(rows); rec['kd_rows']=sig(rows)
  (OUT/f'{name}_raw.json').write_text(json.dumps(j,indent=2))
 report['calls'].append(rec)
 print(name,m,'rows',rec.get('kd_row_count'),flush=True)
(OUT/'report.json').write_text(json.dumps(report,indent=2))
if not any(c.get('kd_row_count') is not None for c in report['calls']): raise SystemExit('No successful BoxScoreMatchupsV3 response')
