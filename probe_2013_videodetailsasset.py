from __future__ import annotations
import json,re,time
from pathlib import Path
import requests

GAME='0021300124'; EVENT=225; PLAYER=203500; TEAM=1610612760
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://www.nba.com/','Origin':'https://www.nba.com','Accept':'application/json,text/plain,*/*','x-nba-stats-origin':'stats','x-nba-stats-token':'true'}
BASE={
 'AheadBehind':'','ClutchTime':'','ContextFilter':'','ContextMeasure':'FGM','DateFrom':'','DateTo':'',
 'EndPeriod':10,'EndRange':28800,'GameID':GAME,'GameSegment':'','LastNGames':0,'LeagueID':'00','Location':'','Month':0,
 'OpponentTeamID':0,'Outcome':'','Period':0,'PlayerID':PLAYER,'PointDiff':'','Position':'','RangeType':0,'RookieYear':'',
 'Season':'2013-14','SeasonSegment':'','SeasonType':'Regular Season','StartPeriod':1,'StartRange':0,'TeamID':TEAM,
 'VsConference':'','VsDivision':''
}

def walk(v,path='$',out=None):
 if out is None: out=[]
 if isinstance(v,dict):
  for k,x in v.items(): walk(x,path+'.'+str(k),out)
 elif isinstance(v,list):
  for i,x in enumerate(v): walk(x,path+f'[{i}]',out)
 elif isinstance(v,(str,int,float,bool)):
  s=str(v)
  if str(EVENT)==s or GAME in s or any(z in s.lower() for z in ('video','media','.mp4','.m3u8','wsc','turner','uuid')):
   out.append({'path':path,'value':s[:2000]})
 return out

out=[]
for host in ['https://stats.gleague.nba.com','https://stats.nba.com']:
 for endpoint in ['videodetailsasset','videodetails']:
  for context in ['FGM','FGA']:
   p=dict(BASE); p['ContextMeasure']=context
   rec={'host':host,'endpoint':endpoint,'context':context}
   t=time.time()
   try:
    r=requests.get(host+'/stats/'+endpoint,params=p,headers=H,timeout=(8,35))
    rec.update({'status':r.status_code,'elapsed':round(time.time()-t,2),'url':r.url,'content_type':r.headers.get('content-type'),'bytes':len(r.content),'head':r.text[:1000]})
    if r.ok:
     try:
      j=r.json(); rec['json']=j; rec['hits']=walk(j)[:1000]
      # Collect video URL objects and event-like rows compactly.
      rs=j.get('resultSets',{}) if isinstance(j,dict) else {}
      rec['resultsets_type']=type(rs).__name__
      if isinstance(rs,dict):
       meta=rs.get('Meta') or {}; rec['videoUrls']=meta.get('videoUrls') or []
       rec['playlist']=meta.get('playlist') or []
     except Exception as e: rec['json_error']=repr(e)
   except Exception as e: rec.update({'elapsed':round(time.time()-t,2),'error':repr(e)})
   out.append(rec)
   print('\n',host,endpoint,context,rec.get('status'),rec.get('elapsed'),rec.get('error'))
   print('VIDEOURLS',json.dumps(rec.get('videoUrls',[]),indent=2)[:10000])
   print('HITS',json.dumps(rec.get('hits',[])[:80],indent=2)[:12000])
Path('videodetailsasset_2013_probe.json').write_text(json.dumps(out,indent=2))
