from __future__ import annotations
import json,re,time
from pathlib import Path
from urllib.parse import urlencode
import requests

GAME='0021300124'; EVENT='225'
PARAMS={'gi':GAME,'ei':EVENT,'st':'01:36:07:00','en':'01:36:13:00','y':'2013','m':'11','d':'14'}
Q=urlencode(PARAMS)
URLS=[
 f'https://secure.nba.com/cvp/content.html?{Q}',
 f'http://secure.nba.com/cvp/content.html?{Q}',
 f'https://www.nba.com/cvp/content.html?{Q}',
 f'http://www.nba.com/cvp/content.html?{Q}',
 f'https://nba.com/cvp/content.html?{Q}',
 f'https://stats.gleague.nba.com/cvp/content.html?{Q}',
 f'https://stats.nba.com/cvp/content.html?{Q}',
]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
S=requests.Session(); S.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer':'https://www.nba.com/'})

def extract(text):
 urls=[]
 for raw in re.findall(r'https?://[^\s"\'<>\\]+',text):
  u=raw.replace('&amp;','&')
  if any(x in u.lower() for x in ('video','media','akamai','turner','.mp4','.m3u8','.f4m','.xml','wsc','cvp')) and u not in urls: urls.append(u)
 ids=[]
 for pat in [r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',r'\b\d{6,9}\b']:
  for x in re.findall(pat,text):
   if x not in ids: ids.append(x)
 return {'urls':urls[:300],'ids':ids[:300]}

out=[]
for u in URLS:
 rec={'url':u}; t=time.time()
 try:
  r=S.get(u,timeout=(8,30),allow_redirects=True)
  rec.update({'status':r.status_code,'elapsed':round(time.time()-t,2),'final_url':r.url,'content_type':r.headers.get('content-type'),'bytes':len(r.content),'headers':dict(r.headers)})
  text=r.text
  rec['head']=text[:10000]
  rec['extract']=extract(text)
  print('\nURL',u,'=>',rec['status'],rec['final_url'],rec['content_type'],rec['bytes'])
  print('EXTRACT',json.dumps(rec['extract'],indent=2)[:12000])
  print('HEAD',text[:3000].replace('\n',' ')[:3000])
 except Exception as e:
  rec.update({'elapsed':round(time.time()-t,2),'error':repr(e)})
  print('\nURL',u,'ERROR',rec['error'])
 out.append(rec)
Path('cvp_2013_event_player_probe.json').write_text(json.dumps(out,indent=2))
