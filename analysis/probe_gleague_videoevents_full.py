from __future__ import annotations
import json,re,urllib.parse,urllib.request,urllib.error,time

EVENTS=[('2017','0021700015','438'),('2019','0041800163','215'),('2025','0022500375','608')]
HOSTS=['https://stats.gleague.nba.com/stats','https://stats.nba.com/stats']
ENDPOINTS=['videoevents','videoeventsasset']
H={
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36',
 'Accept':'application/json, text/plain, */*','Accept-Language':'en-US,en;q=0.9','Origin':'https://www.nba.com','Referer':'https://www.nba.com/',
 'x-nba-stats-origin':'stats','x-nba-stats-token':'true','Cache-Control':'no-cache','Pragma':'no-cache'
}
SENSITIVE=('token','sig','signature','key','auth','jwt')
def safe_url(u):
 try:
  p=urllib.parse.urlsplit(str(u)); return {'host':p.hostname,'path':p.path,'query_keys':sorted(k for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True) if k.lower() not in SENSITIVE)}
 except Exception:return {'raw_type':type(u).__name__}
def sanitize(obj,depth=0):
 if depth>8:return '<depth>'
 if isinstance(obj,dict):
  return {k:('[MASKED]' if k.lower() in SENSITIVE else sanitize(v,depth+1)) for k,v in obj.items()}
 if isinstance(obj,list):return [sanitize(x,depth+1) for x in obj[:80]]
 if isinstance(obj,str):
  if obj.startswith('http://') or obj.startswith('https://'):return {'url':safe_url(obj)}
  return obj[:1000]
 return obj

def get_json(url,timeout=18):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.headers,r.read()
for label,g,e in EVENTS:
 print('\n=== EVENT',label,g,e,'===',flush=True)
 q=urllib.parse.urlencode({'GameID':g,'GameEventID':e,'_cb':time.time_ns()})
 for host in HOSTS:
  for ep in ENDPOINTS:
   url=f'{host}/{ep}?{q}'
   print('\nCALL',host,ep,flush=True)
   try:
    st,final,h,raw=get_json(url)
    print('HTTP',st,'bytes',len(raw),'ct',h.get('Content-Type'),'final',safe_url(final),flush=True)
    try:data=json.loads(raw)
    except Exception:
     print('NONJSON',raw[:500].decode('utf-8','replace'),flush=True);continue
    print('TOP_KEYS',list(data.keys()) if isinstance(data,dict) else type(data).__name__,flush=True)
    # Print complete small sanitized result structure. This is diagnostic metadata, not signed URLs.
    san=sanitize(data)
    blob=json.dumps(san,ensure_ascii=False,sort_keys=True)
    print('PAYLOAD',blob[:30000],flush=True)
   except urllib.error.HTTPError as ex:
    try:b=ex.read(800).decode('utf-8','replace')
    except Exception:b=''
    print('HTTP_ERROR',ex.code,b[:500],flush=True)
   except Exception as ex:print('ERROR',type(ex).__name__,str(ex)[:180],flush=True)
