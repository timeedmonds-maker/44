from __future__ import annotations
import hashlib, html, json, re, time, urllib.parse, urllib.request, urllib.error

EVENTS=[('0021700015','438','2017'),('0041800163','215','2019'),('0022500375','608','2025-control')]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Referer':'https://clips.nba.com/','Accept':'*/*','Cache-Control':'no-cache'}

def get(url,limit=524288,timeout=18):
 req=urllib.request.Request(url,headers=H)
 try:
  with urllib.request.urlopen(req,timeout=timeout) as r:
   raw=r.read(limit);return {'status':r.status,'final':r.geturl(),'ct':r.headers.get('Content-Type'),'len':len(raw),'prefix':raw[:80]}
 except urllib.error.HTTPError as e:
  try:raw=e.read(4096)
  except Exception:raw=b''
  return {'status':e.code,'final':e.geturl(),'ct':e.headers.get('Content-Type'),'len':len(raw),'prefix':raw[:80]}
 except Exception as e:return {'error':type(e).__name__+': '+str(e)[:120]}

def page(g,e):
 url=f'https://clips.nba.com/?gameNo={g}&eventNum={e}&source=grs&_cb={time.time_ns()}'
 r=get(url,2_000_000)
 if r.get('status')!=200:return ''
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=20) as x:return x.read().decode('utf-8','replace')

def options(s):
 out=[]
 for u,a,t in re.findall(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',s,re.S):
  u=html.unescape(u)
  if '.m3u8' in u:
   out.append((re.sub('<[^>]+>','',t).strip(),u,'selected' in a))
 return out

def safe_result(label,r):
 x={'label':label}
 for k in ('status','ct','len','error'): 
  if k in r:x[k]=r[k]
 if 'final' in r:
  p=urllib.parse.urlsplit(r['final']);x['host']=p.hostname;x['path']=p.path;x['query_keys']=sorted({k for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True)})
 pref=r.get('prefix',b'')
 if pref:
  x['prefix_hash']=hashlib.sha256(pref).hexdigest()[:16]
  try:
   txt=pref.decode('utf-8','replace').replace('\n',' ')[:70]
   if txt.startswith('#EXT') or txt.startswith('<?xml') or '<' in txt:x['prefix_text']=txt
  except Exception:pass
 return x

for g,e,label in EVENTS:
 s=page(g,e); oo=options(s)
 print('\nEVENT',label,g,e,'options',len(oo),flush=True)
 if not oo:continue
 # selected broadcast, then first alternate broadcast if present
 chosen=[]
 for item in oo:
  if item[2]:chosen.append(item)
 for item in oo:
  if 'broadcast' in item[0].lower() and item not in chosen:chosen.append(item)
 chosen=chosen[:2]
 for angle,u,sel in chosen:
  p=urllib.parse.urlsplit(u); q=p.query
  stem=p.path.rsplit('/',1)[0]
  origin=f'{p.scheme}://{p.netloc}'
  probes=[
   ('baseline',u),
   ('dash',origin+stem+'/manifest.mpd'+(('?'+q) if q else '')),
   ('smooth',origin+stem+'/Manifest'+(('?'+q) if q else '')),
   ('f4m',origin+stem+'/manifest.f4m'+(('?'+q) if q else '')),
   ('smil',origin+stem+'/jwplayer.smil'+(('?'+q) if q else '')),
   ('chunklist',origin+stem+'/chunklist.m3u8'+(('?'+q) if q else '')),
   ('stream_root',origin+stem+('?' + q if q else '')),
  ]
  # Same stream through common Wowza VOD app aliases; token may be stream-bound rather than app-bound.
  stream=stem.split('/')[-1]
  for app in ('vod','VOD','ondemand','NBAPBP','nba'):
   probes.append((f'app_{app}',f'{origin}/{app}/_definst_/{stream}/playlist.m3u8'+(('?'+q) if q else '')))
  print('ANGLE',angle,'selected',sel,'stream',stream,flush=True)
  for pl,url in probes:
   r=get(url)
   print('PROBE',json.dumps(safe_result(pl,r),sort_keys=True),flush=True)
