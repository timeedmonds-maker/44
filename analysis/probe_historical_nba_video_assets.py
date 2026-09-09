import json, re, urllib.parse, urllib.request, urllib.error

EVENTS=[
 ('0021300124','225','2013'),
 ('0021700015','438','2017'),
 ('0041800163','215','2019'),
 ('0022500375','608','2025-control'),
]
BASE_HEADERS={
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36',
 'Accept':'application/json, text/plain, */*',
 'Accept-Language':'en-US,en;q=0.9',
 'Origin':'https://www.nba.com',
 'Referer':'https://www.nba.com/',
 'Cache-Control':'no-cache',
 'Pragma':'no-cache',
}

def request_json(endpoint,gid,eid):
 url=f'https://stats.nba.com/stats/{endpoint}?GameID={gid}&GameEventID={eid}'
 req=urllib.request.Request(url,headers=BASE_HEADERS)
 with urllib.request.urlopen(req,timeout=25) as r:
  raw=r.read()
 return json.loads(raw), url

def walk(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items(): yield from walk(v,f'{path}.{k}' if path else k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): yield from walk(v,f'{path}[{i}]')
 elif isinstance(obj,str):
  yield path,obj

def safe_url(u):
 try:
  p=urllib.parse.urlsplit(u)
  return {'host':p.hostname,'path':p.path,'scheme':p.scheme,'query_keys':sorted(dict(urllib.parse.parse_qsl(p.query)).keys())}
 except Exception:
  return {'raw_prefix':u[:120]}

for gid,eid,label in EVENTS:
 print('\nEVENT',label,gid,eid,flush=True)
 for ep in ('videoevents','videoeventsasset'):
  try:
   data,url=request_json(ep,gid,eid)
   print(' endpoint',ep,'ok top_keys',list(data.keys()),flush=True)
   strings=list(walk(data))
   urls=[]
   for p,v in strings:
    if v.startswith('http://') or v.startswith('https://'):
     urls.append((p,v))
   print(' url_count',len(urls),flush=True)
   for p,u in urls[:30]:
    print('  url',p,json.dumps(safe_url(u),sort_keys=True),flush=True)
   # print compact non-URL scalar fields that look useful for asset identity
   interesting=[]
   for p,v in strings:
    lp=p.lower(); lv=v.lower()
    if any(x in lp for x in ('uuid','duration','video','asset','media','source','file','stream','playlist')) and not v.startswith('http'):
     interesting.append((p,v[:180]))
   for p,v in interesting[:50]: print('  field',p,repr(v),flush=True)
  except urllib.error.HTTPError as e:
   print(' endpoint',ep,'HTTP',e.code,flush=True)
  except Exception as e:
   print(' endpoint',ep,'ERR',type(e).__name__,str(e)[:240],flush=True)
