import html, json, re, urllib.parse, urllib.request, urllib.error

EVENTS=[
 ('0021300124','225','2013'),
 ('0021700015','438','2017'),
 ('0041800163','215','2019'),
 ('0022500375','608','2025-control'),
]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36'
STATS_HEADERS={
 'User-Agent':UA,
 'Accept':'application/json, text/plain, */*',
 'Accept-Language':'en-US,en;q=0.9',
 'Origin':'https://stats.nba.com',
 'Referer':'https://www.nba.com/',
 'x-nba-stats-origin':'stats',
 'x-nba-stats-token':'true',
 'Cache-Control':'no-cache',
 'Pragma':'no-cache',
 'Connection':'keep-alive',
}
CLIPS_HEADERS={'User-Agent':UA,'Referer':'https://clips.nba.com/','Cache-Control':'no-cache','Pragma':'no-cache'}
UUID_RE=re.compile(r'(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b')
MEDIA_RE=re.compile(r'mp4:(\d{10})_1509kbps\.mp4')

def read(url,headers,timeout=12):
 req=urllib.request.Request(url,headers=headers)
 with urllib.request.urlopen(req,timeout=timeout) as r:
  return r.read(),r.status,r.geturl()

def request_stats(host,endpoint,gid,eid):
 q=urllib.parse.urlencode({'GameEventID':eid,'GameID':gid,'LeagueID':'00'})
 raw,status,_=read(f'https://{host}/stats/{endpoint}?{q}',STATS_HEADERS)
 return json.loads(raw),status

def walk(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items(): yield from walk(v,f'{path}.{k}' if path else k)
 elif isinstance(obj,list):
  for i,v in enumerate(obj): yield from walk(v,f'{path}[{i}]')
 elif isinstance(obj,(str,int,float,bool)) or obj is None:
  yield path,obj

def extract_stats(data):
 out={'uuids':[],'urls':[],'fields':[]}
 for p,v in walk(data):
  if isinstance(v,str):
   out['uuids'] += UUID_RE.findall(v)
   if v.startswith(('http://','https://')):
    u=urllib.parse.urlsplit(v)
    out['urls'].append({'field':p,'host':u.hostname,'path':u.path})
   if any(k in p.lower() for k in ('uuid','video','asset','playlist','media')) and not v.startswith('http'):
    out['fields'].append({'field':p,'value':v[:120]})
 return out

def probe_secure_xml(identifier):
 url=f'https://secure.nba.com/video/wsc/league/{identifier}.secure.xml'
 try:
  raw,status,_=read(url,{'User-Agent':UA,'Referer':'https://www.nba.com/'},8)
  text=raw.decode('utf-8','replace')
  files=re.findall(r'<file[^>]*>(.*?)</file>',text,re.I|re.S)
  return {'status':status,'bytes':len(raw),'file_count':len(files),'hosts':sorted({urllib.parse.urlsplit(html.unescape(x.strip())).hostname for x in files if x.strip().startswith('http')})}
 except urllib.error.HTTPError as e:
  return {'http_error':e.code}
 except Exception as e:
  return {'error':type(e).__name__+': '+str(e)[:120]}

def scan_clips(gid,eid):
 url=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
 raw,status,_=read(url,CLIPS_HEADERS,15)
 text=raw.decode('utf-8','replace')
 title=re.findall(r'<title>(.*?)</title>',text,re.I|re.S)
 uuids=sorted(set(UUID_RE.findall(text)))
 media_ids=sorted(set(MEDIA_RE.findall(text)))
 secure_refs=[]
 for u in re.findall(r'https?://[^\"\'<>\s]+',html.unescape(text)):
  if 'secure.nba.com' in u.lower() or '/video/wsc/' in u.lower():
   p=urllib.parse.urlsplit(u);secure_refs.append({'host':p.hostname,'path':p.path})
 # Keep only attribute/value snippets that contain identifiers, not signed URLs.
 id_snippets=[]
 for line in text.splitlines():
  if any(x in line.lower() for x in ('uuid','videoid','mediaid','assetid','wsc')):
   cleaned=re.sub(r'https?://[^\"\'<>\s]+','[url]',line.strip())
   if cleaned and cleaned not in id_snippets:id_snippets.append(cleaned[:300])
 return {'status':status,'title':html.unescape(title[0].strip()) if title else None,'html_bytes':len(raw),'uuids':uuids,'media_ids':media_ids,'secure_refs':secure_refs[:20],'identifier_snippets':id_snippets[:20]}

for gid,eid,label in EVENTS:
 print('\nEVENT',label,gid,eid,flush=True)
 try:
  c=scan_clips(gid,eid);print(' clips',json.dumps(c,sort_keys=True),flush=True)
 except Exception as e:
  c={'uuids':[],'media_ids':[]};print(' clips ERR',type(e).__name__,str(e)[:180],flush=True)
 seen=set(c.get('uuids',[]))
 for ident in list(seen)[:10]:print(' secure_uuid',ident,probe_secure_xml(ident),flush=True)
 for ident in c.get('media_ids',[])[:3]:print(' secure_numeric',ident,probe_secure_xml(ident),flush=True)
 for host in ('stats.nba.com','stats.gleague.nba.com'):
  for ep in ('videoevents','videoeventsasset'):
   try:
    data,status=request_stats(host,ep,gid,eid);x=extract_stats(data)
    print(' stats',host,ep,'HTTP',status,'uuids',sorted(set(x['uuids'])),'urls',json.dumps(x['urls'][:10],sort_keys=True),'fields',json.dumps(x['fields'][:20],sort_keys=True),flush=True)
    for ident in sorted(set(x['uuids']))[:10]:print(' secure_from_stats',ident,probe_secure_xml(ident),flush=True)
   except urllib.error.HTTPError as e:print(' stats',host,ep,'HTTP',e.code,flush=True)
   except Exception as e:print(' stats',host,ep,'ERR',type(e).__name__,str(e)[:180],flush=True)
