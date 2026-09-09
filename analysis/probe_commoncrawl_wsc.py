from __future__ import annotations
import gzip, io, json, re, urllib.parse, urllib.request, urllib.error

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36'
H={'User-Agent':UA}
EVENTS=[
 {
  'label':'2017','date':'2017/10/19','uuid':'5328a3f3-b7fd-f0a2-52e3-70bc6bc87ee4',
  'crawls':['CC-MAIN-2017-43','CC-MAIN-2017-47','CC-MAIN-2017-51','CC-MAIN-2018-05','CC-MAIN-2018-09']
 },
 {
  'label':'2019','date':'2019/04/19','uuid':'995096ac-c2a1-b2fa-89ed-a345759b84be',
  'crawls':['CC-MAIN-2019-18','CC-MAIN-2019-22','CC-MAIN-2019-26','CC-MAIN-2019-30','CC-MAIN-2019-35','CC-MAIN-2019-39']
 },
]
URL_RE=re.compile(rb'https?://[^\x00-\x20<>"\']+')

def get(url, headers=None, timeout=25):
 h=dict(H); h.update(headers or {})
 req=urllib.request.Request(url,headers=h)
 with urllib.request.urlopen(req,timeout=timeout) as r:
  return r.status,r.geturl(),r.headers,r.read()

def cdx(crawl,url,match='exact'):
 q={'url':url,'output':'json','filter':'status:200','fl':'url,timestamp,mime,status,digest,length,offset,filename','collapse':'urlkey'}
 if match!='exact': q['matchType']=match
 endpoint=f'https://index.commoncrawl.org/{crawl}-index?'+urllib.parse.urlencode(q)
 try:
  st,_,_,raw=get(endpoint,timeout=20)
  text=raw.decode('utf-8','replace').strip()
  if not text:return []
  out=[]
  for line in text.splitlines():
   try:out.append(json.loads(line))
   except Exception:pass
  return out
 except urllib.error.HTTPError as e:
  if e.code in (404,400):return []
  raise

def warc_record(rec):
 start=int(rec['offset']); length=int(rec['length']); end=start+length-1
 url='https://data.commoncrawl.org/'+rec['filename']
 try:
  st,_,hdr,raw=get(url,headers={'Range':f'bytes={start}-{end}'},timeout=40)
  # Individual WARC records in Common Crawl are gzip members.
  try:data=gzip.decompress(raw)
  except Exception:data=raw
  return data
 except Exception as e:
  return b''

def extract_urls(data):
 out=[]
 for b in URL_RE.findall(data):
  try:u=b.decode('utf-8','replace').rstrip(').,;')
  except Exception:continue
  low=u.lower()
  if 'cdn.turner.com' in low or '/nba/wsc/' in low or 'secure.nba.com/video/wsc' in low:
   out.append(u.replace('&amp;','&'))
 return list(dict.fromkeys(out))

def safe(u):
 p=urllib.parse.urlsplit(u)
 return {'host':p.hostname,'path':p.path,'query':bool(p.query)}

for ev in EVENTS:
 uuid=ev['uuid']; date=ev['date']
 print('\nEVENT',ev['label'],uuid,flush=True)
 targets=[
  ('secure_https',f'https://secure.nba.com/video/wsc/league/{uuid}.secure.xml','exact'),
  ('secure_http',f'http://secure.nba.com/video/wsc/league/{uuid}.secure.xml','exact'),
  ('turner_https',f'https://ssl.cdn.turner.com/nba/big/nba/wsc/{date}/{uuid}','prefix'),
  ('turner_http',f'http://ssl.cdn.turner.com/nba/big/nba/wsc/{date}/{uuid}','prefix'),
 ]
 seen=set()
 for crawl in ev['crawls']:
  for label,target,match in targets:
   try:rows=cdx(crawl,target,match)
   except Exception as e:
    print('CDX_ERROR',crawl,label,type(e).__name__,str(e)[:120],flush=True);continue
   if rows: print('CDX_HIT',crawl,label,'count',len(rows),flush=True)
   for rec in rows[:12]:
    key=(rec.get('url'),rec.get('digest'))
    if key in seen:continue
    seen.add(key)
    result={'crawl':crawl,'kind':label,'capture_url':safe(rec.get('url','')),'timestamp':rec.get('timestamp'),'mime':rec.get('mime'),'length':rec.get('length')}
    data=warc_record(rec)
    urls=extract_urls(data) if data else []
    result['warc_bytes']=len(data)
    result['embedded']=[safe(u) for u in urls]
    result['turner_paths']=[urllib.parse.urlsplit(u).path for u in urls if 'cdn.turner.com' in (urllib.parse.urlsplit(u).hostname or '')]
    # A direct WSC capture URL itself carries the missing numeric asset ID.
    if 'cdn.turner.com' in (urllib.parse.urlsplit(rec.get('url','')).hostname or ''):
     result['direct_turner_path']=urllib.parse.urlsplit(rec['url']).path
    print('CAPTURE',json.dumps(result,sort_keys=True),flush=True)
