from __future__ import annotations
import json, re, urllib.parse, urllib.request, urllib.error

EVENTS=[
 ('5328a3f3-b7fd-f0a2-52e3-70bc6bc87ee4','2017'),
 ('995096ac-c2a1-b2fa-89ed-a345759b84be','2019'),
]
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36'
H={'User-Agent':UA}
URL_RE=re.compile(r'https?://[^<"\'\s]+')

def get(url,timeout=25):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.read()

def safe(url):
 p=urllib.parse.urlsplit(url);return {'host':p.hostname,'path':p.path,'has_query':bool(p.query)}

def cdx(original):
 q=urllib.parse.urlencode({'url':original,'output':'json','filter':'statuscode:200','fl':'timestamp,original,statuscode,mimetype,digest','collapse':'digest'})
 status,_,raw=get('https://web.archive.org/cdx/search/cdx?'+q)
 rows=json.loads(raw)
 return rows[1:] if len(rows)>1 else []

def snapshot(ts,original):
 return f'https://web.archive.org/web/{ts}id_/{original}'

def wsc_urls(text):
 found=[]
 # unescape minimal XML entities before extracting
 text=text.replace('&amp;','&')
 for u in URL_RE.findall(text):
  u=u.rstrip(').,;')
  if any(x in u.lower() for x in ('cdn.turner.com','nba/wsc','secure.nba.com','videos.nba.com')):
   found.append(u)
 return list(dict.fromkeys(found))

for uuid,label in EVENTS:
 original=f'https://secure.nba.com/video/wsc/league/{uuid}.secure.xml'
 print('\nEVENT',label,uuid,flush=True)
 try:
  rows=cdx(original);print('CDX_COUNT',len(rows),flush=True)
 except Exception as e:
  print('CDX_ERROR',type(e).__name__,str(e)[:180],flush=True);continue
 for row in rows[-8:]:
  ts,orig,status,mime,digest=row
  rec={'timestamp':ts,'original':safe(orig),'status':status,'mimetype':mime,'digest':digest}
  try:
   st,final,raw=get(snapshot(ts,orig));text=raw.decode('utf-8','replace');urls=wsc_urls(text)
   rec.update(snapshot_status=st,snapshot_final=safe(final),bytes=len(raw),wsc_urls=[safe(u) for u in urls])
   # Emit exact Turner paths (no signed query strings) because asset ID/path identity is needed for deterministic recovery.
   rec['turner_paths']=[urllib.parse.urlsplit(u).path for u in urls if 'cdn.turner.com' in (urllib.parse.urlsplit(u).hostname or '')]
  except urllib.error.HTTPError as e:rec['snapshot_http_error']=e.code
  except Exception as e:rec['snapshot_error']=type(e).__name__+': '+str(e)[:150]
  print('CAPTURE',json.dumps(rec,sort_keys=True),flush=True)
