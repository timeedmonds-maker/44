from __future__ import annotations
import html, json, re, urllib.parse, urllib.request, urllib.error

PAGES=[
 ('2017','https://www.nbaplaydb.com/games/20170409-OKCDEN','VhbPgkcUui3'),
 ('2019','https://www.nbaplaydb.com/games/20190102-OKCLAL','vday_Lm0axe'),
]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/javascript,application/json,*/*;q=0.8'}
KEYS=('videourl','video_url','videoUrl','video url','duplicate video','mixtape','download','merge','collectionName:"nba-plays"','nba-plays','typesense','algolia','meilisearch','supabase','apiFetch','nba.com/stats/events','videoeventsasset','m3u8','mp4','/api/')

def get(url,timeout=25):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:
  return r.status,r.geturl(),r.headers,r.read()

def text_get(url,timeout=25):
 st,final,h,raw=get(url,timeout)
 return st,final,h,raw.decode('utf-8','replace')

def clean(s): return re.sub(r'\s+',' ',html.unescape(s))

def endpoints(blob):
 out=set()
 for pat in [r'[\"\'](/api/[^\"\'`\\\s]{1,240})[\"\']',r'[\"\'](https?://[^\"\'`\\\s]{1,320})[\"\']']:
  for m in re.finditer(pat,blob):
   u=html.unescape(m.group(1)).replace('\\/','/')
   if '/api/' in u or any(x in u.lower() for x in ('typesense','algolia','meili','supabase','nba.com','video','clip')):
    out.add(u)
 return sorted(out)

def contexts(blob,key,maxn=10,span=1300):
 low=blob.lower(); k=key.lower(); pos=0; out=[]
 while len(out)<maxn:
  i=low.find(k,pos)
  if i<0:break
  out.append(clean(blob[max(0,i-span):min(len(blob),i+span*2)]))
  pos=i+max(1,len(k))
 return out

for label,page_url,play_id in PAGES:
 print('\n=== GAME',label,page_url,'PLAY_ID',play_id,'===',flush=True)
 st,final,h,text=text_get(page_url)
 print('PAGE',st,'bytes',len(text),'final',final,flush=True)
 scripts=[urllib.parse.urljoin(page_url,html.unescape(x)) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
 print('SCRIPTS',len(scripts),flush=True)
 seen=set()
 for su in scripts:
  if su in seen or 'cloudflareinsights' in su:continue
  seen.add(su)
  try:sst,sfinal,sh,js=text_get(su,20)
  except Exception as e:continue
  jl=js.lower()
  hits=[k for k in KEYS if k.lower() in jl]
  ep=endpoints(js)
  if not hits and not ep:continue
  path=urllib.parse.urlsplit(su).path
  print('\nCHUNK',path,'bytes',len(js),'hits',hits,flush=True)
  if ep:print('ENDPOINTS',json.dumps(ep[:120],ensure_ascii=False),flush=True)
  # targeted snippets around high-value terms only
  for key in ('videourl','video url','duplicate video','nba-plays','typesense','algolia','meilisearch','supabase','apiFetch','mixtape','download','merge','nba.com/stats/events','/api/'):
   if key.lower() not in jl:continue
   for sn in contexts(js,key,maxn=6,span=900):
    print('CTX',key,sn[:3200],flush=True)
  # Try sourceMappingURL then conventional .map.
  maps=[]
  mm=re.findall(r'//# sourceMappingURL=([^\s]+)',js)
  for m in mm: maps.append(urllib.parse.urljoin(su,m))
  maps.append(su+'.map')
  done=set()
  for mu in maps:
   if mu in done:continue
   done.add(mu)
   try:mst,mfinal,mh,mraw=get(mu,18)
   except Exception:continue
   if mst!=200 or len(mraw)<50:continue
   try:obj=json.loads(mraw.decode('utf-8','replace'))
   except Exception:continue
   sources=obj.get('sources') or []
   contents=obj.get('sourcesContent') or []
   print('SOURCEMAP',urllib.parse.urlsplit(mu).path,'sources',len(sources),'bytes',len(mraw),flush=True)
   for idx,(src,srcblob) in enumerate(zip(sources,contents)):
    if not srcblob:continue
    low=srcblob.lower(); shits=[k for k in KEYS if k.lower() in low]
    sep=endpoints(srcblob)
    if not shits and not sep:continue
    print(' SOURCE',idx,src,'hits',shits,flush=True)
    if sep: print(' SOURCE_ENDPOINTS',json.dumps(sep[:80],ensure_ascii=False),flush=True)
    for key in ('videourl','video url','duplicate video','nba-plays','typesense','apiFetch','mixtape','download','merge','/api/'):
     if key.lower() in low:
      for sn in contexts(srcblob,key,maxn=4,span=700):
       print(' SOURCE_CTX',key,sn[:2600],flush=True)

# Probe obvious public read-only endpoints if they were previously surfaced by app bundles.
CANDIDATES=[
 'https://www.nbaplaydb.com/api/search?q=VhbPgkcUui3',
 'https://www.nbaplaydb.com/api/search?query=VhbPgkcUui3',
 'https://www.nbaplaydb.com/api/plays/VhbPgkcUui3',
 'https://www.nbaplaydb.com/api/play/VhbPgkcUui3',
 'https://www.nbaplaydb.com/api/search?q=vday_Lm0axe',
 'https://www.nbaplaydb.com/api/plays/vday_Lm0axe',
]
print('\n=== CONSERVATIVE PUBLIC API PROBES ===',flush=True)
for u in CANDIDATES:
 try:
  st,final,h,raw=get(u,15)
  ct=h.get('Content-Type',''); snippet=raw[:1000].decode('utf-8','replace')
  print('API',u,'status',st,'ct',ct,'bytes',len(raw),'snippet',clean(snippet)[:900],flush=True)
 except urllib.error.HTTPError as e:
  try:body=e.read(700).decode('utf-8','replace')
  except Exception:body=''
  print('API',u,'status',e.code,'snippet',clean(body)[:500],flush=True)
 except Exception as e:print('API_ERR',u,type(e).__name__,str(e)[:120],flush=True)
