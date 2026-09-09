from __future__ import annotations
import html, json, re, urllib.parse, urllib.request, urllib.error

PAGES=[
 ('2017','https://www.nbaplaydb.com/games/20170409-OKCDEN','Adams Dunk (6 PTS) (Westbrook 4 AST)'),
 ('2019','https://www.nbaplaydb.com/games/20190102-OKCLAL',"Adams 1' Dunk (11 PTS) (Westbrook 5 AST)"),
]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8'}

def get(url,timeout=25):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.headers,r.read()

def safe(u):
 p=urllib.parse.urlsplit(u);return {'host':p.hostname,'path':p.path,'query_keys':sorted({k for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True)})}

def urls(text):
 raw=set(html.unescape(x) for x in re.findall(r'https?://[^\"\'<>\\s]+',text))
 return [u.rstrip('),.;') for u in raw]

for label,url,needle in PAGES:
 print('\nPAGE',label,url,flush=True)
 try:st,final,hdr,raw=get(url)
 except Exception as e:
  print('ERROR',type(e).__name__,str(e)[:160],flush=True);continue
 text=raw.decode('utf-8','replace')
 print('STATUS',st,'bytes',len(raw),'ct',hdr.get('Content-Type'),'final',safe(final),flush=True)
 print('NEEDLE_FOUND',needle.lower() in html.unescape(re.sub('<[^>]+>',' ',text)).lower(),flush=True)
 # Print a bounded sanitized HTML neighborhood around the exact play.
 low=text.lower(); idx=low.find('adams')
 hits=[]
 start=0
 while True:
  i=low.find('adams',start)
  if i<0:break
  chunk=text[max(0,i-900):min(len(text),i+1800)]
  plain=html.unescape(re.sub(r'\s+',' ',chunk))
  if 'westbrook' in plain.lower() and ('dunk' in plain.lower()):hits.append(plain)
  start=i+5
 print('MATCHING_CHUNKS',len(hits),flush=True)
 for c in hits[:4]:
  # mask long tokens, preserve href/src/path structure and IDs
  c=re.sub(r'([?&](?:token|sig|signature|key|auth|jwt)=[^&\"\' ]+)',r'\1[MASKED]',c,flags=re.I)
  print('CHUNK',c[:2600],flush=True)
 # Surface hrefs near Adams/Westbrook/dunk and all NBA/media/API URLs.
 hrefs=[]
 for m in re.finditer(r'href=[\"\']([^\"\']+)',text,re.I):
  p=m.start(); context=html.unescape(re.sub('<[^>]+>',' ',text[max(0,p-900):p+1200]))
  if 'adams' in context.lower() and 'westbrook' in context.lower() and 'dunk' in context.lower():hrefs.append(html.unescape(m.group(1)))
 print('NEARBY_HREFS',list(dict.fromkeys(hrefs))[:30],flush=True)
 allurls=urls(text)
 interesting=[]
 for u in allurls:
  host=(urllib.parse.urlsplit(u).hostname or '').lower(); path=urllib.parse.urlsplit(u).path.lower()
  if any(x in host for x in ('nba.com','nbaplaydb.com','cloudfront','akamai','turner')) or any(x in path for x in ('api','video','clip','play','media','m3u8','mp4')):
   interesting.append(safe(u))
 print('INTERESTING_URLS',json.dumps(interesting[:80],sort_keys=True),flush=True)
 # Inspect Next.js data and script chunks for API route names / video fields.
 scripts=[html.unescape(x) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
 print('SCRIPTS',scripts[-30:],flush=True)
 for src in scripts[-20:]:
  su=urllib.parse.urljoin(url,src)
  try:sst,sfinal,sh,sraw=get(su,timeout=20)
  except Exception:continue
  stext=sraw.decode('utf-8','replace')
  keys=[]
  for pat in ('videoUrl','video_url','clipUrl','clip_url','nba.com','videos.nba','m3u8','mp4','merge','download','playId','eventId','event_id','gameEvent'):
   if pat.lower() in stext.lower():keys.append(pat)
  if keys:
   print('SCRIPT_MATCH',safe(su),'bytes',len(sraw),'keys',keys,flush=True)
   # collect literal endpoint-ish strings
   ep=set(re.findall(r'[\"\']((?:/api/|https?://)[^\"\']{3,240})[\"\']',stext))
   print(' ENDPOINTS',[x[:240] for x in list(ep)[:40]],flush=True)
 # Next data payload if present.
 nd=re.findall(r'<script[^>]+id=[\"\']__NEXT_DATA__[\"\'][^>]*>(.*?)</script>',text,re.S|re.I)
 if nd:
  print('NEXT_DATA_BYTES',len(nd[0]),flush=True)
  try:
   obj=json.loads(html.unescape(nd[0])); blob=json.dumps(obj)
   for k in ('video','clip','event','play'):
    print(' NEXT_CONTAINS',k,k in blob.lower(),flush=True)
  except Exception as e:print('NEXT_PARSE_ERR',type(e).__name__,flush=True)
