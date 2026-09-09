from __future__ import annotations
import html,json,re,urllib.parse,urllib.request

PAGES=[
 ('2017','https://www.nbaplaydb.com/plays/VhbPgkcUui3/2017-04-09-thunder-vs-nuggets-steven-adams-2pt-video'),
 ('2019','https://www.nbaplaydb.com/plays/vday_Lm0axe/2019-01-02-thunder-vs-lakers-steven-adams-2pt-video')]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/javascript,*/*;q=0.8'}

def get(url):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=25) as r:return r.status,r.read().decode('utf-8','replace')

def clean(s):return re.sub(r'\s+',' ',html.unescape(s))
KEYS=('videourl','videoUrl','video_url','download','merge','fetch(','/api/','serverreference','createserverreference','next-action','event_id','eventId','actionnumber','nba.com','m3u8','mp4')
for label,url in PAGES:
 print('\nPAGE',label,url,flush=True)
 st,text=get(url);print('STATUS',st,'bytes',len(text),flush=True)
 # Page snippets around resolver-relevant fields.
 low=text.lower()
 for key in KEYS:
  k=key.lower();pos=0;count=0
  while True:
   i=low.find(k,pos)
   if i<0 or count>=5:break
   sn=clean(text[max(0,i-500):min(len(text),i+1300)])
   print('PAGE_SNIP',key,sn[:1800],flush=True)
   pos=i+len(k);count+=1
 scripts=[urllib.parse.urljoin(url,html.unescape(x)) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
 print('SCRIPT_COUNT',len(scripts),flush=True)
 for su in scripts:
  if 'cloudflareinsights' in su:continue
  try:sst,js=get(su)
  except Exception:continue
  jl=js.lower();matches=[k for k in KEYS if k.lower() in jl]
  if not matches:continue
  print('SCRIPT',urllib.parse.urlsplit(su).path,'bytes',len(js),'keys',matches,flush=True)
  # snippets centered on important calls/refs; dedupe approximate locations
  locs=[]
  for key in ('videourl','download','merge','fetch(','createserverreference','/api/','next-action','nba.com'):
   pos=0
   while True:
    i=jl.find(key.lower(),pos)
    if i<0:break
    if all(abs(i-j)>350 for j in locs):locs.append(i)
    pos=i+len(key)
  for i in sorted(locs)[:18]:
   print(' JS_SNIP',clean(js[max(0,i-700):min(len(js),i+1800)])[:2500],flush=True)
