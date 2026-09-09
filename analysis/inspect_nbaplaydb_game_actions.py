from __future__ import annotations
import html,re,urllib.parse,urllib.request
PAGES=[('2017','https://www.nbaplaydb.com/games/20170409-OKCDEN'),('2019','https://www.nbaplaydb.com/games/20190102-OKCLAL')]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/javascript,*/*;q=0.8'}
def get(url):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=25) as r:return r.status,r.read().decode('utf-8','replace')
def clean(s):return re.sub(r'\s+',' ',html.unescape(s))
KEYS=('videourl','download','merge','fetch(','/api/','createserverreference','serverreference','next-action','event_id','eventid','actionnumber','nba.com','m3u8','mp4','/plays/')
for label,url in PAGES:
 print('\nGAME',label,url,flush=True)
 st,text=get(url);print('STATUS',st,'bytes',len(text),flush=True)
 scripts=[urllib.parse.urljoin(url,html.unescape(x)) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
 print('SCRIPT_COUNT',len(scripts),flush=True)
 for su in scripts:
  if 'cloudflareinsights' in su:continue
  try:sst,js=get(su)
  except Exception as e:continue
  jl=js.lower();matches=[k for k in KEYS if k.lower() in jl]
  if not matches:continue
  # Only print modules containing likely resolver/action semantics.
  strong=any(k in jl for k in ('videourl','createserverreference','next-action','/api/','download','/plays/'))
  if not strong:continue
  print('SCRIPT',urllib.parse.urlsplit(su).path,'bytes',len(js),'keys',matches,flush=True)
  locs=[]
  for key in KEYS:
   pos=0;k=key.lower()
   while True:
    i=jl.find(k,pos)
    if i<0:break
    if all(abs(i-j)>550 for j in locs):locs.append(i)
    pos=i+len(k)
  for i in sorted(locs)[:35]:
   sn=clean(js[max(0,i-1000):min(len(js),i+3200)])
   print('JS_SNIP',sn[:4200],flush=True)
