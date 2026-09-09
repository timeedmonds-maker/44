from __future__ import annotations
import html,json,re,urllib.parse,urllib.request

PAGE='https://www.nba.com/nuggets/game/0021601198-thunder-vs-nuggets-denver-co-04-09-2017'
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/javascript,*/*;q=0.9'}
KEYS=('Watch Replay','watch replay','videoAvailableFlag','watchLinkGameTracker','league-pass-stream','gameId','videoAvailable','replay','watchLink')
def get(url,timeout=25):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.read().decode('utf-8','replace')
def clean(s):return re.sub(r'\s+',' ',html.unescape(s))
def ctx(blob,key,n=8,span=1500):
 low=blob.lower();k=key.lower();p=0;out=[]
 while len(out)<n:
  i=low.find(k,p)
  if i<0:break
  out.append(clean(blob[max(0,i-span):min(len(blob),i+span*2)]));p=i+len(k)
 return out
st,final,text=get(PAGE)
print('PAGE',st,final,'bytes',len(text),flush=True)
scripts=[urllib.parse.urljoin(final,html.unescape(x)) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
print('SCRIPTS',len(scripts),flush=True)
for su in scripts:
 try:sst,sfinal,js=get(su,20)
 except Exception:continue
 low=js.lower(); hits=[k for k in KEYS if k.lower() in low]
 if not hits:continue
 print('\nSCRIPT',urllib.parse.urlsplit(su).path,'bytes',len(js),'hits',hits,flush=True)
 for k in ('Watch Replay','videoAvailableFlag','watchLinkGameTracker','league-pass-stream','videoAvailable','replay'):
  if k.lower() in low:
   for sn in ctx(js,k,n=10,span=1200): print('CTX',k,sn[:5200],flush=True)
 # literal href/API strings involving watch/replay
 literals=set()
 for m in re.finditer(r'[\"\']([^\"\']{1,500})[\"\']',js):
  x=html.unescape(m.group(1)).replace('\\/','/')
  xl=x.lower()
  if ('watch' in xl or 'replay' in xl or 'league-pass' in xl) and len(x)<500:literals.add(x)
 print('LITERALS',json.dumps(sorted(literals)[:150],ensure_ascii=False),flush=True)
