from __future__ import annotations
import html,json,re,urllib.parse,urllib.request,urllib.error

GAMES=[
 ('2017','0021601198',[
  'https://www.nba.com/nuggets/game/0021601198-thunder-vs-nuggets-denver-co-04-09-2017',
  'https://www.nba.com/game/okc-vs-den-0021601198',
  'https://www.nba.com/game/0021601198']),
 ('2019','0021800562',[
  'https://www.nba.com/lakers/game/0021800562',
  'https://www.nba.com/game/okc-vs-lal-0021800562',
  'https://www.nba.com/game/0021800562'])]
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/xhtml+xml,application/json,*/*;q=0.8','Accept-Language':'en-US,en;q=0.9'}

def get(url,timeout=30):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.headers,r.read()
def safe(u):
 p=urllib.parse.urlsplit(u);return {'host':p.hostname,'path':p.path,'query_keys':sorted({k for k,v in urllib.parse.parse_qsl(p.query,keep_blank_values=True)})}
def clean(s):return re.sub(r'\s+',' ',html.unescape(s))
def contexts(text,key,maxn=8,span=1400):
 out=[];low=text.lower();k=key.lower();pos=0
 while len(out)<maxn:
  i=low.find(k,pos)
  if i<0:break
  out.append(clean(text[max(0,i-span):min(len(text),i+span*2)]));pos=i+len(k)
 return out

for label,gid,urls in GAMES:
 print('\n=== GAME',label,gid,'===',flush=True)
 for url in urls:
  try:st,final,h,raw=get(url,25)
  except urllib.error.HTTPError as e:
   try:b=e.read(1000).decode('utf-8','replace')
   except Exception:b=''
   print('PAGE_ERROR',url,e.code,clean(b)[:500],flush=True);continue
  except Exception as e:
   print('PAGE_ERR',url,type(e).__name__,str(e)[:140],flush=True);continue
  text=raw.decode('utf-8','replace')
  print('PAGE_OK',url,'status',st,'bytes',len(raw),'final',safe(final),'ct',h.get('Content-Type'),flush=True)
  # all anchor/button-ish attributes involving replay/watch/league
  refs=[]
  for m in re.finditer(r'(?:href|src|data-href|data-url|action)=[\"\']([^\"\']+)',text,re.I):
   v=html.unescape(m.group(1));ctx=clean(text[max(0,m.start()-1000):m.end()+1000]).lower()
   if any(k in (v+' '+ctx).lower() for k in ('watch replay','replay','league pass','league-pass','watch/','gamecast')):
    refs.append(urllib.parse.urljoin(final,v))
  print('REPLAY_REFS',[safe(x) for x in list(dict.fromkeys(refs))[:50]],flush=True)
  for key in ('Watch Replay','watchReplay','replay','league pass','leaguePass','league-pass','watchUrl','watchURL','streamUrl','playback','entitlement','gameId'):
   if key.lower() in text.lower():
    for sn in contexts(text,key,maxn=5,span=900):print('CTX',key,sn[:3000],flush=True)
  # embedded absolute URLs potentially related to watch/replay APIs
  absurls=set(html.unescape(x).replace('\\/','/') for x in re.findall(r'https?:\\?/\\?/[^\"\'<>\\s]+',text))
  interesting=[]
  for u in absurls:
   u=u.replace('https:\\/\\/','https://').replace('http:\\/\\/','http://')
   low=u.lower()
   if any(k in low for k in ('watch','replay','league','content-api','api-hub','playback','stream','m3u8')):
    try:interesting.append(safe(u.rstrip('),.;')))
    except Exception:pass
  print('ABS_INTERESTING',interesting[:80],flush=True)
  break
