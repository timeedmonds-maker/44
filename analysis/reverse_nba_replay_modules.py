from __future__ import annotations
import html,re,urllib.parse,urllib.request

PAGE='https://www.nba.com/nuggets/game/0021601198-thunder-vs-nuggets-denver-co-04-09-2017'
H={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36','Accept':'text/html,application/javascript,*/*;q=0.9'}
TARGETS=('68079','94779','91643')
KEYS=('linkAppend','watchReplay','watch replay','gameId','teamTricode','/watch/','?watch','league-pass-stream','game/')

def get(url,timeout=25):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:return r.status,r.geturl(),r.read().decode('utf-8','replace')
def clean(s):return re.sub(r'\s+',' ',html.unescape(s))
def contexts(blob,needle,maxn=8,span=6500):
 out=[];low=blob.lower();k=needle.lower();p=0
 while len(out)<maxn:
  i=low.find(k,p)
  if i<0:break
  out.append(clean(blob[max(0,i-span):min(len(blob),i+span)]));p=i+len(k)
 return out

st,final,text=get(PAGE)
print('PAGE',st,'bytes',len(text),flush=True)
scripts=[urllib.parse.urljoin(final,html.unescape(x)) for x in re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',text,re.I)]
print('SCRIPTS',len(scripts),flush=True)
for su in scripts:
 try:sst,sfinal,js=get(su,20)
 except Exception:continue
 path=urllib.parse.urlsplit(su).path
 found=[]
 for t in TARGETS:
  # webpack module forms like 68079:(e,t,a)=> or 68079:e=>
  if re.search(r'(?<!\d)'+re.escape(t)+r'\s*:',js):found.append(t)
 if found:
  print('\nMODULE_CHUNK',path,'bytes',len(js),'targets',found,flush=True)
  for t in found:
   for sn in contexts(js,t+':',maxn=3,span=9000):
    print('MODULE',t,sn[:18000],flush=True)
 # also find small chunks where linkAppend + game exist, even if target module IDs differ after build splits
 low=js.lower()
 if 'linkappend' in low and ('?watch' in low or 'watch replay' in low):
  print('\nLINKAPPEND_CHUNK',path,'bytes',len(js),flush=True)
  for sn in contexts(js,'linkAppend',maxn=6,span=5000):print('LINKAPPEND_CTX',sn[:12000],flush=True)
