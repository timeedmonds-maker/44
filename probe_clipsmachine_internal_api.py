from __future__ import annotations
import json,re
from pathlib import Path
from urllib.parse import urljoin
import requests

UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36'
S=requests.Session(); S.headers.update({'User-Agent':UA,'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8','Referer':'https://clips.nba.com/'})
EVENTS=[('2013','0021300124',225),('2019','0021801220',7)]
KEYS=['gameNo','eventNum','source=grs','inTime','outTime','startTime','endTime','timecode','clip','angle','lrmedia','m3u8','event','game','media','api','ajax','fetch(','axios','XMLHttpRequest']

def snippets(text):
 out=[]
 for key in KEYS:
  pos=0;n=0
  while True:
   i=text.lower().find(key.lower(),pos)
   if i<0:break
   s=text[max(0,i-1000):min(len(text),i+2200)]
   if s not in out:out.append(s)
   pos=i+len(key);n+=1
   if n>=20:break
 return out[:250]

out={'events':[],'scripts':{}}
script_urls=[]
for tag,gid,eid in EVENTS:
 u=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
 r=S.get(u,timeout=30); r.raise_for_status(); txt=r.text
 rec={'tag':tag,'game':gid,'event':eid,'status':r.status_code,'bytes':len(txt),'title':''}
 tm=re.search(r'<title>(.*?)</title>',txt,re.I|re.S); rec['title']=tm.group(1).strip() if tm else ''
 rec['options']=re.findall(r'<option\s+value=["\']([^"\']+)["\'][^>]*>(.*?)</option>',txt,re.I|re.S)
 rec['scripts']=[]
 for m in re.finditer(r'<script[^>]+src=["\']([^"\']+)["\']',txt,re.I):
  su=urljoin(u,m.group(1)); rec['scripts'].append(su)
  if su not in script_urls:script_urls.append(su)
 rec['inline_snippets']=snippets(txt)
 rec['interesting_urls']=list(dict.fromkeys(re.findall(r'https?://[^\s"\'<>]+',txt)))[:1000]
 out['events'].append(rec)
 print('\nEVENT',tag,gid,eid,'title',rec['title'],'options',len(rec['options']),'scripts',len(rec['scripts']))
 print('SCRIPTS',rec['scripts'])
 print('INLINE')
 for s in rec['inline_snippets'][:20]: print('---',s[:3500].replace('\n',' '))

for su in script_urls:
 try:
  rr=S.get(su,timeout=30); text=rr.text
  ss=snippets(text)
  if ss:
   out['scripts'][su]={'status':rr.status_code,'bytes':len(text),'snippets':ss}
   print('\nSCRIPT',su,'bytes',len(text),'snips',len(ss))
   for s in ss[:30]:print('---',s[:3500].replace('\n',' '))
 except Exception as e:
  out['scripts'][su]={'error':repr(e)}
Path('clipsmachine_internal_api_probe.json').write_text(json.dumps(out,indent=2))
