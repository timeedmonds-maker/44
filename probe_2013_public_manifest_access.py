from __future__ import annotations
import json,time
from pathlib import Path
import requests

URLS=[
 'http://nbalp-vh.akamaihd.net/z/nba/big/leaguepass/2013/11/14/0021300124_home_,high,.mp4.csmil/manifest.f4m',
 'https://nbalp-vh.akamaihd.net/z/nba/big/leaguepass/2013/11/14/0021300124_home_,high,.mp4.csmil/manifest.f4m',
]
# Deliberately no auth, cookies, NBA tokens, signed query strings, or subscriber headers.
s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0','Accept':'*/*'})
out=[]
for u in URLS:
 rec={'url':u}; t=time.time()
 try:
  r=s.get(u,timeout=(8,20),allow_redirects=True,stream=True)
  first=next(r.iter_content(8192),b'')
  rec.update({'status':r.status_code,'elapsed':round(time.time()-t,2),'final_url':r.url,'content_type':r.headers.get('content-type'),'content_length':r.headers.get('content-length'),'www_authenticate':r.headers.get('www-authenticate'),'first_bytes':first.decode('utf-8','replace')[:6000]})
 except Exception as e: rec['error']=repr(e)
 out.append(rec)
 print(json.dumps(rec,indent=2))
Path('public_manifest_access_2013.json').write_text(json.dumps(out,indent=2))
