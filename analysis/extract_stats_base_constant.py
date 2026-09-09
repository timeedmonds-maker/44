#!/usr/bin/env python3
import re,urllib.parse
from curl_cffi import requests
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
P='https://www.nba.com/stats/events?GameEventID=438&GameID=0021700015&Season=2017-18&flag=1'
s=requests.Session(impersonate='chrome120'); s.get('https://www.nba.com/',headers={'User-Agent':UA},timeout=15)
r=s.get(P,headers={'User-Agent':UA,'Referer':'https://www.nba.com/'},timeout=15)
for src in re.findall(r'<script[^>]+src=["\']([^"\']+)',r.text,re.I):
 u=urllib.parse.urljoin(r.url,src)
 try: js=s.get(u,headers={'User-Agent':UA,'Referer':P},timeout=10).text
 except: continue
 if '18744:' not in js and 'bL' not in js: continue
 for pat in [r'bL\s*:\s*([^,}]+)',r'bL\s*=\s*([^,;]+)',r'"bL"\s*:\s*([^,}]+)',r'\.bL\s*=\s*([^,;]+)']:
  for m in re.finditer(pat,js):
   print('MATCH',pat,m.group(0)[:1000]); print('CTX',re.sub(r'\s+',' ',js[max(0,m.start()-2500):m.end()+2500])[:6000])
 for needle in ['stats.nba.com/stats','https://stats.nba.com','stats.nba.com']:
  pos=0
  while True:
   i=js.find(needle,pos)
   if i<0:break
   print('URL_CTX',needle,re.sub(r'\s+',' ',js[max(0,i-1200):i+2200])[:3500]);pos=i+len(needle)
