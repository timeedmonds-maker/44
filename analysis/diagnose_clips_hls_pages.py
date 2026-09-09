import hashlib, html, json, re, time, urllib.parse, urllib.request

EVENTS=[
 ('0021300124','225','2013'),
 ('0021700015','438','2017'),
 ('0041800163','215','2019'),
 ('0022500375','608','2025-control'),
]
H={
 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152.0.0.0 Safari/537.36',
 'Referer':'https://clips.nba.com/',
 'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
 'Cache-Control':'no-cache','Pragma':'no-cache'
}

def get_page(g,e,n):
 url=f'https://clips.nba.com/?gameNo={g}&eventNum={e}&source=grs&_cb={time.time_ns()}_{n}'
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=20) as r: return r.read().decode('utf-8','replace')

def opts(s):
 out=[]
 for u,a,t in re.findall(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',s,re.S):
  u=html.unescape(u)
  if '.m3u8' in u: out.append((re.sub('<[^>]+>','',t).strip(),u,'selected' in a))
 return out

def safe(u):
 p=urllib.parse.urlsplit(u); q=urllib.parse.parse_qsl(p.query,keep_blank_values=True)
 return {
  'host':p.hostname,
  'path':p.path,
  'query_keys':sorted({k for k,v in q}),
  'query_hash':hashlib.sha256(p.query.encode()).hexdigest()[:16] if p.query else None,
  'url_hash':hashlib.sha256(u.encode()).hexdigest()[:16],
 }

for g,e,label in EVENTS:
 pages=[]
 for n in range(3):
  s=get_page(g,e,n); oo=opts(s)
  pages.append(oo)
  time.sleep(.25)
 title=re.findall(r'<title>(.*?)</title>',s,re.S)
 print('\nEVENT',label,g,e,'title=',html.unescape(title[0].strip()) if title else None)
 print('option_counts',[len(x) for x in pages])
 for i,oo in enumerate(pages):
  print('request',i+1)
  for name,u,sel in oo[:3]: print(json.dumps({'angle':name,'selected':sel,**safe(u)},sort_keys=True))
 if pages and pages[0]:
  full_hash_sets=[[hashlib.sha256(u.encode()).hexdigest() for _,u,_ in x] for x in pages]
  print('all_option_urls_identical_across_requests',full_hash_sets[0]==full_hash_sets[1]==full_hash_sets[2])
 # Discover any non-option HLS URLs or obvious media/API references without exposing tokens.
 urls=set(html.unescape(u) for u in re.findall(r'https?://[^\"\'<>\\s]+',s) if 'm3u8' in u.lower() or 'lrmedia' in u.lower())
 print('all_hls_or_lrmedia_refs',len(urls))
 for u in list(urls)[:10]: print(' ref',json.dumps(safe(u),sort_keys=True))
 scripts=re.findall(r'<script[^>]+src=[\"\']([^\"\']+)',s,re.I)
 print('script_src_count',len(scripts))
 for x in scripts[:20]: print(' script',urllib.parse.urljoin('https://clips.nba.com/',html.unescape(x)))
