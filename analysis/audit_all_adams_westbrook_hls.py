from __future__ import annotations
import concurrent.futures, csv, html, json, re, urllib.error, urllib.parse, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'public_manifest/adams_westbrook_dunk_event_links_public.csv'
OUT=ROOT/'adams_westbrook_hls_coverage';OUT.mkdir(exist_ok=True)
UA='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/152 Safari/537.36'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*','Cache-Control':'no-cache','Pragma':'no-cache'}

def read(url,limit=None,timeout=15):
 req=urllib.request.Request(url,headers=H)
 with urllib.request.urlopen(req,timeout=timeout) as r:
  b=r.read() if limit is None else r.read(limit)
  return r.status,r.headers.get('content-type',''),r.geturl(),b

def playlist(url):
 status,ct,final,b=read(url,timeout=15);text=b.decode('utf-8','replace')
 if status!=200 or not text.lstrip().startswith('#EXTM3U'):raise ValueError(f'not_m3u8_{status}_{ct}')
 # Follow up to two master levels, choosing highest BANDWIDTH.
 cur=final
 for _ in range(2):
  if '#EXT-X-STREAM-INF' not in text:break
  lines=text.splitlines();variants=[]
  for i,line in enumerate(lines):
   if not line.startswith('#EXT-X-STREAM-INF'):continue
   m=re.search(r'BANDWIDTH=(\d+)',line);bw=int(m.group(1)) if m else 0
   nxt=next((x for x in lines[i+1:] if x and not x.startswith('#')),None)
   if nxt:variants.append((bw,nxt))
  if not variants:raise ValueError('master_without_variant')
  cur=urllib.parse.urljoin(cur,max(variants)[1]);_,_,cur,b=read(cur,timeout=15);text=b.decode('utf-8','replace')
  if not text.lstrip().startswith('#EXTM3U'):raise ValueError('variant_not_m3u8')
 seg=next((x for x in text.splitlines() if x and not x.startswith('#')),None)
 if not seg:raise ValueError('media_playlist_without_segment')
 segurl=urllib.parse.urljoin(cur,seg);s,ct,_,data=read(segurl,limit=4096,timeout=15)
 if s!=200 or len(data)<188:raise ValueError(f'segment_bad_{s}_{len(data)}_{ct}')
 return len(data)

def one(row):
 gid=str(row['game_id']);eid=str(row['event_num']);page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
 rec={k:row.get(k) for k in ('rank','season','season_type','game_date','game_id','event_num','description')};rec.update(status='unresolved',angle_count=0,attempts=[])
 try:
  _,_,_,raw=read(page,timeout=15);text=raw.decode('utf-8','replace')
  tm=re.search(r'<title>(.*?)</title>',text,re.I|re.S);title=html.unescape(tm.group(1).strip()) if tm else ''
  rec['title']=title
  if not all(x in title.lower() for x in ('adams','westbrook','dunk')):
   rec['status']='title_mismatch';return rec
  opts=[]
  for u,a,t in re.findall(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',text,re.I|re.S):
   u=html.unescape(u.strip());lab=re.sub(r'<[^>]+>','',html.unescape(t)).strip()
   if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():opts.append((u,'selected' in a.lower(),lab))
  rec['angle_count']=len(opts)
  if not opts:rec['status']='no_hls_advertised';return rec
  chosen=[o for o in opts if o[1]]+[o for o in opts if not o[1] and o[2].lower() in ('broadcast','other broadcast','mobile broadcast')]
  seen=set();chosen=[o for o in chosen if not (o[0] in seen or seen.add(o[0]))]
  for u,sel,lab in chosen:
   a={'label':lab,'selected':sel}
   try:
    a['segment_probe_bytes']=playlist(u);a['status']='ok';rec['attempts'].append(a);rec['status']='hls_segment_retrieved';rec['working_label']=lab;return rec
   except urllib.error.HTTPError as e:a['http_error']=e.code
   except Exception as e:a['error']=type(e).__name__+': '+str(e)[:120]
   rec['attempts'].append(a)
  rec['status']='hls_advertised_but_not_retrieved';return rec
 except urllib.error.HTTPError as e:rec['status']='clips_page_http_error';rec['page_http_error']=e.code
 except Exception as e:rec['status']='clips_page_error';rec['error']=type(e).__name__+': '+str(e)[:140]
 return rec

rows=list(csv.DictReader(MANIFEST.open()))
assert len(rows)==381 and len({(r['game_id'],r['event_num']) for r in rows})==381
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
 futs=[ex.submit(one,r) for r in rows]
 for f in concurrent.futures.as_completed(futs):
  q=f.result();results.append(q);print(len(results),q['rank'],q['season'],q['game_id'],q['event_num'],q['status'],flush=True)
by={(r['game_id'],r['event_num']):r for r in results};ordered=[by[(r['game_id'],r['event_num'])] for r in rows]
(OUT/'coverage.json').write_text(json.dumps(ordered,indent=2))
summary={'total':len(ordered),'overall':dict(Counter(r['status'] for r in ordered)),'by_season':{}}
for season,grp in __import__('itertools').groupby(ordered,key=lambda r:r['season']):
 g=list(grp);summary['by_season'][season]={'total':len(g),'statuses':dict(Counter(r['status'] for r in g)),'working_ranks':[int(r['rank']) for r in g if r['status']=='hls_segment_retrieved']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
