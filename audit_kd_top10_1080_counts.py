from __future__ import annotations
import html as htmlmod, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

UA='Mozilla/5.0'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}
EVENTS=[
(1,'0022500375',608),(2,'0022500491',88),(3,'0022500628',553),(4,'0022500054',256),(5,'0022500816',676),
(6,'0022501178',79),(7,'0022500679',102),(8,'0022501058',735),(9,'0022500782',225),(10,'0022500527',498),
]

def get(url,timeout=30):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def inspect_master(url):
    text=get(url).decode('utf-8','replace')
    out=[]
    for line in text.splitlines():
        if line.startswith('#EXT-X-STREAM-INF:'):
            rm=re.search(r'RESOLUTION=(\d+)x(\d+)',line)
            bm=re.search(r'BANDWIDTH=(\d+)',line)
            if rm: out.append((int(rm.group(1)),int(rm.group(2)),int(bm.group(1)) if bm else 0))
    if not out:return (0,0,0)
    return max(out,key=lambda x:(x[0]*x[1],x[2]))

def inspect_event(item):
    rank,gid,eid=item
    page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
    text=get(page).decode('utf-8','replace')
    opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',text,flags=re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip())
        lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():opts.append((lab,u))
    rows=[]
    with ThreadPoolExecutor(max_workers=min(12,len(opts))) as pool:
        futs={pool.submit(inspect_master,u):(lab,u) for lab,u in opts}
        for f in as_completed(futs):
            lab,_=futs[f]
            try:w,h,bw=f.result();err=None
            except Exception as e:w=h=bw=0;err=repr(e)
            rows.append({'label':lab,'width':w,'height':h,'bandwidth':bw,'error':err})
    rows.sort(key=lambda r:r['label'])
    return {'rank':rank,'game_id':gid,'event_id':eid,'angles':rows,'count_1080':sum(1 for r in rows if r['width']==1920 and r['height']==1080),'labels_1080':[r['label'] for r in rows if r['width']==1920 and r['height']==1080]}

results=[]
with ThreadPoolExecutor(max_workers=10) as pool:
    fs=[pool.submit(inspect_event,e) for e in EVENTS]
    for f in as_completed(fs):results.append(f.result())
results.sort(key=lambda r:r['rank'])
summary={'total_angle_clips':sum(len(r['angles']) for r in results),'total_1080_angles':sum(r['count_1080'] for r in results),'events_with_1080':sum(1 for r in results if r['count_1080']>0),'results':results}
open('kd_top10_1080_counts.json','w').write(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
