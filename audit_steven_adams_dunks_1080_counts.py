from __future__ import annotations
import html as htmlmod, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
UA='Mozilla/5.0'; H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}
EVENTS=[
(1,'0022500093',43),(2,'0022500131',86),(3,'0022500146',146),(4,'0022500160',489),(5,'0022500191',561),(6,'0022500218',96),(7,'0022500218',315),(8,'0022500301',489),(9,'0022501205',505),(10,'0022500375',121),(11,'0022500438',248),(12,'0022500453',13),(13,'0022500505',41)]
def get(url,timeout=30):
    with urllib.request.urlopen(urllib.request.Request(url,headers=H),timeout=timeout) as r:return r.read()
def master(url):
    text=get(url).decode('utf-8','replace'); vals=[]
    for line in text.splitlines():
        if line.startswith('#EXT-X-STREAM-INF:'):
            rm=re.search(r'RESOLUTION=(\d+)x(\d+)',line); bm=re.search(r'BANDWIDTH=(\d+)',line)
            if rm: vals.append((int(rm.group(1)),int(rm.group(2)),int(bm.group(1)) if bm else 0))
    return max(vals,key=lambda x:(x[0]*x[1],x[2])) if vals else (0,0,0)
def event(item):
    rank,gid,eid=item; page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'; text=get(page).decode('utf-8','replace'); opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',text,re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower(): opts.append((lab,u))
    rows=[]
    with ThreadPoolExecutor(max_workers=min(12,len(opts))) as p:
        fs={p.submit(master,u):lab for lab,u in opts}
        for f in as_completed(fs):
            lab=fs[f]
            try:w,h,bw=f.result();err=None
            except Exception as e:w=h=bw=0;err=repr(e)
            rows.append({'label':lab,'width':w,'height':h,'bandwidth':bw,'error':err})
    rows.sort(key=lambda r:r['label']); hi=[r['label'] for r in rows if (r['width'],r['height'])==(1920,1080)]
    return {'rank':rank,'game_id':gid,'event_id':eid,'angle_count':len(rows),'count_1080':len(hi),'labels_1080':hi,'angles':rows}
results=[]
with ThreadPoolExecutor(max_workers=13) as p:
    for f in as_completed([p.submit(event,e) for e in EVENTS]):results.append(f.result())
results.sort(key=lambda r:r['rank'])
summary={'total_angle_clips':sum(r['angle_count'] for r in results),'total_1080_angles':sum(r['count_1080'] for r in results),'events_with_1080':sum(r['count_1080']>0 for r in results),'results':results}
open('steven_adams_dunks_1080_counts.json','w').write(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
