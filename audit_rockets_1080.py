from __future__ import annotations
import csv, html as htmlmod, io, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PBP='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
UA='Mozilla/5.0'
H={'User-Agent':UA,'Referer':'https://clips.nba.com/','Accept':'*/*'}
ROCKETS_TOKENS={'HOU','Houston Rockets','1610612745'}

def get(url,timeout=45):
    req=urllib.request.Request(url,headers=H)
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def team_row(row):
    vals={str(v).strip() for v in row.values() if v is not None}
    return bool(vals & ROCKETS_TOKENS)

def event_num(row):
    for k in ('event_num','event_id','action_number','eventnum','EVENTNUM'):
        v=row.get(k)
        if v not in (None,''):
            try:return int(float(v))
            except:pass
    return None

def game_id(row):
    for k in ('game_id','GAME_ID','gameId'):
        v=row.get(k)
        if v:return str(v).strip()
    return ''

def inspect_master(url):
    txt=get(url).decode('utf-8','replace')
    out=[]
    for line in txt.splitlines():
        if line.startswith('#EXT-X-STREAM-INF:'):
            m=re.search(r'RESOLUTION=(\d+)x(\d+)',line); b=re.search(r'BANDWIDTH=(\d+)',line)
            if m: out.append((int(m.group(1)),int(m.group(2)),int(b.group(1)) if b else 0))
    return max(out,key=lambda x:(x[0]*x[1],x[2])) if out else (0,0,0)

def inspect_event(gid,eid):
    page=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
    txt=get(page).decode('utf-8','replace')
    opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower(): opts.append((lab,u))
    if not opts:return {'event_id':eid,'ok':False,'angles':[],'has_1080':False}
    angles=[]
    with ThreadPoolExecutor(max_workers=min(12,len(opts))) as ex:
        fut={ex.submit(inspect_master,u):(lab,u) for lab,u in opts}
        for f in as_completed(fut):
            lab,_=fut[f]
            try:w,h,b=f.result(); angles.append({'label':lab,'width':w,'height':h,'bandwidth':b})
            except Exception as e:angles.append({'label':lab,'error':repr(e)})
    return {'event_id':eid,'ok':True,'angles':angles,'has_1080':any(a.get('width')==1920 and a.get('height')==1080 for a in angles)}

def audit_game(item):
    gid, evs=item
    # separated events: ~1/3 and ~2/3 of all observed events
    evs=sorted(set(evs)); picks=[]
    if evs:
        picks=[evs[len(evs)//3],evs[(2*len(evs))//3]]
    checks=[]
    for e in picks:
        try:checks.append(inspect_event(gid,e))
        except Exception as ex:checks.append({'event_id':e,'ok':False,'error':repr(ex),'has_1080':False})
    return {'game_id':gid,'events_checked':checks,'has_1080':any(c.get('has_1080') for c in checks)}

def main():
    raw=get(PBP,120).decode('utf-8','replace')
    rd=csv.DictReader(io.StringIO(raw))
    by={}
    for row in rd:
        if not team_row(row):continue
        gid=game_id(row); e=event_num(row)
        if gid and e is not None:by.setdefault(gid,[]).append(e)
    games=sorted(by.items())
    print('ROCKETS_GAMES_FOUND',len(games))
    results=[]
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut=[ex.submit(audit_game,x) for x in games]
        for f in as_completed(fut):results.append(f.result())
    results.sort(key=lambda x:x['game_id'])
    has=[r for r in results if r['has_1080']]
    payload={'pbp_source':PBP,'rockets_games_found':len(results),'games_with_1080':len(has),'pct':round(100*len(has)/len(results),1) if results else None,'game_ids_with_1080':[r['game_id'] for r in has],'games':results}
    open('rockets_1080_audit.json','w').write(json.dumps(payload,indent=2))
    print(json.dumps({k:payload[k] for k in ('rockets_games_found','games_with_1080','pct','game_ids_with_1080')},indent=2))
if __name__=='__main__':main()
