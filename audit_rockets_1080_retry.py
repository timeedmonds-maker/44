from __future__ import annotations
import csv, html as htmlmod, io, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
PBP='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
GAMES=set('0022500001 0022500012 0022500032 0022500042 0022500054 0022500066 0022500093 0022500116 0022500131 0022500146 0022500160 0022500176 0022500375 0022500505 0022500527 0022500540 0022500557 0022500565'.split())
H={'User-Agent':'Mozilla/5.0','Referer':'https://clips.nba.com/','Accept':'*/*'}
def get(url,timeout=45,retries=3):
    last=None
    for i in range(retries):
        try:
            req=urllib.request.Request(url,headers=H)
            with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()
        except Exception as e:last=e; time.sleep(.4*(i+1))
    raise last
def eid(row):
    for k in ('event_num','event_id','action_number'):
        try:
            if row.get(k) not in ('',None):return int(float(row[k]))
        except:pass
    return None
def gid(row):return str(row.get('game_id') or row.get('GAME_ID') or '').strip()
def master(url):
    txt=get(url).decode('utf-8','replace'); vals=[]
    for line in txt.splitlines():
        if line.startswith('#EXT-X-STREAM-INF:'):
            m=re.search(r'RESOLUTION=(\d+)x(\d+)',line); b=re.search(r'BANDWIDTH=(\d+)',line)
            if m:vals.append((int(m.group(1)),int(m.group(2)),int(b.group(1)) if b else 0))
    return max(vals,key=lambda x:(x[0]*x[1],x[2])) if vals else (0,0,0)
def event(game,event):
    txt=get(f'https://clips.nba.com/?gameNo={game}&eventNum={event}&source=grs').decode('utf-8','replace')
    opts=[]
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():opts.append((lab,u))
    angles=[]
    with ThreadPoolExecutor(max_workers=min(8,len(opts) or 1)) as ex:
        fs={ex.submit(master,u):(lab,u) for lab,u in opts}
        for f in as_completed(fs):
            lab,_=fs[f]
            try:w,h,b=f.result();angles.append({'label':lab,'width':w,'height':h,'bandwidth':b})
            except Exception as x:angles.append({'label':lab,'error':repr(x)})
    return {'event_id':event,'angles':angles,'has_1080':any(a.get('width')==1920 and a.get('height')==1080 for a in angles),'unresolved':sum(1 for a in angles if not a.get('width'))}
def main():
    raw=get(PBP,120).decode('utf-8','replace'); by={g:[] for g in GAMES}
    for r in csv.DictReader(io.StringIO(raw)):
        g=gid(r)
        if g in GAMES:
            e=eid(r)
            if e is not None:by[g].append(e)
    out=[]
    for g in sorted(GAMES):
        ev=sorted(set(by[g])); picks=[ev[len(ev)//4],ev[len(ev)//2],ev[(3*len(ev))//4]]
        checks=[event(g,e) for e in picks]
        out.append({'game_id':g,'checks':checks,'has_1080':any(c['has_1080'] for c in checks),'unresolved':sum(c['unresolved'] for c in checks)})
        print(g,'1080',out[-1]['has_1080'],'unresolved',out[-1]['unresolved'],flush=True)
    payload={'games':out,'games_with_1080':[x['game_id'] for x in out if x['has_1080']],'remaining_unresolved_games':[x['game_id'] for x in out if x['unresolved']]}
    open('rockets_1080_retry.json','w').write(json.dumps(payload,indent=2))
    print(json.dumps({'games_with_1080':payload['games_with_1080'],'remaining_unresolved_games':payload['remaining_unresolved_games']},indent=2))
if __name__=='__main__':main()
