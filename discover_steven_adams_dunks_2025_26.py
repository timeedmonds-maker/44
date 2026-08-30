from __future__ import annotations
import csv, json, urllib.request
from datetime import datetime

URL='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
OUT='steven_adams_dunks_manifest.json'
UA='Mozilla/5.0'
ADAMS='203500 Steven Adams'

def main():
    req=urllib.request.Request(URL,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=180) as r, open('pbp.csv','wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)

    out=[]
    with open('pbp.csv',newline='',encoding='utf-8-sig',errors='replace') as f:
        reader=csv.DictReader(f)
        print('FIELDS='+json.dumps(reader.fieldnames or []))
        for row in reader:
            if (row.get('player1_name') or '').strip() != ADAMS:
                continue
            if (row.get('is_field_goal') or '').strip() != '1':
                continue
            if (row.get('shot_result') or '').strip().lower() != 'made':
                continue
            subtype=(row.get('sub_type') or '').strip()
            desc=(row.get('description') or '').strip()
            if 'dunk' not in (subtype+' '+desc).lower():
                continue
            gid=(row.get('game_id') or '').strip()
            try: eid=int(float((row.get('event_num') or '').strip()))
            except: continue
            out.append({
                'game_id':gid,
                'event_id':eid,
                'event_num':eid,
                'game_date':(row.get('game_date') or '').strip(),
                'team_home':(row.get('team_home') or '').strip(),
                'team_away':(row.get('team_away') or '').strip(),
                'period':(row.get('period') or '').strip(),
                'game_clock':(row.get('clock') or '').strip(),
                'action_type':(row.get('action_type') or '').strip(),
                'sub_type':subtype,
                'descriptor':(row.get('descriptor') or '').strip(),
                'description':desc,
                'player_id':203500,
            })

    def dtkey(x):
        try:return datetime.strptime(x['game_date'],'%Y-%m-%d')
        except:return datetime.max
    out.sort(key=lambda x:(dtkey(x),x['game_id'],x['event_id']))
    for i,e in enumerate(out,1): e['rank']=i
    payload={
        'source':URL,
        'player':'Steven Adams',
        'player_id':203500,
        'season':'2025-26',
        'filter':'player1_name == 203500 Steven Adams AND is_field_goal == 1 AND shot_result == Made AND dunk in sub_type/description',
        'expected_count':len(out),
        'events':out,
    }
    open(OUT,'w').write(json.dumps(payload,indent=2))
    print('DUNK_COUNT',len(out))
    for e in out: print(f"{e['rank']:02d} {e['game_date']} {e['team_away']} @ {e['team_home']} {e['game_id']}/{e['event_id']} P{e['period']} {e['game_clock']} {e['sub_type']} :: {e['description']}")
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
