from __future__ import annotations
import csv, json, re, urllib.request
from datetime import datetime

URL='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
OUT='steven_adams_dunks_manifest.json'
UA='Mozilla/5.0'

def val(row,*names):
    for n in names:
        if n in row and row[n] not in (None,''):
            return row[n]
    return ''

def norm(s): return re.sub(r'\s+',' ',str(s or '')).strip()

def main():
    req=urllib.request.Request(URL,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=180) as r, open('pbp.csv','wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
    with open('pbp.csv',newline='',encoding='utf-8-sig',errors='replace') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        print('FIELDS='+json.dumps(fields))
        adams_rows=[]
        for row in reader:
            blob=' | '.join(norm(v) for v in row.values())
            if 'Steven Adams' not in blob and 'steven adams' not in blob.lower():
                continue
            adams_rows.append(row)
    print('ADAMS_ROWS',len(adams_rows))
    for r in adams_rows[:8]: print('SAMPLE',json.dumps(r,ensure_ascii=False))

    out=[]
    seen=set()
    for row in adams_rows:
        # Prefer explicit player identity if available; fallback to description fields containing his name.
        pid=norm(val(row,'playerId','player_id','personId','person_id','PLAYER_ID'))
        pname=norm(val(row,'playerName','player_name','PLAYER_NAME','personName','person_name'))
        desc=norm(val(row,'description','eventDescription','event_description','text','actionDescription','action_description','DESCRIPTION'))
        action=norm(val(row,'actionType','action_type','ACTION_TYPE','action','shotActionType','shot_action_type'))
        result=norm(val(row,'result','RESULT','shotResult','shot_result'))
        event_type=norm(val(row,'eventType','event_type','EVENT_TYPE','type'))
        combined=' | '.join([pname,desc,action,event_type,result])
        if 'steven adams' not in combined.lower() and pid not in {'203500'}:
            continue
        if 'dunk' not in combined.lower():
            continue
        # Require a make if a make/miss field or description gives enough signal.
        low=combined.lower()
        if result and result.lower() not in {'made','make','made shot','1','true'} and 'makes' not in low and 'made' not in low:
            continue
        if any(x in low for x in ['misses','missed dunk','missed']):
            continue
        gid=norm(val(row,'gameId','game_id','GAME_ID','gameID'))
        eid=norm(val(row,'eventNum','event_num','EVENTNUM','eventId','event_id','EVENT_ID','actionNumber','action_number'))
        if not gid or not eid: continue
        try: ei=int(float(eid))
        except: continue
        key=(gid,ei)
        if key in seen: continue
        seen.add(key)
        date=norm(val(row,'gameDate','game_date','GAME_DATE','date'))
        period=norm(val(row,'period','PERIOD','quarter'))
        clock=norm(val(row,'gameClock','game_clock','GAME_CLOCK','clock','time'))
        matchup=norm(val(row,'matchup','MATCHUP'))
        out.append({'game_id':gid,'event_id':ei,'event_num':ei,'game_date':date,'matchup':matchup,'period':period,'game_clock':clock,'action_type':action,'description':desc,'player_id':203500})

    def dtkey(x):
        s=x.get('game_date','')
        for fmt in ('%Y-%m-%d','%m/%d/%Y','%Y/%m/%d'):
            try:return datetime.strptime(s,fmt)
            except: pass
        return datetime.max
    out.sort(key=lambda x:(dtkey(x),x['game_id'],x['event_id']))
    for i,e in enumerate(out,1): e['rank']=i
    payload={'source':URL,'player':'Steven Adams','player_id':203500,'season':'2025-26','filter':'made dunks','expected_count':len(out),'events':out}
    open(OUT,'w').write(json.dumps(payload,indent=2))
    print('DUNK_COUNT',len(out))
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
