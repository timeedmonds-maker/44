from __future__ import annotations
# PR trigger: inspect Steven Adams 2025-26 block rows and shot-location fields.
import csv, json, urllib.request

URL='https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
ADAMS_ID='203500'
UA='Mozilla/5.0'

def main():
    req=urllib.request.Request(URL,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=180) as r, open('pbp.csv','wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
    rows=[]
    with open('pbp.csv',newline='',encoding='utf-8-sig',errors='replace') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        print('FIELDS='+json.dumps(fields))
        coord=[c for c in fields if any(k in c.lower() for k in ('loc_','shot_distance','distance','x','y'))]
        player_cols=[c for c in fields if c.lower().startswith('player')]
        print('COORD_CANDIDATES='+json.dumps(coord))
        print('PLAYER_COLS='+json.dumps(player_cols))
        for row in reader:
            desc=(row.get('description') or '')
            vals=' | '.join((row.get(c) or '') for c in player_cols)
            if ADAMS_ID not in vals and 'ADAMS' not in desc.upper():
                continue
            if (row.get('is_field_goal') or '').strip()!='1' and 'BLOCK' not in desc.upper():
                continue
            rec={c:row.get(c) for c in fields if c in set(['game_id','event_num','game_date','team_home','team_away','period','clock','msg_type','is_field_goal','shot_result','shot_type','action_type','sub_type','descriptor','description','team_abb','player1_name','player2_name','player3_name','loc_x','loc_y','shot_distance']) or c in coord or c in player_cols}
            rows.append(rec)
    print('ADAMS_BLOCK_CANDIDATE_ROWS='+str(len(rows)))
    for r in rows: print('ROW='+json.dumps(r,ensure_ascii=False))
    open('adams_block_diagnostic.json','w').write(json.dumps({'fields':fields,'coord_candidates':coord,'player_cols':player_cols,'rows':rows},indent=2))

if __name__=='__main__': main()
