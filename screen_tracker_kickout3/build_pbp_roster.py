#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PLAYERS={
  1628988:('Aaron Holiday','0'), 1631095:('Jabari Smith Jr.','10'), 1642263:('Reed Sheppard','15'), 201142:('Kevin Durant','7'), 203500:('Steven Adams','12'),
  1641708:('Amen Thompson','1'),
  1628366:('Lonzo Ball','2'), 1628386:('Jarrett Allen','31'), 1629631:("De'Andre Hunter",'12'), 1641772:("Nae'Qwan Tomlin",'35'), 1642878:('Tyrese Proctor','24'),
  1627742:('Brandon Ingram','3'), 1629628:('RJ Barrett','9'), 1630193:('Immanuel Quickley','5'), 1641711:('Gradey Dick','1'), 1642867:('Collin Murray-Boyles','12'),
}

def ids(s): return [int(x) for x in re.findall(r'(?<!\d)(\d{6,7})(?!\d)',str(s))]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--context',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 p=(json.loads(a.context.read_text()).get('pbp') or {})
 home,away=ids(p.get('lineup_home')),ids(p.get('lineup_away'))
 assert len(home)==5 and len(away)==5,(home,away)
 missing=[i for i in home+away if i not in PLAYERS]
 if missing: raise RuntimeError(f'missing jersey metadata for exact PBP ids {missing}')
 def team(xs): return {'players':[{'id':i,'name':PLAYERS[i][0],'jersey':PLAYERS[i][1]} for i in xs]}
 out={'teams':{'home':team(home),'away':team(away)},'source':'exact context.json PBP lineup + 2025-26 jersey metadata','lineup_home':p.get('lineup_home'),'lineup_away':p.get('lineup_away')}
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
