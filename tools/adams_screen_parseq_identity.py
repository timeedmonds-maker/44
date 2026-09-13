#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd


def parse_lineup_ids(text):
    return [int(x) for x in re.findall(r'(?<!\d)(\d{6,7})(?!\d)',str(text))]


def load_roster(path):
    raw=json.loads(Path(path).read_text()); out={}
    for team,t in raw['teams'].items():
        out[team]=[{**p,'id':int(p['id']),'jersey':str(p['jersey'])} for p in t['players']]
    return out


def read_crops(path):
    out=[]
    for p in sorted(Path(path).glob('*.jpg')):
        im=cv2.imread(str(p))
        if im is not None and im.size: out.append(im)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--candidates',required=True); ap.add_argument('--crop-root',required=True); ap.add_argument('--roster',required=True); ap.add_argument('--nbacv-src',required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(Path(a.nbacv_src)))
    from nbacv.jersey import load_models, _legibility_scores, _parseq_read, canon_number, LEGIBLE_TH, MIN_VOTE_WEIGHT, MIN_VOTE_SHARE
    parseq,leg,device=load_models(device='cpu')
    roster=load_roster(a.roster); df=pd.read_csv(a.candidates,dtype={'game_id':str}); rows=[]; read_detail=[]
    roles=['screener','ballhandler','screened_defender','screener_defender']
    for _,r in df.iterrows():
        rec={'candidate_id':r.candidate_id,'game_id':str(r.game_id).zfill(10),'possession_uid':r.possession_uid,'period':r.period,'start_time':r.start_time,'end_time':r.end_time,'event_num':r.event_num,'start_s':r.start_s,'end_s':r.end_s,'pts_poss':r.pts_poss,'type_end':r.type_end}
        lineups={'HOU':set(parse_lineup_ids(r.lineup_team)),'OKC':set(parse_lineup_ids(r.lineup_opp))}
        for role in roles:
            cs=read_crops(Path(a.crop_root)/str(r.candidate_id)/role)
            votes=defaultdict(float); n_leg=0; n_reads=0; evidence=[]
            if cs:
                scores=_legibility_scores(leg,cs,device); legible=[(c,float(s),i) for i,(c,s) in enumerate(zip(cs,scores)) if s>=LEGIBLE_TH]; legible.sort(key=lambda x:-x[1]); n_leg=len(legible)
                if legible:
                    reads=_parseq_read(parseq,[x[0] for x in legible],device)
                    for (text,conf),(_,ls,idx) in zip(reads,legible):
                        n=canon_number(text); evidence.append({'crop_index':idx,'legibility':round(ls,3),'raw':text,'canon':n,'parseq_conf':round(float(conf),3)})
                        if n is not None and conf>.5: votes[n]+=float(conf); n_reads+=1
            jersey=None; conf=0.; best_weight=0.; share=0.
            if votes:
                total=sum(votes.values()); jersey,best_weight=max(votes.items(),key=lambda kv:kv[1]); share=best_weight/total
                if best_weight<MIN_VOTE_WEIGHT or share<MIN_VOTE_SHARE: jersey=None
                else: conf=float(share*min(1.,best_weight/3.))
            team='HOU' if role in ('screener','ballhandler') else 'OKC'
            matches=[]
            if jersey is not None:
                matches=[p for p in roster[team] if p['id'] in lineups[team] and p['jersey']==str(jersey)]
            player=matches[0] if len(matches)==1 else None
            rec[f'{role}_jersey']=jersey; rec[f'{role}_jersey_conf']=round(conf,3); rec[f'{role}_player_id']=None if player is None else player['id']; rec[f'{role}_name']=None if player is None else player['name']; rec[f'{role}_ocr_crops']=len(cs); rec[f'{role}_legible_crops']=n_leg
            read_detail.append({'candidate_id':r.candidate_id,'role':role,'team':team,'jersey':jersey,'confidence':round(conf,3),'votes':dict(sorted(votes.items(),key=lambda kv:-kv[1])),'evidence':evidence})
        sid=rec.get('screener_player_id')
        if sid==203500: status='confirmed_adams_screen'
        elif pd.notna(sid): status='rejected_other_screener'
        elif str(rec.get('screener_jersey') or '')=='12': status='probable_adams_screen'
        else: status='identity_ambiguous'
        rec['identity_status']=status; rows.append(rec)
    out=pd.DataFrame(rows); out.to_csv(a.out/'screen_identity.csv',index=False); (a.out/'ocr_evidence.json').write_text(json.dumps(read_detail,indent=2))
    qa={'candidates':len(out),'confirmed_adams':int((out.identity_status=='confirmed_adams_screen').sum()),'probable_adams':int((out.identity_status=='probable_adams_screen').sum()),'rejected_other_screener':int((out.identity_status=='rejected_other_screener').sum()),'ambiguous':int((out.identity_status=='identity_ambiguous').sum()),'named_ballhandlers':int(out.ballhandler_name.notna().sum()),'named_screened_defenders':int(out.screened_defender_name.notna().sum()),'named_screener_defenders':int(out.screener_defender_name.notna().sum())}
    (a.out/'qa.json').write_text(json.dumps(qa,indent=2)); print(json.dumps(qa,indent=2))
if __name__=='__main__': main()
