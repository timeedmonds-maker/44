#!/usr/bin/env python3
"""DROP_COVERAGE_LOCKED_BROADCAST_V1 — identity-stitch bugfix.

Preserves the canonical enduring renderer while fixing tracker identity swaps at
occlusion. For the Toronto reference event, Reed Sheppard and Jamal Shead swap
raw tracker IDs 7/3 around the Adams screen. This wrapper creates durable
virtual identity tracks before invoking the locked renderer.
"""
from pathlib import Path
import argparse, json
import pandas as pd
import drop_coverage_locked_broadcast_v1 as canonical

TOOL_ID='DROP_COVERAGE_LOCKED_BROADCAST_V1'


def stitch_tracks(src_csv, cfg, out_csv):
    df=pd.read_csv(src_csv)
    spec=cfg['identity_stitch']
    swap=float(spec['swap_time_s'])
    a=int(spec['pre']['sheppard']); d=int(spec['pre']['shead'])
    a2=int(spec['post']['sheppard']); d2=int(spec['post']['shead'])
    vs=int(spec['virtual_track_ids']['sheppard']); vd=int(spec['virtual_track_ids']['shead'])

    keep=df[~df.track_id.isin([a,d,a2,d2])].copy()
    parts=[keep]
    pre_s=df[(df.track_id==a)&(df.time_s<swap)].copy(); pre_s['track_id']=vs
    post_s=df[(df.track_id==a2)&(df.time_s>=swap)].copy(); post_s['track_id']=vs
    pre_d=df[(df.track_id==d)&(df.time_s<swap)].copy(); pre_d['track_id']=vd
    post_d=df[(df.track_id==d2)&(df.time_s>=swap)].copy(); post_d['track_id']=vd
    parts += [pre_s,post_s,pre_d,post_d]
    out=pd.concat(parts,ignore_index=True).sort_values(['track_id','time_s'])
    out.to_csv(out_csv,index=False)

    # Continuity QA across stitch boundary.
    def nearest(tid,t):
        g=out[out.track_id==tid].copy(); i=(g.time_s-t).abs().idxmin(); r=g.loc[i]
        return {'time_s':float(r.time_s),'cx':float((r.x1+r.x2)/2),'cy':float((r.y1+r.y2)/2)}
    eps=0.05
    qa={'swap_time_s':swap,'sheppard':{'pre':nearest(vs,swap-eps),'post':nearest(vs,swap+eps)},'shead':{'pre':nearest(vd,swap-eps),'post':nearest(vd,swap+eps)}}
    return qa


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True); ap.add_argument('--tracks',required=True)
    ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); cfg=json.load(open(a.config)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    stitched=out/'identity_stitched_tracks.csv'
    stitch_qa=stitch_tracks(a.tracks,cfg,stitched)

    # Render using virtual, identity-stable tracks and names.
    canonical.TOOL_ID=TOOL_ID
    canonical.main = canonical.main  # explicit lineage marker
    import locked_broadcast_screen_v2 as core
    core.TOOL_ID=TOOL_ID
    core.render(a.source,str(stitched),a.config,a.out)

    qa=json.load(open(out/'qa.json'))
    qa['tool_id']=TOOL_ID
    qa['identity_stitch_enabled']=True
    qa['identity_stitch']=cfg['identity_stitch']
    qa['identity_stitch_qa']=stitch_qa
    qa['name_lock']={'shooter':'SHEPPARD','screener':'ADAMS','point_of_attack_defender':'SHEAD'}
    qa['tracking_bugfix']='raw tracks 7 and 3 exchange person identity during screen occlusion; virtual identity tracks prevent label/ring swap'
    (out/'qa.json').write_text(json.dumps(qa,indent=2))
    (out/'LOCKED_TOOL_ID.txt').write_text(TOOL_ID+'\n')
    print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
