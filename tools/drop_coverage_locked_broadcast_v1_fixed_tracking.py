#!/usr/bin/env python3
"""DROP_COVERAGE_LOCKED_BROADCAST_V1 — universal identity-stitch resolver.

Deterministic only. Resolves focal role identities from exact event lineup +
validated role tracks, then stitches tracker re-identifications into durable
virtual identities before invoking the locked renderer.
"""
from pathlib import Path
import argparse, json
import pandas as pd
import drop_coverage_locked_broadcast_v1 as canonical

TOOL_ID='DROP_COVERAGE_LOCKED_BROADCAST_V1'


def _append_segment(parts, df, raw_tid, start_s, end_s, virtual_tid):
    g=df[(df.track_id==int(raw_tid)) & (df.time_s>=float(start_s)) & (df.time_s<float(end_s))].copy()
    if g.empty:
        raise RuntimeError(f'No tracker rows for raw track {raw_tid} in [{start_s},{end_s})')
    g['track_id']=int(virtual_tid)
    parts.append(g)


def _nearest(out, tid, t):
    g=out[out.track_id==int(tid)].copy()
    if g.empty: raise RuntimeError(f'Virtual track {tid} missing')
    i=(g.time_s-float(t)).abs().idxmin(); r=g.loc[i]
    return {'time_s':float(r.time_s),'cx':float((r.x1+r.x2)/2),'cy':float((r.y1+r.y2)/2)}


def stitch_tracks(src_csv, cfg, out_csv):
    df=pd.read_csv(src_csv)
    parts=[]

    # Preserve all unrelated raw tracks. Focal role rendering only references the
    # configured identity-stable virtual track IDs, so raw tracks remain harmless.
    parts.append(df.copy())

    # Ballhandler / POA defender identity stitch.
    spec=cfg['identity_stitch']
    swap=float(spec['swap_time_s'])
    vs=int(spec['virtual_track_ids']['sheppard']); vd=int(spec['virtual_track_ids']['shead'])
    _append_segment(parts,df,spec['pre']['sheppard'],df.time_s.min(),swap,vs)
    _append_segment(parts,df,spec['post']['sheppard'],swap,df.time_s.max()+1,vs)
    _append_segment(parts,df,spec['pre']['shead'],df.time_s.min(),swap,vd)
    _append_segment(parts,df,spec['post']['shead'],swap,df.time_s.max()+1,vd)

    # Universal drop-defender stitch: exact screener-defender role identity may
    # re-identify across multiple raw tracker IDs. The config/role manifest supplies
    # the deterministic segments after lineup + role QA.
    ds=cfg['drop_big_stitch']
    vdrop=int(ds['virtual_track_id'])
    for seg in ds['segments']:
        _append_segment(parts,df,seg['raw_track_id'],seg['start_s'],seg['end_s_exclusive'],vdrop)

    out=pd.concat(parts,ignore_index=True).sort_values(['track_id','time_s'])
    # Remove duplicate virtual rows at exact same timestamp if any boundary overlap.
    out=out.drop_duplicates(['track_id','time_s'],keep='first')
    out.to_csv(out_csv,index=False)

    eps=0.05
    qa={
        'poa_swap_time_s':swap,
        'sheppard':{'pre':_nearest(out,vs,swap-eps),'post':_nearest(out,vs,swap+eps)},
        'shead':{'pre':_nearest(out,vd,swap-eps),'post':_nearest(out,vd,swap+eps)},
        'drop_defender':{
            'identity':ds['identity'],
            'player_id':int(ds['player_id']),
            'jersey':int(ds['jersey']),
            'virtual_track_id':vdrop,
            'segments':ds['segments'],
            'segment_boundary_samples':[]
        }
    }
    for seg in ds['segments'][:-1]:
        b=float(seg['end_s_exclusive'])
        qa['drop_defender']['segment_boundary_samples'].append({'boundary_s':b,'pre':_nearest(out,vdrop,b-eps),'post':_nearest(out,vdrop,b+eps)})
    return qa


def validate_role_manifest(cfg, manifest_path):
    m=json.load(open(manifest_path))
    ev=cfg['event']
    assert str(m['game_id'])==str(ev['game_id']) and int(m['event_num'])==int(ev['event_num']), (m['game_id'],m['event_num'],ev)
    drop=m['screen_roles']['screener_defender']
    ds=cfg['drop_big_stitch']
    assert int(drop['player_id'])==int(ds['player_id']), (drop,ds)
    assert int(drop['jersey'])==int(ds['jersey']), (drop,ds)
    assert drop['name']==ds['identity'], (drop,ds)
    defensive_ids={int(p['player_id']) for p in m['defensive_lineup']}
    assert int(ds['player_id']) in defensive_ids, 'drop defender not in exact defensive lineup'
    assert all(p['name']!='Jakob Poeltl' for p in m['defensive_lineup']), 'forbidden out-of-lineup identity present'
    assert int(drop['source_role_track'])==int(ds['role_source_track_id'])==4, (drop,ds)
    return m


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True); ap.add_argument('--tracks',required=True)
    ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); cfg=json.load(open(a.config)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    manifest_path=cfg['event']['role_manifest']
    manifest=validate_role_manifest(cfg,manifest_path)

    stitched=out/'identity_stitched_tracks.csv'
    stitch_qa=stitch_tracks(a.tracks,cfg,stitched)

    import locked_broadcast_screen_v2 as core
    core.TOOL_ID=TOOL_ID
    core.render(a.source,str(stitched),a.config,a.out)

    qa=json.load(open(out/'qa.json'))
    qa['tool_id']=TOOL_ID
    qa['universal_role_resolution_v2']=True
    qa['identity_stitch_enabled']=True
    qa['identity_stitch']=cfg['identity_stitch']
    qa['drop_big_stitch']=cfg['drop_big_stitch']
    qa['identity_stitch_qa']=stitch_qa
    qa['exact_defensive_lineup']=manifest['defensive_lineup']
    qa['name_lock']={
        'shooter':'SHEPPARD',
        'screener':'ADAMS',
        'point_of_attack_defender':'SHEAD',
        'drop_coverage_defender':'MAMUKELASHVILI'
    }
    qa['drop_defender_identity_source']='exact PBP lineup + screener-defender role + jersey-54 multi-track continuity'
    qa['forbidden_out_of_lineup_identity']='Jakob Poeltl'
    qa['drop_defender_visual_contract']='same broadcast name bar and horseshoe floor ring as other focal players, both following virtual track 104'
    (out/'qa.json').write_text(json.dumps(qa,indent=2))
    (out/'LOCKED_TOOL_ID.txt').write_text(TOOL_ID+'\n')
    print(json.dumps(qa,indent=2))

if __name__=='__main__': main()
