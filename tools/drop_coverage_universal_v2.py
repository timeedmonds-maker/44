#!/usr/bin/env python3
"""DROP_COVERAGE_UNIVERSAL_V2

Deterministic enduring drop-coverage resolver + renderer.

Universal role identity order:
1. exact game/event;
2. exact PBP defensive lineup;
3. validated screen roles from the tracker;
4. drop defender = screener defender (NOT a guessed center / tallest player);
5. identity must belong to exact event lineup;
6. stitch raw tracker re-identifications into one durable role identity;
7. render both name bar and floor ring from that identity-stable role track.

Universal team-colour rule:
- use each team's declared primary colour by default;
- if the two primary colours are too similar, preserve the defensive team's
  primary colour and switch the offensive team to its declared secondary;
- fail QA rather than guess a colour when a required secondary is absent or
  remains too similar.

No generated imagery and no AI super-resolution.
"""
from pathlib import Path
import argparse, copy, json, math
import pandas as pd
import locked_broadcast_screen_v2 as core
import drop_coverage_locked_broadcast_v1_fixed_tracking as fixed

TOOL_ID='DROP_COVERAGE_UNIVERSAL_V2'


def _player_ids(lineup):
    return {int(p['player_id']) for p in lineup}


def _player_names(lineup):
    return {str(p['name']) for p in lineup}


def _rgb_distance(a,b):
    aa=[int(x) for x in a]; bb=[int(x) for x in b]
    if len(aa)!=3 or len(bb)!=3:
        raise ValueError((a,b))
    return math.sqrt(sum((x-y)**2 for x,y in zip(aa,bb)))


def _hex_from_rgb(rgb):
    return '#%02X%02X%02X' % tuple(int(x) for x in rgb)


def resolve_team_colours(cfg, manifest):
    runtime=copy.deepcopy(cfg)
    policy=runtime.get('team_colour_collision_policy',{})
    enabled=bool(policy.get('enabled',True))
    threshold=float(policy.get('rgb_distance_threshold',80.0))
    offense=str(manifest['offense']); defense=str(manifest['defense'])
    if offense not in runtime['teams'] or defense not in runtime['teams']:
        raise RuntimeError(f'Missing team palette for offense={offense} defense={defense}')

    op=runtime['teams'][offense]; de=runtime['teams'][defense]
    op_primary=[int(x) for x in op['primary_rgb']]
    de_primary=[int(x) for x in de['primary_rgb']]
    primary_distance=_rgb_distance(op_primary,de_primary)
    collision=enabled and primary_distance < threshold

    qa={
        'enabled':enabled,
        'rule':'defense_keeps_primary_offense_switches_secondary_when_primaries_similar',
        'rgb_distance_threshold':threshold,
        'offense':offense,
        'defense':defense,
        'offense_primary_rgb':op_primary,
        'defense_primary_rgb':de_primary,
        'primary_rgb_distance':primary_distance,
        'collision_detected':collision,
        'offense_render_rgb':op_primary,
        'defense_render_rgb':de_primary,
        'offense_render_source':'primary',
        'defense_render_source':'primary'
    }

    if collision:
        if 'secondary_rgb' not in op:
            raise RuntimeError(f'Primary-colour collision but {offense} has no secondary_rgb')
        secondary=[int(x) for x in op['secondary_rgb']]
        secondary_distance=_rgb_distance(secondary,de_primary)
        if bool(policy.get('fail_if_secondary_collision',True)) and secondary_distance < threshold:
            raise RuntimeError(
                f'Primary-colour collision and {offense} secondary remains too similar: '
                f'distance={secondary_distance:.2f} threshold={threshold:.2f}'
            )
        runtime['teams'][offense]['primary_rgb']=secondary
        runtime['teams'][offense]['hex']=_hex_from_rgb(secondary)
        qa.update({
            'offense_render_rgb':secondary,
            'offense_render_hex':_hex_from_rgb(secondary),
            'offense_render_source':'secondary',
            'secondary_vs_defense_primary_rgb_distance':secondary_distance,
            'collision_resolution':'offense_secondary'
        })
    else:
        qa['offense_render_hex']=_hex_from_rgb(op_primary)
        qa['collision_resolution']='none'
    qa['defense_render_hex']=_hex_from_rgb(de_primary)
    return runtime, qa


def add_drop_identity_track(stitched_csv, cfg, out_csv):
    df=pd.read_csv(stitched_csv)
    spec=cfg['drop_big_stitch']
    virtual=int(spec['virtual_track_id'])
    pieces=[]
    segment_qa=[]
    for seg in spec['segments']:
        tid=int(seg['raw_track_id']); s=float(seg['start_s']); e=float(seg['end_s_exclusive'])
        g=df[(df.track_id==tid)&(df.time_s>=s)&(df.time_s<e)].copy()
        if g.empty:
            raise RuntimeError(f'No raw-track rows for drop-big segment track={tid} {s}-{e}')
        g['track_id']=virtual
        pieces.append(g)
        segment_qa.append({
            'raw_track_id':tid,'start_s':float(g.time_s.min()),'end_s':float(g.time_s.max()),'rows':int(len(g)),
            'first_center':[float((g.iloc[0].x1+g.iloc[0].x2)/2),float((g.iloc[0].y1+g.iloc[0].y2)/2)],
            'last_center':[float((g.iloc[-1].x1+g.iloc[-1].x2)/2),float((g.iloc[-1].y1+g.iloc[-1].y2)/2)]
        })
    drop=pd.concat(pieces,ignore_index=True).sort_values('time_s')
    drop=drop.drop_duplicates(subset=['time_s'],keep='last')
    out=pd.concat([df,drop],ignore_index=True).sort_values(['track_id','time_s'])
    out.to_csv(out_csv,index=False)

    jumps=[]
    for a,b in zip(segment_qa,segment_qa[1:]):
        ax,ay=a['last_center']; bx,by=b['first_center']
        jump=((ax-bx)**2+(ay-by)**2)**0.5
        jumps.append(float(jump))
        if jump>80:
            raise RuntimeError(f'Drop-big identity stitch jump too large: {jump:.1f}px')
    return {'segments':segment_qa,'boundary_center_jumps_px':jumps,'virtual_track_id':virtual}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True)
    ap.add_argument('--tracks',required=True)
    ap.add_argument('--source-qa',required=True)
    ap.add_argument('--config',required=True)
    ap.add_argument('--role-manifest',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()

    cfg=json.load(open(a.config)); manifest=json.load(open(a.role_manifest)); source_qa=json.load(open(a.source_qa))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    assert str(cfg['event']['game_id'])==str(manifest['game_id'])==str(source_qa['game_id'])
    assert int(cfg['event']['event_num'])==int(manifest['event_num'])==int(source_qa['event_num'])
    assert source_qa['source']['angle']=='Broadcast', source_qa['source']

    defense=manifest['defensive_lineup']; def_ids=_player_ids(defense); def_names=_player_names(defense)
    drop=manifest['screen_roles']['screener_defender']; drop_cfg=cfg['drop_big_stitch']

    assert int(drop['player_id']) in def_ids, (drop,defense)
    assert str(drop['name']) in def_names, (drop,defense)
    assert int(drop_cfg['player_id'])==int(drop['player_id'])
    assert str(drop_cfg['identity'])==str(drop['name'])
    assert int(source_qa['roles']['adams_defender'])==int(drop_cfg['role_source_track_id'])==int(drop['source_role_track'])
    assert str(drop_cfg['forbidden_identity']) not in def_names, (drop_cfg['forbidden_identity'],defense)
    assert int(drop_cfg['jersey'])==54 and int(drop_cfg['role_source_track_id'])==4

    stage1=out/'identity_stitched_primary_tracks.csv'
    primary_qa=fixed.stitch_tracks(a.tracks,cfg,stage1)
    final_tracks=out/'identity_stitched_all_roles.csv'
    drop_qa=add_drop_identity_track(stage1,cfg,final_tracks)

    players={int(p['track_id']):p['label'] for p in cfg['players']}
    drop_virtual=int(drop_cfg['virtual_track_id'])
    assert players[drop_virtual]=='MAMUKELASHVILI'
    assert cfg['roles']['drop_coverage_defender_track_id']==drop_virtual

    runtime_cfg, colour_qa=resolve_team_colours(cfg,manifest)
    runtime_cfg_path=out/'runtime_resolved_config.json'
    runtime_cfg_path.write_text(json.dumps(runtime_cfg,indent=2))

    core.TOOL_ID=TOOL_ID
    core.render(a.source,str(final_tracks),str(runtime_cfg_path),a.out)

    qa=json.load(open(out/'qa.json'))
    qa.update({
        'tool_id':TOOL_ID,
        'canonical_tool_family':'DROP_COVERAGE_LOCKED_BROADCAST',
        'universal_role_resolution_v2':True,
        'universal_team_colour_collision_rule_v1':True,
        'team_colour_resolution':colour_qa,
        'exact_event':{'game_id':manifest['game_id'],'event_num':manifest['event_num'],'period':manifest['period'],'clock':manifest['clock']},
        'exact_defensive_lineup':defense,
        'drop_coverage_role':{
            'definition':'screener defender in the validated screen-role model',
            'source_role_key':'adams_defender',
            'source_track_id':int(drop_cfg['role_source_track_id']),
            'player_id':int(drop_cfg['player_id']),
            'identity':drop_cfg['identity'],
            'jersey':int(drop_cfg['jersey']),
            'virtual_track_id':drop_virtual
        },
        'primary_identity_stitch_qa':primary_qa,
        'drop_big_identity_stitch_qa':drop_qa,
        'name_lock':{
            'shooter':'SHEPPARD','screener':'ADAMS','point_of_attack_defender':'SHEAD','drop_coverage_defender':'MAMUKELASHVILI'
        },
        'visual_lock':{'drop_defender_name_bar':True,'drop_defender_floor_ring':True},
        'forbidden_identity_assertion':'Jakob Poeltl is not in the exact event defensive lineup and cannot pass QA',
        'universal_failure_policy':'fail if role identity or colour contrast cannot be reconciled; never guess'
    })
    (out/'qa.json').write_text(json.dumps(qa,indent=2))
    (out/'LOCKED_TOOL_ID.txt').write_text(TOOL_ID+'\n')
    print(json.dumps(qa,indent=2))

if __name__=='__main__':
    main()
