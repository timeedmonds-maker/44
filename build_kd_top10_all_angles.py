from __future__ import annotations
import argparse, html as htmlmod, json, re, shutil, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import nba_video_worker as w

ORDER={'Broadcast':0,'Other Broadcast':1,'Mobile Broadcast':2,'Play by Play':3,'In Arena':4,'High Tight':5,'Left Slash':6,'Right Slash':7,'Left HandHeld':8,'Right HandHeld':9,'Left Above Rim':10,'Right Above Rim':11}

def inventory(gid,eid):
    url=f'https://clips.nba.com/?gameNo={gid}&eventNum={eid}&source=grs'
    txt=w.http_bytes(url,w.H).decode('utf-8','replace')
    tm=re.search(r'<title>(.*?)</title>',txt,re.I|re.S); title=htmlmod.unescape(tm.group(1).strip()) if tm else ''
    opts=[]; seen=set()
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>',txt,re.I|re.S):
        u=htmlmod.unescape(m.group(1).strip()); lab=re.sub(r'<[^>]+>','',htmlmod.unescape(m.group(3))).strip(); sel='selected' in m.group(2).lower()
        if '.m3u8' not in u.lower() or 'lrmedia.nba.com' not in u.lower(): continue
        if (lab,u) in seen: continue
        seen.add((lab,u)); opts.append({'label':lab,'url':u,'page_selected':sel})
    if not opts: raise RuntimeError(f'No HLS options for {gid}/{eid}')
    return url,title,sorted(opts,key=lambda x:(ORDER.get(x['label'],99),x['label']))

def one(item, outdir):
    started=time.perf_counter(); gid=item['game_id']; eid=item['event_id']; lab=item['label']; safe=re.sub(r'[^A-Za-z0-9]+','_',lab).strip('_') or 'Angle'
    rec={k:item[k] for k in ('global_rank','shot_rank','angle_rank','game_id','event_id','label','page_selected','clips_page_title')}; rec['status']='failed'
    try:
        dst=outdir/f"{item['global_rank']:03d}_R{item['shot_rank']:02d}_{gid}_{eid}_{safe}_SOURCE.mp4"
        w.download_hls_source(item['url'],dst); q=w.probe_video(dst); rec['probe']=q
        if not q['ok']: raise RuntimeError(q['reason'])
        rec['source_path']=str(dst); rec['status']='ok'; rec['seconds']=round(time.perf_counter()-started,3)
    except Exception as e: rec['error']=repr(e); rec['seconds']=round(time.perf_counter()-started,3)
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--request',required=True); ap.add_argument('--out',default='output'); ap.add_argument('--workers',type=int,default=8); a=ap.parse_args()
    req=json.loads(Path(a.request).read_text()); events=req['events']; out=Path(a.out); shutil.rmtree(out,ignore_errors=True); clips=out/'clips'; clips.mkdir(parents=True)
    expanded=[]; plans=[]; g=0
    for e in sorted(events,key=lambda x:x['rank']):
        gid=str(e['game_id']); eid=int(e.get('event_id') or e.get('event_num')); page,title,opts=inventory(gid,eid)
        plans.append({'rank':e['rank'],'game_id':gid,'event_id':eid,'matchup':e.get('matchup'),'game_date':e.get('game_date'),'difficulty_score':e.get('hybrid_difficulty_score',e.get('xfg_pct')),'angle_count':len(opts),'angle_labels':[o['label'] for o in opts]})
        for ar,o in enumerate(opts,1):
            g+=1; expanded.append({'global_rank':g,'shot_rank':int(e['rank']),'angle_rank':ar,'game_id':gid,'event_id':eid,'label':o['label'],'url':o['url'],'page_selected':o['page_selected'],'clips_page_title':title})
    workers=max(1,min(a.workers,8,len(expanded))); results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(one,x,clips) for x in expanded]
        for f in as_completed(fs): results.append(f.result())
    results.sort(key=lambda x:x['global_rank'])

    # Exact byte-identical media across distinct events is a hard failure.
    # Coarse visual fingerprints are intentionally NOT a hard duplicate test: similar
    # NBA camera views can collide at 16x9 aHash resolution even for different plays.
    sha=defaultdict(list); fp=defaultdict(list)
    for r in results:
        if r['status']=='ok':
            sha[r['probe']['sha256']].append(r)
            fp[tuple(r['probe']['visual_fingerprints'])].append(r)

    bad=set()
    exact_duplicate_groups=[]
    for digest,group in sha.items():
        ev={(r['game_id'],r['event_id']) for r in group}
        if len(ev)>1:
            bad |= ev
            exact_duplicate_groups.append({'sha256':digest,'events':sorted([list(x) for x in ev]),'clips':[{'shot_rank':r['shot_rank'],'label':r['label']} for r in group]})

    visual_warnings=[]
    for fingerprints,group in fp.items():
        ev={(r['game_id'],r['event_id']) for r in group}
        if len(ev)>1:
            visual_warnings.append({
                'visual_fingerprints':list(fingerprints),
                'events':sorted([list(x) for x in ev]),
                'clips':[{'shot_rank':r['shot_rank'],'label':r['label'],'duration':(r.get('probe') or {}).get('duration'),'sha256':(r.get('probe') or {}).get('sha256')} for r in group],
                'disposition':'warning_only_not_duplicate_failure',
            })

    for r in results:
        if (r['game_id'],r['event_id']) in bad:
            r['status']='failed'; r['global_qa_failure']='exact_sha256_duplicate_media_across_distinct_events'
    ok=[r for r in results if r['status']=='ok']
    if len(ok)==len(results):
        concat=out/'concat.txt'; concat.write_text('\n'.join("file '"+str(Path(r['source_path']).resolve()).replace("'","'\\''")+"'" for r in ok)+'\n')
        try: w.run([w.FFMPEG,'-nostdin','-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(out/'reel_SOURCE.mp4')])
        except Exception as e: print('REEL_ASSEMBLY_FAILED',repr(e),flush=True)
    payload={
        'mode':'all_camera_angles_source',
        'policy':'For every ranked exact NBA event, include every clips.nba.com lrmedia camera option; each angle uses its highest native HLS rendition. No scaling/re-encode in source clips. Known placeholder QA remains enforced. Exact cross-event SHA256 duplication is a hard failure; coarse visual-fingerprint collisions are warning-only.',
        'request':req,
        'shot_plans':plans,
        'expanded_clip_count':len(results),
        'valid_clip_count':len(ok),
        'exact_cross_event_duplicate_groups':exact_duplicate_groups,
        'visual_fingerprint_collision_warnings':visual_warnings,
        'clips':results,
    }
    (out/'qa.json').write_text(json.dumps(payload,indent=2))
    print(f'SHOTS={len(events)} ANGLE_CLIPS={len(results)} VALID={len(ok)}',flush=True)
    print(f'EXACT_CROSS_EVENT_DUPLICATE_GROUPS={len(exact_duplicate_groups)} VISUAL_COLLISION_WARNINGS={len(visual_warnings)}',flush=True)
    for p in plans: print(f"R{p['rank']} {p['game_id']}/{p['event_id']} angles={p['angle_count']}",flush=True)
    if len(ok)!=len(results): sys.exit(2)
if __name__=='__main__': main()
