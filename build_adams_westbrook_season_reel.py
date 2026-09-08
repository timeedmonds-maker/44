from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nba_video_worker as w
from build_adams_westbrook_381_reel import download_event, concat_copy

EXPECTED = {'2013-14': 6, '2014-15': 31, '2015-16': 80, '2016-17': 80, '2017-18': 111, '2018-19': 73}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='adams_westbrook_dunk_event_links.json')
    ap.add_argument('--season', required=True, choices=sorted(EXPECTED))
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    payload = json.loads(Path(args.manifest).read_text())
    all_events = payload.get('events') or []
    if len(all_events) != 381:
        raise SystemExit(f'Full manifest must contain 381 events, found {len(all_events)}')
    events = [dict(e) for e in all_events if e.get('season') == args.season]
    if len(events) != EXPECTED[args.season]:
        raise SystemExit(f'{args.season}: expected {EXPECTED[args.season]} events, found {len(events)}')
    for i, e in enumerate(events, 1):
        e['rank'] = i
        e['event_id'] = int(e.get('event_id') or e.get('event_num'))
        e['game_id'] = str(e['game_id']).zfill(10)

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    clips = out / 'clips'
    clips.mkdir(parents=True)
    workers = max(1, min(args.workers, 6))

    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f'aw-{args.season}') as pool:
        fs = [pool.submit(download_event, e, clips) for e in events]
        for f in as_completed(fs):
            results.append(f.result())
    results.sort(key=lambda r: r['rank'])
    failures = [r for r in results if r.get('status') != 'ok']
    if failures:
        (out / 'qa_partial.json').write_text(json.dumps({'season': args.season, 'events': results}, indent=2))
        raise SystemExit(f'{args.season}: {len(failures)} video failures; refusing incomplete season reel')

    sha_groups = defaultdict(list)
    for r in results:
        sha_groups[r['probe']['sha256']].append((r['game_id'], r['event_id']))
    dupes = {sha: g for sha, g in sha_groups.items() if len(set(g)) > 1}
    if dupes:
        raise SystemExit(f'{args.season}: exact duplicate media across distinct events')

    files = [Path(r['source_path']) for r in results]
    source_duration = sum(float(r['probe']['duration']) for r in results)
    slug = args.season.replace('-', '_')
    reel = out / f'{slug}_steven_adams_westbrook_dunks_NATIVE.mp4'
    concat_copy(files, reel)
    probe = w.probe_video(reel)
    if not probe.get('ok'):
        raise SystemExit(f'{args.season}: reel QA failed: {probe}')
    if abs(float(probe['duration']) - source_duration) > max(3.0, source_duration * 0.005):
        raise SystemExit(f'{args.season}: reel duration mismatch')

    qa = {
        'season': args.season,
        'expected_events': EXPECTED[args.season],
        'validated_events': len(results),
        'source_duration_seconds_sum': source_duration,
        'reel_path': str(reel),
        'reel_bytes': reel.stat().st_size,
        'reel_probe': probe,
        'events': results,
    }
    (out / f'{slug}_qa.json').write_text(json.dumps(qa, indent=2))
    shutil.rmtree(clips)
    print(f'SEASON_OK {args.season} EVENTS={len(results)} REEL={reel} BYTES={reel.stat().st_size} DUR={probe["duration"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
