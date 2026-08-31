from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import nba_video_worker as w
from build_kd_top10_all_angles import inventory

MAX_REEL_SOURCE_BYTES = 85_000_000


def safe(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_') or 'clip'


def choose_broadcast(options: list[dict]) -> dict:
    for o in options:
        if (o.get('label') or '').strip().lower() == 'broadcast':
            return o
    for o in options:
        if o.get('page_selected'):
            return o
    return options[0]


def concat_copy(files: list[Path], out: Path) -> None:
    lst = out.with_suffix('.txt')
    lst.write_text('\n'.join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in files) + '\n')
    w.run([w.FFMPEG, '-nostdin', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', '-movflags', '+faststart', str(out)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', default='request.json')
    ap.add_argument('--out', default='adams_rim_blocks_output')
    a = ap.parse_args()

    req = json.loads(Path(a.request).read_text())
    events = req.get('events') or []
    if not events:
        raise SystemExit('No rim-block events in request')

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    clips = out / 'clips'
    reels = out / 'reels'
    clips.mkdir(parents=True)
    reels.mkdir(parents=True)

    clip_rows = []
    sha_seen = {}
    for e in sorted(events, key=lambda x: int(x['rank'])):
        rank = int(e['rank'])
        gid = str(e['game_id'])
        eid = int(e['event_id'])
        page, title, opts = inventory(gid, eid)
        chosen = choose_broadcast(opts)
        label = chosen.get('label') or 'selected'
        dst = clips / f"R{rank:02d}_{gid}_{eid}_{safe(label)}_SOURCE.mp4"
        w.download_hls_source(chosen['url'], dst)
        probe = w.probe_video(dst)
        if not probe.get('ok'):
            raise RuntimeError(f'Video QA failed for {gid}/{eid}: {probe}')
        sha = probe['sha256']
        if sha in sha_seen:
            raise RuntimeError(f'Exact duplicate media across distinct block events: {sha_seen[sha]} vs {(gid,eid)}')
        sha_seen[sha] = (gid, eid)
        clip_rows.append({
            'rank': rank,
            'game_id': gid,
            'event_id': eid,
            'game_date': e.get('game_date'),
            'matchup': f"{e.get('team_away')} @ {e.get('team_home')}",
            'period': e.get('period'),
            'clock': e.get('clock'),
            'shooter': e.get('shooter'),
            'description': e.get('description'),
            'shot_distance_ft': e.get('shot_distance_ft'),
            'area': e.get('area'),
            'area_detail': e.get('area_detail'),
            'clips_page': page,
            'clips_page_title': title,
            'angle_label': label,
            'angle_count_available': len(opts),
            'source_path': str(dst),
            'source_bytes': dst.stat().st_size,
            'probe': probe,
        })
        print(f"R{rank:02d} {gid}/{eid} {e.get('shot_distance_ft')}ft angle={label} bytes={dst.stat().st_size}", flush=True)

    groups = []
    cur = []
    cur_bytes = 0
    for row in clip_rows:
        p = Path(row['source_path'])
        sz = p.stat().st_size
        if cur and cur_bytes + sz > MAX_REEL_SOURCE_BYTES:
            groups.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(row)
        cur_bytes += sz
    if cur:
        groups.append(cur)

    reel_rows = []
    for i, group in enumerate(groups, 1):
        first_rank = int(group[0]['rank'])
        last_rank = int(group[-1]['rank'])
        rp = reels / f'steven_adams_rim_blocks_2025_26_reel_{i:02d}_R{first_rank:02d}-R{last_rank:02d}_NATIVE.mp4'
        concat_copy([Path(r['source_path']) for r in group], rp)
        probe = w.probe_video(rp)
        if not probe.get('ok'):
            raise RuntimeError(f'Reel QA failed {rp}: {probe}')
        if rp.stat().st_size >= 100_000_000:
            raise RuntimeError(f'Reel exceeds 100 MB hard cap: {rp} {rp.stat().st_size}')
        reel_rows.append({
            'reel': i,
            'path': str(rp),
            'bytes': rp.stat().st_size,
            'mb_decimal': round(rp.stat().st_size / 1_000_000, 2),
            'first_rank': first_rank,
            'last_rank': last_rank,
            'event_count': len(group),
            'event_ranks': [int(r['rank']) for r in group],
            'probe': probe,
        })
        print(f"REEL_{i:02d}={rp} MB={rp.stat().st_size/1_000_000:.2f} EVENTS={len(group)}", flush=True)

    qa = {
        'player': 'Steven Adams',
        'player_id': 203500,
        'season': '2025-26',
        'definition': req.get('rim_definition'),
        'event_count': len(events),
        'validated_clip_count': len(clip_rows),
        'video_policy': 'official clips.nba.com Broadcast angle when available; highest native signed NBA HLS rendition; stream-copy only; no scaling or video re-encode',
        'clips': clip_rows,
        'reels': reel_rows,
    }
    (out / 'qa.json').write_text(json.dumps(qa, indent=2, ensure_ascii=False))
    (out / 'reels.json').write_text(json.dumps(reel_rows, indent=2))
    print(f'RIM_BLOCK_EVENTS={len(events)} VALIDATED_CLIPS={len(clip_rows)} REELS={len(reel_rows)}', flush=True)


if __name__ == '__main__':
    main()
