from __future__ import annotations

import csv
import html as htmlmod
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from download_nba_angle_compare import download_hls, get, probe

GAME_ID = '0022500301'
ADAMS_ID = 203500
DUNK_EVENT_ID = 489
PBP_URL = 'https://github.com/ramirobentes/nba_pbp_data/releases/download/pbp-final-2026/data.csv'
PBP_FILE = Path('.runtime_pbp_2025_26.csv')
OUT = Path('deliveries/adams_jazz_0022500301_overhead')
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'


def norm_gid(v: object) -> str:
    s = str(v or '').strip()
    s = re.sub(r'\.0$', '', s)
    return s.zfill(10)


def clock_seconds(clock: object) -> float:
    s = str(clock or '').strip()
    m = re.fullmatch(r'PT(?:(\d+)M)?([0-9.]+)S', s, flags=re.I)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2))
    m = re.fullmatch(r'(?:(\d+):)?(\d{1,2}):([0-9.]+)', s)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r'(\d{1,2}):([0-9.]+)', s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return -1.0


def event_id(row: dict) -> int:
    for k in ('event_num', 'event_id', 'actionNumber', 'action_id'):
        try:
            return int(float(str(row.get(k) or '').strip()))
        except Exception:
            pass
    return 0


def download_pbp() -> None:
    if PBP_FILE.exists() and PBP_FILE.stat().st_size > 100_000_000:
        return
    req = urllib.request.Request(PBP_URL, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=180) as r, PBP_FILE.open('wb') as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def row_hay(row: dict) -> str:
    keys = (
        'description', 'event_type', 'action_type', 'sub_type', 'descriptor',
        'player1_name', 'player2_name', 'player3_name', 'player4_name',
        'block_player_name', 'blocker_player_name', 'shot_result',
    )
    return ' '.join(str(row.get(k) or '') for k in keys).lower()


def resolve_exact_events_from_pbp() -> dict[str, dict]:
    download_pbp()
    nearby: list[dict] = []
    with PBP_FILE.open(newline='', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if norm_gid(row.get('game_id')) != GAME_ID:
                continue
            try:
                period = int(float(str(row.get('period') or '0')))
            except Exception:
                period = 0
            if period != 3:
                continue
            sec = clock_seconds(row.get('clock') or row.get('game_clock'))
            if 235 <= sec <= 258:
                nearby.append(dict(row))

    if not nearby:
        raise RuntimeError('No Q3 PBP rows found around 4:11-4:03 for game ' + GAME_ID)

    block_candidates = []
    dunk_candidates = []
    for row in nearby:
        sec = clock_seconds(row.get('clock') or row.get('game_clock'))
        hay = row_hay(row)
        player_fields = ' '.join(str(row.get(k) or '') for k in ('player1_name','player2_name','player3_name','player4_name')).lower()
        has_adams = str(ADAMS_ID) in player_fields or 'steven adams' in hay or re.search(r'\badams\b', hay)
        if abs(sec - 251.0) <= 2.0 and has_adams and 'block' in hay:
            block_candidates.append(row)

        p1 = str(row.get('player1_name') or '').strip().lower()
        made = str(row.get('shot_result') or '').strip().lower() == 'made'
        is_fg = str(row.get('is_field_goal') or '').strip() in {'1', '1.0', 'true', 'True'}
        if abs(sec - 243.0) <= 2.0 and (p1.startswith(str(ADAMS_ID)) or 'steven adams' in p1) and made and is_fg and 'dunk' in hay:
            dunk_candidates.append(row)

    if not block_candidates:
        compact = [
            {
                'event_num': event_id(r),
                'clock': r.get('clock') or r.get('game_clock'),
                'description': r.get('description'),
                'player1_name': r.get('player1_name'),
                'player2_name': r.get('player2_name'),
                'event_type': r.get('event_type'),
                'action_type': r.get('action_type'),
                'sub_type': r.get('sub_type'),
            }
            for r in nearby
        ]
        raise RuntimeError('Could not resolve Adams block near 4:11 Q3. Nearby rows=' + json.dumps(compact, ensure_ascii=False))

    block_candidates.sort(key=lambda r: (abs(clock_seconds(r.get('clock') or r.get('game_clock')) - 251.0), event_id(r)))
    block = block_candidates[0]

    if dunk_candidates:
        dunk_candidates.sort(key=lambda r: (abs(clock_seconds(r.get('clock') or r.get('game_clock')) - 243.0), event_id(r)))
        dunk = dunk_candidates[0]
        if event_id(dunk) != DUNK_EVENT_ID:
            raise RuntimeError(f'Dunk cross-check failed: expected event {DUNK_EVENT_ID}, resolved {event_id(dunk)}')
    else:
        dunk = {
            'event_num': DUNK_EVENT_ID,
            'period': '3',
            'clock': '04:03',
            'description': 'Steven Adams dunk vs Utah',
        }

    return {'block': block, 'dunk': dunk}


def top_master_resolution(url: str) -> tuple[int, int]:
    try:
        txt = get(url).decode('utf-8', 'replace')
    except Exception:
        return (0, 0)
    best = (0, 0)
    for line in txt.splitlines():
        if not line.startswith('#EXT-X-STREAM-INF:'):
            continue
        m = re.search(r'RESOLUTION=(\d+)x(\d+)', line)
        if m:
            wh = (int(m.group(1)), int(m.group(2)))
            if wh[0] * wh[1] > best[0] * best[1]:
                best = wh
    return best


def above_rim_option(eid: int) -> tuple[str, str, list[str], tuple[int, int]]:
    page = f'https://clips.nba.com/?gameNo={GAME_ID}&eventNum={eid}&source=grs'
    txt = get(page).decode('utf-8', 'replace')
    opts = []
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', txt, flags=re.I | re.S):
        u = htmlmod.unescape(m.group(1).strip())
        lab = re.sub(r'<[^>]+>', '', htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in u.lower() and 'lrmedia.nba.com' in u.lower():
            opts.append((lab, u))
    if not opts:
        raise RuntimeError(f'No signed HLS angles for event {eid}')
    above = [(lab, u) for lab, u in opts if 'above rim' in lab.lower()]
    if not above:
        raise RuntimeError(f'No Above Rim angle for event {eid}; have {[x[0] for x in opts]}')

    priority = {'left above rim': 0, 'right above rim': 1, 'above rim': 2}
    scored = []
    for lab, u in above:
        res = top_master_resolution(u)
        scored.append((res[0] * res[1], -priority.get(lab.lower(), 9), lab, u, res))
    scored.sort(reverse=True)
    _, _, lab, url, advertised_res = scored[0]
    return lab, url, [x[0] for x in opts], advertised_res


def to_uhd(src: Path, dst: Path) -> None:
    subprocess.run([
        'ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(src),
        '-vf', 'scale=3840:2160:flags=lanczos,fps=30',
        '-c:v', 'libx264', '-profile:v', 'high', '-crf', '18',
        '-maxrate', '28M', '-bufsize', '56M',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(dst)
    ], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = resolve_exact_events_from_pbp()
    qa = {'game_id': GAME_ID, 'pbp_source': PBP_URL, 'events': {}}
    names = {
        'block': 'steven_adams_block_vs_jazz_above_rim_UHD.mp4',
        'dunk': 'steven_adams_dunk_vs_jazz_above_rim_UHD.mp4',
    }
    for kind in ('block', 'dunk'):
        row = events[kind]
        eid = event_id(row)
        label, hls, all_labels, advertised_res = above_rim_option(eid)
        source = OUT / f'.{kind}_above_rim_source.mp4'
        final = OUT / names[kind]
        download_hls(hls, source)
        source_qa = probe(source)
        if not source_qa.get('width') or not source_qa.get('height') or source_qa.get('duration', 0) < 2:
            raise RuntimeError(f'Invalid Above Rim source for {kind}: {source_qa}')
        to_uhd(source, final)
        final_qa = probe(final)
        if (final_qa.get('width'), final_qa.get('height')) != (3840, 2160):
            raise RuntimeError(f'UHD QA failed for {kind}: {final_qa}')
        source.unlink(missing_ok=True)
        qa['events'][kind] = {
            'event_id': eid,
            'period': int(float(str(row.get('period') or 3))),
            'clock': row.get('clock') or row.get('game_clock'),
            'description': row.get('description'),
            'angle_label': label,
            'advertised_best_resolution': {'width': advertised_res[0], 'height': advertised_res[1]},
            'available_angle_labels': all_labels,
            'source_qa': source_qa,
            'final_qa': final_qa,
            'file': final.name,
            'clips_page': f'https://clips.nba.com/?gameNo={GAME_ID}&eventNum={eid}&source=grs',
        }
    PBP_FILE.unlink(missing_ok=True)
    (OUT / 'qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
