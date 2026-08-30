from __future__ import annotations

import html as htmlmod
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from download_nba_angle_compare import download_hls, get, probe

GAME_ID = '0022500301'
ADAMS_ID = 203500
OUT = Path('deliveries/adams_jazz_0022500301_overhead')
UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'


def clock_seconds(clock: str) -> float:
    m = re.fullmatch(r'PT(?:(\d+)M)?([0-9.]+)S', str(clock or ''))
    if not m:
        return -1.0
    return int(m.group(1) or 0) * 60 + float(m.group(2))


def fetch_actions() -> list[dict]:
    url = f'https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{GAME_ID}.json'
    req = urllib.request.Request(url, headers={'User-Agent': UA, 'Referer': 'https://www.nba.com/'})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)['game']['actions']


def find_exact_events(actions: list[dict]) -> dict[str, dict]:
    block_candidates = []
    dunk_candidates = []
    for a in actions:
        if int(a.get('period') or 0) != 3:
            continue
        sec = clock_seconds(a.get('clock'))
        desc = str(a.get('description') or '')
        hay = ' '.join([desc, str(a.get('actionType') or ''), str(a.get('subType') or '')]).lower()
        try:
            pid = int(a.get('personId') or 0)
        except Exception:
            pid = 0
        blocker_ids = []
        for k in ('blockPersonId', 'blockPlayerId', 'blockerPersonId', 'blockerPlayerId'):
            try:
                blocker_ids.append(int(a.get(k) or 0))
            except Exception:
                pass
        block_name = ' '.join(str(a.get(k) or '') for k in ('blockPlayerName', 'blockerPlayerName'))

        if abs(sec - 251.0) <= 2.0 and (
            ADAMS_ID in blocker_ids
            or ('block' in hay and ('adams' in hay or 'adams' in block_name.lower()))
        ):
            block_candidates.append(a)

        is_fg = a.get('isFieldGoal') in {1, True, '1'}
        made = str(a.get('shotResult') or '').lower() == 'made'
        if abs(sec - 243.0) <= 2.0 and pid == ADAMS_ID and is_fg and made and 'dunk' in hay:
            dunk_candidates.append(a)

    if len(block_candidates) != 1:
        raise RuntimeError(f'Expected 1 Adams block near 4:11 Q3; got {len(block_candidates)}: {block_candidates}')
    if len(dunk_candidates) != 1:
        raise RuntimeError(f'Expected 1 Adams dunk near 4:03 Q3; got {len(dunk_candidates)}: {dunk_candidates}')
    return {'block': block_candidates[0], 'dunk': dunk_candidates[0]}


def event_id(a: dict) -> int:
    return int(a.get('actionNumber') or a.get('actionId') or 0)


def above_rim_option(eid: int) -> tuple[str, str, list[str]]:
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
    # Prefer an explicitly side-labelled Above Rim camera when supplied by NBA.
    priority = {'left above rim': 0, 'right above rim': 1, 'above rim': 2}
    above.sort(key=lambda x: (priority.get(x[0].lower(), 9), x[0].lower()))
    lab, url = above[0]
    return lab, url, [x[0] for x in opts]


def to_uhd(src: Path, dst: Path) -> None:
    subprocess.run([
        'ffmpeg', '-nostdin', '-y', '-v', 'error', '-i', str(src),
        '-vf', 'scale=3840:2160:flags=lanczos,fps=30',
        '-c:v', 'libx264', '-profile:v', 'high', '-crf', '16',
        '-maxrate', '36M', '-bufsize', '72M',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', str(dst)
    ], check=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = find_exact_events(fetch_actions())
    qa = {'game_id': GAME_ID, 'events': {}}
    names = {
        'block': 'steven_adams_block_vs_jazz_above_rim_UHD.mp4',
        'dunk': 'steven_adams_dunk_vs_jazz_above_rim_UHD.mp4',
    }
    for kind in ('block', 'dunk'):
        a = events[kind]
        eid = event_id(a)
        label, hls, all_labels = above_rim_option(eid)
        source = OUT / f'.{kind}_above_rim_source_1080.mp4'
        final = OUT / names[kind]
        download_hls(hls, source)
        source_qa = probe(source)
        if (source_qa.get('width'), source_qa.get('height')) != (1920, 1080):
            raise RuntimeError(f'Expected native 1920x1080 Above Rim source for {kind}, got {source_qa}')
        to_uhd(source, final)
        final_qa = probe(final)
        if (final_qa.get('width'), final_qa.get('height')) != (3840, 2160):
            raise RuntimeError(f'UHD QA failed for {kind}: {final_qa}')
        source.unlink(missing_ok=True)
        qa['events'][kind] = {
            'event_id': eid,
            'period': int(a.get('period') or 0),
            'clock': a.get('clock'),
            'description': a.get('description'),
            'angle_label': label,
            'available_angle_labels': all_labels,
            'source_qa': source_qa,
            'final_qa': final_qa,
            'file': final.name,
            'clips_page': f'https://clips.nba.com/?gameNo={GAME_ID}&eventNum={eid}&source=grs',
        }
    (OUT / 'qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
