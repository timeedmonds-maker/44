from __future__ import annotations

import argparse
import hashlib
import html as htmlmod
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import requests

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
KNOWN_PLACEHOLDER_AHASH = [
    '0000180039fc39fc38c839fc39fc10000000',
    '0000180019fc39fc38c819fc39fc18000000',
    '0000100011fc11fc18ec11fc19fc10000000',
]
H = {
    'User-Agent': UA,
    'Referer': 'https://clips.nba.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def run(cmd: list[str]) -> None:
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def ahash_frame(path: Path, t: float) -> str:
    p = subprocess.run(
        ['ffmpeg', '-v', 'error', '-ss', str(t), '-i', str(path),
         '-vf', 'scale=16:9,format=gray', '-frames:v', '1',
         '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
        capture_output=True,
    )
    if p.returncode or len(p.stdout) != 144:
        return ''
    vals = list(p.stdout)
    avg = sum(vals) / len(vals)
    bits = ''.join('1' if x >= avg else '0' for x in vals)
    return f'{int(bits, 2):036x}'


def hamming_hex(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 999
    return (int(a, 16) ^ int(b, 16)).bit_count()


def probe_video(path: Path) -> dict:
    p = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=codec_name,width,height,avg_frame_rate:format=duration,bit_rate',
         '-of', 'json', str(path)],
        capture_output=True, text=True,
    )
    if p.returncode:
        return {'ok': False, 'reason': 'ffprobe_failed'}
    try:
        j = json.loads(p.stdout)
        s = j['streams'][0]
        fmt = j.get('format') or {}
        duration = float(fmt.get('duration') or 0)
    except Exception:
        return {'ok': False, 'reason': 'ffprobe_parse'}

    times = [min(max(duration * f, 0.25), max(duration - 0.25, 0.25)) for f in (0.25, 0.5, 0.75)]
    fps = [ahash_frame(path, t) for t in times]
    distances = [hamming_hex(a, b) for a, b in zip(fps, KNOWN_PLACEHOLDER_AHASH)]
    placeholder_like = all(d <= 12 for d in distances)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = duration >= 2.0 and all(fps) and not placeholder_like
    return {
        'ok': ok,
        'reason': None if ok else ('known_nba_video_not_available_placeholder' if placeholder_like else 'invalid_media'),
        'duration': duration,
        'codec': s.get('codec_name'),
        'width': s.get('width'),
        'height': s.get('height'),
        'avg_frame_rate': s.get('avg_frame_rate'),
        'bit_rate': fmt.get('bit_rate'),
        'sha256': sha,
        'visual_fingerprints': fps,
        'placeholder_hamming_distances': distances,
    }


def parse_clips_page(game_id: str, event_id: int) -> dict:
    page_url = f'https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs'
    r = requests.get(page_url, headers=H, timeout=45)
    r.raise_for_status()
    text = r.text
    title_match = re.search(r'<title>(.*?)</title>', text, flags=re.I | re.S)
    title = htmlmod.unescape(title_match.group(1).strip()) if title_match else ''

    options = []
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, flags=re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        attrs = m.group(2).lower()
        label = re.sub(r'<[^>]+>', '', htmlmod.unescape(m.group(3))).strip()
        options.append({'url': url, 'selected': 'selected' in attrs, 'label': label})

    hls = [o for o in options if '.m3u8' in o['url'].lower() and 'lrmedia.nba.com' in o['url'].lower()]
    if not hls:
        raise RuntimeError(f'No signed lrmedia HLS found for {game_id}/{event_id}')
    selected = next((o for o in hls if o['selected']), hls[0])
    return {'page_url': page_url, 'title': title, 'selected': selected, 'angle_count': len(hls)}


def download_hls_source(url: str, out: Path) -> None:
    headers = f'User-Agent: {UA}\r\nReferer: https://clips.nba.com/\r\n'
    run([
        'ffmpeg', '-y', '-v', 'error', '-rw_timeout', '30000000', '-headers', headers,
        '-i', url, '-map', '0:v:0', '-map', '0:a:0?',
        '-c', 'copy', '-movflags', '+faststart', str(out),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', default='output')
    args = ap.parse_args()

    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    events = request.get('events') or []
    expected_count = int(request.get('expected_count', len(events)))
    if not events or expected_count != len(events):
        raise SystemExit('Request must contain events and expected_count must match event count')

    out = Path(args.out)
    clips_dir = out / 'clips'
    shutil.rmtree(out, ignore_errors=True)
    clips_dir.mkdir(parents=True)

    qa = []
    for i, event in enumerate(events, 1):
        rank = int(event.get('rank', i))
        gid = str(event['game_id'])
        eid = int(event.get('event_id') or event.get('event_num'))
        rec = {'rank': rank, 'game_id': gid, 'event_id': eid, 'status': 'failed'}
        try:
            clip = parse_clips_page(gid, eid)
            rec['clips_page_url'] = clip['page_url']
            rec['clips_page_title'] = clip['title']
            rec['angle_count'] = clip['angle_count']
            rec['selected_angle_label'] = clip['selected']['label']

            dst = clips_dir / f'{rank:02d}_{gid}_{eid}_SOURCE.mp4'
            download_hls_source(clip['selected']['url'], dst)
            q = probe_video(dst)
            rec['probe'] = q
            if not q['ok']:
                raise RuntimeError(q['reason'])
            rec['source_path'] = str(dst)
            rec['status'] = 'ok'
        except Exception as exc:
            rec['error'] = repr(exc)
        qa.append(rec)

    sha_groups = defaultdict(list)
    fp_groups = defaultdict(list)
    for rec in qa:
        if rec.get('status') != 'ok':
            continue
        q = rec['probe']
        sha_groups[q['sha256']].append(rec)
        fp_groups[tuple(q['visual_fingerprints'])].append(rec)

    duplicate_keys = set()
    for group in list(sha_groups.values()) + list(fp_groups.values()):
        keys = {(r['game_id'], r['event_id']) for r in group}
        if len(keys) > 1:
            duplicate_keys |= keys
    for rec in qa:
        if (rec['game_id'], rec['event_id']) in duplicate_keys:
            rec['status'] = 'failed'
            rec['global_qa_failure'] = 'duplicate_media_across_distinct_events'

    ok = [r for r in qa if r.get('status') == 'ok']
    payload = {
        'mode': 'source',
        'processing': 'NONE: signed NBA HLS remuxed to MP4 with ffmpeg -c copy; no scaling, denoise, sharpening, interpolation, or re-encoding',
        'source_policy': 'clips.nba.com exact event page -> fresh signed lrmedia.nba.com HLS',
        'request': request,
        'events': qa,
    }
    (out / 'qa.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    if len(ok) == len(events) and len(events) > 1:
        concat = out / 'concat.txt'
        files = [Path(r['source_path']) for r in sorted(ok, key=lambda r: r['rank'])]
        concat.write_text('\n'.join("file '" + str(f.resolve()).replace("'", "'\\''") + "'" for f in files) + '\n', encoding='utf-8')
        try:
            run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(out / 'reel_SOURCE.mp4')])
        except Exception:
            print('Source clips validated; stream-copy reel assembly failed. Individual source clips remain available.', flush=True)

    print(f'VALID_REAL_CLIPS={len(ok)}/{len(events)}', flush=True)
    if len(ok) != expected_count:
        sys.exit(2)


if __name__ == '__main__':
    main()
