from __future__ import annotations

import argparse
import hashlib
import html as htmlmod
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36'
FFMPEG = os.environ.get('FFMPEG_BINARY', 'ffmpeg')
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
        [FFMPEG, '-v', 'error', '-ss', str(t), '-i', str(path),
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_duration_seconds(text: str) -> float:
    m = re.search(r'Duration:\s*(\d+):(\d+):([0-9.]+)', text)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def probe_video(path: Path) -> dict:
    # Use ffmpeg itself for metadata so the fast worker only needs one bundled binary.
    # -c copy and a tiny output duration avoid a decode/transcode pass.
    p = subprocess.run(
        [FFMPEG, '-hide_banner', '-i', str(path), '-map', '0:v:0',
         '-c', 'copy', '-t', '0.01', '-f', 'null', '-'],
        capture_output=True, text=True,
    )
    text = p.stderr or ''
    duration = parse_duration_seconds(text)
    video_line = next((line.strip() for line in text.splitlines() if 'Video:' in line), '')

    codec_m = re.search(r'Video:\s*([^\s,(]+)', video_line)
    res_m = re.search(r'\b(\d{2,5})x(\d{2,5})\b', video_line)
    fps_m = re.search(r'([0-9.]+)\s*fps\b', video_line)
    bitrate_m = re.search(r'Duration:.*?bitrate:\s*([0-9.]+)\s*kb/s', text, flags=re.S)

    if not video_line or not res_m or duration <= 0:
        return {'ok': False, 'reason': 'ffmpeg_probe_parse'}

    times = [min(max(duration * f, 0.25), max(duration - 0.25, 0.25)) for f in (0.25, 0.5, 0.75)]
    fingerprints = [ahash_frame(path, t) for t in times]
    distances = [hamming_hex(a, b) for a, b in zip(fingerprints, KNOWN_PLACEHOLDER_AHASH)]
    placeholder_like = all(d <= 12 for d in distances)
    sha = sha256_file(path)
    ok = duration >= 2.0 and all(fingerprints) and not placeholder_like

    return {
        'ok': ok,
        'reason': None if ok else ('known_nba_video_not_available_placeholder' if placeholder_like else 'invalid_media'),
        'duration': duration,
        'codec': codec_m.group(1) if codec_m else None,
        'width': int(res_m.group(1)),
        'height': int(res_m.group(2)),
        'avg_frame_rate': float(fps_m.group(1)) if fps_m else None,
        'bit_rate': int(float(bitrate_m.group(1)) * 1000) if bitrate_m else None,
        'sha256': sha,
        'visual_fingerprints': fingerprints,
        'placeholder_hamming_distances': distances,
    }


def parse_clips_page(game_id: str, event_id: int) -> dict:
    page_url = f'https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs'
    request = urllib.request.Request(page_url, headers=H)
    with urllib.request.urlopen(request, timeout=45) as response:
        text = response.read().decode('utf-8', errors='replace')

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
        FFMPEG, '-nostdin', '-y', '-v', 'error', '-rw_timeout', '30000000', '-headers', headers,
        '-i', url, '-map', '0:v:0', '-map', '0:a:0?',
        '-c', 'copy', '-movflags', '+faststart', str(out),
    ])


def process_event(event: dict, fallback_rank: int, clips_dir: Path) -> dict:
    started = time.perf_counter()
    rank = int(event.get('rank', fallback_rank))
    gid = str(event['game_id'])
    eid = int(event.get('event_id') or event.get('event_num'))
    rec = {'rank': rank, 'game_id': gid, 'event_id': eid, 'status': 'failed'}
    try:
        t0 = time.perf_counter()
        clip = parse_clips_page(gid, eid)
        t1 = time.perf_counter()
        rec['clips_page_url'] = clip['page_url']
        rec['clips_page_title'] = clip['title']
        rec['angle_count'] = clip['angle_count']
        rec['selected_angle_label'] = clip['selected']['label']

        dst = clips_dir / f'{rank:03d}_{gid}_{eid}_SOURCE.mp4'
        download_hls_source(clip['selected']['url'], dst)
        t2 = time.perf_counter()
        q = probe_video(dst)
        t3 = time.perf_counter()
        rec['probe'] = q
        rec['timing_seconds'] = {
            'resolve': round(t1 - t0, 3),
            'download_remux': round(t2 - t1, 3),
            'qa': round(t3 - t2, 3),
            'total': round(t3 - started, 3),
        }
        if not q['ok']:
            raise RuntimeError(q['reason'])
        rec['source_path'] = str(dst)
        rec['status'] = 'ok'
    except Exception as exc:
        rec['error'] = repr(exc)
        rec.setdefault('timing_seconds', {})['total'] = round(time.perf_counter() - started, 3)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--out', default='output')
    ap.add_argument('--workers', type=int, default=0, help='0 = request value or automatic')
    args = ap.parse_args()

    overall_started = time.perf_counter()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    events = request.get('events') or []
    expected_count = int(request.get('expected_count', len(events)))
    if not events or expected_count != len(events):
        raise SystemExit('Request must contain events and expected_count must match event count')

    ranks = [int(event.get('rank', i)) for i, event in enumerate(events, 1)]
    if len(set(ranks)) != len(ranks):
        raise SystemExit('Event ranks must be unique')

    out = Path(args.out)
    clips_dir = out / 'clips'
    shutil.rmtree(out, ignore_errors=True)
    clips_dir.mkdir(parents=True)

    requested_workers = args.workers or int(request.get('workers', 0) or 0)
    workers = requested_workers if requested_workers > 0 else min(4, len(events))
    workers = max(1, min(workers, len(events), 8))

    if workers == 1:
        qa = [process_event(event, i, clips_dir) for i, event in enumerate(events, 1)]
    else:
        qa = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='nba-video') as pool:
            futures = {
                pool.submit(process_event, event, i, clips_dir): i
                for i, event in enumerate(events, 1)
            }
            for future in as_completed(futures):
                qa.append(future.result())
        qa.sort(key=lambda r: r['rank'])

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
    reel_seconds = 0.0
    if len(ok) == len(events) and len(events) > 1:
        concat = out / 'concat.txt'
        files = [Path(r['source_path']) for r in sorted(ok, key=lambda r: r['rank'])]
        concat.write_text('\n'.join("file '" + str(f.resolve()).replace("'", "'\\''") + "'" for f in files) + '\n', encoding='utf-8')
        reel_started = time.perf_counter()
        try:
            run([FFMPEG, '-nostdin', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(out / 'reel_SOURCE.mp4')])
        except Exception:
            print('Source clips validated; stream-copy reel assembly failed. Individual source clips remain available.', flush=True)
        reel_seconds = time.perf_counter() - reel_started

    payload = {
        'mode': 'source',
        'processing': 'NONE: signed NBA HLS remuxed to MP4 with ffmpeg -c copy; no scaling, denoise, sharpening, interpolation, or re-encoding',
        'source_policy': 'clips.nba.com exact event page -> fresh signed lrmedia.nba.com HLS',
        'worker': {
            'parallel_workers': workers,
            'overall_seconds': round(time.perf_counter() - overall_started, 3),
            'reel_assembly_seconds': round(reel_seconds, 3),
        },
        'request': request,
        'events': qa,
    }
    (out / 'qa.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f'VALID_REAL_CLIPS={len(ok)}/{len(events)}', flush=True)
    print(f'PARALLEL_WORKERS={workers}', flush=True)
    print(f'WORKER_SECONDS={payload["worker"]["overall_seconds"]}', flush=True)
    for rec in qa:
        p = rec.get('probe') or {}
        t = rec.get('timing_seconds') or {}
        print(
            f"EVENT {rec['game_id']}/{rec['event_id']}: status={rec['status']} "
            f"{p.get('width')}x{p.get('height')} codec={p.get('codec')} "
            f"fps={p.get('avg_frame_rate')} bitrate={p.get('bit_rate')} "
            f"duration={p.get('duration')} total_s={t.get('total')}",
            flush=True,
        )

    if len(ok) != expected_count:
        sys.exit(2)


if __name__ == '__main__':
    main()
