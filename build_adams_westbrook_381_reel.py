from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nba_video_worker as w
from build_kd_top10_all_angles import inventory as raw_inventory

ORDER = {
    'Broadcast': 0,
    'Other Broadcast': 1,
    'Mobile Broadcast': 2,
    'Play by Play': 3,
    'In Arena': 4,
    'High Tight': 5,
    'Left Slash': 6,
    'Right Slash': 7,
    'Left HandHeld': 8,
    'Right HandHeld': 9,
    'Left Above Rim': 10,
    'Right Above Rim': 11,
}
RETRY_DELAYS = (1.0, 2.0, 4.0, 8.0, 12.0)
TRANSIENT = {403, 408, 409, 425, 429, 500, 502, 503, 504}
ORIGINAL_HTTP_BYTES = w.http_bytes
MAX_NATIVE_ARTIFACT_BYTES = 1_750_000_000


def retry_http_bytes(url, headers=None, timeout=45):
    attempts = len(RETRY_DELAYS) + 1
    for attempt in range(attempts):
        try:
            return ORIGINAL_HTTP_BYTES(url, headers, timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in TRANSIENT or attempt == attempts - 1:
                raise
            delay = RETRY_DELAYS[attempt]
            print(f'HTTP_RETRY status={exc.code} attempt={attempt + 1}/{attempts} sleep={delay} url={url}', flush=True)
            time.sleep(delay)
        except urllib.error.URLError as exc:
            if attempt == attempts - 1:
                raise
            delay = RETRY_DELAYS[attempt]
            print(f'URL_RETRY reason={exc.reason!r} attempt={attempt + 1}/{attempts} sleep={delay} url={url}', flush=True)
            time.sleep(delay)


w.http_bytes = retry_http_bytes


def fresh_inventory(gid: str, eid: int):
    last = None
    for attempt, delay in enumerate((0.0, 1.0, 2.0, 4.0), 1):
        if delay:
            time.sleep(delay)
        try:
            return raw_inventory(gid, eid)
        except Exception as exc:
            last = exc
            print(f'INVENTORY_RETRY {gid}/{eid} attempt={attempt} error={exc!r}', flush=True)
    raise RuntimeError(f'Unable to resolve clips page for {gid}/{eid}: {last!r}')


def ordered_options(opts: list[dict]) -> list[dict]:
    def key(o):
        label = (o.get('label') or '').strip()
        selected_bonus = -1 if o.get('page_selected') else 0
        return (ORDER.get(label, 50), selected_bonus, label)
    return sorted(opts, key=key)


def download_event(event: dict, clips_dir: Path) -> dict:
    rank = int(event['rank'])
    gid = str(event['game_id']).zfill(10)
    eid = int(event.get('event_id') or event.get('event_num'))
    record = {
        'rank': rank,
        'game_id': gid,
        'event_id': eid,
        'season': event.get('season'),
        'season_type': event.get('season_type'),
        'game_date': event.get('game_date'),
        'description': event.get('description'),
        'status': 'failed',
        'attempts': [],
    }
    tried = set()
    for refresh_cycle in range(1, 4):
        page, title, opts = fresh_inventory(gid, eid)
        record['clips_page'] = page
        record['clips_page_title'] = title
        record['angle_count_available'] = len(opts)
        for option in ordered_options(opts):
            label = (option.get('label') or 'Unknown').strip()
            key = (refresh_cycle, label, option.get('url'))
            if key in tried:
                continue
            tried.add(key)
            safe = ''.join(c if c.isalnum() else '_' for c in label).strip('_') or 'Angle'
            dst = clips_dir / f'{rank:03d}_{gid}_{eid}_{safe}_SOURCE.mp4'
            dst.unlink(missing_ok=True)
            attempt = {'refresh_cycle': refresh_cycle, 'label': label, 'page_selected': bool(option.get('page_selected'))}
            try:
                w.download_hls_source(option['url'], dst)
                probe = w.probe_video(dst)
                attempt['probe'] = probe
                if not probe.get('ok'):
                    raise RuntimeError(probe.get('reason') or 'video_qa_failed')
                record.update({
                    'status': 'ok',
                    'angle_label': label,
                    'page_selected': bool(option.get('page_selected')),
                    'source_path': str(dst),
                    'source_bytes': dst.stat().st_size,
                    'probe': probe,
                })
                attempt['status'] = 'ok'
                record['attempts'].append(attempt)
                print(f"OK R{rank:03d} {gid}/{eid} angle={label} {probe.get('width')}x{probe.get('height')} dur={probe.get('duration'):.3f}s bytes={dst.stat().st_size}", flush=True)
                return record
            except Exception as exc:
                attempt['status'] = 'failed'
                attempt['error'] = repr(exc)
                record['attempts'].append(attempt)
                dst.unlink(missing_ok=True)
                print(f'ANGLE_FAIL R{rank:03d} {gid}/{eid} angle={label} refresh={refresh_cycle} error={exc!r}', flush=True)
        time.sleep(refresh_cycle)
    record['error'] = 'all_official_angles_failed_after_fresh_signed_hls_retries'
    print(f"EVENT_FAIL R{rank:03d} {gid}/{eid}", flush=True)
    return record


def concat_copy(files: list[Path], out: Path) -> None:
    lst = out.with_suffix('.concat.txt')
    lst.write_text('\n'.join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in files) + '\n')
    w.run([w.FFMPEG, '-nostdin', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', '-movflags', '+faststart', str(out)])


def artifact_size_encode(src: Path, dst: Path) -> None:
    # Delivery-only fallback if the native stream-copy reel is too large for reliable artifact transport.
    # Resolution is preserved; this is deterministic H.264/AAC, not generative enhancement.
    w.run([
        w.FFMPEG, '-nostdin', '-y', '-v', 'error', '-i', str(src),
        '-map', '0:v:0', '-map', '0:a:0?',
        '-c:v', 'libx264', '-preset', 'slow', '-b:v', '2200k', '-maxrate', '2600k', '-bufsize', '5200k',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', str(dst)
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='adams_westbrook_dunk_event_links.json')
    ap.add_argument('--out', default='adams_westbrook_381_reel_output')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    payload = json.loads(Path(args.manifest).read_text())
    events = payload.get('events') or []
    if len(events) != 381 or int(payload.get('total', len(events))) != 381:
        raise SystemExit(f'Expected exactly 381 events, found {len(events)}')
    for i, e in enumerate(events, 1):
        e['rank'] = i
        e['event_id'] = int(e.get('event_id') or e.get('event_num'))
        e['game_id'] = str(e['game_id']).zfill(10)

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    clips = out / 'clips'
    clips.mkdir(parents=True)

    workers = max(1, min(int(args.workers), 6))
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='adams-westbrook-video') as pool:
        futures = [pool.submit(download_event, e, clips) for e in events]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda r: r['rank'])

    failures = [r for r in results if r.get('status') != 'ok']
    if failures:
        (out / 'qa_partial.json').write_text(json.dumps({'events': results}, indent=2))
        raise SystemExit(f'{len(failures)} event videos failed; refusing to build an incomplete reel')

    sha_groups = defaultdict(list)
    for r in results:
        sha_groups[r['probe']['sha256']].append((r['game_id'], r['event_id'], r['rank']))
    dupes = {sha: group for sha, group in sha_groups.items() if len({(g,e) for g,e,_ in group}) > 1}
    if dupes:
        (out / 'duplicate_qa.json').write_text(json.dumps(dupes, indent=2))
        raise SystemExit(f'Exact duplicate media detected across {len(dupes)} SHA groups')

    resolutions = sorted({(r['probe']['width'], r['probe']['height']) for r in results})
    codecs = sorted({r['probe']['codec'] for r in results})
    files = [Path(r['source_path']) for r in results]
    source_duration = sum(float(r['probe']['duration']) for r in results)

    native_reel = out / 'steven_adams_all_381_westbrook_assisted_dunks_NATIVE.mp4'
    concat_copy(files, native_reel)
    native_probe = w.probe_video(native_reel)
    if not native_probe.get('ok'):
        raise SystemExit(f'Native reel QA failed: {native_probe}')
    duration_gap = abs(float(native_probe['duration']) - source_duration)
    if duration_gap > max(5.0, source_duration * 0.005):
        raise SystemExit(f'Native reel duration mismatch: reel={native_probe["duration"]} clips={source_duration}')

    final_reel = native_reel
    delivery_mode = 'native_stream_copy'
    if native_reel.stat().st_size > MAX_NATIVE_ARTIFACT_BYTES:
        delivery = out / 'steven_adams_all_381_westbrook_assisted_dunks_960x540_H264.mp4'
        artifact_size_encode(native_reel, delivery)
        delivery_probe = w.probe_video(delivery)
        if not delivery_probe.get('ok'):
            raise SystemExit(f'Delivery reel QA failed: {delivery_probe}')
        delivery_gap = abs(float(delivery_probe['duration']) - source_duration)
        if delivery_gap > max(5.0, source_duration * 0.005):
            raise SystemExit(f'Delivery reel duration mismatch: reel={delivery_probe["duration"]} clips={source_duration}')
        native_reel.unlink()
        final_reel = delivery
        delivery_mode = 'deterministic_h264_delivery_encode_native_resolution'

    qa = {
        'title': 'Steven Adams — all 381 dunks assisted by Russell Westbrook',
        'event_count_expected': 381,
        'event_count_validated': len(results),
        'event_definition': payload.get('definition'),
        'coverage': payload.get('coverage'),
        'event_source': payload.get('source'),
        'video_source_policy': 'Exact GameID/EventID -> fresh clips.nba.com page -> signed lrmedia.nba.com HLS -> highest native HLS rendition. Broadcast preferred; official alternate angles used only when the preferred angle fails QA. Known NBA Video Not Available placeholder is rejected. No event is silently skipped.',
        'delivery_mode': delivery_mode,
        'source_resolutions': resolutions,
        'source_codecs': codecs,
        'source_duration_seconds_sum': source_duration,
        'final_path': str(final_reel),
        'final_bytes': final_reel.stat().st_size,
        'final_probe': w.probe_video(final_reel),
        'events': results,
    }
    (out / 'qa.json').write_text(json.dumps(qa, indent=2))
    shutil.copy2('adams_westbrook_dunk_event_links.csv', out / 'adams_westbrook_dunk_event_links.csv')

    # Keep the artifact compact: after final QA, remove 381 intermediate MP4s.
    shutil.rmtree(clips)
    print(f'FINAL={final_reel} BYTES={final_reel.stat().st_size} MODE={delivery_mode} DURATION={qa["final_probe"]["duration"]:.3f}s', flush=True)
    print(f'VALIDATED_EVENTS={len(results)} RESOLUTIONS={resolutions} CODECS={codecs}', flush=True)


if __name__ == '__main__':
    main()
