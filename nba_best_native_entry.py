from __future__ import annotations

import html as htmlmod
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import nba_video_worker as worker

# At equal native resolution, preserve the most generally useful basketball view.
CAMERA_PREFERENCE = {
    'Broadcast': 0,
    'Other Broadcast': 1,
    'In Arena': 2,
    'High Tight': 3,
    'Left Slash': 4,
    'Right Slash': 5,
    'Left HandHeld': 6,
    'Right HandHeld': 7,
    'Left Above Rim': 8,
    'Right Above Rim': 9,
    'Play by Play': 10,
    'Mobile Broadcast': 11,
}


def _inspect_master(option: dict) -> dict:
    rec = dict(option)
    rec.update({'width': 0, 'height': 0, 'bandwidth': 0, 'inspection_error': None})
    try:
        text = worker.http_bytes(option['url'], worker.MEDIA_H, timeout=30).decode('utf-8', errors='replace')
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        variants = []
        for i, line in enumerate(lines):
            if not line.startswith('#EXT-X-STREAM-INF:'):
                continue
            bw_m = re.search(r'BANDWIDTH=(\d+)', line)
            res_m = re.search(r'RESOLUTION=(\d+)x(\d+)', line)
            bw = int(bw_m.group(1)) if bw_m else 0
            width = int(res_m.group(1)) if res_m else 0
            height = int(res_m.group(2)) if res_m else 0
            variants.append((width * height, bw, width, height))
        if variants:
            _, bw, width, height = max(variants)
            rec.update({'width': width, 'height': height, 'bandwidth': bw})
    except Exception as exc:
        rec['inspection_error'] = repr(exc)
    return rec


def parse_clips_page_best_native(game_id: str, event_id: int) -> dict:
    page_url = f'https://clips.nba.com/?gameNo={game_id}&eventNum={event_id}&source=grs'
    text = worker.http_bytes(page_url, worker.H).decode('utf-8', errors='replace')
    title_match = re.search(r'<title>(.*?)</title>', text, flags=re.I | re.S)
    title = htmlmod.unescape(title_match.group(1).strip()) if title_match else ''

    options = []
    for m in re.finditer(r'<option\s+value="([^"]+)"([^>]*)>(.*?)</option>', text, flags=re.I | re.S):
        url = htmlmod.unescape(m.group(1).strip())
        attrs = m.group(2).lower()
        label = re.sub(r'<[^>]+>', '', htmlmod.unescape(m.group(3))).strip()
        if '.m3u8' in url.lower() and 'lrmedia.nba.com' in url.lower():
            options.append({'url': url, 'selected': 'selected' in attrs, 'label': label})

    if not options:
        raise RuntimeError(f'No signed lrmedia HLS found for {game_id}/{event_id}')

    # Always inspect all camera masters, but do so concurrently to keep this off the critical path.
    inspected = []
    with ThreadPoolExecutor(max_workers=min(12, len(options)), thread_name_prefix='angle-probe') as pool:
        futures = [pool.submit(_inspect_master, option) for option in options]
        for future in as_completed(futures):
            inspected.append(future.result())

    default = next((o for o in inspected if o.get('selected')), inspected[0])
    native_hd = [o for o in inspected if o.get('height', 0) >= 720 and o.get('width', 0) >= 1280]

    if native_hd:
        # Resolution outranks camera label; camera preference only breaks equal-resolution ties.
        chosen = max(
            native_hd,
            key=lambda o: (
                o.get('width', 0) * o.get('height', 0),
                -CAMERA_PREFERENCE.get(o.get('label', ''), 99),
                o.get('bandwidth', 0),
            ),
        )
        reason = 'best_native_hd_available'
    else:
        chosen = default
        reason = 'no_native_hd_available_fallback_to_default_broadcast'

    angle_inventory = [
        {
            'label': o.get('label'),
            'selected_on_page': bool(o.get('selected')),
            'width': o.get('width'),
            'height': o.get('height'),
            'bandwidth': o.get('bandwidth'),
            'inspection_error': o.get('inspection_error'),
        }
        for o in sorted(inspected, key=lambda x: CAMERA_PREFERENCE.get(x.get('label', ''), 99))
    ]

    # Keep worker compatibility: it expects clip['selected']['url'/'label'].
    chosen = dict(chosen)
    chosen['selection_reason'] = reason
    chosen['native_hd_found'] = bool(native_hd)
    chosen['available_native_hd_labels'] = [o['label'] for o in native_hd]

    return {
        'page_url': page_url,
        'title': title,
        'selected': chosen,
        'angle_count': len(options),
        'angle_inventory': angle_inventory,
    }


def parse_hls_playlist_resolution_first(url: str, depth: int = 0):
    if depth > 3:
        raise RuntimeError('HLS playlist nesting too deep')
    text = worker.http_bytes(url, worker.MEDIA_H).decode('utf-8', errors='replace')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != '#EXTM3U':
        raise RuntimeError('Invalid HLS playlist')

    for line in lines:
        if line.startswith('#EXT-X-KEY:') and 'METHOD=NONE' not in line:
            raise RuntimeError('Encrypted HLS playlist is not supported by fast source path')

    variants = []
    for i, line in enumerate(lines):
        if not line.startswith('#EXT-X-STREAM-INF:'):
            continue
        bw_m = re.search(r'BANDWIDTH=(\d+)', line)
        res_m = re.search(r'RESOLUTION=(\d+)x(\d+)', line)
        bw = int(bw_m.group(1)) if bw_m else 0
        width = int(res_m.group(1)) if res_m else 0
        height = int(res_m.group(2)) if res_m else 0
        for child in lines[i + 1:]:
            if not child.startswith('#'):
                variants.append((width * height, bw, worker.signed_join(url, child)))
                break
    if variants:
        return parse_hls_playlist_resolution_first(max(variants, key=lambda x: (x[0], x[1]))[2], depth + 1)

    init_url = None
    segments = []
    for line in lines:
        if line.startswith('#EXT-X-MAP:'):
            m = re.search(r'URI="([^"]+)"', line)
            if m:
                init_url = worker.signed_join(url, m.group(1))
        elif not line.startswith('#'):
            segments.append(worker.signed_join(url, line))
    if not segments:
        raise RuntimeError('No HLS media segments found')
    return init_url, segments


# Patch the validated worker at runtime so all existing QA/concurrency/reel behavior is preserved.
worker.parse_clips_page = parse_clips_page_best_native
worker.parse_hls_playlist = parse_hls_playlist_resolution_first

_original_process_event = worker.process_event


def process_event_with_source_metadata(event, fallback_rank, clips_dir):
    rec = _original_process_event(event, fallback_rank, clips_dir)
    # The original worker records selected_angle_label but not the new decision metadata.
    # Re-resolving would waste time, so the detailed angle inventory is intentionally kept
    # in the page-selection function only for selection. The final probe is authoritative.
    return rec

worker.process_event = process_event_with_source_metadata

if __name__ == '__main__':
    worker.main()
