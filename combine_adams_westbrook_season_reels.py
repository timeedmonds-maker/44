from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nba_video_worker as w
from build_adams_westbrook_381_reel import concat_copy, artifact_size_encode, MAX_NATIVE_ARTIFACT_BYTES

SEASONS = ['2013-14','2014-15','2015-16','2016-17','2017-18','2018-19']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', required=True)
    ap.add_argument('--manifest-dir', required=True)
    ap.add_argument('--out', default='adams_westbrook_381_parallel_final')
    args = ap.parse_args()

    parts = Path(args.parts)
    manifest_dir = Path(args.manifest_dir)
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    reels = []
    season_qa = []
    total_events = 0
    total_duration = 0.0
    for season in SEASONS:
        slug = season.replace('-', '_')
        matches = sorted(parts.rglob(f'{slug}_steven_adams_westbrook_dunks_NATIVE.mp4'))
        qmatches = sorted(parts.rglob(f'{slug}_qa.json'))
        if len(matches) != 1 or len(qmatches) != 1:
            raise SystemExit(f'{season}: expected exactly one reel and QA; got reels={matches} qa={qmatches}')
        q = json.loads(qmatches[0].read_text())
        if int(q.get('validated_events', 0)) != int(q.get('expected_events', -1)):
            raise SystemExit(f'{season}: season QA count mismatch')
        reels.append(matches[0])
        season_qa.append(q)
        total_events += int(q['validated_events'])
        total_duration += float(q['reel_probe']['duration'])

    if total_events != 381:
        raise SystemExit(f'Expected 381 validated season events, got {total_events}')

    native = out / 'steven_adams_all_381_westbrook_assisted_dunks_NATIVE.mp4'
    concat_copy(reels, native)
    native_probe = w.probe_video(native)
    if not native_probe.get('ok'):
        raise SystemExit(f'Final native reel QA failed: {native_probe}')
    if abs(float(native_probe['duration']) - total_duration) > max(5.0, total_duration * 0.005):
        raise SystemExit(f'Final duration mismatch reel={native_probe["duration"]} seasons={total_duration}')

    final = native
    mode = 'native_stream_copy_from_six_validated_season_reels'
    if native.stat().st_size > MAX_NATIVE_ARTIFACT_BYTES:
        delivery = out / 'steven_adams_all_381_westbrook_assisted_dunks_960x540_H264.mp4'
        artifact_size_encode(native, delivery)
        probe = w.probe_video(delivery)
        if not probe.get('ok'):
            raise SystemExit(f'Final delivery QA failed: {probe}')
        if abs(float(probe['duration']) - total_duration) > max(5.0, total_duration * 0.005):
            raise SystemExit('Final delivery duration mismatch')
        native.unlink()
        final = delivery
        mode = 'deterministic_h264_delivery_encode_native_resolution'

    src_csv = manifest_dir / 'adams_westbrook_dunk_event_links.csv'
    src_json = manifest_dir / 'adams_westbrook_dunk_event_links.json'
    if not src_csv.exists() or not src_json.exists():
        raise SystemExit('Exact event manifest artifact missing')
    shutil.copy2(src_csv, out / src_csv.name)
    shutil.copy2(src_json, out / src_json.name)

    qa = {
        'title':'Steven Adams — all 381 dunks assisted by Russell Westbrook',
        'validated_events':381,
        'season_order':SEASONS,
        'season_qa':season_qa,
        'video_source_policy':'Exact GameID/EventID -> fresh clips.nba.com -> signed lrmedia.nba.com HLS. Broadcast preferred; official alternate angles only after failed/placeholder QA. No event silently skipped.',
        'delivery_mode':mode,
        'final_path':str(final),
        'final_bytes':final.stat().st_size,
        'final_probe':w.probe_video(final),
    }
    (out / 'qa.json').write_text(json.dumps(qa, indent=2))
    print(f'FINAL_OK EVENTS=381 FILE={final} BYTES={final.stat().st_size} DUR={qa["final_probe"]["duration"]:.3f} MODE={mode}', flush=True)


if __name__ == '__main__':
    main()
