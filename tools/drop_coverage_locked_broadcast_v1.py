#!/usr/bin/env python3
"""DROP_COVERAGE_LOCKED_BROADCAST_V1

Canonical enduring deterministic broadcast explainer tool.

Lineage:
- Christmas-event V7 visual language
- LOCKED_BROADCAST_SCREEN_V1/V2/V3 validation lineage

This tool preserves the locked broadcast presentation while making the generic
coverage concept explicit. The canonical main panel label is "DROP COVERAGE".

No generated imagery. No AI super-resolution. Real source pixels plus
deterministic OpenCV/Pillow overlays only.
"""
from pathlib import Path
import argparse, json
import locked_broadcast_screen_v2 as core

TOOL_ID = 'DROP_COVERAGE_LOCKED_BROADCAST_V1'
CANONICAL_LABEL = 'DROP COVERAGE'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = json.load(open(a.config))

    # Enduring visual/data contract.
    assert cfg['screen']['value'] == CANONICAL_LABEL, cfg['screen']
    assert len(cfg['timing']['contact_phases']) == 1, cfg['timing']['contact_phases']
    sim = cfg['shot']['similar_xfg']
    assert abs(float(sim['distance_tolerance_ft']) - 1.5) < 1e-12, sim
    assert abs(float(sim['xfg_tolerance_percentage_points']) - 2.0) < 1e-12, sim
    assert sim.get('criteria_logic') == 'intersection', sim
    assert sim.get('action_type_filter') is None, sim

    core.TOOL_ID = TOOL_ID
    core.render(a.source, a.tracks, a.config, a.out)

    out = Path(a.out)
    qa = json.load(open(out / 'qa.json'))
    qa['tool_id'] = TOOL_ID
    qa['parent_tool_id'] = 'LOCKED_BROADCAST_SCREEN_V3'
    qa['canonical_tool_family'] = 'DROP_COVERAGE_LOCKED_BROADCAST'
    qa['canonical_ui_label'] = CANONICAL_LABEL
    qa['enduring_model_lock'] = True
    qa['universal_default'] = True
    qa['v1_contract'] = {
        'visual_model': 'locked Christmas-V7 broadcast lineage',
        'main_label': CANONICAL_LABEL,
        'deterministic_overlays_only': True,
        'event_xfg_separate': True,
        'matched_xfg_distance_tolerance_ft': 1.5,
        'matched_xfg_difficulty_tolerance_percentage_points': 2.0,
        'matched_xfg_logic': 'intersection',
        'matched_xfg_action_type_filter': None,
        'contact_window_authority': 'validated source-frame anchors when available',
        'required_outputs': ['native', 'UHD master', 'streamable 4K', 'QA package']
    }
    (out / 'qa.json').write_text(json.dumps(qa, indent=2))
    (out / 'LOCKED_TOOL_ID.txt').write_text(TOOL_ID + '\n')
    print(json.dumps(qa, indent=2))


if __name__ == '__main__':
    main()
