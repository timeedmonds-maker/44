#!/usr/bin/env python3
"""Audit operational text files for non-approved HD/UHD render paths."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.yml', '.yaml', '.sh', '.md', '.txt', '.json', '.ts', '.tsx', '.js'}
SKIP_DIRS = {'.git', 'node_modules', 'vendor', 'outputs', 'deliveries', '__pycache__'}
APPROVED_RENDERER = 'tools/render_deterministic_master.py'
APPROVED_RENDERER_TOKEN = 'render_deterministic_master'
THIS_FILE = 'tools/audit_video_render_policy.py'

# Keep obsolete implementation names out of ordinary repository text while
# still detecting them here without storing the literal historical strings.
OBSOLETE_PATTERNS = [
    ''.join(['real', '-', 'esrgan']),
    ''.join(['real', 'esrgan']),
    ''.join(['esr', 'gan']),
    ''.join(['top', 'az']),
    ''.join(['swin', 'ir']),
    ''.join(['basic', 'vsr']),
    ''.join(['waifu', '2x']),
]


def iter_files():
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        yield rel, p


def main() -> None:
    violations = []
    render_refs = []
    for rel, p in iter_files():
        rels = rel.as_posix()
        if rels == THIS_FILE:
            continue
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except Exception:
            continue
        lower = text.lower()
        calls_shared = APPROVED_RENDERER_TOKEN in lower
        for pat in OBSOLETE_PATTERNS:
            if pat in lower:
                violations.append({'path': rels, 'type': 'obsolete-render-reference', 'pattern': pat})
        if 'scale=3840:2160' in lower or 'scale=1280:720' in lower:
            render_refs.append(rels)
            approved_inline = all(x in lower for x in (
                'hqdn3d=0.6:0.6:2.0:2.0',
                'flags=lanczos',
                'cas=0.22',
                'crf 16',
            ))
            if rels != APPROVED_RENDERER and not calls_shared and not approved_inline:
                violations.append({'path': rels, 'type': 'nonstandard-inline-render'})
        if '2160p' in lower and 'scale=3840:2160' not in lower and not calls_shared:
            # Documentation may describe 2160p without implementing a renderer.
            if rel.suffix.lower() in {'.py', '.yml', '.yaml', '.sh'}:
                violations.append({'path': rels, 'type': '2160p-output-without-shared-renderer'})

    report = {
        'approved_renderer': APPROVED_RENDERER,
        'render_references': sorted(set(render_refs)),
        'violations': violations,
        'violation_count': len(violations),
    }
    Path('video_render_policy_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if violations:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
