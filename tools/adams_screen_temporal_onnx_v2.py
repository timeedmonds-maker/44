#!/usr/bin/env python3
"""Temporal screen detector v2.

Wraps the original detector but fixes sequence persistence semantics: multiple
opponent hypotheses in one sampled frame may not satisfy the repeated-screen
gate. A retained sequence must contain evidence on at least two distinct
sampled frames. Within each frame, only the highest-scoring hypothesis is kept.
"""
from __future__ import annotations

import adams_screen_temporal_onnx as base

_ORIG = base.detect_sequences


def detect_sequences_distinct_frames(*args, **kwargs):
    seqs, bh_tids = _ORIG(*args, **kwargs)
    fixed = []
    for s in seqs:
        by_frame = {}
        for h in s.get("hits", []):
            f = int(h["frame"])
            if f not in by_frame or float(h.get("score", 0.0)) > float(by_frame[f].get("score", 0.0)):
                by_frame[f] = h
        hits = [by_frame[f] for f in sorted(by_frame)]
        if len(hits) < 2:
            continue
        s = dict(s)
        s["hits"] = hits
        s["start_frame"] = int(hits[0]["frame"])
        s["end_frame"] = int(hits[-1]["frame"])
        s["best"] = max(hits, key=lambda h: float(h.get("score", 0.0)))
        fixed.append(s)
    return fixed, bh_tids


base.detect_sequences = detect_sequences_distinct_frames

if __name__ == "__main__":
    base.main()
