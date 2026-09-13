#!/usr/bin/env python3
"""Production POC v2: time-consistent BoT-SORT wrapper.

This deliberately reuses adams_screen_prod_poc.py rather than forking the full
pipeline. The v1 POC samples detections at --analysis-fps but configured
BoT-SORT at the native source FPS and called update() without timestamps. That
made track ageing / motion prediction operate on the wrong temporal cadence.

V2 injects the actual sample timestamps and configures the tracker at the
analysis cadence. Event 646 remains an engineering benchmark only; historical
identity QA does not validate it as a Steven Adams screen.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import adams_screen_prod_poc as base
from trackers import BoTSORTTracker as _BoTSORTTracker


def _arg_float(name: str, default: float) -> float:
    try:
        i = sys.argv.index(name)
        return float(sys.argv[i + 1])
    except (ValueError, IndexError, TypeError):
        return float(default)


ANALYSIS_FPS = max(0.25, _arg_float('--analysis-fps', 10.0))


class TimeConsistentBoTSORT(_BoTSORTTracker):
    """BoT-SORT configured for sampled detections, with explicit timestamps."""

    # trackers.BaseTracker validates every inherited search-space key against
    # the subclass __init__ signature. This wrapper intentionally owns no tuning
    # space, so clear the inherited registry.
    search_space = {}

    def __init__(self, *args, frame_rate=30.0, **kwargs):
        self._analysis_fps = ANALYSIS_FPS
        self._sample_index = 0
        # Keep the tracker's seconds-based lost-track semantics, but make its
        # frame cadence match the frames it actually receives.
        super().__init__(*args, frame_rate=self._analysis_fps, **kwargs)

    def update(self, detections, frame=None, timestamp=None):
        if timestamp is None:
            timestamp = self._sample_index / self._analysis_fps
        self._sample_index += 1
        return super().update(detections, frame=frame, timestamp=float(timestamp))


# main() resolves this module-global symbol at runtime, so the rest of the
# production POC stays byte-for-byte shared with v1.
base.BoTSORTTracker = TimeConsistentBoTSORT


if __name__ == '__main__':
    base.main()
