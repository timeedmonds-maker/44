#!/usr/bin/env python3
"""DROP_COVERAGE_UNIVERSAL_V3

Compatibility wrapper around the validated universal-v2 role resolver. The
role/lineup/colour logic remains unchanged; only the renderer is upgraded to
LOCKED_BROADCAST_SCREEN_V4 so every camera angle uses the universal
perspective-normalized floor-ring policy.
"""
import drop_coverage_universal_v2 as base
import locked_broadcast_screen_v4 as ring_renderer

base.core = ring_renderer
base.main()
