from __future__ import annotations

"""Compatibility runner for v33i.

The v33i renderer wants an explicit projection-valid mask for two call sites,
while the shared legacy helper intentionally returns only (uv, depth).  Keep the
shared helper untouched: expose a local proxy to v33i that adds the third value,
without changing the helper's own internal calls.
"""

import numpy as np

from freeze_spin import build_three_camera_freeview_preview_v1 as _shared
from freeze_spin import build_v33i_rar_anchored_native_freeview as _v33i


class _ProjectionProxy:
    def __getattr__(self, name):
        return getattr(_shared, name)

    @staticmethod
    def project_points(K, R, C, P):
        uv, depth = _shared.project_points(K, R, C, P)
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
        return uv, depth, valid


if __name__ == '__main__':
    _v33i.p = _ProjectionProxy()
    _v33i.main()
