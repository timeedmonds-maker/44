from __future__ import annotations

"""v33i2 hotfixes for the accepted v33h native arc.

This wrapper changes only input plumbing around the existing v33i renderer:
1. preserve the full accepted v33e +/-20 frame-reader window;
2. bridge the v32j camera mapping shape (K/R/C) to inherited render layers that
   use either named mapping access or legacy tuple access (C,R,K).

No camera is refit. No centre, exact-state selection, ball world point, source
frame, renderer geometry, RGB source, resolution or appearance rule changes.
"""

from freeze_spin import build_v33i_native_arc_from_v33h as v33i


class _CameraCompat(dict):
    """Modern K/R/C mapping plus legacy tuple semantics (C,R,K)."""
    _ORDER = ('C', 'R', 'K')

    def __iter__(self):
        for key in self._ORDER:
            yield dict.__getitem__(self, key)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += 3
            if key not in (0, 1, 2):
                raise IndexError(key)
            key = self._ORDER[key]
        return dict.__getitem__(self, key)


def main() -> None:
    v33i.v33d.RELS = v33i.RELS
    original_cam = v33i.v32j.cam

    def compat_cam(scene, label):
        d = original_cam(scene, label)
        return _CameraCompat(K=d['K'], R=d['R'], C=d['C'])

    v33i.v32j.cam = compat_cam
    try:
        v33i.main()
    finally:
        v33i.v32j.cam = original_cam


if __name__ == '__main__':
    main()
