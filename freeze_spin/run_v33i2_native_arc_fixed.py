from __future__ import annotations

"""v33i2 hotfix: preserve the full accepted v33e +/-20 frame window in v33i.

The v33i adapter correctly declares RELS=-20..20, but its inherited v33d.frames()
reader uses v33d.RELS. v33h explicitly sets that global before reading the burst;
v33i omitted the assignment, so accepted Broadcast rel +18 was not indexed and
the render aborted before any image synthesis. This wrapper changes only that
reader window. Cameras, centres, exact-state selection, ball world point and
renderer remain untouched.
"""

from freeze_spin import build_v33i_native_arc_from_v33h as v33i


def main() -> None:
    v33i.v33d.RELS = v33i.RELS
    v33i.main()


if __name__ == '__main__':
    main()
