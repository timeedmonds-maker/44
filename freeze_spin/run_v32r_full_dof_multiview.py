from __future__ import annotations

"""v32r: expose valid unbounded MHR body DOFs to the v32q image-space solve.

v32q intentionally removes fixed lo==hi MHR parameters, but its first run also
removed every parameter whose declared limit was +/-inf. Those are valid body
pose DOFs, not fixed parameters. The result had only 22 active model parameters
and could fit RAR while remaining ~20 px off in Broadcast.

This wrapper changes no observations, cameras, thresholds, identity logic, or
objective. It only presents unbounded MHR parameters to the existing v32q solver
with conservative numerical caps (+/-3.2 rad; flexible body dimensions +/-2.2),
while true equal-bound parameters remain fixed and excluded.
"""

import numpy as np
from freeze_spin import fit_v32q_mhr_direct_multiview_2d as v32q


class CharacterWithFiniteOptimizationEnvelope:
    def __init__(self, character):
        self._character = character

    def __getattr__(self, name):
        return getattr(self._character, name)

    @property
    def model_parameter_limits(self):
        lo, hi = self._character.model_parameter_limits
        lo = np.asarray(lo, float).copy()
        hi = np.asarray(hi, float).copy()
        # Preserve real finite MHR limits and exact fixed parameters. Only replace
        # open numerical bounds so scipy can expose those valid DOFs.
        lo[~np.isfinite(lo)] = -3.2
        hi[~np.isfinite(hi)] = 3.2
        return lo, hi


_original_ensure_assets = v32q.ensure_assets


def ensure_assets_with_free_dofs(work):
    geo, character, fbx, model_path = _original_ensure_assets(work)
    return geo, CharacterWithFiniteOptimizationEnvelope(character), fbx, model_path


def main():
    v32q.ensure_assets = ensure_assets_with_free_dofs
    v32q.main()


if __name__ == "__main__":
    main()
