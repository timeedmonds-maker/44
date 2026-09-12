from __future__ import annotations

"""v32q: execute v32p while removing fixed MHR parameters from SciPy's solve vector.

v32p correctly uses MHR's declared parameter bounds, but scipy.optimize.least_squares
requires every optimized lower bound to be strictly smaller than its upper bound.
MHR legitimately contains fixed parameters where lo == hi.  Those are not degrees
of freedom and must not be perturbed merely to satisfy SciPy.

This wrapper replaces v32p's local least_squares callable with an adapter that:
- identifies lo < hi parameters as genuinely free;
- holds lo == hi parameters exactly fixed at their declared value;
- optimizes only the free subvector with SciPy;
- reconstructs a full-size OptimizeResult.x for v32p downstream code.

No camera, pose observation, mesh, pixel, resolution, threshold or player identity
logic is changed.  Three calibrated cameras only; native 960x540 only.
"""

import numpy as np
from scipy.optimize import least_squares as scipy_least_squares
from scipy.optimize import OptimizeResult

from freeze_spin import fit_v32p_pymomentum_geometry_to_v32m_pose as v32p


def free_bound_least_squares(fun, x0, bounds=(-np.inf, np.inf), *args, **kwargs):
    x0 = np.asarray(x0, dtype=float)
    lo = np.broadcast_to(np.asarray(bounds[0], dtype=float), x0.shape).copy()
    hi = np.broadcast_to(np.asarray(bounds[1], dtype=float), x0.shape).copy()

    invalid = lo > hi
    if np.any(invalid):
        ii = np.flatnonzero(invalid)
        raise ValueError(f"invalid parameter bounds lo>hi at indices {ii.tolist()}")

    # Strictly free dimensions only. Equal-bound parameters are true constants.
    free = lo < hi
    fixed = ~free
    if np.any(fixed):
        fixed_values = lo[fixed]
    else:
        fixed_values = np.empty((0,), dtype=float)

    base = x0.copy()
    if np.any(fixed):
        base[fixed] = fixed_values
    # Numerical guard only for genuinely free dimensions: keep initial point inside.
    if np.any(free):
        eps = 1e-12
        base[free] = np.minimum(np.maximum(base[free], lo[free] + eps), hi[free] - eps)

    def expand(z):
        x = base.copy()
        x[free] = z
        return x

    def wrapped(z, *fargs, **fkwargs):
        return fun(expand(z), *fargs, **fkwargs)

    if not np.any(free):
        # Degenerate but well-defined: evaluate once and return a full result.
        r = np.asarray(fun(base, *args), dtype=float)
        return OptimizeResult(
            x=base,
            cost=0.5 * float(np.dot(r, r)),
            fun=r,
            optimality=0.0,
            active_mask=np.zeros_like(base, dtype=int),
            nfev=1,
            njev=0,
            status=1,
            message="All parameters fixed by declared bounds.",
            success=True,
        )

    # Preserve user kwargs but operate on the reduced variable vector.
    result = scipy_least_squares(
        wrapped,
        base[free],
        bounds=(lo[free], hi[free]),
        *args,
        **kwargs,
    )
    full_x = expand(result.x)

    # Return an OptimizeResult compatible with v32p, retaining all SciPy diagnostics.
    out = OptimizeResult(result)
    out.x = full_x
    active = np.zeros_like(base, dtype=int)
    if hasattr(result, "active_mask"):
        active[free] = np.asarray(result.active_mask, dtype=int)
    out.active_mask = active
    out.fixed_parameter_count = int(np.sum(fixed))
    out.free_parameter_count = int(np.sum(free))
    return out


def main():
    v32p.least_squares = free_bound_least_squares
    v32p.main()


if __name__ == "__main__":
    main()
