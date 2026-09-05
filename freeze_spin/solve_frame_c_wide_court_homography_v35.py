from __future__ import annotations

"""v35: computationally efficient reproduction of the v34 wide-court proof.

v34's geometry was intentionally rigorous but computationally wasteful: every
half-pixel perturbation reran the full eight-root multistart solve even though all
64 perturbations are local changes around an already accepted optimum.  The first
Actions run therefore hit its 30-minute job limit before producing a verdict.

v35 keeps the exact same observations, 64 perturbations and numerical thresholds.
It changes only the perturbation optimizer:
  * the nominal solve remains the original deterministic multistart solve;
  * the root-reduction test remains the original deterministic multistart solve;
  * each half-pixel perturbation is solved once from the accepted nominal root;
  * eight deterministic sentinel perturbations are ALSO solved with the original
    full multistart solver and must agree with the warm solution to <=0.05 px p95
    on both dense regulation curves.

This is a runtime change, not a weaker geometry gate.  Passing still authorizes
only the immutable Frame C floor homography, never the 3D event camera or replay.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from freeze_spin import solve_frame_c_wide_court_homography_v34 as base


ORIGINAL_SOLVE = base.solve
SENTINEL_TRIALS = {0, 7, 15, 23, 31, 47, 55, 63}
SENTINEL_MAX_P95_DISAGREEMENT_PX = 0.05
WARM_MAX_NFEV = 10000

_warm_call_count = 0
_sentinel_rows: list[dict] = []


def solve_warm_once(h0: np.ndarray, groups: dict, warm: np.ndarray) -> np.ndarray:
    """Local solve with the exact v34 objective, starting at the proved root."""
    fit = least_squares(
        lambda z: base.residual(z, h0, groups),
        np.asarray(warm, dtype=np.float64),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=WARM_MAX_NFEV,
    )
    z = np.asarray(fit.x, dtype=np.float64)
    Hm = base.H_from_z(z, h0)
    if not np.isfinite(z).all() or not np.isfinite(Hm).all() or abs(float(np.linalg.det(Hm))) < 1e-12:
        raise RuntimeError("v35 warm perturbation solve produced a non-finite/degenerate homography")
    return z


def accelerated_solve(h0: np.ndarray, groups: dict, *, warm: np.ndarray | None = None) -> np.ndarray:
    global _warm_call_count

    # Nominal solve remains the original full multistart solve.
    if warm is None:
        return ORIGINAL_SOLVE(h0, groups, warm=None)

    _warm_call_count += 1

    # The first warm solve in v34 main() is the reduced-support root-consistency
    # test. Preserve it exactly as the original full multistart solve.
    if _warm_call_count == 1:
        return ORIGINAL_SOLVE(h0, groups, warm=warm)

    trial = _warm_call_count - 2
    fast = solve_warm_once(h0, groups, warm)

    if trial in SENTINEL_TRIALS:
        full = ORIGINAL_SOLVE(h0, groups, warm=warm)
        H_fast = base.H_from_z(fast, h0)
        H_full = base.H_from_z(full, h0)
        d3 = np.linalg.norm(
            base.project_h(H_fast, base.dense_three()) - base.project_h(H_full, base.dense_three()), axis=1
        )
        df = np.linalg.norm(
            base.project_h(H_fast, base.dense_ft()) - base.project_h(H_full, base.dense_ft()), axis=1
        )
        row = {
            "trial": int(trial),
            "three_point_p95_disagreement_px": float(np.percentile(d3, 95)),
            "three_point_max_disagreement_px": float(np.max(d3)),
            "free_throw_p95_disagreement_px": float(np.percentile(df, 95)),
            "free_throw_max_disagreement_px": float(np.max(df)),
        }
        _sentinel_rows.append(row)
        if (
            row["three_point_p95_disagreement_px"] > SENTINEL_MAX_P95_DISAGREEMENT_PX
            or row["free_throw_p95_disagreement_px"] > SENTINEL_MAX_P95_DISAGREEMENT_PX
        ):
            raise RuntimeError(f"v35 warm/full multistart sentinel disagreement: {row}")

    return fast


def main() -> None:
    # Reuse the complete v34 proof and its unchanged gates, substituting only the
    # accelerated perturbation solver above.
    base.solve = accelerated_solve
    base.main()

    out = None
    for i, token in enumerate(sys.argv):
        if token == "--out" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])
            break
    if out is None:
        raise RuntimeError("--out is required")

    src_json = out / "frame_c_wide_court_homography_v34.json"
    src_overlay = out / "frame_c_wide_court_overlay_v34.png"
    payload = json.loads(src_json.read_text(encoding="utf-8"))

    if len(_sentinel_rows) != len(SENTINEL_TRIALS):
        raise RuntimeError(
            f"Expected {len(SENTINEL_TRIALS)} multistart sentinels, got {len(_sentinel_rows)}"
        )
    max_three = max(x["three_point_p95_disagreement_px"] for x in _sentinel_rows)
    max_ft = max(x["free_throw_p95_disagreement_px"] for x in _sentinel_rows)
    equivalence_pass = bool(
        max_three <= SENTINEL_MAX_P95_DISAGREEMENT_PX
        and max_ft <= SENTINEL_MAX_P95_DISAGREEMENT_PX
    )
    if not equivalence_pass:
        raise RuntimeError("v35 perturbation acceleration equivalence gate failed")

    payload["version"] = "v35_wide_regulation_court_accelerated"
    payload["solver_acceleration"] = {
        "nominal_solver": "unchanged v34 deterministic 8-root multistart",
        "root_reduction_solver": "unchanged v34 deterministic 8-root multistart",
        "perturbation_trial_count": int(payload["half_pixel_training_annotation_perturbation"]["trial_count"]),
        "perturbation_solver": "single exact-objective least_squares warm-started from accepted nominal root",
        "warm_max_nfev": WARM_MAX_NFEV,
        "sentinel_trial_indices": sorted(SENTINEL_TRIALS),
        "sentinel_full_multistart_results": _sentinel_rows,
        "max_three_point_p95_warm_vs_full_disagreement_px": float(max_three),
        "max_free_throw_p95_warm_vs_full_disagreement_px": float(max_ft),
        "max_allowed_p95_disagreement_px": SENTINEL_MAX_P95_DISAGREEMENT_PX,
        "equivalence_gate": equivalence_pass,
        "note": "All 64 half-pixel perturbations and every v34 geometric threshold are unchanged; sentinels prove the faster local solve reproduces the original multistart solution in source-pixel projection.",
    }
    payload["gates"]["accelerated_perturbation_solver_matches_full_multistart_sentinels"] = equivalence_pass
    payload["status"] = "PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35" if all(payload["gates"].values()) else "FAIL_WIDE_COURT_FLOOR_HOMOGRAPHY_V35"
    payload["floor_homography_allowed"] = bool(all(payload["gates"].values()))
    payload["metric_event_camera_allowed"] = False
    payload["replay_render_allowed"] = False

    dst_json = out / "frame_c_wide_court_homography_v35.json"
    dst_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if src_overlay.exists():
        (out / "frame_c_wide_court_overlay_v35.png").write_bytes(src_overlay.read_bytes())

    print(json.dumps({
        "status": payload["status"],
        "heldout_three_p95_px": payload["heldout_pixel_curve_error"]["three_point_arc"]["p95_px"],
        "heldout_ft_p95_px": payload["heldout_pixel_curve_error"]["free_throw_front_semicircle"]["p95_px"],
        "max_half_pixel_three_p95_shift_px": payload["half_pixel_training_annotation_perturbation"]["max_three_point_p95_shift_px"],
        "max_half_pixel_ft_p95_shift_px": payload["half_pixel_training_annotation_perturbation"]["max_free_throw_p95_shift_px"],
        "sentinel_max_three_p95_disagreement_px": max_three,
        "sentinel_max_ft_p95_disagreement_px": max_ft,
        "floor_homography_allowed": payload["floor_homography_allowed"],
    }, indent=2), flush=True)

    if payload["status"] != "PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
