from __future__ import annotations

"""v32x: full-MHR sweep of the strongest coherent LAR states.

v32w solved the LAR identity problem and found several plausible whole-body
states.  Its rank-1 t+03 state passed raw LAR<->RAR consistency and produced
excellent median MHR reprojection, but missed the strict LAR max/p90 and held-out
Broadcast gates.  Rank-2 t+04 had a slightly worse epipolar median but a much
more balanced tail.

v32x therefore does not change a single geometry or QA threshold.  It runs the
unchanged v32v articulated solve independently for the top coherent LAR
candidates and lets downstream fit + independent Broadcast holdout decide.

No per-joint temporal mixing, no camera refinement, no generated RGB, no
upscale, and no gate relaxation.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from freeze_spin import run_v32v_lar_rar_global_state_mhr as v32v
from freeze_spin import run_v32w_lar_all_candidate_epipolar_mhr as v32w
from freeze_spin import solve_v32j_rfdetr_freeze_pose as v32j
from freeze_spin import solve_v32l_rar_temporal_denseflow as v32l

LAR = v32v.LAR
BODY = v32v.BODY
MIN_CONF = v32v.MIN_CONF


def all_coherent_candidates(model, stage: Path, scene: dict, rar_xy, rar_valid):
    """Return full candidate objects, sorted exactly by the v32w pre-fit policy."""
    frames = v32v.load_burst(stage, LAR, range(0, 7))
    candidates = []
    for rel in (0, 2, 3, 4, 5, 6):
        dets = v32j.infer(model, frames[rel])
        for i, d in enumerate(dets):
            dark, torso, confident = v32w.appearance(d, frames[rel])
            if confident < 6:
                continue
            if rel == 0:
                xy0 = np.asarray(d["xy"], float)
                valid0 = np.asarray(d["conf"], float) >= MIN_CONF
                trace = None
                source = "direct_t00"
            else:
                xy0, valid0, trace = v32l.dense_track_back(frames, rel, d["xy"], d["conf"])
                source = f"temporal_t+{rel}"
            c = v32w.coherent_candidate(
                scene, source, rel, i, d, frames[rel], xy0, valid0, trace,
                rar_xy, rar_valid,
            )
            if int(c["epipolar"]["joint_count"]) >= 6:
                candidates.append(c)
    if not candidates:
        raise RuntimeError("v32x: no coherent LAR candidate has >=6 RAR-comparable joints")
    candidates.sort(key=lambda c: (
        c["selection_score"], c["epipolar"]["median_px"], -c["torso_dark_score"]
    ))
    return candidates, frames


def serial_candidate(rank: int, c: dict) -> dict:
    return {
        "rank": int(rank),
        "source": c["source"],
        "anchor_rel": int(c["anchor_rel"]),
        "selected_index": int(c["selected_index"]),
        "box_xyxy": c["box_xyxy"],
        "selection_score": float(c["selection_score"]),
        "dark_fraction": float(c["dark_fraction"]),
        "torso_dark_score": float(c["torso_dark_score"]),
        "confident_body_joints": int(c["confident_body_joints"]),
        "epipolar": c["epipolar"],
    }


def selector_for_rank(rank: int, audit_sink: dict):
    def choose(model, stage, qual, scene, rar_xy, rar_valid):
        candidates, frames = all_coherent_candidates(model, stage, scene, rar_xy, rar_valid)
        if rank < 1 or rank > len(candidates):
            raise RuntimeError(f"v32x: candidate rank {rank} unavailable; have {len(candidates)}")
        selected = candidates[rank - 1]
        serial = [serial_candidate(i + 1, c) for i, c in enumerate(candidates)]
        audit_sink.clear()
        audit_sink.update({
            "candidate_count": len(candidates),
            "selected": serial[rank - 1],
            "top_candidates": serial[:12],
        })
        return selected, serial, frames
    return choose


def positive_excess(value: float, limit: float) -> float:
    return max(0.0, float(value) / float(limit) - 1.0)


def gate_deficit(qa: dict) -> float:
    p = qa["per_camera_reprojection"]
    l = p[v32v.LAR]; r = p[v32v.RAR]; b = p[v32v.BCAST]
    # Exact existing thresholds, expressed only as a ranking distance for failed
    # candidates. This does NOT alter pass/fail.
    vals = [
        positive_excess(l["median_px"], 8.0), positive_excess(l["p90_px"], 16.0), positive_excess(l["max_px"], 26.0),
        positive_excess(r["median_px"], 8.0), positive_excess(r["p90_px"], 16.0), positive_excess(r["max_px"], 26.0),
        positive_excess(b["median_px"], 24.0), positive_excess(b["p75_px"], 36.0),
    ]
    return float(sum(vals))


def run_rank(args, rank: int) -> dict:
    out = args.out / f"rank{rank:02d}"
    out.mkdir(parents=True, exist_ok=True)
    audit = {}
    original_selector = v32v.choose_global_lar_state
    original_argv = sys.argv[:]
    v32v.choose_global_lar_state = selector_for_rank(rank, audit)
    sys.argv = [
        "run_v32v_lar_rar_global_state_mhr",
        "--b32-root", str(args.b32_root),
        "--work", str(args.work),
        "--out", str(out),
        "--max-nfev", str(args.max_nfev),
    ]
    exit_code = 0
    try:
        v32v.main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    finally:
        v32v.choose_global_lar_state = original_selector
        sys.argv = original_argv

    qpath = out / "v32v_qa.json"
    if not qpath.exists():
        raise RuntimeError(f"v32x rank {rank}: v32v QA missing (exit={exit_code})")
    qa = json.loads(qpath.read_text())
    apath = out / "v32v_state_audit.json"
    state = json.loads(apath.read_text()) if apath.exists() else {}
    selected = audit.get("selected", {})
    row = {
        "rank": int(rank),
        "solver_exit_code": int(exit_code),
        "candidate": selected,
        "selected_lar_source": state.get("selected_lar_source"),
        "selected_lar_anchor_rel": state.get("selected_lar_anchor_rel"),
        "selected_lar_detection_index": state.get("selected_lar_detection_index"),
        "status": qa.get("status"),
        "gate": qa.get("gate"),
        "per_camera_reprojection": qa.get("per_camera_reprojection"),
        "optimizer": qa.get("optimizer"),
        "gate_deficit": gate_deficit(qa),
    }
    (out / "v32x_rank_summary.json").write_text(json.dumps(row, indent=2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b32-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ranks", default="1,2,3")
    ap.add_argument("--max-nfev", type=int, default=360)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True); a.out.mkdir(parents=True, exist_ok=True)
    ranks = [int(x.strip()) for x in a.ranks.split(",") if x.strip()]
    rows = []
    for rank in ranks:
        print(f"\n=== v32x full MHR candidate rank {rank} ===", flush=True)
        rows.append(run_rank(a, rank))

    def key(row):
        g = row.get("gate") or {}
        return (
            0 if g.get("three_camera_leave_one_out_gate") else 1,
            0 if g.get("two_view_geometry_gate") else 1,
            0 if g.get("raw_state_consistency") else 1,
            float(row["gate_deficit"]),
            float(row["per_camera_reprojection"][v32v.BCAST]["median_px"]),
        )
    rows.sort(key=key)
    winner = rows[0]
    unlocked = bool((winner.get("gate") or {}).get("static_0_15_geometry_unlocked"))
    summary = {
        "version": "v32x_lar_candidate_full_mhr_sweep",
        "policy": "unchanged v32v geometry/gates; full articulated fit decides among coherent whole-body LAR candidates",
        "tested_ranks": ranks,
        "winner_rank": winner["rank"],
        "winner": winner,
        "all_results": rows,
        "static_0_15_geometry_unlocked": unlocked,
        "next_if_pass": "render native 960x540 static 0/5/10/15 proof; inspect anatomy before animation",
        "next_if_fail": "diagnose only the remaining gate residuals of the best full-fit candidate; do not relax thresholds",
    }
    (a.out / "v32x_summary.json").write_text(json.dumps(summary, indent=2))
    # Copy the winning three-camera overlay to the root for immediate visual QA.
    src = a.out / f"rank{winner['rank']:02d}" / "v32v_three_camera_overlay.png"
    if src.exists():
        shutil.copy2(src, a.out / "v32x_winner_three_camera_overlay.png")
    print(json.dumps(summary, indent=2), flush=True)
    if not unlocked:
        raise SystemExit(6)


if __name__ == "__main__":
    main()
