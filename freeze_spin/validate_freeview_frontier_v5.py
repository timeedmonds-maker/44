from __future__ import annotations

"""Fail-closed guardrail for the authoritative Adams/Jazz free-view frontier.

This validator exists to prevent stale historical registries or later fourth-camera
experiments from silently regressing the already-certified three-camera state.
It intentionally validates project-state semantics, not camera geometry itself.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v5.json"
V4 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v4.json"
FRONTIER = ROOT / "freeze_spin" / "CURRENT_FREEVIEW_FRONTIER.md"

EXPECTED_GAME = "0022500301"
EXPECTED_ACCEPTED = ["Left Above Rim", "Right Above Rim", "Broadcast"]
EXPECTED_V5 = "freeze_spin/adams_jazz_game_camera_registry_v5.json"


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def main() -> None:
    v5 = json.loads(V5.read_text())
    v4 = json.loads(V4.read_text())
    md = FRONTIER.read_text()

    require(v5["game_id"] == EXPECTED_GAME, "Authoritative game changed")
    require(v5["accepted_camera_count"] == 3, "Accepted camera count regressed from three")
    require(v5["accepted_camera_names"] == EXPECTED_ACCEPTED, "Accepted camera names/order changed")
    require(v5["authoritative_frontier"] == "THREE_DISTINCT_METRIC_CAMERAS_SOLVED_RIGHT_SLASH_IS_CAMERA_4", "Authoritative frontier changed")
    require(v5["state_invariants"]["accepted_camera_state_is_monotonic"] is True, "Monotonic acceptance rule disabled")
    require(v5["state_invariants"]["active_target_rule"], "Active-target guardrail missing")

    for camera in EXPECTED_ACCEPTED:
        c = v5["accepted_cameras"].get(camera)
        require(c is not None, f"Accepted camera missing: {camera}")
        require(c.get("revoked") is False, f"Camera silently revoked without a new registry: {camera}")
        require(c["permissions"].get("counts_as_distinct_metric_camera") is True, f"Camera no longer counts as distinct: {camera}")

    require(v5["accepted_cameras"]["Right Above Rim"]["status"] == "PASS_RIGHT_ABOVE_RIM_FIXED_MOUNT_ANCHOR_V74", "Right Above Rim v74 lock lost")
    require(v5["accepted_cameras"]["Broadcast"]["status"] == "PASS_BROADCAST_SHARED_OPTICAL_CENTER_V90", "Broadcast v90 lock lost")
    require(v5["accepted_cameras"]["Broadcast"]["hard_gate_results"]["all_v90_gates_passed"] is True, "Broadcast v90 gate state changed")

    active = v5["active_candidate"]
    require(active["camera"] == "Right Slash", "Active fourth camera is no longer Right Slash")
    require(active["ordinal"] == 4, "Right Slash is no longer marked as camera #4")
    require(active["status"] == "UNSOLVED_FOURTH_CAMERA_ACTIVE_FRONTIER", "Right Slash frontier status changed")

    gp = v5["global_permissions"]
    require(gp["three_distinct_metric_cameras_validated"] is True, "Three-camera validation flag regressed")
    require(gp["four_distinct_metric_cameras_validated"] is False, "Four cameras cannot be claimed without a new certified registry")
    require(gp["replay_render_allowed"] is False, "Replay render unlocked before four-camera proof")

    require(v4.get("historical_snapshot_only") is True, "v4 is no longer marked historical")
    require(v4.get("superseded_by") == EXPECTED_V5, "v4 no longer points to v5")
    require("DO_NOT_USE_FOR_CURRENT_CAMERA_COUNT" in v4, "v4 stale-state warning removed")

    required_md = [
        EXPECTED_GAME,
        "THREE distinct metric cameras solved and locked",
        "Left Above Rim",
        "Right Above Rim",
        "Broadcast",
        "Right Slash is camera #4",
        "Portland Sidy Cissoko / Steven Adams block work is earlier prototype/architecture reference",
        "MONOTONIC ACCEPTANCE RULE",
    ]
    for text in required_md:
        require(text in md, f"Human-readable frontier lost required guardrail: {text}")

    print(json.dumps({
        "status": "PASS_FREEVIEW_FRONTIER_V5_INVARIANTS",
        "game_id": EXPECTED_GAME,
        "accepted_camera_count": 3,
        "accepted_cameras": EXPECTED_ACCEPTED,
        "active_candidate": "Right Slash",
        "active_candidate_ordinal": 4,
        "historical_registry_v4_blocked": True,
        "portland_not_current_active_solve": True,
    }, indent=2))


if __name__ == "__main__":
    main()
