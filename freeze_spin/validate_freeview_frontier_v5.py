from __future__ import annotations

"""Compatibility validator for the historical v5 free-view frontier.

v5 originally made Right Slash the active camera-4 frontier. On 2026-09-13 the
user explicitly locked the active solve to the three already-calibrated cameras.
If registry v6 exists, v5 must not re-impose the obsolete fourth-camera path;
instead it delegates to the v6 fail-closed guardrail.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v6.json"
V5 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v5.json"
V4 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v4.json"
FRONTIER = ROOT / "freeze_spin" / "CURRENT_FREEVIEW_FRONTIER.md"

EXPECTED_GAME = "0022500301"
EXPECTED_ACCEPTED = ["Left Above Rim", "Right Above Rim", "Broadcast"]
EXPECTED_V5 = "freeze_spin/adams_jazz_game_camera_registry_v5.json"


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def validate_historical_v5() -> None:
    v5 = json.loads(V5.read_text())
    v4 = json.loads(V4.read_text())
    md = FRONTIER.read_text()

    require(v5["game_id"] == EXPECTED_GAME, "Authoritative game changed")
    require(v5["accepted_camera_count"] == 3, "Accepted camera count regressed from three")
    require(v5["accepted_camera_names"] == EXPECTED_ACCEPTED, "Accepted camera names/order changed")
    require(v5["authoritative_frontier"] == "THREE_DISTINCT_METRIC_CAMERAS_SOLVED_RIGHT_SLASH_IS_CAMERA_4", "Historical v5 frontier changed")
    require(v5["state_invariants"]["accepted_camera_state_is_monotonic"] is True, "Monotonic acceptance rule disabled")

    for camera in EXPECTED_ACCEPTED:
        c = v5["accepted_cameras"].get(camera)
        require(c is not None, f"Accepted camera missing: {camera}")
        require(c.get("revoked") is False, f"Camera silently revoked: {camera}")
        require(c["permissions"].get("counts_as_distinct_metric_camera") is True, f"Camera no longer counts as distinct: {camera}")

    require(v4.get("historical_snapshot_only") is True, "v4 is no longer marked historical")
    require(v4.get("superseded_by") == EXPECTED_V5, "v4 no longer points to v5")
    require("DO_NOT_USE_FOR_CURRENT_CAMERA_COUNT" in v4, "v4 stale-state warning removed")

    required_md = [EXPECTED_GAME, "Left Above Rim", "Right Above Rim", "Broadcast"]
    for text in required_md:
        require(text in md, f"Human-readable frontier lost required state: {text}")

    print(json.dumps({
        "status": "PASS_FREEVIEW_FRONTIER_V5_HISTORICAL_INVARIANTS",
        "game_id": EXPECTED_GAME,
        "accepted_camera_count": 3,
        "accepted_cameras": EXPECTED_ACCEPTED,
        "v6_present": False,
    }, indent=2))


def main() -> None:
    if V6.exists():
        # v6 explicitly supersedes the v5 camera-4 frontier. Never let this
        # compatibility workflow resurrect Right Slash as an active path.
        from freeze_spin.validate_freeview_frontier_v6 import main as validate_v6
        validate_v6()
        return
    validate_historical_v5()


if __name__ == "__main__":
    main()
