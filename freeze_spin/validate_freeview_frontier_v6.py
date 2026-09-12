from __future__ import annotations

"""Fail-closed guardrail for the explicit three-camera Adams/Jazz solve.

The active camera set is user-locked to Left Above Rim, Right Above Rim and
Broadcast. A dynamic exact-state failure must never cause an automatic switch
to a fourth camera or a physical-camera recalibration.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "freeze_spin" / "adams_jazz_game_camera_registry_v6.json"
FRONTIER = ROOT / "freeze_spin" / "CURRENT_FREEVIEW_FRONTIER.md"
EXPECTED_GAME = "0022500301"
EXPECTED = ["Left Above Rim", "Right Above Rim", "Broadcast"]
EXPECTED_FRONTIER = "THREE_CALIBRATED_CAMERAS_LOCKED_EXACT_STATE_SYNC_ACTIVE"


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def main() -> None:
    reg = json.loads(V6.read_text())
    md = FRONTIER.read_text() if FRONTIER.exists() else ""

    require(reg["schema_version"] == 6, "v6 schema version changed")
    require(reg["game_id"] == EXPECTED_GAME, "Active game changed")
    require(reg["authoritative_frontier"] == EXPECTED_FRONTIER, "Three-camera exact-state frontier changed")
    require(reg["accepted_camera_count"] == 3, "Accepted camera count must remain exactly three")
    require(reg["accepted_camera_names"] == EXPECTED, "Accepted camera set/order changed")

    lock = reg["active_camera_lock"]
    require(lock["mode"] == "THREE_CAMERA_LOCKED", "Three-camera lock disabled")
    require(lock["locked_cameras"] == EXPECTED, "Locked camera set changed")
    require(lock["allow_camera_addition"] is False, "Camera addition silently enabled")
    require(lock["allow_camera_substitution"] is False, "Camera substitution silently enabled")
    require(lock["allow_physical_center_recalibration"] is False, "Physical-centre recalibration silently enabled")
    require(lock["allow_camera_frontier_switch_on_sync_failure"] is False, "Sync failure may not switch camera frontier")
    require(lock["unlock_requires_explicit_user_instruction"] is True, "Camera lock can be silently unlocked")
    require(lock["failure_policy"] == "EXPAND_OR_REFINE_TEMPORAL_STATE_SEARCH_WITHIN_LOCKED_CAMERAS_ONLY", "Failure policy changed")

    for name in EXPECTED:
        cam = reg["accepted_cameras"].get(name)
        require(cam is not None, f"Missing locked camera: {name}")
        require(cam.get("revoked") is False, f"Locked camera silently revoked: {name}")
        perms = cam["permissions"]
        require(perms["counts_as_distinct_metric_camera"] is True, f"Camera stopped counting as metric: {name}")
        require(perms["physical_camera_center_recalibration_allowed"] is False, f"Physical center unlocked: {name}")
        require(perms["temporal_state_search_allowed"] is True, f"Temporal search disabled: {name}")

    require(reg["accepted_cameras"]["Right Above Rim"]["status"] == "PASS_RIGHT_ABOVE_RIM_FIXED_MOUNT_ANCHOR_V74", "RAR v74 calibration lost")
    require(reg["accepted_cameras"]["Broadcast"]["status"] == "PASS_BROADCAST_SHARED_OPTICAL_CENTER_V90", "Broadcast v90 calibration lost")
    require(reg["accepted_cameras"]["Broadcast"]["hard_gate_results"]["all_v90_gates_passed"] is True, "Broadcast v90 hard gates changed")

    gates = reg["exact_state_gates"]
    require(gates["pair_min_joint_count"] >= 6, "Exact-state joint support weakened")
    require(gates["pair_median_epipolar_px_max"] <= 18.0, "Median epipolar gate weakened")
    require(gates["pair_p90_epipolar_px_max"] <= 35.0, "p90 epipolar gate weakened")
    require(gates["threshold_policy"] == "DO_NOT_WEAKEN", "Threshold anti-weakening rule removed")

    gp = reg["global_permissions"]
    require(gp["three_distinct_metric_cameras_validated"] is True, "Three calibrated cameras no longer validated")
    require(gp["camera_set_locked"] is True, "Camera set lock removed")
    require(gp["camera_addition_allowed"] is False, "Fourth-camera work silently enabled")
    require(gp["camera_substitution_allowed"] is False, "Camera substitution silently enabled")
    require(gp["physical_center_recalibration_allowed"] is False, "Physical centers silently unlocked")
    require(gp["native_resolution_only"] is True, "Native-resolution-only rule removed")
    require(gp["replay_render_allowed"] is False, "Render unlocked before exact-state/body geometry proof")

    # Human-readable source of truth must carry the same lock. Once CURRENT is
    # promoted to v6, these assertions prevent stale camera-4 guidance from returning.
    if "adams_jazz_game_camera_registry_v6.json" in md:
        required = [
            "THREE-CAMERA LOCK",
            "Left Above Rim",
            "Right Above Rim",
            "Broadcast",
            "Do not add or substitute a fourth camera",
            "joint temporal/state search",
            "native 960×540",
        ]
        for text in required:
            require(text in md, f"CURRENT frontier lost v6 guardrail: {text}")
        require("Right Slash is camera #4 and is the active unsolved frontier" not in md, "Stale camera-4 frontier returned")

    print(json.dumps({
        "status": "PASS_FREEVIEW_FRONTIER_V6_THREE_CAMERA_LOCK",
        "game_id": EXPECTED_GAME,
        "accepted_camera_count": 3,
        "locked_cameras": EXPECTED,
        "camera_addition_allowed": False,
        "camera_substitution_allowed": False,
        "physical_center_recalibration_allowed": False,
        "failure_policy": lock["failure_policy"],
        "next_stage": reg["current_dynamic_state"]["next_stage"],
    }, indent=2))


if __name__ == "__main__":
    main()
