# CURRENT FREEVIEW FRONTIER — AUTHORITATIVE

Updated: 2026-09-07
Game: `0022500301` (HOU @ UTA, 2025-11-30)
Branch: `codex/multi-angle-freeze-spin-poc`
Machine-readable authority: `freeze_spin/adams_jazz_game_camera_registry_v5.json`

## READ THIS BEFORE ANY OLDER CAMERA REGISTRY

The project has **THREE distinct metric cameras solved and locked for the current game**:

1. **Left Above Rim** — accepted through v41/v42.
2. **Right Above Rim** — metric camera v73 PASS; fixed-centre PTZ anchor v74 PASS.
3. **Broadcast** — shared optical centre v90 PASS; explicitly counts as a distinct metric camera.

**Right Slash is camera #4 and is the active unsolved frontier.**

Do not report the project as having only one solved camera. That error occurs if `adams_jazz_game_camera_registry_v4.json` is read as current state. v4 stops at the v57a frontier and predates the successful v73/v74 and v90 certifications. It is historical only and is explicitly superseded by v5.

## MONOTONIC ACCEPTANCE RULE

Accepted-camera state is monotonic. A solved camera remains solved unless a later explicit revocation record:

- names the camera;
- identifies the exact falsification test;
- cites the workflow/artifact that failed it; and
- explicitly changes that camera's accepted status.

Work on a different camera does **not** revoke or reopen already accepted cameras. A stale registry does **not** override later successful hard-gated workflows.

Therefore, before claiming fewer than three solved cameras, a future agent must first find an explicit later revocation for Left Above Rim, Right Above Rim or Broadcast. If no such record exists, the accepted count remains three.

## CURRENT EVIDENCE

### Left Above Rim

Accepted as a non-coplanar physical/event camera in v41/v42. Current basket-local physical centre prior is retained in registry v5. It counts as distinct metric camera #1.

### Right Above Rim

`PASS_RIGHT_ABOVE_RIM_METRIC_CAMERA_V73` was followed by `PASS_RIGHT_ABOVE_RIM_FIXED_MOUNT_ANCHOR_V74` in workflow run `33954197690`.

The v74 architecture hard-locks the **physical camera centre** across same-rig states and solves per state for:

- orientation;
- focal length;
- principal point / digital crop.

This is a key architectural lesson for camera #4.

### Broadcast

`PASS_BROADCAST_SHARED_OPTICAL_CENTER_V90` in workflow run `33954197537`.

The proof used two native source states plus independent rim validation, wide multistarts, support removal, and 64 independent half-pixel perturbation trials. It explicitly set:

- `broadcast_frame_c_metric_event_camera_allowed = true`;
- `broadcast_physical_camera_center_allowed = true`;
- `broadcast_counts_as_distinct_metric_camera = true`.

Its strongest stability numbers were:

- multistart centre spread: `0.001862 cm`;
- support-removal max centre shift: `70.877 cm` against a `75 cm` gate;
- half-pixel perturbation max centre shift: `67.718 cm` against a `75 cm` gate.

It counts as distinct metric camera #3.

## ACTIVE FRONTIER: RIGHT SLASH = CAMERA #4

The v92–v111 Right Slash work has not invalidated any of the three solved cameras. It is solely an attempt to promote the **fourth** camera.

The next solve should exploit what the three successful cameras already established rather than rebuilding the world from scratch:

- keep the established NBA basket/court metric world fixed;
- solve **one shared Right Slash physical centre** over multiple same-game native states;
- allow per-state orientation, focal length and crop/principal point to vary;
- use only the target/rim/court geometry genuinely visible in each state;
- reserve independent holdout geometry;
- enforce unchanged multistart, support-removal, half-pixel perturbation, physical-plausibility and full-resolution visual-overlay gates.

This should be implemented as the decisive v112-style fourth-camera attempt.

## DO NOT DO

- Do not reopen Left Above Rim, Right Above Rim or Broadcast merely because Right Slash is difficult.
- Do not use `adams_jazz_game_camera_registry_v4.json` as current state.
- Do not reduce the accepted camera count without an explicit later revocation artifact.
- Do not weaken the Right Slash gates to get to four cameras.
- Do not render the free-view arc until the fourth camera passes cross-camera metric QA.

## NEXT MILESTONE AFTER RIGHT SLASH PASSES

1. Four-camera cross-camera metric consistency QA.
2. Exact-state synchronization at the target moment.
3. Source-grounded static views at approximately `0°, 5°, 10°, 15°, 20°, 25°`.
4. Only after those stills pass visual QA: animate the short orbit and resume real video.
