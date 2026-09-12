# CURRENT FREEVIEW FRONTIER — AUTHORITATIVE

Updated: 2026-09-13
Game: `0022500301` (HOU @ UTA, 2025-11-30)
Active calibration lineage: event-489 / Frame-C Adams-Jazz lineage
Branch: `codex/multi-angle-freeze-spin-poc`
Machine-readable authority: `freeze_spin/adams_jazz_game_camera_registry_v6.json`

## READ THIS BEFORE ANY OLDER CAMERA REGISTRY OR PORTLAND PROTOTYPE NOTES

The current active solve is the Adams-Jazz HOU @ UTA game `0022500301`. The Portland Sidy Cissoko / Steven Adams block work is earlier prototype/architecture reference and must not replace this frontier unless the user explicitly changes targets.

## THREE-CAMERA LOCK

The active solve has **exactly THREE calibrated physical cameras**, and the camera set is explicitly locked:

1. **Left Above Rim** — accepted through v41/v42.
2. **Right Above Rim** — metric camera v73 PASS; fixed-centre PTZ anchor v74 PASS.
3. **Broadcast** — shared optical centre v90 PASS.

**Do not add or substitute a fourth camera. Do not change the camera set after a synchronization failure. Do not re-solve the accepted physical camera centres from player or ball residuals.**

This instruction supersedes the older v5/right-slash frontier and the older project note that made a four-camera proof the next milestone. Right Slash and all other candidate-camera work are historical/inactive for the present solve unless the user explicitly unlocks the camera set.

A dynamic-state mismatch does not revoke a camera calibration. The three accepted cameras remain calibrated and accepted unless a later explicit calibration falsification/revocation artifact says otherwise.

## CURRENT DYNAMIC RESULT

v33c deliberately failed closed with `FAIL_CLOSED_V33C_NO_RAR_EXACT_STATE`.

That result means only that the tested temporal combination did not produce one mutually consistent Steven Adams state across all three views. v33c held verified LAR and Broadcast reference states fixed and swept RAR only over rel `-6..+6`. It therefore does **not** exhaust the possible joint temporal alignment across the three cameras.

The strongest v33c facts remain:

- verified LAR ↔ Broadcast state passed the existing pairwise epipolar gate;
- best RAR candidate was rel `-3` / `frame0253`;
- the RAR static camera transfer for that real frame was excellent and preserved the accepted physical camera centre;
- RAR agreed with LAR but not with Broadcast at that selected state;
- exact three-view state was not found;
- no render was unlocked.

Nothing in v33c invalidates the RAR v73/v74 calibration.

## ACTIVE FRONTIER: JOINT THREE-CAMERA EXACT-STATE SYNCHRONIZATION

The next engineering stage is `v33d_locked_three_camera_joint_state_search`.

It must perform a **joint temporal/state search** across **Left Above Rim + Right Above Rim + Broadcast**, rather than fixing two views and sweeping only the third. The physical camera calibrations stay locked.

Required method:

- search wider real native-frame windows around the known event moment in all three cameras;
- lock Steven Adams identity independently in each view before using geometry to rank states;
- use one real decoded frame per camera per tested triplet;
- never mix different temporal frames joint-by-joint;
- preserve the accepted physical camera centre for every camera;
- for RAR, exact static frame-state transfer may update orientation/focal/principal-point/crop only when the existing fixed-centre transfer gate passes;
- evaluate all three pairwise epipolar relationships for candidate triplets;
- retain the existing minimum pair support and epipolar thresholds or make them stricter, never weaker;
- after a temporal triplet passes pairwise gates, triangulate and run three-view reprojection/held-out geometry QA;
- if no triplet passes, expand/refine the temporal search **inside these same three cameras** and fail closed.

The current pairwise state gate is at least 6 supported joints, median epipolar error no greater than 18 px, and p90 no greater than 35 px. These are ceilings, not targets, and must not be loosened to create a pass.

## SOURCE AND OUTPUT QUALITY

Use real NBA source frames at **native 960×540** for this solve. No UHD/upscaling, generated RGB, generative fill, temporal interpolation, blur, or crossfade is allowed to hide a synchronization or geometry failure.

## RENDER LOCK

Do not render a free-view arc yet.

A novel-view render remains locked until:

1. one exact three-camera Steven Adams state passes the joint temporal/epipolar gates;
2. a source-grounded three-view body/ball reconstruction passes reprojection and visual QA; and
3. the intended virtual viewpoint is actually supported by those three calibrated source views.

The engine must earn a view geometrically; it must not invent missing coverage.

## FAILURE POLICY

If v33d or a later exact-state solve fails:

- keep Left Above Rim, Right Above Rim and Broadcast as the active calibrated camera set;
- do not reopen camera selection;
- do not switch to Right Slash, Left Slash, handheld, In Arena or another feed;
- do not move physical camera centres to fit player poses;
- widen/refine native temporal search, improve identity tracking/visibility handling, or diagnose source-state mismatch inside the same three views;
- expose the failure and preserve diagnostics.

The camera lock may change only after an explicit user instruction to change cameras.
