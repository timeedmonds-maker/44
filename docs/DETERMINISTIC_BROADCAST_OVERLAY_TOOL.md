# Deterministic NBA Broadcast Overlay Tool

## Status

Durable production pattern established from the HOU @ LAL Christmas Day 2025 Kevin Durant / Steven Adams screen clip.

Execution/heavy runners remain in `timeedmonds-maker/44`. Durable methodology and reusable presentation logic should be mirrored into `long-rebound-era`; do not use `long-rebound-era` as the heavy Actions runner repository.

## Deterministic-only rule

The overlay pipeline operates on official NBA source video and deterministic computer-vision/data outputs. It must not use AI image generation, generative fill, synthesized frames, invented identities or invented measurements.

## Locked broadcast presentation defaults

Analytics information panels use the reusable helper `tools/broadcast_overlay_style.py`.

Default panel treatment:

- neutral semi-transparent grey fill;
- neutral grey border;
- no coloured accent bars/bands/rules;
- white/grey broadcast typography;
- clipped-corner geometry where appropriate;
- stable on-screen placement.

Team-colour player rings/name plates are separate from analytics information panels and are not affected by the neutral-panel rule.

## Player rings and labels

- Centre the projected ring on the player's foot/base position.
- Use the same ring geometry for comparable players in the frame.
- Never use a solid black rear arc.
- Occlude the rear portion behind legs/feet where geometry supports it.
- Temporally smooth ring position to avoid jitter.
- Keep surname labels compact, stable and centred above the relevant player.
- Label only players involved in the play when that is the requested use-case.

## Reusable analytics blocks

The durable pattern supports league xFG, player-conditioned xFG, screen coverage, screen-contact time, and a release freeze with dotted defender-distance annotation.

## Player-conditioned xFG definition

Canonical entry point in repo 44: `tools/player_conditioned_xfg_authoritative.py`.

For any player and exact shot event:

1. Resolve exact focal shot by `game_id + event_num + player_id`.
2. League xFG comes from official `shotqualityvideologs` `shotQuality`.
3. Shot distance comes from official shot-level `shotchartdetail.SHOT_DISTANCE` and is the source of truth.
4. Do not derive comparison distance from `locX/locY` when `SHOT_DISTANCE` is available.
5. Comparison pool is the same player's other tracked regular-season shots within `±1.0 ft` of focal `SHOT_DISTANCE` and `±2.5 percentage points` of focal league xFG.
6. Exclude the focal shot from its own comparison set.
7. Player-conditioned xFG is the empirical make rate of that comparison pool.
8. Preserve comparison rows and QA output for auditability.

## Current validated example

For Kevin Durant, HOU @ LAL, 2025-12-25, event 94:

- league xFG: 42.6%;
- official shot distance: 17 ft;
- KD comparable pool: 45 FGA, 23 FGM;
- KD xFG: 51.1%;
- distance join coverage across KD tracked xFG shots: 100% (1,281 / 1,281).

## Christmas clip validated overlay semantics

- Clip begins before screen development; player graphics are live from the start.
- Screen coverage: Deep Drop.
- Screen-contact timer begins at visually validated Adams–LaRavia contact and stops at Durant release for this clip.
- xFG block is visible from one second before release through two seconds after release.
- Release freeze keeps the Durant-to-LaRavia dotted line and validated distance annotation.
- Final presentation output is deterministic UHD presentation upscale, not native 4K.

## Reuse contract

Future clips should supply clip-specific configuration (event, players, coverage label, timing anchors and defender target) while reusing the common data definition and common neutral analytics-panel style.
