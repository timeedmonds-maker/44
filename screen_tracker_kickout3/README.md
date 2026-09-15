# Screen Tracker - Kickout 3

This is a derivative application family built **from** Screen Tracker 1.1 without modifying the canonical Screen Tracker runtime.

## Base lock

- Canonical branch: `screen-tracker`
- Canonical Screen Tracker version: `1.1.0`
- Branch point / protected source SHA: `15cde6374333c1956299c2355183549298a466db`
- This derivative may import/reuse the canonical visual/tracking primitives, but must not write to or change `screen_tracker/`, its workflows, or the private canonical source.

## Kickout 3 event definition

For the initial 2025-26 proof:

1. Steven Adams records an individual offensive rebound.
2. Houston's **first subsequent field-goal attempt** is a Kevin Durant 3PA.
3. The Durant 3PA is made.
4. Shot occurs no more than 3.0 seconds after the Adams OREB.
5. Initial examples prefer plays where the official play-by-play credits Adams with the assist.
6. Two examples must come from different games.

## Presentation contract

Each selected play is rendered from two validated distinct official NBA camera angles and the four full-length segments are concatenated.

Only four players receive name bars + perspective-normalized floor rings throughout the clip:

- Steven Adams
- Adams's defender
- Kevin Durant
- Durant's defender

At the exact frame Durant first controls the Adams kickout:

- freeze the real source frame;
- draw a high-visibility white dotted line from Durant to the closest defender;
- label the measured defender separation in feet;
- show official shot distance above Durant;
- show exact-event official NBA xFG in the upper-left;
- show Durant player-conditioned xFG for similar 2025-26 shots in a second upper-left tile, including sample size.

The frozen defender is the opponent closest to Durant at the catch; that same identified player remains Durant's defender track for the whole clip. Adams's defender is resolved from the opponent guarding/contesting Adams at the rebound and remains identity-stable for the whole clip.

## Similar-shot definition

Reuse the validated `tools/player_conditioned_xfg_v2.py` definition:

- same player (Kevin Durant), 2025-26 regular season;
- official NBA `SHOT_DISTANCE` within ±1.0 ft of the target;
- official league xFG within ±2.5 percentage points of the target;
- target shot excluded;
- player-conditioned xFG is the empirical FG% of that comparison sample;
- show `n`.

## Video/data policy

Use the project-locked production route:

exact event -> `clips.nba.com` -> fresh signed `lrmedia.nba.com` HLS -> semantic QA -> deterministic rendering.

No generative frames, generative image edits, or AI super-resolution.
