# Screen Tracker

`Screen Tracker` is the public application runner for the durable screen-analysis system.

Canonical ownership is split deliberately:

- **Private canonical source / contracts / registry:** `timeedmonds-maker/long-rebound-era` (`screen_tracker/`).
- **Public per-play applications and GitHub Actions execution:** `timeedmonds-maker/44`, branch `screen-tracker`.

The public repository must never require private repository contents at runtime. A new play is represented by a self-contained **application manifest** plus its event-specific role/config manifests. The generic runner stays stable.

## Screen Tracker 1.1 default

Every new Screen Tracker application is a **two-angle delivery**.

- Primary camera: **Broadcast** when available.
- Secondary camera: **High Tight** when available; otherwise the highest-priority distinct official NBA angle from `screen_tracker/angle_policy.py`.
- The same basketball analysis and role identity contract applies to both angles, with camera-specific tracking and label placement.
- Floor rings use the universal perspective-normalized policy: Broadcast is the minimum visual ring size and tighter angles scale up from focal-player/camera scale, then hold one constant ring size for the entire angle.
- Each angle is encoded independently to the final UHD/streamable profile and only then concat-copied into the combined delivery.
- The combined video must equal the sum of the two angle durations within the QA tolerance. A shortened or missing second angle fails the run.
- Never duplicate Broadcast as a fake second angle. If a distinct official second angle is unavailable or cannot pass role/identity QA, fail rather than fabricate it.

Version `1.0.0` reference applications remain reproducible only as explicit `legacy_single_angle` applications. Version `1.1.0` defaults to `two_angle_default`.

## What Screen Tracker does

For a specified NBA screen play, the application must deterministically resolve and render:

1. exact game/event identity;
2. two distinct official NBA event-video angles;
3. exact PBP lineup state;
4. ballhandler/shooter, screener, point-of-attack defender, and coverage/screener defender in each angle;
5. identity-stable tracking through occlusion / tracker ID swaps;
6. screen-contact timing from validated source frames;
7. event shot xFG and player-season matched-shot xFG when applicable;
8. team-colour selection, including automatic primary-colour collision handling;
9. perspective-normalized broadcast-style rings, name bars and data panels;
10. deterministic native + UHD + streamable 4K output and QA package;
11. full-length two-angle assembly QA.

No generated imagery. No AI super-resolution. No synthetic player/ball/court frames.

## Screen Tracker Search

Game-level discovery uses **Screen Tracker Search 1.0.1** through:

`.github/workflows/screen-tracker-search.yml`

The required search method is:

1. input one exact `game_id`, screener player ID and target player ID;
2. query the 2025-26 possession table and retain **every offensive possession whose `lineup_team` contains both players**;
3. join exact PBP events to each possession using exact game, period and possession start/end clock window;
4. retain every exact PBP event number in every retained possession;
5. resolve every event independently through `clips.nba.com` to fresh signed `lrmedia.nba.com` HLS;
6. scan every resolvable event video using deterministic frame sampling and temporal screen geometry;
7. aggregate event evidence to possession-level candidates;
8. rank by visual screen evidence, with exact-PBP target/screener actor signals used only as ranking features;
9. fetch native official preview clips and dense contact sheets for the top candidates;
10. pass the selected candidate through the full Screen Tracker role/identity QA before rendering.

There is deliberately **no pre-filter for target scoring, shot type, assist credit or manually guessed screen events**. Those can improve rank but cannot define the scan universe.

Public implementation:

- `screen_tracker/search_game.py` — authoritative shared-possession/PBP event join + candidate ranking.
- `screen_tracker/search_previews.py` — official native top-candidate previews/contact sheets.
- `screen_tracker/angle_policy.py` — universal two-angle selection policy.
- `screen_tracker/combine_two_angles.py` — full-length two-angle assembler and duration QA.
- the existing deterministic temporal screen detector is reused internally as the visual scan engine.

Default search pair is Steven Adams (`203500`) as screener and Amen Thompson (`1641708`) as target, but the workflow accepts any pair.

## Application model

New `1.1.0` applications live under `screen_tracker/applications/` and declare exactly two entries in `angles`. Each angle points to:

- its validated source/tracking artifact;
- its camera label;
- an event-specific render config;
- an exact role manifest;
- the backend engine to use.

Both angles must represent the same exact game/event and pass the same role/lineup contract. The first locked reference, HOU @ TOR game `0022500131` event `483`, predates 1.1 and is retained as an explicit legacy single-angle reproducibility reference.

## Universal invariants

- Exact PBP lineup gates player identity.
- The coverage defender is resolved by **basketball role** (screener defender), never by reputation/height/nominal position.
- Out-of-lineup identities fail QA.
- Identity stitching is deterministic and role-stable.
- Name bar and floor ring use the same identity-stable track.
- Floor rings are perspective-normalized by camera angle and never smaller than the canonical Broadcast minimum.
- If team primary colours are visually similar, the defensive team keeps primary and the offensive team switches to its declared secondary colour. If that does not create adequate separation, fail QA.
- Contact timing uses validated source-frame boundaries when available.
- Event xFG and matched-shot xFG are separate statistics.
- Two-angle output requires two distinct official camera labels and full duration from each angle.
- Final-profile angle files are assembled by concat-copy; a combined master is never re-encoded across discontinuous source timestamps.
- The canonical matched-shot rule for the validated reference is distance ±1.5 ft AND xFG difficulty ±2.0 percentage points, with sample size displayed.

## Running an application

Use `.github/workflows/screen-tracker.yml` and provide the application JSON path.

For Screen Tracker 1.1 the workflow downloads both application-declared source/tracking artifacts, renders and validates each camera independently, performs the two-angle duration gate, and uploads the combined UHD, combined streamable 4K file, QA and package. A run cannot pass if angle 2 is shortened.

## Adding a new play

Use Screen Tracker Search first when a game rather than an exact event is supplied. Once a candidate is selected, do **not** fork the renderer. Prepare both official camera-angle tracking artifacts using the default selection policy, add one `1.1.0` two-angle application manifest with angle-specific config/role manifests, then run the same workflow. If a genuinely new screen/coverage behavior requires engine work, update the generic engine first and version the contract.
