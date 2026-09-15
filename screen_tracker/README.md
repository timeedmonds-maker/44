# Screen Tracker

`Screen Tracker` is the public application runner for the durable screen-analysis system.

Canonical ownership is split deliberately:

- **Private canonical source / contracts / registry:** `timeedmonds-maker/long-rebound-era` (`screen_tracker/`).
- **Public per-play applications and GitHub Actions execution:** `timeedmonds-maker/44`, branch `screen-tracker`.

The public repository must never require private repository contents at runtime. A new play is represented by a self-contained **application manifest** plus its event-specific role/config manifests. The generic runner stays stable.

## What Screen Tracker does

For a specified NBA screen play, the application must deterministically resolve and render:

1. exact game/event identity;
2. official broadcast event video;
3. exact PBP lineup state;
4. ballhandler/shooter, screener, point-of-attack defender, and coverage/screener defender;
5. identity-stable tracking through occlusion / tracker ID swaps;
6. screen-contact timing from validated source frames;
7. event shot xFG and player-season matched-shot xFG when applicable;
8. team-colour selection, including automatic primary-colour collision handling;
9. broadcast-style rings, name bars and data panels;
10. deterministic native + UHD + streamable 4K output and QA package.

No generated imagery. No AI super-resolution. No synthetic player/ball/court frames.

## Application model

Every play lives under `screen_tracker/applications/` and points to:

- the validated source/tracking artifact;
- an event-specific render config;
- an exact role manifest;
- the backend engine to use;
- output naming and QA requirements.

The first locked reference is HOU @ TOR, game `0022500131`, event `483`.

## Universal invariants

- Exact PBP lineup gates player identity.
- The coverage defender is resolved by **basketball role** (screener defender), never by reputation/height/nominal position.
- Out-of-lineup identities fail QA.
- Identity stitching is deterministic and role-stable.
- Name bar and floor ring use the same identity-stable track.
- If team primary colours are visually similar, the defensive team keeps primary and the offensive team switches to its declared secondary colour. If that does not create adequate separation, fail QA.
- Contact timing uses validated source-frame boundaries when available.
- Event xFG and matched-shot xFG are separate statistics.
- The canonical matched-shot rule for the validated reference is distance ±1.5 ft AND xFG difficulty ±2.0 percentage points, with sample size displayed.

## Running an application

Use `.github/workflows/screen-tracker.yml` and provide the application JSON path, for example:

`screen_tracker/applications/0022500131_483.json`

The workflow downloads the application-declared source artifact, runs `screen_tracker/run_application.py`, validates the backend QA, creates a streamable 4K copy, and uploads the complete package.

## Adding a new play

Do **not** fork the renderer. Add a new application manifest and play-specific config/role manifest, then run the same workflow. If a genuinely new screen/coverage behavior requires engine work, update the generic engine first and version the contract.
