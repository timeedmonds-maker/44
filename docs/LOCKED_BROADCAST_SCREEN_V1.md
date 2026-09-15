# LOCKED_BROADCAST_SCREEN_V1

This is the durable production renderer for standard deterministic NBA screen-analysis videos.

## Source-of-truth lineage

The visual baseline is the recovered Christmas Day HOU @ LAL event 94 V7 build:

- repository: `timeedmonds-maker/44`
- historical branch: `codex/adams-screen-okc-opener`
- successful workflow run: `34826047210`
- recovered commit: `40d0ed3974ecbd0fcd433670cddd462c26f9e397`
- workflow: `.github/workflows/adams-lal-christmas-broadcast-v7.yml`
- renderer: `tools/christmas_event94_broadcast_v7.py`
- artifact: `adams-lal-christmas-broadcast-v7-authoritative-distance-grey-panels`

Do not substitute `adams_screen_prod_v3.py`, the Toronto experimental V8 renderer, V9, free-view, or any other later experiment when the user asks for the locked standard screen-analysis look.

## Locked presentation

- full-frame real NBA broadcast footage
- deterministic OpenCV/Pillow overlays only
- no AI image generation
- no generative video
- no AI super-resolution
- neutral translucent grey analytics panels (`neutral_translucent_grey_v1`)
- no coloured accent bars on analytics panels
- wide flat open horseshoe rings using the recovered V7 geometry
- ring colour is the exact primary RGB supplied for that player's team
- compact team-coloured nameplates
- high-contrast white dotted shooter-to-primary-defender release line
- release freeze
- shot-distance release tag parsed from the event shot-description text (for example `17'` -> `17 FT`); do not infer shot distance from image geometry
- deterministic UHD presentation profile: `hqdn3d=0.6:0.6:2.0:2.0 -> Lanczos 3840x2160 -> CAS 0.22 -> 30fps`, H.264 High, CRF 16, maxrate 36M, AAC 192k when audio is present

## Version rule

The durable tool identifier is **`LOCKED_BROADCAST_SCREEN_V1`**.

Future changes must receive a new durable tool identifier. Do not silently change the visual treatment under this identifier. Event-specific configs may change player IDs, labels, timings, team RGB values, shot description and supported analytics, but not the locked presentation primitives.

## Team RGB rule

The config must contain explicit RGB triplets. Do not use approximate colours. If both teams share the same primary RGB, the rings are intentionally the same colour rather than inventing a substitute colour.

## Data-integrity rule

Only display metrics supported by the event data/config. Missing xFG, defender distance or coverage classification must be omitted rather than estimated. The shot-distance release tag is taken from the supplied shot-description field and is recorded in `qa.json` with its source.
