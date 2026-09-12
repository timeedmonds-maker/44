# Reed Sheppard — Steven Adams impact, 2025-26 regular season

## Scope and provenance

- Subject: Reed Sheppard (1642263), Houston Rockets.
- Teammate: Steven Adams (203500).
- Main-data native WOWY: PBP Stats `get-on-off/nba/player`, the same source/semantics used by the private 104 pair builder.
- Exact lineup layer: `ramirobentes/nba_pbp_data`, `lineup-final2026/data.csv`.
- Shot quality: official NBA/G League `shotqualityvideologs`, joined by exact game/event to the 2025-26 PBP lineups.
- Long-rebound possession mechanism: validated missed-three possession continuation output in `timeedmonds-maker/long-rebound-era`.
- All findings are descriptive associations, not causal estimates.

## Native player WOWY (same source semantics as 104 pair layer)

Shared exposure from the native endpoint: Adams ON 416 minutes; Adams OFF 1,687 minutes.

| Metric | Adams ON | Adams OFF | ON − OFF |
|---|---:|---:|---:|
| eFG% | 54.5% | 55.4% | -0.9 pp |
| TS% | 56.5% | 56.7% | -0.2 pp |
| 2P% | 48.7% | 48.6% | +0.2 pp (rounded) |
| 3P% | 39.4% | 39.7% | -0.3 pp |
| Points / 100 possessions | 24.793 | 25.476 | -0.683 |
| Usage | 19.942 | 21.729 | -1.788 |
| Turnovers / 100 possessions | 2.361 | 2.990 | -0.629 |
| At-rim accuracy | 76.2% | 51.4% | +24.8 pp |
| At-rim frequency | 11.7% | 10.0% | +1.7 pp |
| 3PA share | 55.3% | 62.3% | -7.0 pp |
| 2PA blocked rate | 6.2% | 9.0% | -2.7 pp |
| Shooting-fouls-drawn rate | 4.9% | 2.9% | +2.0 pp |
| 2PT shooting-fouls-drawn rate | 9.4% | 5.8% | +3.6 pp |
| Native ShotQualityAvg | 51.8% | 52.1% | -0.3 pp |

The native split therefore does **not** show higher realized overall shooting or scoring with Adams. Its positive player-level signals are shot location/rim outcome, turnover suppression, fewer blocked twos, and more shooting fouls drawn.

## Repo 44: official shot-level xFG joined to exact lineups

QA: 946 Reed FGA in the authoritative PBP; 179 Adams-ON FGA and 767 Adams-OFF FGA. Official shot-level xFG was available for 890/946 FGA (94.1%).

| Metric | Adams ON | Adams OFF | Raw ON − OFF | Exact-other-three-teammate effect |
|---|---:|---:|---:|---:|
| xFG make probability | 42.506% | 40.021% | +2.485 pp | +2.087 pp |
| Expected eFG% | 52.500% | 51.333% | +1.167 pp | +0.874 pp |
| Expected points / FGA | 1.0500 | 1.0267 | +0.02335 | +0.01747 |
| Actual eFG% | 54.469% | 55.150% | -0.681 pp | -2.472 pp |
| eFG above expected | +3.932 pp | +5.691 pp | -1.758 pp | -3.140 pp |
| 3PA share | 55.307% | 62.190% | -6.883 pp | -6.636 pp |
| Rim FGA share | 15.642% | 12.386% | +3.257 pp | +2.474 pp |
| Average shot distance | 18.795 ft | 20.115 ft | -1.319 ft | -1.286 ft |

The exact-other-three-teammate model holds Reed's other three teammates fixed and compares Adams as the fifth player against alternate fifth-player minutes. The shot-level fixed-effect sample covers 38-39 matched teammate trios depending on xFG availability.

By shot type, the shot-quality effect is concentrated inside the arc:

| Shot type | Adams ON FG% | Adams OFF FG% | Adams ON xFG | Adams OFF xFG |
|---|---:|---:|---:|---:|
| 2PT | 48.75% (80 FGA) | 48.62% (290 FGA) | 52.745% | 47.927% |
| 3PT | 39.394% (99 FGA) | 39.413% (477 FGA) | 34.879% | 35.516% |

Thus Adams improved Reed's **2PT expected quality by +4.82 percentage points of xFG**, while actual 2P% was essentially unchanged. His 3P% was also essentially unchanged.

## Reed-lineup performance with Adams

Exact Houston lineup-stint aggregation (Reed on court):

| Team result while Reed is on court | Adams ON | Adams OFF | Raw ON − OFF |
|---|---:|---:|---:|
| Minutes | 415.835 | 1,731.088 | — |
| Offensive Rating | 120.000 | 116.883 | +3.117 |
| Defensive Rating | 106.100 | 115.249 | -9.149 |
| Net Rating | +13.900 | +1.634 | +12.266 |
| Pace / 48 | 97.596 | 97.201 | +0.395 |

Exact-other-three-teammate possession-balanced comparison:

- 47 matched teammate trios.
- 835 balanced-weight possessions.
- Offensive Rating effect: **+4.036**.
- Defensive Rating effect: **-9.847**.
- Net Rating effect: **+13.883**.

This is a much stronger lineup-level effect than Reed's individual scoring split, and the raw result is not explained by pace.

## Long Rebound Era: missed-three possession value

For Houston 2025-26, the validated possession output shows that when Adams was on the floor, possessions containing a missed three retained materially more value team-wide.

All possessions containing a missed three:
- PPP: 0.4901 Adams ON vs 0.4138 OFF = **+0.0763**.
- Scored rate: 22.925% vs 19.332% = **+3.593 pp**.
- Multi-FGA rate: 37.154% vs 31.755% = **+5.399 pp**.

Cleaner continuation test: possession's first FGA was a missed three:
- PPP: 0.5148 Adams ON vs 0.4142 OFF = **+0.1005**.
- Scored rate: 24.051% vs 19.414% = **+4.636 pp**.
- Multi-FGA rate: 32.911% vs 26.444% = **+6.468 pp**.

This is a **Houston team-wide mechanism**, not a Reed-miss-only split. It is relevant to Reed because a majority of his FGA were threes in both Adams states, but it should not be presented as a Reed-specific recovery rate.

## Bottom line

The evidence supports the claim that Adams improved the *environment and process* around Reed more strongly than Reed's realized box-score efficiency. With Adams, Reed took closer and easier shots, especially better 2PT opportunities, reached the rim more often, turned the ball over less, and played in dramatically better-performing lineups. Reed did not convert those advantages into a higher overall eFG% or TS% in the observed 2025-26 sample.
