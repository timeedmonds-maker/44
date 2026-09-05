# Jazz v15 visual failure and v16 correction

## v15 is rejected

The v15 workflow completed successfully numerically, but user visual QA invalidated it on 2026-09-02 for two independent reasons:

1. **Wrong freeze state** — the output shows Steven Adams already hanging on the rim after the dunk. The v15 `9.635413333166447` Left Above Rim reference time must not be reused as an accepted apex.
2. **Wrong camera motion** — the MP4 reads as a zoom/push rather than a spatial orbit.

Per project policy, visual QA is authoritative when numerical QA is underconstrained. A green workflow is not a successful free-view proof.

## Root cause: camera motion

`build_portable_moge_pnp_freeview_v12.py` used `slerp_vector()` for the camera centre. That function interpolated the radius from the action pivot:

`r=(1-alpha)*ra+alpha*rb`

so the virtual camera contained a radial dolly component while orientation was separately interpolated. This can present as zoom rather than rotation.

## v16 orbit gate

`build_portable_moge_true_orbit_v16.py` replaces that with a rigid camera rotation about the 3D action pivot:

- reference camera centre is rotated around the pivot at constant radius;
- world-to-camera rotation is the inverse of that same rigid rotation (`R = Q.T`);
- reference intrinsics/focal length are held fixed;
- the action pivot preserves its reference camera coordinates;
- radius drift must be <= `1e-6`;
- pivot projection drift must be <= `0.05 px`;
- no focal interpolation or radial camera-distance interpolation is permitted.

## v16 freeze gate

`select_jazz_predunk_ball_apex_v4.py` treats the v15 state as explicitly forbidden. The new freeze must be:

- a semantically anchored basketball trajectory;
- before the rejected hanging state;
- at the highest verified ball position before the first observed rim-plane approach/crossing;
- supported independently by Right Slash and Right HandHeld height-sensitive views;
- within 70 ms cross-camera consensus.

Do not weaken these gates merely to obtain a render.
