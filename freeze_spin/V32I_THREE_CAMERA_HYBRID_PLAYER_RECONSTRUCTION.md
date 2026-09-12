# v32i — three-camera hybrid player reconstruction

## Decision

The production solve remains locked to the three accepted calibrated physical cameras for event `0022500301 / 489`:

1. Left Above Rim
2. Broadcast
3. Right Above Rim

No fourth camera is required by v32i. Native output remains `960x540` unless explicitly requested otherwise.

v32i changes the **foreground representation**, not the solved NBA camera geometry. The previous ray-volume and capsule/surfel foreground experiments are rejected as final representations because they did not preserve exact player pose, silhouette, identity or anatomy.

## External ideas accepted only where they improve the existing architecture

### 1. Local sub-frame synchronization residual — ACCEPT

Use the existing ±6-frame real bursts and calibrated epipolar geometry to refine a small temporal offset per camera around the current v32 freeze. The objective may borrow the useful idea from VisualSync: correctly synchronized moving points should satisfy cross-view epipolar constraints.

This is an **additional residual**, not a replacement synchronization pipeline. The project already has stronger event-specific anchors: ball, hands, rim relationship, shoulders, hips and contact state.

Engineering target: minimize exact-state disagreement to substantially below one 30-fps frame. Do not declare a universal 0.25-frame guarantee unless measured on this event.

### 2. Temporal player masks — ACCEPT

Use a video segmentation model such as SAM 2 only as an observation generator over the 13-frame bursts. Its output is a per-pixel mask plus confidence, not geometry and not truth.

Masks must remain editable/rejectable. Ambiguous occluded pixels receive low/zero weight rather than fabricated ownership.

### 3. Single-view human mesh recovery — ACCEPT AS INITIALIZER ONLY

A model such as SAM 3D Body may provide per-camera articulated mesh, hand/foot/body joint and silhouette proposals.

Its monocular 3D prediction must **never** become final world geometry directly. Per-view proposals are transformed into observations and jointly reconciled in the calibrated NBA world frame.

Learned appearance/texture from the human model is discarded.

### 4. Mesh-aware cross-view association / triangulation — ACCEPT

Borrow the useful MAEM concepts:

- dense mesh/keypoint projections for cross-view association;
- epipolar filtering;
- robust per-joint triangulation;
- RANSAC/outlier rejection;
- one shared identity across the three views.

The project has only a small number of relevant paint-area players, so known event context and existing v32 player masks can constrain association further.

### 5. Shared articulated multi-view mesh solve — ACCEPT; THIS IS THE CORE v32i CHANGE

For each focal/interacting player, optimize **one shared articulated human mesh** against all three cameras at the exact frozen state.

Suggested objective:

`L = wkpt*L_keypoint + wsil*L_silhouette + wepi*L_epipolar + wfloor*L_floor + wbone*L_anatomy + wtemp*L_temporal + wocc*L_occlusion + wprior*L_mesh_prior`

where:

- `L_keypoint`: robust 2D reprojection error across visible joints;
- `L_silhouette`: differentiable silhouette boundary / distance-transform disagreement;
- `L_epipolar`: cross-view consistency for matched dynamic landmarks;
- `L_floor`: physically valid foot/floor contact when applicable;
- `L_anatomy`: stable limb lengths and joint-limit constraints;
- `L_temporal`: support from the 13-frame burst without changing the chosen freeze state;
- `L_occlusion`: correct depth ordering between overlapping players, rim, backboard and ball;
- `L_mesh_prior`: regularization only, never allowed to override source observations.

The optimizer must expose per-camera residuals and fail closed if one player's pose is unsupported.

### 6. Analytical ball / rim / backboard / court — RETAIN

Keep the existing metric NBA world model. The ball is one metric sphere with a triangulated center. Rim/backboard/court are analytic static geometry. Do not ask a learned human or Gaussian model to solve these.

### 7. Real-pixel view-dependent texturing — ACCEPT

After geometry is solved, project the original synchronized source images onto the player mesh.

For every visible surface element:

1. z-buffer against player/player, player/basket and self-occlusion;
2. score source cameras by visibility, projected resolution/sharpness, incidence angle and mask confidence;
3. use the strongest source as the dominant texture;
4. blend a second source only where the two observations are geometrically and photometrically consistent;
5. leave unsupported surfaces unfilled/black rather than hallucinating them.

Do **not** bake a single averaged texture across all cameras. Do **not** use generated/inpainted jersey, face, hand or shoe pixels.

### 8. Gaussian representation — RETAIN ONLY AS A RESIDUAL DETAIL LAYER

Generic sparse-view 3D Gaussian optimization is no longer responsible for primary player anatomy.

Optional Gaussian/surfel residuals may model only source-supported detail such as:

- hair boundary;
- loose jersey/shorts edges;
- net strands;
- tiny geometry gaps/disocclusions;
- background detail not captured by analytic/static geometry.

Residual support must be tied to the explicit solved geometry and source visibility. Free-floating body Gaussians, anatomy-changing splats and generative priors are forbidden.

## Ideas explicitly rejected

- adding a fourth camera to v32i;
- using SAM 3D Body's monocular mesh as final geometry;
- using SAM 2 masks as geometry;
- making FSGS/SparseGS/generic 3DGS the primary human solver;
- unrestricted 360-degree synthesis;
- a promised 50–80 degree orbit before this event proves it;
- diffusion/SDS/generative completion for unseen player surfaces;
- manually painting missing player pixels;
- moving-player pixels driving camera calibration;
- returning to capsule, billboard or ray-cloud bodies as the production foreground.

## Camera calibration policy

The existing accepted v32e three-camera calibration is the prior. Only a tightly bounded **micro-refinement** is permitted, using static court/rim/backboard landmarks only.

Player observations cannot move the cameras to make the body fit.

Any calibration refinement must preserve:

- full rim/board/court visual QA;
- physically valid camera pose;
- positive scene depth;
- one common metric world frame.

## Synchronization policy

Current v32 frame choices remain the initial state. Optimize only a small local temporal offset around them using the real ±6 frame burst.

Dynamic features may include:

- ball center/trajectory;
- Adams/Cissoko hands, wrists, elbows, shoulders, hips;
- silhouette boundary motion;
- rim/contact relationship.

Audio remains coarse timing only.

## Validation protocol

### Geometry leave-one-view-out tests

Run three diagnostics where possible:

- solve from LAR + Broadcast observations, render/project into RAR;
- solve from Broadcast + RAR observations, render/project into LAR;
- solve from RAR + LAR observations, render/project into Broadcast.

These diagnose whether a body is genuinely constrained rather than simply fitting all observations at once. They are diagnostics, not the final production fit.

### Appearance holdout

For a target camera, withhold that camera's RGB texture, texture only from the other two cameras, and render the target pose. Compare against the real target RGB using masks plus direct visual inspection.

### Production fit

Only after diagnostics pass, jointly solve geometry from all three calibrated cameras and texture from all three real source images for the restricted `0° -> 25°` orbit.

## Pass/fail gates

A frame passes only when:

- Adams is one coherent, correctly positioned articulated body;
- Cissoko/interacting player is one coherent, correctly positioned articulated body;
- silhouettes align in all three real source views;
- hands/arms at the rim agree with source state;
- feet/body contacts are physically plausible;
- one stable ball;
- rim/backboard remain fixed;
- jersey/skin/shoe appearance comes from real source pixels;
- no doubled limbs, stretched anatomy, floaters, blobs or player drift;
- unsupported surfaces are honest holes rather than inventions.

Only then render the native `960x540` virtual orbit.

## Integration order

1. freeze current v32e camera solution;
2. refine sub-frame temporal offsets from all three 13-frame bursts;
3. produce temporal player masks/confidences;
4. produce per-view articulated mesh/keypoint proposals;
5. associate identities across the three cameras;
6. robustly triangulate joints;
7. jointly optimize one shared mesh per player;
8. validate leave-one-view-out geometry;
9. attach source RGB with view-dependent visibility weighting;
10. optionally add source-supported Gaussian residual detail;
11. run appearance holdout QA;
12. render only the restricted native 0–25 degree arc.

## Non-negotiable provenance

Every generated artifact must record which source camera/frame contributed to:

- each keypoint/mesh observation;
- each mask;
- each texture region;
- each temporal residual;
- each residual Gaussian/surfel, if used.

No stage may silently substitute generated RGB or learned appearance for missing official NBA pixels.
