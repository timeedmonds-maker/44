# Multi-angle freeze + spin replay: feasibility and prototype result

Research date: 2026-09-01  
Test event: HOU @ POR, 2026-01-07, game `0022500527`, event `18`

## Verdict

There is no mature open-source program that can take ordinary NBA broadcast-angle clips and automatically produce a broadcast-quality free-viewpoint replay.

The strongest practical architecture is two-tiered:

1. ship a deterministic Level-C replay now: automatically retrieve all angles, audio-synchronize them, visually refine the impact frame, lock the action, and travel rapidly through the real views with deterministic perspective/blur transitions;
2. run a separate Level-A research track using 4C4D/InstantSplat-style reconstruction after per-frame basketball-court camera calibration.

Confidence: high for the Level-C conclusion; moderate that a useful Level-A freeze-frame reconstruction can be achieved after calibration; low that the current broadcast feeds can yield Intel/Canon-grade continuous free-viewpoint video.

## What the actual Sidy Cissoko material shows

- 12 validated official angles, each 960×540 at approximately 29.92–29.97 fps and approximately 20.39 seconds.
- All files have 48 kHz stereo AAC audio.
- The same nominal timestamp is not the same basketball instant across feeds.
- Manually selected block-contact times span 8.83–9.80 seconds: a 0.97-second range.
- FFT audio cross-correlation against Broadcast estimated offsets from −0.5653 to +0.3520 seconds. Ten of the eleven non-reference feeds produced moderate/high correlation confidence.
- Sparse SIFT matching between adjacent orbit views produced only 4–11 RANSAC homography inliers. That is inadequate for a trustworthy full-frame view morph; zero-ish reprojection errors on four-point fits are overfitting, not evidence of valid geometry.

These results are preserved in `audio_offsets.json` and `view_overlap.json` when the workflow runs.

## Why commercial replay looks better

Intel's NBA installation used 38 fixed 5K cameras positioned around each arena. Canon describes its Free Viewpoint Video System as using more than 100 synchronized 4K cameras. Those are calibrated capture arrays with deliberate overlap—not a collection of operated broadcast cameras with changing zoom, crops, exposure, motion and occlusion.

Primary references:

- NBA/Intel: https://www.nba.com/news/ap-teams-enhancing-fan-experience-high-tech-replays
- Canon FVS: https://www.usa.canon.com/newsroom/2024/20240104-ces

The current 12–13 NBA angles therefore cannot substitute for a fixed volumetric array. The count is not the decisive variable; geometric overlap, synchronization, calibration, exposure consistency and the visibility of every surface are.

## Candidate assessment

| Candidate | Normal input contract | License | Fit to these clips |
|---|---|---:|---|
| 4C4D | Four synchronized video cameras, valid camera parameters and point-cloud initialization; its custom-data path expects COLMAP-format intrinsics/extrinsics and recommends MASt3R because sparse-view COLMAP is weak | MIT for the main repo; dependencies/weights require separate review | Closest open-source research candidate, but not plug-and-play. It assumes coherent multi-camera sequences and trains to a 30,000-step checkpoint. Runtime/hardware are not published in the README. |
| InstantSplat | Sparse, unposed images of a static scene; joint scene/pose optimization initialized from a learned geometry prior | Apache-2.0 for the repo; dependency/model terms require review | Best one-instant sparse-view experiment. Risk: dynamic inconsistencies, low overlap and learned priors can create unsupported geometry. |
| Nerfstudio Splatfacto / gsplat | Static images plus camera poses, normally produced by COLMAP | Apache-2.0 | Mature and reproducible after calibration, but ordinary COLMAP is unlikely to register this sparse wide-baseline set reliably. |
| OpenSplat | Static images with valid poses/sparse points; CPU/GPU implementation | AGPL-3.0 | Useful portable trainer after calibration. CPU fallback exists, but camera reconstruction—not rasterization—is the blocker. |
| 4DGaussians / EasyVolcap | Calibrated multi-view video organized by camera and time, normally synchronized | Apache-2.0 for 4DGaussians; EasyVolcap terms must be checked | Strong controlled-capture tools, poor direct fit to operated broadcast feeds. |
| EasyMocap multi-person NVS | Calibrated camera array; its published sparse-human example uses eight GoPros and recommends four RTX 3090 GPUs | Research/non-profit terms; commercial use requires permission | Useful for pose/foreground priors, not a complete photorealistic NBA replay system. Its own documentation lists moving cameras and general backgrounds as limitations. |
| RIFE / FILM | Temporally adjacent or near-duplicate images of the same scene/view | MIT / Apache-2.0 | Good for slow motion inside one angle. Not camera-view interpolation; wide-baseline use produces warping/occlusion artefacts. |

Primary references:

- 4C4D: https://github.com/yangzf-1023/4C4D
- InstantSplat: https://research.nvidia.com/labs/avg/publication/fan.wen.etal.cvprnri2024/
- Nerfstudio custom data: https://docs.nerf.studio/quickstart/custom_dataset.html
- gsplat: https://github.com/nerfstudio-project/gsplat
- OpenSplat: https://github.com/WebODM/OpenSplat
- 4DGaussians: https://github.com/hustvl/4DGaussians
- EasyVolcap: https://github.com/zju3dv/EasyVolcap
- EasyMocap sparse multi-view NVS: https://chingswy.github.io/easymocap-public-doc/works/multinb.html
- RIFE: https://github.com/hzwer/ECCV2022-RIFE
- FILM: https://github.com/google-research/frame-interpolation

## Camera calibration

Known NBA court geometry materially helps, but only up to a point.

- Court lines provide a strong floor-plane homography.
- Four or more 2D/3D correspondences can initialize a planar pose; non-planar basket/backboard points help resolve full pose and focal length.
- Because broadcast cameras pan, tilt and zoom, intrinsics/extrinsics must be estimated per freeze frame or tracked continuously—not once per named angle.
- Floor-plane calibration does not reconstruct airborne players, the ball, hidden limbs or surfaces facing away from every camera.

Useful primary implementations:

- Basketball camera-calibration challenge: https://github.com/DeepSportradar/camera-calibration-challenge
- OpenCV PnP: https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
- COLMAP: https://colmap.github.io/tutorial.html

## Recommended automatic synchronization

1. Extract mono audio at 4 kHz from every feed.
2. Estimate whole-clip delay with FFT cross-correlation against Broadcast.
3. Reject low-correlation estimates and fall back to scoreboard/visual cues.
4. Search ±6 frames around the coarse offset for the chosen event anchor.
5. Refine using ball–rim distance, player pose and/or manually approved impact frames.
6. Persist both the automatic lag and the final selected frame in the event manifest.

The new `estimate_audio_offsets.py` implements step 1–3 and already works on this event.

Research on unsynchronized dynamic Gaussian reconstruction exists, but it still assumes coherent multi-view video and performs learned coarse-to-fine temporal optimization; it does not remove the need for camera calibration or adequate overlap: https://arxiv.org/abs/2511.11175

## Damage caused by source mismatch

| Condition | Severity | Practical consequence |
|---|---:|---|
| 29.92 vs 29.97 fps | Low | Normalize timestamps/frame rate after measuring delay. |
| Different crops and zoom | High | Intrinsics change; a single camera model per feed is invalid. |
| Moving cameras | High | Extrinsics change every frame; background SfM becomes less stable. |
| 0.3–1.0 s feed delay | Critical before sync | Players occupy different poses, so reconstruction creates duplicate/ghost geometry. |
| 960×540 compression | Medium/high | Ball, hands, faces and thin limbs have too few clean pixels for robust matching. |
| Player occlusion | Critical for true 3D | Unseen surfaces cannot be recovered faithfully without a prior or hallucination. |
| Exposure/flash differences | Medium | Photometric losses and feature matching become unreliable. |

## Prototype implemented

The deterministic proof of concept is:

`normal Broadcast → slowed approach → exact impact freeze → action-locked whip orbit through eight real synchronized views → Broadcast resumes`

It uses only official NBA pixels. It does not use AI enhancement, generative fill, RIFE/FILM, depth hallucination or synthetic player reconstruction. The output is deliberately native 960×540 because the current project policy says to preserve native source quality and upscale later in VN if desired.

This is a valid Level-C result, not a claim of true 3D reconstruction.

## Production roadmap

1. Generalize audio sync and visual impact-frame refinement across the seven known Adams block events.
2. Add per-frame court/rim/backboard calibration and rank angle pairs by baseline plus overlap.
3. Run a static one-instant 4C4D/InstantSplat experiment on the best calibrated 6–8 views.
4. Reject any rendered viewpoint whose geometry changes player/ball relationships; keep it out of production footage.
5. Use Level-C automatically when calibration or reconstruction QA fails.

The eventual automated interface remains feasible at Level C:

`DATABASE QUERY → EVENT → ALL ANGLES → AUDIO COARSE SYNC → VISUAL FRAME REFINE → ACTION-LOCKED SPIN → MP4`

True Level-A output should remain an opt-in experimental branch until calibration and geometric QA are reliable.
