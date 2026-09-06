from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from freeze_spin.prove_right_above_rim_functional_camera_v47 import (
    action_volume,
    deterministic_starts,
    functional_p95,
)
from freeze_spin.solve_right_above_rim_fixed_geometry_v1 import (
    FOOT_CM,
    IMAGE_H,
    IMAGE_W,
    RIM_RADIUS_CM,
    RIM_X_CM,
    RIM_Z_CM,
    RESTRICTED_RADIUS_CM,
    circle,
    nearest_curve_distances,
    project,
)

FT_CENTER_X_CM = 15.0 * FOOT_CM
FT_RADIUS_CM = 6.0 * FOOT_CM


def geometry(samples: int = 240):
    return (
        circle(RIM_X_CM, RIM_Z_CM, RIM_RADIUS_CM, samples),
        circle(RIM_X_CM, 0.0, RESTRICTED_RADIUS_CM, samples),
        circle(FT_CENTER_X_CM, 0.0, FT_RADIUS_CM, samples),
    )


def residual_for(p, rim_obs, restricted_obs, ft_obs, hash_obj, hash_obs):
    rim3, restricted3, ft3 = geometry(240)
    rim_uv, rim_cam, _ = project(p, rim3)
    restricted_uv, restricted_cam, _ = project(p, restricted3)
    ft_uv, ft_cam, _ = project(p, ft3)
    hash_uv, hash_cam, _ = project(p, hash_obj)
    rim_d = nearest_curve_distances(rim_obs, rim_uv)
    restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
    ft_d = nearest_curve_distances(ft_obs, ft_uv)
    hash_res = (hash_uv - hash_obs).ravel()
    depth_min = min(
        float(np.min(rim_cam[:,2])),
        float(np.min(restricted_cam[:,2])),
        float(np.min(ft_cam[:,2])),
        float(np.min(hash_cam[:,2])),
    )
    priors = np.asarray([
        (p[7] - IMAGE_W/2.0)/100.0,
        (p[8] - IMAGE_H/2.0)/100.0,
        (p[6] - math.log(1000.0))/1.5,
        max(0.0,20.0-depth_min)/2.0,
    ])
    return np.concatenate([rim_d, restricted_d/1.5, ft_d, hash_res, priors])


def bounds():
    lower = np.r_[[-np.inf]*3,[-1000.0,-1000.0,250.0],math.log(150.0),300.0,100.0]
    upper = np.r_[[np.inf]*3,[1000.0,1000.0,3000.0],math.log(4000.0),660.0,440.0]
    return lower, upper


def optimize(p0, rim_obs, restricted_obs, ft_obs, hash_obj, hash_obs, max_nfev=1500):
    lo, hi = bounds()
    fit = least_squares(
        lambda p: residual_for(p,rim_obs,restricted_obs,ft_obs,hash_obj,hash_obs),
        p0,
        bounds=(lo,hi),
        loss='soft_l1',
        f_scale=2.0,
        x_scale='jac',
        max_nfev=max_nfev,
    )
    return fit.x, float(fit.cost)


def summarize(p, rim_obs, restricted_obs, ft_obs, hash_obj, hash_obs):
    rim3, restricted3, ft3 = geometry(720)
    rim_uv, rim_cam, R = project(p, rim3)
    restricted_uv, restricted_cam, _ = project(p, restricted3)
    ft_uv, ft_cam, _ = project(p, ft3)
    hash_uv, hash_cam, _ = project(p, hash_obj)
    rim_d = nearest_curve_distances(rim_obs, rim_uv)
    restricted_d = nearest_curve_distances(restricted_obs, restricted_uv)
    ft_d = nearest_curve_distances(ft_obs, ft_uv)
    hash_d = np.linalg.norm(hash_uv-hash_obs,axis=1)
    all_d = np.r_[rim_d,restricted_d,ft_d,hash_d]
    plausible = (
        min(
            float(np.min(rim_cam[:,2])),
            float(np.min(restricted_cam[:,2])),
            float(np.min(ft_cam[:,2])),
            float(np.min(hash_cam[:,2])),
        ) > 20.0
        and 150.0 <= math.exp(float(p[6])) <= 4000.0
        and -1000.0 <= float(p[3]) <= 1000.0
        and -1000.0 <= float(p[4]) <= 1000.0
        and 250.0 <= float(p[5]) <= 3000.0
    )
    return {
        'plausible': bool(plausible),
        'combined_anchor_rms_px': float(np.sqrt(np.mean(all_d**2))),
        'combined_anchor_p95_px': float(np.percentile(all_d,95)),
        'rim_curve_rms_px': float(np.sqrt(np.mean(rim_d**2))),
        'restricted_curve_rms_px': float(np.sqrt(np.mean(restricted_d**2))),
        'free_throw_circle_rms_px': float(np.sqrt(np.mean(ft_d**2))),
        'hash_point_errors_px': [float(v) for v in hash_d],
        'max_hash_point_error_px': float(np.max(hash_d)),
        'camera_center_world_cm': [float(v) for v in p[3:6]],
        'focal_px': float(math.exp(float(p[6]))),
        'principal_point_px': [float(p[7]),float(p[8])],
        'R_world_to_camera': R.tolist(),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--observations',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    spec=json.loads(args.observations.read_text(encoding='utf-8'))
    rim=np.asarray(spec['rim_inner_edge_samples_px'],dtype=np.float64)
    restricted=np.asarray(spec['restricted_area_centerline_samples_px'],dtype=np.float64)
    ft=np.asarray(spec['free_throw_circle_dash_centroids_px'],dtype=np.float64)
    hash_names=['negative_y_hash','positive_y_hash']
    hash_obj=np.asarray([spec['corresponded_hash_world_cm'][k] for k in hash_names],dtype=np.float64)
    hash_obs=np.asarray([spec['corresponded_hash_centroids_px'][k] for k in hash_names],dtype=np.float64)
    volume=action_volume()

    def one(item):
        i,p0=item
        p,cost=optimize(p0,rim,restricted,ft,hash_obj,hash_obs)
        qa=summarize(p,rim,restricted,ft,hash_obj,hash_obs)
        return {'index':i,'p':p,'cost':cost,'qa':qa}

    with ThreadPoolExecutor(max_workers=4) as pool:
        roots=list(pool.map(one,list(enumerate(deterministic_starts()))))
    roots.sort(key=lambda r:(not r['qa']['plausible'],r['qa']['combined_anchor_rms_px'],r['cost']))
    base=roots[0]
    gate=spec['gates']
    competitive=[r for r in roots if r['qa']['plausible']
        and r['qa']['combined_anchor_rms_px'] <= base['qa']['combined_anchor_rms_px']+0.35
        and r['qa']['combined_anchor_p95_px'] <= float(gate['max_combined_anchor_p95_px'])
        and r['qa']['max_hash_point_error_px'] <= float(gate['max_hash_point_error_px'])]
    pairwise=[]
    max_shift=0.0
    for i in range(len(competitive)):
        for j in range(i+1,len(competitive)):
            s=functional_p95(competitive[i]['p'],competitive[j]['p'],volume)
            pairwise.append({'a':competitive[i]['index'],'b':competitive[j]['index'],'p95_px':s})
            max_shift=max(max_shift,s)
    report={
        'status':'DIAGNOSTIC_ONLY_NO_PROMOTION',
        'base':base['qa'],
        'root_summaries':[{
            'index':r['index'],
            'combined_anchor_rms_px':r['qa']['combined_anchor_rms_px'],
            'combined_anchor_p95_px':r['qa']['combined_anchor_p95_px'],
            'max_hash_point_error_px':r['qa']['max_hash_point_error_px'],
            'hash_point_errors_px':r['qa']['hash_point_errors_px'],
            'rim_curve_rms_px':r['qa']['rim_curve_rms_px'],
            'restricted_curve_rms_px':r['qa']['restricted_curve_rms_px'],
            'free_throw_circle_rms_px':r['qa']['free_throw_circle_rms_px'],
            'camera_center_world_cm':r['qa']['camera_center_world_cm'],
            'focal_px':r['qa']['focal_px'],
            'principal_point_px':r['qa']['principal_point_px'],
        } for r in roots],
        'competitive_root_count':len(competitive),
        'max_competitive_pairwise_action_volume_p95_shift_px':max_shift,
        'pairwise':pairwise,
        'direct_hash_correspondences_appear_to_resolve_v49_two_root_ambiguity': bool(len(competitive)>=3 and max_shift <= float(gate['max_competitive_root_action_volume_p95_shift_px'])),
        'permissions':{'camera_promotion_allowed':False,'static_novel_view_allowed':False,'replay_render_allowed':False},
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'right_above_rim_hash_roots_v50a.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
