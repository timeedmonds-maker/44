from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FOOT_CM = 30.48
RIM_X_CM = 15.0 * 2.54
RESTRICTED_RADIUS_CM = 4.0 * FOOT_CM
FT_CENTER_X_CM = 15.0 * FOOT_CM
FT_RADIUS_CM = 6.0 * FOOT_CM


def point_rows(X: float, Y: float, u: float, v: float) -> list[np.ndarray]:
    return [
        np.asarray([-X,-Y,-1.0, 0.0,0.0,0.0, u*X,u*Y,u], dtype=np.float64),
        np.asarray([0.0,0.0,0.0, -X,-Y,-1.0, v*X,v*Y,v], dtype=np.float64),
    ]


def line_rows(world_line: np.ndarray, image_line: np.ndarray) -> list[np.ndarray]:
    A,B,C = [float(v) for v in world_line]
    a,b,c = [float(v) for v in image_line]
    # q = H^T l_img. Enforce l_world x q = 0 with two independent equations.
    q1 = np.asarray([a,0,0, b,0,0, c,0,0], dtype=np.float64)
    q2 = np.asarray([0,a,0, 0,b,0, 0,c,0], dtype=np.float64)
    q3 = np.asarray([0,0,a, 0,0,b, 0,0,c], dtype=np.float64)
    r1 = B*q3 - C*q2
    r2 = C*q1 - A*q3
    r3 = A*q2 - B*q1
    candidates = [r1,r2,r3]
    norms = [float(np.linalg.norm(r)) for r in candidates]
    order = np.argsort(norms)[::-1]
    chosen = []
    for idx in order:
        r = candidates[int(idx)]
        if np.linalg.norm(r) < 1e-12:
            continue
        if not chosen:
            chosen.append(r)
        else:
            # keep a second row not collinear with first
            M = np.vstack([chosen[0],r])
            if np.linalg.matrix_rank(M, tol=1e-10) == 2:
                chosen.append(r)
                break
    if len(chosen) != 2:
        raise RuntimeError('Could not form two independent line constraints')
    return chosen


def solve_h(spec: dict) -> np.ndarray:
    rows=[]
    for key in ('negative_y_hash','positive_y_hash'):
        X,Y,_ = spec['corresponded_hash_world_cm'][key]
        u,v = spec['corresponded_hash_centroids_px'][key]
        rows.extend(point_rows(float(X),float(Y),float(u),float(v)))

    # v48 image line fits: x = slope*y + intercept => x - slope*y - intercept = 0.
    # Common world convention: negative-Y lane is image-left, positive-Y lane image-right.
    left = spec['lane_image_lines']['negative_y_lane']
    right = spec['lane_image_lines']['positive_y_lane']
    lane_half = 8.0 * FOOT_CM
    for y0, item in ((-lane_half,left),(+lane_half,right)):
        world_line=np.asarray([0.0,1.0,-y0],dtype=np.float64)  # Y-y0=0
        image_line=np.asarray([1.0,-float(item['x_as_function_of_y']['slope']),-float(item['x_as_function_of_y']['intercept'])],dtype=np.float64)
        rows.extend(line_rows(world_line,image_line))
    A=np.vstack(rows)
    if A.shape != (8,9) or np.linalg.matrix_rank(A) < 8:
        raise RuntimeError(f'Degenerate floor system shape={A.shape} rank={np.linalg.matrix_rank(A)}')
    _,_,vh=np.linalg.svd(A)
    H=vh[-1].reshape(3,3)
    if abs(H[2,2]) > 1e-12:
        H=H/H[2,2]
    return H


def project_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    q=np.column_stack([xy,np.ones(len(xy))])
    p=(H@q.T).T
    return p[:,:2]/p[:,2:3]


def circle_xy(cx: float, radius: float, n: int=1440) -> np.ndarray:
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.column_stack([cx+radius*np.cos(t), radius*np.sin(t)])


def nearest(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    d2=np.sum((obs[:,None,:]-pred[None,:,:])**2,axis=2)
    return np.sqrt(np.min(d2,axis=1))


def line_point_distance(line: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a,b,c=[float(v) for v in line]
    return np.abs(a*pts[:,0]+b*pts[:,1]+c)/np.hypot(a,b)


def evaluate(H: np.ndarray, spec: dict) -> dict:
    ft_obs=np.asarray(spec['free_throw_circle_dash_centroids_px'],dtype=np.float64)
    restricted_obs=np.asarray(spec['restricted_area_centerline_samples_px'],dtype=np.float64)
    ft_pred=project_h(H,circle_xy(FT_CENTER_X_CM,FT_RADIUS_CM))
    restricted_pred=project_h(H,circle_xy(RIM_X_CM,RESTRICTED_RADIUS_CM))
    ft_d=nearest(ft_obs,ft_pred)
    restricted_d=nearest(restricted_obs,restricted_pred)
    hash_names=('negative_y_hash','positive_y_hash')
    hash_world=np.asarray([[*spec['corresponded_hash_world_cm'][k][:2]] for k in hash_names],dtype=np.float64)
    hash_obs=np.asarray([spec['corresponded_hash_centroids_px'][k] for k in hash_names],dtype=np.float64)
    hash_pred=project_h(H,hash_world)
    hash_d=np.linalg.norm(hash_pred-hash_obs,axis=1)

    lane_half=8.0*FOOT_CM
    line_errors=[]
    for y0,key in ((-lane_half,'negative_y_lane'),(+lane_half,'positive_y_lane')):
        xs=np.linspace(-4.0*FOOT_CM,15.0*FOOT_CM,100)
        pred=project_h(H,np.column_stack([xs,np.full_like(xs,y0)]))
        item=spec['lane_image_lines'][key]
        img_line=np.asarray([1.0,-float(item['x_as_function_of_y']['slope']),-float(item['x_as_function_of_y']['intercept'])],dtype=np.float64)
        line_errors.extend(line_point_distance(img_line,pred).tolist())
    all_hold=np.r_[ft_d,restricted_d]
    return {
        'finite_nondegenerate_homography': bool(np.all(np.isfinite(H)) and abs(np.linalg.det(H))>1e-12),
        'free_throw_circle_rms_px': float(np.sqrt(np.mean(ft_d**2))),
        'free_throw_circle_p95_px': float(np.percentile(ft_d,95)),
        'restricted_circle_rms_px': float(np.sqrt(np.mean(restricted_d**2))),
        'restricted_circle_p95_px': float(np.percentile(restricted_d,95)),
        'pooled_heldout_rms_px': float(np.sqrt(np.mean(all_hold**2))),
        'pooled_heldout_p95_px': float(np.percentile(all_hold,95)),
        'hash_anchor_errors_px': [float(v) for v in hash_d],
        'max_lane_line_reprojection_error_px': float(max(line_errors)),
    }


def perturb(spec: dict, trials: int=128) -> dict:
    rng=np.random.default_rng(20260903)
    H0=solve_h(spec)
    grid=[]
    for x in np.linspace(-4*FOOT_CM,15*FOOT_CM,9):
        for y in np.linspace(-8*FOOT_CM,8*FOOT_CM,9):
            grid.append([x,y])
    grid=np.asarray(grid,dtype=np.float64)
    base=project_h(H0,grid)
    shifts=[]
    heldout=[]
    for _ in range(trials):
        q=json.loads(json.dumps(spec))
        for key in ('negative_y_hash','positive_y_hash'):
            p=np.asarray(q['corresponded_hash_centroids_px'][key],dtype=float)+rng.uniform(-0.5,0.5,size=2)
            q['corresponded_hash_centroids_px'][key]=p.tolist()
        # perturb fitted image lane lines by sampling two points on each line, shifting each point +-0.5px, refitting x(y)
        for key in ('negative_y_lane','positive_y_lane'):
            item=q['lane_image_lines'][key]
            a=float(item['x_as_function_of_y']['slope']); b=float(item['x_as_function_of_y']['intercept'])
            ys=np.asarray([120.0,360.0])
            xs=a*ys+b
            pts=np.column_stack([xs,ys])+rng.uniform(-0.5,0.5,size=(2,2))
            anew=(pts[1,0]-pts[0,0])/(pts[1,1]-pts[0,1])
            bnew=pts[0,0]-anew*pts[0,1]
            item['x_as_function_of_y']['slope']=float(anew)
            item['x_as_function_of_y']['intercept']=float(bnew)
        try:
            H=solve_h(q)
            pred=project_h(H,grid)
            shifts.append(float(np.percentile(np.linalg.norm(pred-base,axis=1),95)))
            heldout.append(evaluate(H,q)['pooled_heldout_p95_px'])
        except Exception:
            shifts.append(float('inf')); heldout.append(float('inf'))
    return {
        'trials':trials,
        'max_half_pixel_floor_grid_p95_shift_px':float(max(shifts)),
        'p95_half_pixel_floor_grid_p95_shift_px':float(np.percentile(shifts,95)),
        'max_perturbed_heldout_p95_px':float(max(heldout)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--observations',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    spec=json.loads(args.observations.read_text(encoding='utf-8'))
    H=solve_h(spec)
    qa=evaluate(H,spec)
    sens=perturb(spec,128)
    report={
        'status':'DIAGNOSTIC_ONLY_NO_PROMOTION',
        'method':'linear floor homography from two regulation lane-line correspondences + two directly corresponded 13-foot hash marks; free-throw and restricted circles held out',
        'floor_homography_world_to_image':H.tolist(),
        'qa':qa,
        'sensitivity':sens,
        'interpretation':{
            'lane_lines_used_only_as_infinite_line_correspondences':True,
            'no_longitudinal_pixel_to_world_correspondence_assumed_on_lane_lines':True,
            'free_throw_and_restricted_circles_are_heldout':True,
        },
        'permissions':{'floor_homography_allowed':False,'metric_camera_allowed':False,'static_novel_view_allowed':False,'replay_render_allowed':False},
    }
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'right_above_rim_floor_homography_v51a.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
