from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares

W,H=960,540
FT=30.48; IN=2.54
RIM_X=15*IN; FT_X=15*FT; FT_R=6*FT; THREE_R=23.75*FT; CORNER_Y=22*FT; PAINT_HALF=8*FT
SCALES=np.asarray([1.,1.,300.,1.,1.,300.,.002,.002],float)
SENTINELS={0,7,15,23,31,47,55,63}

def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def project_h(M,xy):
    a=np.asarray(xy,float);q=(M@np.c_[a,np.ones(len(a))].T).T
    return q[:,:2]/q[:,2:]

def pvec(M):
    M=np.asarray(M,float)/float(M[2,2]);return np.r_[M[0],M[1],M[2,:2]]

def H_from_z(z,h0):
    q=h0+np.asarray(z,float)*SCALES
    return np.asarray([[q[0],q[1],q[2]],[q[3],q[4],q[5]],[q[6],q[7],1.]],float)

def split_groups(obs,held):
    tr={};te={}
    for k,rows in obs.items():
        a=np.asarray(rows,float);ids=np.asarray(held[k],int)
        if np.any(ids<0) or np.any(ids>=len(a)):raise RuntimeError(f'bad held index {k}')
        m=np.ones(len(a),bool);m[ids]=False;tr[k]=a[m];te[k]=a[~m]
        if len(tr[k])<3 or len(te[k])<1:raise RuntimeError(f'insufficient support {k}')
    return tr,te

def world_res(M,g):
    G=np.linalg.inv(M);o={}
    p=project_h(G,g['three_point_arc']);o['three_point_arc']=np.sqrt((p[:,0]-RIM_X)**2+p[:,1]**2)-THREE_R
    p=project_h(G,g['free_throw_front_semicircle']);o['free_throw_front_semicircle']=np.sqrt((p[:,0]-FT_X)**2+p[:,1]**2)-FT_R
    p=project_h(G,g['free_throw_line']);o['free_throw_line']=p[:,0]-FT_X
    p=project_h(G,g['lane_sideline']);o['lane_sideline']=p[:,1]-PAINT_HALF
    p=project_h(G,g['opposite_lane_sideline']);o['opposite_lane_sideline']=p[:,1]+PAINT_HALF
    return o

def residual(z,h0,g):
    try:
        M=H_from_z(z,h0)
        if not np.isfinite(M).all() or abs(float(np.linalg.det(M)))<1e-12:raise ValueError
        r=world_res(M,g)
        return np.r_[r['three_point_arc']/3.,r['free_throw_front_semicircle']/3.,r['free_throw_line']/3.,r['lane_sideline']/3.,r['opposite_lane_sideline']/3.,np.asarray(z)*.001]
    except Exception:
        return np.full(sum(len(v) for v in g.values())+8,1e6,float)

def multistart_candidates(h0,g,warm=None,seed=650904):
    seeds=[]
    if warm is not None:seeds.append(np.asarray(warm,float))
    seeds.append(np.zeros(8));rng=np.random.default_rng(seed)
    for _ in range(7):seeds.append(rng.uniform(-.35,.35,8))
    rows=[]
    for i,s in enumerate(seeds):
        try:
            fit=least_squares(lambda z:residual(z,h0,g),s,loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=18000)
            M=H_from_z(fit.x,h0);wr=world_res(M,g)
            med=float(np.median(np.abs(np.concatenate(list(wr.values())))))
            rows.append({'seed_index':i,'cost':float(fit.cost),'median_abs_world_cm':med,'z':np.asarray(fit.x,float),'H':M})
        except Exception:
            continue
    if not rows:raise RuntimeError('all multistarts failed')
    rows.sort(key=lambda r:(r['cost'],r['median_abs_world_cm']))
    return rows

def solve_full(h0,g,warm=None,seed=650904):return multistart_candidates(h0,g,warm,seed)[0]['z']

def solve_warm(h0,g,warm):
    fit=least_squares(lambda z:residual(z,h0,g),np.asarray(warm,float),loss='soft_l1',f_scale=2.,x_scale='jac',max_nfev=6000)
    return np.asarray(fit.x,float)

def dense_three(n=2001):
    tmax=math.asin(CORNER_Y/THREE_R);t=np.linspace(-tmax,tmax,n)
    return np.c_[RIM_X+THREE_R*np.cos(t),THREE_R*np.sin(t)]

def dense_ft(n=1601):
    t=np.linspace(0,2*math.pi,n);return np.c_[FT_X+FT_R*np.cos(t),FT_R*np.sin(t)]

def dense_ft_line(n=1001):return np.c_[np.full(n,FT_X),np.linspace(-8*FT,8*FT,n)]
def dense_lane(n=1001):return np.c_[np.linspace(-4*FT,FT_X,n),np.full(n,PAINT_HALF)]
def dense_opp_lane(n=1001):return np.c_[np.linspace(-4*FT,FT_X,n),np.full(n,-PAINT_HALF)]

def visible_support(M,spec):
    G=np.linalg.inv(M);obs=spec['observations_px']
    w3=project_h(G,np.asarray(obs['three_point_arc'],float));t3=np.arctan2(w3[:,1],w3[:,0]-RIM_X)
    wf=project_h(G,np.asarray(obs['free_throw_front_semicircle'],float));tf=np.arctan2(wf[:,1],wf[:,0]-FT_X)
    wl=project_h(G,np.asarray(obs['free_throw_line'],float))
    wa=project_h(G,np.asarray(obs['lane_sideline'],float));wb=project_h(G,np.asarray(obs['opposite_lane_sideline'],float))
    return {'three_theta_rad':[float(t3.min()),float(t3.max())],'ft_theta_rad':[float(tf.min()),float(tf.max())],'ft_line_y_cm':[float(wl[:,1].min()),float(wl[:,1].max())],'lane_x_cm':[float(wa[:,0].min()),float(wa[:,0].max())],'opposite_lane_x_cm':[float(wb[:,0].min()),float(wb[:,0].max())]}

def support_curves(M,spec,n=1601):
    sp=visible_support(M,spec);t=np.linspace(*sp['three_theta_rad'],n);p3=np.c_[RIM_X+THREE_R*np.cos(t),THREE_R*np.sin(t)]
    t=np.linspace(*sp['ft_theta_rad'],n);pft=np.c_[FT_X+FT_R*np.cos(t),FT_R*np.sin(t)]
    pfl=np.c_[np.full(n,FT_X),np.linspace(*sp['ft_line_y_cm'],n)]
    pla=np.c_[np.linspace(*sp['lane_x_cm'],n),np.full(n,PAINT_HALF)]
    pob=np.c_[np.linspace(*sp['opposite_lane_x_cm'],n),np.full(n,-PAINT_HALF)]
    return sp,(p3,pft,pfl,pla,pob)

def nearest_metrics(obs,pred):
    d=np.sqrt(((np.asarray(obs)[:,None,:]-pred[None,:,:])**2).sum(2)).min(1)
    return {'count':int(len(d)),'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d)),'per_point_px':[float(x) for x in d]}

def world_metrics(M,g):
    r=world_res(M,g)
    return {k:{'count':int(len(v)),'median_abs_cm':float(np.median(np.abs(v))),'p95_abs_cm':float(np.percentile(np.abs(v),95)),'max_abs_cm':float(np.max(np.abs(v)))} for k,v in r.items()}

def shift(A,B):
    d=np.linalg.norm(A-B,axis=1);return {'median_px':float(np.median(d)),'p95_px':float(np.percentile(d,95)),'max_px':float(np.max(d))}

def draw(im,spec,M,path):
    ov=im.copy()
    for pts,c in ((project_h(M,dense_three()),(0,0,255)),(project_h(M,dense_ft()),(0,165,255)),(project_h(M,dense_ft_line()),(255,255,255)),(project_h(M,dense_lane()),(255,0,255)),(project_h(M,dense_opp_lane()),(0,255,255))):
        q=np.round(pts).astype(int);ok=(q[:,0]>=0)&(q[:,0]<W)&(q[:,1]>=0)&(q[:,1]<H)
        for x,y in q[ok][::3]:cv2.circle(ov,(int(x),int(y)),1,c,-1,cv2.LINE_AA)
    colors={'three_point_arc':(255,255,0),'free_throw_front_semicircle':(0,255,0),'free_throw_line':(0,255,255),'lane_sideline':(255,0,255),'opposite_lane_sideline':(0,255,255)}
    held={k:set(v) for k,v in spec['held_out_indices'].items()}
    for k,rows in spec['observations_px'].items():
        for i,p in enumerate(np.asarray(rows,int)):
            cv2.circle(ov,tuple(p),5 if i in held[k] else 3,colors[k],2 if i in held[k] else 1,cv2.LINE_AA)
    cv2.imwrite(str(path),ov)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frame',type=Path,required=True);ap.add_argument('--observations',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    spec=json.loads(a.observations.read_text());im=cv2.imread(str(a.frame))
    if im is None or im.shape[:2]!=(H,W):raise RuntimeError('immutable Play by Play frame must be native 960x540')
    actual=sha256(a.frame)
    if actual!=spec['image_sha256']:raise RuntimeError(f'immutable SHA mismatch {actual}')
    expected={'three_point_arc','free_throw_front_semicircle','free_throw_line','lane_sideline','opposite_lane_sideline'}
    if set(spec['observations_px'])!=expected:raise RuntimeError('v65 accepts only five static regulation floor families')
    tr,te=split_groups(spec['observations_px'],spec['held_out_indices'])
    h0=pvec(np.asarray(spec['numerical_seed_homography_world_to_source_px'],float))
    nominal_roots=multistart_candidates(h0,tr,seed=650904);z=nominal_roots[0]['z'];M=nominal_roots[0]['H']
    visible,world_support=support_curves(M,spec);w3,wft,wfl,wla,wob=world_support
    p3=project_h(M,w3);pft=project_h(M,wft);pfl=project_h(M,wfl);plane=project_h(M,wla);opp=project_h(M,wob)
    heldpx={'three_point_arc':nearest_metrics(te['three_point_arc'],project_h(M,dense_three())),'free_throw_front_semicircle':nearest_metrics(te['free_throw_front_semicircle'],project_h(M,dense_ft())),'free_throw_line':nearest_metrics(te['free_throw_line'],project_h(M,dense_ft_line())),'lane_sideline':nearest_metrics(te['lane_sideline'],project_h(M,dense_lane())),'opposite_lane_sideline':nearest_metrics(te['opposite_lane_sideline'],project_h(M,dense_opp_lane()))}
    low=[r for r in nominal_roots if r['cost']<=nominal_roots[0]['cost']+1.0]
    cluster=[]
    for r in low:
        cluster.append({'seed_index':r['seed_index'],'cost':r['cost'],'three_p95_px':shift(p3,project_h(r['H'],w3))['p95_px'],'ft_p95_px':shift(pft,project_h(r['H'],wft))['p95_px'],'lane_p95_px':shift(plane,project_h(r['H'],wla))['p95_px'],'opposite_lane_p95_px':shift(opp,project_h(r['H'],wob))['p95_px']})
    maxcluster=max(max(x['three_p95_px'],x['ft_p95_px'],x['lane_p95_px'],x['opposite_lane_p95_px']) for x in cluster)
    red={k:v[:-1] for k,v in tr.items()};zr=solve_full(h0,red,warm=z,seed=650905);Mr=H_from_z(zr,h0)
    root={'three_point_arc':shift(p3,project_h(Mr,w3)),'free_throw_circle':shift(pft,project_h(Mr,wft)),'free_throw_line':shift(pfl,project_h(Mr,wfl)),'lane_sideline':shift(plane,project_h(Mr,wla)),'opposite_lane_sideline':shift(opp,project_h(Mr,wob))}
    rng=np.random.default_rng(650906);pert=[];sent=[]
    for trial in range(64):
        pg={k:v+rng.uniform(-.5,.5,v.shape) for k,v in tr.items()};zw=solve_warm(h0,pg,z);Mw=H_from_z(zw,h0)
        row={'trial':trial,'three_point_arc':shift(p3,project_h(Mw,w3)),'free_throw_circle':shift(pft,project_h(Mw,wft)),'free_throw_line':shift(pfl,project_h(Mw,wfl)),'lane_sideline':shift(plane,project_h(Mw,wla)),'opposite_lane_sideline':shift(opp,project_h(Mw,wob))};pert.append(row)
        if trial in SENTINELS:
            zf=solve_full(h0,pg,warm=z,seed=650906+trial);Mf=H_from_z(zf,h0)
            sent.append({'trial':trial,'three_p95_px':shift(project_h(Mw,w3),project_h(Mf,w3))['p95_px'],'ft_p95_px':shift(project_h(Mw,wft),project_h(Mf,wft))['p95_px'],'line_p95_px':shift(project_h(Mw,wfl),project_h(Mf,wfl))['p95_px'],'lane_p95_px':shift(project_h(Mw,wla),project_h(Mf,wla))['p95_px'],'opposite_lane_p95_px':shift(project_h(Mw,wob),project_h(Mf,wob))['p95_px']})
    maxpert={fam:max(r[fam]['p95_px'] for r in pert) for fam in ('three_point_arc','free_throw_circle','free_throw_line','lane_sideline','opposite_lane_sideline')}
    maxsent={k:max(r[k] for r in sent) for k in ('three_p95_px','ft_p95_px','line_p95_px','lane_p95_px','opposite_lane_p95_px')}
    wmte=world_metrics(M,te)
    gates={
      'immutable_native_frame':True,
      'static_regulation_floor_only':True,
      'heldout_three_point_p95_at_most_2px':heldpx['three_point_arc']['p95_px']<=2.,
      'heldout_ft_semicircle_p95_at_most_2px':heldpx['free_throw_front_semicircle']['p95_px']<=2.,
      'heldout_ft_line_p95_at_most_2px':heldpx['free_throw_line']['p95_px']<=2.,
      'heldout_lane_sideline_p95_at_most_2px':heldpx['lane_sideline']['p95_px']<=2.,
      'heldout_opposite_lane_sideline_p95_at_most_2px':heldpx['opposite_lane_sideline']['p95_px']<=2.,
      'heldout_straight_line_world_p95_at_most_8cm':max(wmte['free_throw_line']['p95_abs_cm'],wmte['lane_sideline']['p95_abs_cm'],wmte['opposite_lane_sideline']['p95_abs_cm'])<=8.,
      'nominal_multistart_low_cost_cluster_at_most_0_05px':maxcluster<=.05,
      'support_removal_three_p95_at_most_2_5px':root['three_point_arc']['p95_px']<=2.5,
      'support_removal_ft_p95_at_most_2px':root['free_throw_circle']['p95_px']<=2.,
      'support_removal_lane_p95_at_most_2px':root['lane_sideline']['p95_px']<=2.,
      'support_removal_opposite_lane_p95_at_most_2px':root['opposite_lane_sideline']['p95_px']<=2.,
      'half_pixel_three_p95_shift_at_most_2_5px':maxpert['three_point_arc']<=2.5,
      'half_pixel_ft_p95_shift_at_most_2px':maxpert['free_throw_circle']<=2.,
      'warm_full_multistart_sentinels_at_most_0_05px':max(maxsent.values())<=.05,
    }
    passed=all(gates.values());draw(im,spec,M,a.out/'play_by_play_frame_c_wide_court_overlay_v65.png')
    clean_roots=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items() if k!='H'} for r in nominal_roots]
    report={'status':'PASS_PLAY_BY_PLAY_WIDE_COURT_FLOOR_V65' if passed else 'FAIL_PLAY_BY_PLAY_WIDE_COURT_FLOOR_V65','game_id':spec['game_id'],'event_id':spec['event_id'],'camera_label':spec['camera_label'],'immutable_frame':{'file':a.frame.name,'sha256':actual,'geometry':'960x540 native'},'method':'undistorted regulation-floor homography from 23ft9in arc + isolated visible 6ft FT solid semicircle + FT line + paired +/-8ft paint sidelines; no player/ball/body points','rejected_predecessor_note':'A prior full-circle candidate was rejected because straight paint-boundary pixels contaminated the FT-circle set. A one-parameter Brown fit was also rejected after independent v64a same-centre homography diagnostics preferred near-zero/mild-positive distortion. v65 instead fixes the source segmentation and adds paired independent paint sidelines to remove projective root ambiguity.','homography_world_to_source_px':M.tolist(),'homography_source_to_world':np.linalg.inv(M).tolist(),'training_world_error_cm':world_metrics(M,tr),'heldout_world_error_cm':wmte,'heldout_pixel_curve_error':heldpx,'source_visible_regulation_support_world':visible,'nominal_multistart':{'best_cost':nominal_roots[0]['cost'],'low_cost_root_count':len(low),'max_low_cost_projection_p95_px':maxcluster,'root_cluster':cluster,'roots':clean_roots},'support_removal_root_stability':root,'half_pixel_training_annotation_perturbation':{'trial_count':64,'max_p95_shift_px':maxpert,'trials':pert},'warm_vs_full_multistart_sentinels':{'count':len(sent),'max_p95_disagreement_px':maxsent,'rows':sent},'gates':gates,'permissions':{'floor_homography_allowed':passed,'shared_optical_center_prior_allowed':True,'metric_camera_promotion_allowed':False,'static_novel_view_allowed':False,'replay_render_allowed':False},'next_action':'Only if v65 passes: decompose this floor plane with the independently proven v64a optical/intrinsic prior, add exact 10-ft rim geometry, and require physical camera-centre multistart/half-pixel stability before metric promotion.'}
    (a.out/'play_by_play_frame_c_wide_court_v65.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'heldpx':heldpx,'maxcluster':maxcluster,'root':root,'maxpert':maxpert,'maxsent':maxsent,'gates':gates},indent=2),flush=True)
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
