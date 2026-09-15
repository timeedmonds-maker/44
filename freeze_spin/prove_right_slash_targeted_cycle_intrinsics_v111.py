from __future__ import annotations
"""Right Slash v111: targeted cycle-closure proof around event 415.

Runs in parallel with broad v110.  It does not relax any transfer or promotion gate.
The only change is pair selection: force event pairs that can make 415 vertex-redundant
inside the already-known Right Slash component, and use ORB+FLANN only to shortlist
frame pairs before the unchanged exact mutual-SIFT .72 + topology gate.
"""
import argparse,itertools,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from freeze_spin import diagnose_right_slash_allpair_radial_v104 as base
from freeze_spin import prove_right_slash_expanded_connected_intrinsics_v110 as v110

# Durable tree branches + expanded temporal neighbours.  Pairing these branches can
# close cycles through target 415 instead of adding more leaves to 540.
CORE_LEFT={9,15,25,205,300,621}
CORE_COURT={40,275,405,407,409,411}
CORE_RIGHT={680,685,690}
TARGET=415
HUB=540
KNOWN=v110.KNOWN_FRAME_EDGES

def make_pairs(events):
    ev=set(events)
    pairs=set()
    # Direct target links have highest value: any second route from 415 into the
    # known component creates cycle redundancy.
    pairs.update(tuple(sorted((TARGET,e))) for e in ev if e not in {TARGET})
    # Force cross-branch closures rather than more within-branch leaves.
    for A,B in ((CORE_LEFT,CORE_COURT),(CORE_LEFT,CORE_RIGHT),(CORE_COURT,CORE_RIGHT)):
        pairs.update(tuple(sorted((a,b))) for a in A&ev for b in B&ev if a!=b)
    # Preserve all durable known edges as the backbone.
    pairs.update(tuple(sorted((a,c))) for a,_,c,_ in KNOWN if a in ev and c in ev)
    return sorted(pairs)

def frame_candidates(feat,pa,pb,topn):
    rows=[]
    # Two independent cheap screens; neither can create an accepted edge.
    for z in v110.orb_frame_candidates(feat,pa,pb,topn): rows.append((z[3],z[4]))
    try:
        for z in base.flann_frame_candidates(feat,pa,pb,topn=topn): rows.append((z[3],z[4]))
    except Exception:
        pass
    return list(dict.fromkeys(rows))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--top-frame-pairs',type=int,default=10);ap.add_argument('--perturbation-trials',type=int,default=12)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);cv2.setRNGSeed(0);cv2.setNumThreads(1)
    dirs=sorted(a.bank.glob('event_*_frames'));events=sorted(int(d.name.split('_')[1]) for d in dirs);paths=sorted(a.bank.glob('event_*_frames/f*.png'));feat=base.Features(paths)
    forced=defaultdict(list)
    for ea,fa,eb,fb in KNOWN:
        pa=a.bank/f'event_{ea}_frames'/fa;pb=a.bank/f'event_{eb}_frames'/fb
        if pa.exists() and pb.exists(): forced[tuple(sorted((ea,eb)))].append((pa,pb) if ea<eb else (pb,pa))
    pairs=make_pairs(events);accepted=[];audit=[]
    for n,(x,y) in enumerate(pairs,1):
        pa=sorted((a.bank/f'event_{x}_frames').glob('f*.png'));pb=sorted((a.bank/f'event_{y}_frames').glob('f*.png'))
        tests=frame_candidates(feat,pa,pb,a.top_frame_pairs);tests.extend(forced.get((x,y),[]));tests=list(dict.fromkeys(tests));best=[]
        for p,q in tests:
            z=base.exact_edge(feat,p,q)
            if z is not None: best.append(z)
        if best:
            best.sort(key=lambda z:(z['withheld_error']['p90_px'],z['withheld_error']['median_px'],-z['training_inliers']))
            accepted.append(best[0]);audit.append({'events':[x,y],'screened_frame_pairs':len(tests),'pass':True,'best':v110.serial_edge(best[0])})
        else: audit.append({'events':[x,y],'screened_frame_pairs':len(tests),'pass':False})
        print('PAIR',n,'/',len(pairs),x,y,'PASS' if best else 'FAIL',flush=True)
    nodes=sorted(set([z['a_event'] for z in accepted]+[z['b_event'] for z in accepted]));bics,_=v110.biconnected_components(nodes,accepted) if nodes else ([],{})
    eligible=[c for c in bics if TARGET in c and len(c)>=6];core=max(eligible,key=len) if eligible else []
    rep={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','method':'targeted cycle closure around event 415; ORB+FLANN shortlist only; unchanged exact transfer gate and shared PP + one Brown k1 physical model','pair_count':len(pairs),'accepted_edges':[v110.serial_edge(z) for z in accepted],'pair_audit':audit,'biconnected_components':bics,'robust_core_events':core,'metric_event_camera_allowed':False,'replay_render_allowed':False}
    prior=False
    if core:
        ce=[z for z in accepted if z['a_event'] in core and z['b_event'] in core];full=v110.fit_physical(ce);loo=[]
        for hold in core:
            sub=[z for z in ce if hold not in (z['a_event'],z['b_event'])]
            y=v110.fit_physical(sub,seed_shared=(full['pp'][0],full['pp'][1],full['k1']))
            loo.append({'held_out_event':hold,'principal_point_px':y['pp'],'pp_shift_px':float(np.linalg.norm(np.asarray(y['pp'])-np.asarray(full['pp']))),'k1':y['k1'],'k1_abs_shift':abs(y['k1']-full['k1']),'k1_same_sign':bool(np.sign(y['k1'])==np.sign(full['k1']) and abs(y['k1'])>1e-8),'all_heldout_pass':y['all_heldout_pass'],'focal_at_bound':y['focal_at_bound']})
        maxpp=max(x['pp_shift_px'] for x in loo);maxdk=max(x['k1_abs_shift'] for x in loo);mags=[abs(full['k1'])]+[abs(x['k1']) for x in loo];ratio=max(mags)/max(min(mags),1e-8)
        rng=np.random.default_rng(111);pert=[];mpp=0.;psign=True;pat=False
        for t in range(a.perturbation_trials):
            jit=[]
            for e in ce:
                n=min(len(e['train_idx']),70);jit.append((rng.uniform(-.5,.5,(n,2)),rng.uniform(-.5,.5,(n,2))))
            y=v110.fit_physical(ce,seed_shared=(full['pp'][0],full['pp'][1],full['k1']),jitter=jit,warm_only=True)
            ds=float(np.linalg.norm(np.asarray(y['pp'])-np.asarray(full['pp'])));same=bool(np.sign(y['k1'])==np.sign(full['k1']) and abs(y['k1'])>1e-8);mpp=max(mpp,ds);psign &= same;pat |= y['focal_at_bound'];pert.append({'trial':t,'pp_shift_px':ds,'k1':y['k1'],'k1_same_sign':same,'focal_at_bound':y['focal_at_bound']})
        gates={'robust_biconnected_core_at_least_6_events':len(core)>=6,'target_event_415_in_core':TARGET in core,'physical_model_all_heldout_edges_pass':full['all_heldout_pass'],'no_full_fit_focal_at_bound':not full['focal_at_bound'],'whole_event_loo_pp_shift_at_most_8px':maxpp<=8.,'whole_event_loo_k1_same_sign':all(x['k1_same_sign'] for x in loo),'whole_event_loo_k1_magnitude_ratio_at_most_2x':ratio<=2.,'whole_event_loo_k1_abs_shift_at_most_0_05':maxdk<=.05,'loo_fits_no_focal_at_bound':all(not x['focal_at_bound'] for x in loo),'half_pixel_pp_at_most_5px':mpp<=5.,'half_pixel_k1_sign_stable':psign,'half_pixel_no_focal_at_bound':not pat}
        prior=bool(all(gates.values()));rep.update({'physical_full_fit':full,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':maxpp,'k1_loo_max_min_ratio':ratio,'max_loo_k1_abs_shift':maxdk,'perturbation_trials':pert,'max_half_pixel_pp_shift_px':mpp,'gates':gates})
    else:
        rep['gates']={'robust_biconnected_core_at_least_6_events':False,'target_event_415_in_core':False}
    rep['status']='PASS_RIGHT_SLASH_TARGETED_CYCLE_INTRINSICS_PRIOR_V111' if prior else 'FAIL_RIGHT_SLASH_TARGETED_CYCLE_INTRINSICS_V111';rep['principal_point_prior_allowed']=prior
    (a.out/'right_slash_targeted_cycle_intrinsics_v111.json').write_text(json.dumps(rep,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else o.tolist() if isinstance(o,np.ndarray) else (_ for _ in ()).throw(TypeError(type(o).__name__)))+'\n')
    print(json.dumps({'status':rep['status'],'pairs':len(pairs),'edges':len(accepted),'core':core,'pp':rep.get('physical_full_fit',{}).get('pp'),'k1':rep.get('physical_full_fit',{}).get('k1'),'max_loo_pp':rep.get('max_leave_one_event_out_pp_shift_px'),'max_half_pixel_pp':rep.get('max_half_pixel_pp_shift_px'),'gates':rep.get('gates')},indent=2),flush=True)
if __name__=='__main__':main()
