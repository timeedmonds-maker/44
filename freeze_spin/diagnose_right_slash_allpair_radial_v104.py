from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
from collections import defaultdict, deque
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

W,H=960,540
EVENT_RE=re.compile(r'event_(\d+)_frames$')
REP={15:'f03.png',25:'f00.png',40:'f04.png',140:'f06.png',145:'f06.png',155:'f00.png',275:'f00.png',300:'f01.png',310:'f05.png',405:'f06.png',410:'f00.png',415:'f06.png',540:'f01.png',555:'f06.png',560:'f04.png',680:'f00.png',685:'f02.png',690:'f02.png'}
FORCE={(15,540),(25,540),(415,540),(540,690),(685,690)}

def event_id(p:Path)->int:
    m=EVENT_RE.search(p.parent.name); return int(m.group(1)) if m else -1

def action_core(xy):
    x,y=xy[:,0],xy[:,1]
    return (x>.20*W)&(x<.80*W)&(y>.48*H)&(y<.98*H)

def estats(e):
    if not len(e): return {'n':0,'median_px':None,'p90_px':None,'p95_px':None,'max_px':None}
    return {'n':int(len(e)),'median_px':float(np.median(e)),'p90_px':float(np.percentile(e,90)),'p95_px':float(np.percentile(e,95)),'max_px':float(np.max(e))}

def topology_keep(p,q,k=8,min_overlap=2):
    if len(p)<=k: return np.ones(len(p),bool)
    _,a=cKDTree(p).query(p,k=k+1); _,b=cKDTree(q).query(q,k=k+1)
    return np.asarray([len(set(a[i,1:])&set(b[i,1:]))>=min_overlap for i in range(len(p))],bool)

class Features:
    def __init__(self,paths):
        self.sift={}; self.orb={}
        sf=cv2.SIFT_create(nfeatures=9000,contrastThreshold=.015)
        of=cv2.ORB_create(nfeatures=4500,fastThreshold=10)
        for p in paths:
            im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape!=(H,W): continue
            ks,ds=sf.detectAndCompute(im,None); ko,do=of.detectAndCompute(im,None)
            self.sift[p]=(ks,None if ds is None else ds.astype(np.float32)); self.orb[p]=(ko,do)

def one_to_one_ratio(raw,ratio):
    good=[m for m,n in raw if m.distance<ratio*n.distance]
    best={}
    for m in good:
        if m.trainIdx not in best or m.distance<best[m.trainIdx].distance: best[m.trainIdx]=m
    return list(best.values())

def orb_rank(f:Features,bank:Path):
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    rows=[]; events=sorted(REP)
    for i,a in enumerate(events):
        pa=bank/f'event_{a}_frames'/REP[a]
        ka,da=f.orb.get(pa,(None,None))
        if da is None: continue
        for b in events[i+1:]:
            pb=bank/f'event_{b}_frames'/REP[b]; kb,db=f.orb.get(pb,(None,None))
            if db is None: continue
            raw=bf.knnMatch(da,db,k=2); good=one_to_one_ratio(raw,.78)
            if len(good)<12: continue
            p=np.float32([ka[m.queryIdx].pt for m in good]); q=np.float32([kb[m.trainIdx].pt for m in good])
            M,mask=cv2.findHomography(p,q,cv2.RANSAC,3.0,maxIters=5000,confidence=.995)
            if M is None or mask is None: continue
            nin=int(mask.sum()); rows.append((nin,nin/max(1,len(good)),a,b))
    rows.sort(reverse=True)
    return rows

def flann_frame_candidates(f:Features,pa,pb,topn=10):
    fl=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=40)); rows=[]
    for a in pa:
        ka,da=f.sift.get(a,(None,None))
        if da is None: continue
        for b in pb:
            kb,db=f.sift.get(b,(None,None))
            if db is None: continue
            ab=fl.knnMatch(da,db,k=2); ba=fl.knnMatch(db,da,k=2)
            gab={m.queryIdx:m for m,n in ab if m.distance<.76*n.distance}; gba={m.queryIdx:m for m,n in ba if m.distance<.76*n.distance}
            ms=[m for qi,m in gab.items() if m.trainIdx in gba and gba[m.trainIdx].trainIdx==qi]
            if len(ms)<28: continue
            p=np.float32([ka[m.queryIdx].pt for m in ms]); q=np.float32([kb[m.trainIdx].pt for m in ms])
            xa,ya=p[:,0],p[:,1]; xb,yb=q[:,0],q[:,1]
            tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W))
            tr=tg&~action_core(p)&~action_core(q); wh=~tr&~action_core(p)&~action_core(q)
            rows.append((int(tr.sum()),int(wh.sum()),len(ms),a,b))
    rows.sort(key=lambda z:(-z[0],-z[1],-z[2]))
    return rows[:topn]

def exact_edge(f:Features,a:Path,b:Path):
    ka,da=f.sift.get(a,(None,None)); kb,db=f.sift.get(b,(None,None))
    if da is None or db is None: return None
    bf=cv2.BFMatcher(cv2.NORM_L2)
    ab=bf.knnMatch(da,db,k=2); ba=bf.knnMatch(db,da,k=2)
    gab={m.queryIdx:m for m,n in ab if m.distance<.72*n.distance}; gba={m.queryIdx:m for m,n in ba if m.distance<.72*n.distance}
    ms=[m for qi,m in gab.items() if m.trainIdx in gba and gba[m.trainIdx].trainIdx==qi]
    if len(ms)<30: return None
    p=np.float64([ka[m.queryIdx].pt for m in ms]); q=np.float64([kb[m.trainIdx].pt for m in ms])
    keep=topology_keep(p,q,8,2); p=p[keep]; q=q[keep]
    if len(p)<30: return None
    xa,ya=p[:,0],p[:,1]; xb,yb=q[:,0],q[:,1]
    tg=((ya<.46*H)|(xa<.14*W)|(xa>.86*W))&((yb<.46*H)|(xb<.14*W)|(xb>.86*W))
    tr=tg&~action_core(p)&~action_core(q); wh=~tr&~action_core(p)&~action_core(q)
    if int(tr.sum())<12: return None
    M,mask=cv2.findHomography(p[tr].astype(np.float32),q[tr].astype(np.float32),cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
    if M is None or mask is None: return None
    ii=mask.ravel().astype(bool); ti=np.where(tr)[0][ii]; hi=np.where(wh)[0]
    pred=cv2.perspectiveTransform(p[:,None,:].astype(np.float32),M)[:,0]; e=np.linalg.norm(pred-q,axis=1)
    ts,hs=estats(e[ti]),estats(e[hi])
    gates={'training_inliers_at_least_24':len(ti)>=24,'training_p95_at_most_1_5px':ts['p95_px'] is not None and ts['p95_px']<=1.5,'withheld_matches_at_least_10':len(hi)>=10,'withheld_median_at_most_2_5px':hs['median_px'] is not None and hs['median_px']<=2.5,'withheld_p90_at_most_4px':hs['p90_px'] is not None and hs['p90_px']<=4.0}
    if not all(gates.values()): return None
    return {'a_event':event_id(a),'a_frame':a.name,'b_event':event_id(b),'b_frame':b.name,'a_path':str(a),'b_path':str(b),'match_count_after_topology':int(len(p)),'training_inliers':int(len(ti)),'training_error':ts,'withheld_error':hs,'gates':gates,'H':M.tolist(),'p':p,'q':q,'train_idx':ti,'held_idx':hi}

def components(events,edges):
    adj={e:set() for e in events}
    for z in edges: adj[z['a_event']].add(z['b_event']); adj[z['b_event']].add(z['a_event'])
    out=[]; seen=set()
    for s in events:
        if s in seen: continue
        q=[s]; seen.add(s); c=[]
        while q:
            x=q.pop(); c.append(x)
            for y in adj[x]:
                if y not in seen: seen.add(y); q.append(y)
        out.append(sorted(c))
    return sorted(out,key=len,reverse=True),adj

def connected_after_removal(core,adj,hold):
    rem=[x for x in core if x!=hold]
    if not rem: return False
    seen={rem[0]}; q=[rem[0]]
    while q:
        x=q.pop()
        for y in adj[x]:
            if y==hold or y not in rem or y in seen: continue
            seen.add(y); q.append(y)
    return len(seen)==len(rem)

def find_robust_core(component,adj,must={415,540},min_size=6):
    core=set(component)
    changed=True
    while changed and len(core)>=min_size:
        changed=False
        bad=[x for x in sorted(core) if not connected_after_removal(sorted(core),adj,x)]
        if not bad: return sorted(core)
        candidates=[x for x in bad if x not in must]
        if not candidates: break
        x=min(candidates,key=lambda n:len(adj[n]&core)); core.remove(x); changed=True
    if len(core)>=min_size and all(connected_after_removal(sorted(core),adj,x) for x in core): return sorted(core)
    return []

def undistort_pix(uv,f,cx,cy,k1):
    xd=(uv[:,0]-cx)/f; yd=(uv[:,1]-cy)/f; x=xd.copy(); y=yd.copy()
    for _ in range(7):
        r2=x*x+y*y; s=1+k1*r2; s=np.where(np.abs(s)<1e-8,1e-8,s); x=xd/s; y=yd/s
    return np.column_stack([x,y,np.ones_like(x)])

def project_ray(ray,f,cx,cy,k1):
    z=np.where(np.abs(ray[:,2])<1e-8,1e-8,ray[:,2]); x=ray[:,0]/z; y=ray[:,1]/z; r2=x*x+y*y; s=1+k1*r2
    return np.column_stack([f*x*s+cx,f*y*s+cy])

def fit_physical(edges,seed_shared=None):
    frames=[]
    for e in edges:
        for k in ((e['a_event'],e['a_frame']),(e['b_event'],e['b_frame'])):
            if k not in frames: frames.append(k)
    fi={k:i for i,k in enumerate(frames)}; N=len(frames); E=len(edges)
    def unpack(x): return float(x[0]),float(x[1]),float(x[2]),np.exp(x[3:3+N]),x[3+N:].reshape(E,3)
    def residual(x):
        cx,cy,k1,fs,rv=unpack(x); out=[]
        for j,e in enumerate(edges):
            idx=e['train_idx']
            if len(idx)>70: idx=idx[np.linspace(0,len(idx)-1,70).astype(int)]
            fa=fs[fi[(e['a_event'],e['a_frame'])]]; fb=fs[fi[(e['b_event'],e['b_frame'])]]
            rays=undistort_pix(e['p'][idx],fa,cx,cy,k1); R,_=cv2.Rodrigues(rv[j]); pred=project_ray((R@rays.T).T,fb,cx,cy,k1)
            out.append((pred-e['q'][idx]).ravel()/math.sqrt(max(1,len(idx))/40.0))
        out.append(np.array([(cx-W/2)/350.0,(cy-H/2)/350.0,k1/0.5]))
        out.append((np.log(fs)-math.log(1400.0))/2.0)
        return np.concatenate(out)
    K=np.array([[1400.,0,480.],[0,1400.,270.],[0,0,1.]])
    r0=[]
    for e in edges:
        M=np.linalg.inv(K)@np.asarray(e['H'],float)@K; d=np.linalg.det(M); M=M/np.cbrt(abs(d)); M=-M if d<0 else M; U,_,Vt=np.linalg.svd(M); R=U@Vt
        if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
        r,_=cv2.Rodrigues(R); r0.append(r.ravel())
    seeds=[]
    if seed_shared is not None: seeds.append(seed_shared)
    seeds += [(480.,270.,0.0),(480.,270.,0.25),(480.,270.,-0.25),(500.,280.,0.35)]
    lo=np.r_[0.,0.,-.8,np.repeat(math.log(150.),N),np.repeat(-math.pi,E*3)]
    hi=np.r_[960.,540.,.8,np.repeat(math.log(4000.),N),np.repeat(math.pi,E*3)]
    best=None; bestscore=float('inf')
    for sx,sy,sk in seeds:
        x0=np.r_[sx,sy,sk,np.repeat(math.log(1400.),N),np.concatenate(r0)]
        z=least_squares(residual,x0,bounds=(lo,hi),loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=12000)
        sc=float(np.mean(residual(z.x)**2))
        if np.isfinite(sc) and sc<bestscore: bestscore,best=sc,z.x
    if best is None: raise RuntimeError('physical radial fit failed')
    cx,cy,k1,fs,rv=unpack(best)
    held=[]
    for j,e in enumerate(edges):
        idx=e['held_idx']; fa=fs[fi[(e['a_event'],e['a_frame'])]]; fb=fs[fi[(e['b_event'],e['b_frame'])]]
        rays=undistort_pix(e['p'][idx],fa,cx,cy,k1); R,_=cv2.Rodrigues(rv[j]); pred=project_ray((R@rays.T).T,fb,cx,cy,k1); er=np.linalg.norm(pred-e['q'][idx],axis=1)
        held.append({'a_event':e['a_event'],'a_frame':e['a_frame'],'b_event':e['b_event'],'b_frame':e['b_frame'],'error':estats(er),'pass':bool(len(er)>=10 and np.median(er)<=2.5 and np.percentile(er,90)<=4.0)})
    focal_map=[{'event_id':k[0],'frame':k[1],'focal_px':float(fs[i])} for i,k in enumerate(frames)]
    at_bound=any(v['focal_px']<=155.0 or v['focal_px']>=3950.0 for v in focal_map)
    return {'pp':[cx,cy],'k1':k1,'focals':focal_map,'heldout':held,'all_heldout_pass':all(x['pass'] for x in held),'focal_at_bound':at_bound,'score':bestscore}

def serial_edge(e):
    return {k:v for k,v in e.items() if k not in {'p','q','train_idx','held_idx'}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--bank',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--top-event-pairs',type=int,default=50); ap.add_argument('--top-frame-pairs',type=int,default=10); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    paths=sorted(a.bank.glob('event_*_frames/f*.png')); feat=Features(paths); ranking=orb_rank(feat,a.bank)
    pairs=[]
    for _,_,x,y in ranking[:a.top_event_pairs]: pairs.append(tuple(sorted((x,y))))
    pairs=sorted(set(pairs)|{tuple(sorted(x)) for x in FORCE})
    accepted=[]; audit=[]
    for x,y in pairs:
        pa=sorted((a.bank/f'event_{x}_frames').glob('f*.png')); pb=sorted((a.bank/f'event_{y}_frames').glob('f*.png'))
        cand=flann_frame_candidates(feat,pa,pb,a.top_frame_pairs); best=[]
        for _,_,_,p,q in cand:
            z=exact_edge(feat,p,q)
            if z is not None: best.append(z)
        if best:
            best.sort(key=lambda z:(z['withheld_error']['p90_px'],z['withheld_error']['median_px'],-z['training_inliers']))
            accepted.append(best[0]); audit.append({'events':[x,y],'candidate_count':len(cand),'pass':True,'best':serial_edge(best[0])})
        else: audit.append({'events':[x,y],'candidate_count':len(cand),'pass':False})
        print('PAIR',x,y,'PASS' if best else 'FAIL',flush=True)
    events=sorted(set([z['a_event'] for z in accepted]+[z['b_event'] for z in accepted])); comps,adj=components(events,accepted) if events else ([],{})
    component=[]
    for c in comps:
        if 415 in c and 540 in c: component=c; break
    core=find_robust_core(component,adj,{415,540},6) if component else []
    report={'schema_version':1,'game_id':'0022500301','camera_label':'Right Slash','method':'all-event graph search: ORB/FLANN screening only; exact mutual SIFT .72 + model-free local-neighbour topology filter; unchanged transfer gates; shared PP+k1 physical pair model','guardrail':'Screening cannot create a pass. Every accepted edge independently clears the original 24/1.5/10/2.5/4 px gates. One k1 only; no k2/tangential.','ranked_event_pairs':[{'a':x,'b':y,'orb_inliers':int(n),'orb_inlier_ratio':float(r)} for n,r,x,y in ranking],'pair_audit':audit,'accepted_edges':[serial_edge(z) for z in accepted],'components':comps,'component_containing_415_540':component,'robust_core_events':core}
    status='FAIL_RIGHT_SLASH_ALLPAIR_GRAPH_V104'; prior=False
    if len(core)>=6:
        core_edges=[z for z in accepted if z['a_event'] in core and z['b_event'] in core]
        full=fit_physical(core_edges); loo=[]
        for hold in core:
            sub=[z for z in core_edges if hold not in (z['a_event'],z['b_event'])]
            y=fit_physical(sub,seed_shared=(full['pp'][0],full['pp'][1],full['k1']))
            loo.append({'held_out_event':hold,'principal_point_px':y['pp'],'pp_shift_px':float(np.linalg.norm(np.asarray(y['pp'])-np.asarray(full['pp']))),'k1':float(y['k1']),'k1_abs_shift':abs(float(y['k1']-full['k1'])),'all_heldout_pass':y['all_heldout_pass'],'focal_at_bound':y['focal_at_bound']})
        maxpp=max(x['pp_shift_px'] for x in loo); maxdk=max(x['k1_abs_shift'] for x in loo); sign=all(np.sign(x['k1'])==np.sign(full['k1']) for x in loo) and abs(full['k1'])>1e-4
        graphloo=all(connected_after_removal(core,adj,x) for x in core)
        gates={'robust_core_at_least_6_events':len(core)>=6,'event_graph_connected_after_every_single_event_holdout':graphloo,'physical_model_all_heldout_edges_median_p90_pass':full['all_heldout_pass'],'no_full_fit_focal_at_bound':not full['focal_at_bound'],'whole_event_loo_pp_shift_at_most_8px':maxpp<=8.0,'whole_event_loo_k1_same_sign':sign,'whole_event_loo_k1_abs_shift_at_most_0_05':maxdk<=0.05,'loo_fits_no_focal_at_bound':all(not x['focal_at_bound'] for x in loo)}
        prior=bool(all(gates.values())); status='PASS_RIGHT_SLASH_INTRINSICS_PRIOR_V104' if prior else 'FAIL_RIGHT_SLASH_RADIAL_IDENTIFIABILITY_V104'
        report.update({'physical_full_fit':full,'leave_one_event_out':loo,'max_leave_one_event_out_pp_shift_px':maxpp,'max_leave_one_event_out_k1_abs_shift':maxdk,'k1_sign_stable':sign,'gates':gates})
    else:
        report['gates']={'robust_core_at_least_6_events':False}
    report.update({'status':status,'principal_point_prior_allowed':prior,'metric_event_camera_allowed':False,'replay_render_allowed':False})
    def conv(o):
        if isinstance(o,np.generic): return o.item()
        if isinstance(o,np.ndarray): return o.tolist()
        raise TypeError(type(o).__name__)
    (a.out/'right_slash_allpair_radial_v104.json').write_text(json.dumps(report,indent=2,default=conv)+'\n')
    print(json.dumps({'status':status,'accepted_event_edges':len(accepted),'component':component,'core':core,'pp':report.get('physical_full_fit',{}).get('pp'),'k1':report.get('physical_full_fit',{}).get('k1'),'max_pp_loo':report.get('max_leave_one_event_out_pp_shift_px'),'max_k1_loo':report.get('max_leave_one_event_out_k1_abs_shift'),'prior':prior},indent=2),flush=True)

if __name__=='__main__': main()
