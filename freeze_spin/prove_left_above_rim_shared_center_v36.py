from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
W,H=960,540; FT=30.48; IN=2.54

def normh(h):
 h=np.asarray(h,float); return h/h[2,2]
def terr(h,p,q):
 if len(p)==0:return np.empty(0)
 z=cv2.perspectiveTransform(p[:,None,:],h.astype(float))[:,0]; return np.linalg.norm(z-q,axis=1)
def sift(a,b):
 s=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.015); ka,da=s.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),None); kb,db=s.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),None)
 if da is None or db is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
 g=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.72*n.distance]; d={}
 for m in g:
  if m.trainIdx not in d or m.distance<d[m.trainIdx].distance:d[m.trainIdx]=m
 g=list(d.values()); return np.float32([ka[m.queryIdx].pt for m in g]),np.float32([kb[m.trainIdx].pt for m in g])
def core(x):
 return (x[:,0]>.2*W)&(x[:,0]<.8*W)&(x[:,1]>.48*H)&(x[:,1]<.98*H)
def audit(src,tgt):
 a=cv2.imread(str(src));b=cv2.imread(str(tgt));r={'source':str(src),'pass':False}
 if a is None or b is None or a.shape[:2]!=(H,W) or b.shape[:2]!=(H,W):return r
 p,q=sift(a,b);r['matches']=len(p)
 if len(p)<30:return r
 tr=(((p[:,1]<.46*H)|(p[:,0]<.14*W)|(p[:,0]>.86*W))&((q[:,1]<.46*H)|(q[:,0]<.14*W)|(q[:,0]>.86*W))&~core(p)&~core(q)); wh=~tr&~core(p)&~core(q)
 if tr.sum()<12:return r
 hm,m=cv2.findHomography(p[tr],q[tr],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 if hm is None:return r
 ii=m.ravel().astype(bool);te=terr(hm,p[tr][ii],q[tr][ii]);we=terr(hm,p[wh],q[wh])
 gates=[ii.sum()>=24,len(te)>0 and np.percentile(te,95)<=1.5,len(we)>=10,len(we)>0 and np.median(we)<=2.5,len(we)>0 and np.percentile(we,90)<=4]
 r.update({'pass':bool(all(gates)),'H':hm,'p':p,'q':q,'tr':tr,'wh':wh,'train_inliers':int(ii.sum()),'train_p95':float(np.percentile(te,95)),'withheld_median':float(np.median(we)),'withheld_p90':float(np.percentile(we,90))})
 return r
def best(paths,tgt):
 rr=[audit(p,tgt) for p in paths];rr.sort(key=lambda r:(1 if r.get('pass') else 0,r.get('train_inliers',0),-r.get('withheld_median',1e9),-r.get('withheld_p90',1e9)),reverse=True);return rr[0]

def decomp(h,cx,cy):
 h1,h2,h3=h[:,0],h[:,1],h[:,2];a1=np.array([h1[0]-cx*h1[2],h1[1]-cy*h1[2]]);a2=np.array([h2[0]-cx*h2[2],h2[1]-cy*h2[2]]);c=[]
 if abs(h1[2]*h2[2])>1e-12:
  x=-(a1@a2)/(h1[2]*h2[2]); c += [x] if x>0 else []
 d=h1[2]**2-h2[2]**2
 if abs(d)>1e-12:
  x=-(a1@a1-a2@a2)/d; c += [x] if x>0 else []
 if not c:return None
 f=math.sqrt(float(np.median(c)));ki=np.array([[1/f,0,-cx/f],[0,1/f,-cy/f],[0,0,1.]]);q1,q2,q3=ki@h1,ki@h2,ki@h3;l=2/(np.linalg.norm(q1)+np.linalg.norm(q2));r0=np.c_[l*q1,l*q2,np.cross(l*q1,l*q2)];u,_,v=np.linalg.svd(r0);R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 C=-R.T@(l*q3);rv,_=cv2.Rodrigues(R);return f,C,rv.ravel()
def grid():return np.array([[x,y,0.] for x in np.linspace(-4*FT,28*FT,9) for y in np.linspace(-25*FT,25*FT,11)])
def hobs(h):
 g=grid();z=(h@np.c_[g[:,:2],np.ones(len(g))].T).T;uv=z[:,:2]/z[:,2,None];m=(z[:,2]>0)&(uv[:,0]>20)&(uv[:,0]<940)&(uv[:,1]>20)&(uv[:,1]<520);return g[m],uv[m]
def solve(hd,multi=True,warm=None,nfev=8000):
 ks=list(hd);n=len(ks);wb={};ob={}
 for k in ks:wb[k],ob[k]=hobs(hd[k])
 starts=[]
 if warm is not None and warm['keys']==ks:starts=[warm['x']]
 else:
  for pp in (((480.,270.),(460.,180.),(480.,400.)) if multi else ((480.,270.),)):
   cc=[];bb=[]
   for k in ks:
    d=decomp(hd[k],*pp)
    if d is None:break
    f,C,rv=d;cc.append(C);bb.append([math.log(f),*rv])
   if len(bb)==n:starts.append(np.r_[np.mean(cc,0),pp,np.array(bb).ravel()])
 def un(x):return x[:3],x[3:5],x[5:].reshape(n,4)
 def fun(x):
  C,pp,r=un(x);o=[]
  for i,k in enumerate(ks):
   f=np.exp(r[i,0]);R,_=cv2.Rodrigues(r[i,1:].reshape(3,1));ca=(R@(wb[k]-C).T).T;uv=np.c_[f*ca[:,0]/ca[:,2]+pp[0],f*ca[:,1]/ca[:,2]+pp[1]];o += [(uv-ob[k]).ravel(),np.minimum(ca[:,2]-20,0)*10]
  return np.concatenate(o)
 lo=np.r_[[-5000,-3000,50],[100,50],np.tile(np.r_[[math.log(250)],[-10,-10,-10]],n)];hi=np.r_[[5000,3000,1500],[850,520],np.tile(np.r_[[math.log(2500)],[10,10,10]],n)]
 oo=[least_squares(fun,x,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=nfev) for x in starts];o=min(oo,key=lambda z:z.cost);C,pp,r=un(o.x);return {'keys':ks,'x':o.x,'C':C,'pp':pp,'r':r,'rms':float(np.sqrt(np.mean(fun(o.x)**2)))}
def fitview(h,C,pp):
 P,O=hobs(h);d=decomp(h,*pp);x=np.r_[math.log(d[0]),d[2]]
 def f(z):
  q=np.exp(z[0]);R,_=cv2.Rodrigues(z[1:].reshape(3,1));ca=(R@(P-C).T).T;return (np.c_[q*ca[:,0]/ca[:,2]+pp[0],q*ca[:,1]/ca[:,2]+pp[1]]-O).ravel()
 o=least_squares(f,x,bounds=(np.r_[[math.log(250)],[-10,-10,-10]],np.r_[[math.log(2500)],[10,10,10]]),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=5000);return o.x,float(np.sqrt(np.mean(f(o.x)**2)))
def gh(t,s,pp):
 ft,fs=np.exp(t[0]),np.exp(s[0]);Rt,_=cv2.Rodrigues(t[1:].reshape(3,1));Rs,_=cv2.Rodrigues(s[1:].reshape(3,1));Kt=np.array([[ft,0,pp[0]],[0,ft,pp[1]],[0,0,1.]]);Ks=np.array([[fs,0,pp[0]],[0,fs,pp[1]],[0,0,1.]]);return normh(Kt@Rt@Rs.T@np.linalg.inv(Ks))
def hold(pair,hm):
 p,q=pair['p'][pair['wh']],pair['q'][pair['wh']];keep=terr(pair['H'],p,q)<=4;p,q=p[keep],q[keep];e=terr(hm,p,q);return {'count':len(e),'median':float(np.median(e)),'p95':float(np.percentile(e,95)),'max':float(np.max(e)),'p':p,'q':q}
def proj(C,pp,m,P):
 f=np.exp(m[0]);R,_=cv2.Rodrigues(m[1:].reshape(3,1));ca=(R@(P-C).T).T;return np.c_[f*ca[:,0]/ca[:,2]+pp[0],f*ca[:,1]/ca[:,2]+pp[1]]

def main():
 a=argparse.ArgumentParser();a.add_argument('--target-frame',type=Path,required=True);a.add_argument('--samples',type=Path,required=True);a.add_argument('--floor-proof',type=Path,required=True);a.add_argument('--events',default='40,220,440,620');a.add_argument('--out',type=Path,required=True);a.add_argument('--perturbation-trials',type=int,default=24);z=a.parse_args();z.out.mkdir(parents=True,exist_ok=True)
 fp=json.loads(z.floor_proof.read_text());assert fp['status']=='PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35' and fp['permissions']['floor_homography_allowed'];ht=normh(fp['floor_homography_world_to_image']);ev=[int(x) for x in z.events.split(',')];sel={};hd={'target':ht}
 for e in ev:
  r=best(sorted(z.samples.glob(f'Left_Above_Rim__event{e:04d}__s*.png')),z.target_frame)
  if not r.get('pass'):raise SystemExit(f'event {e} transfer failed')
  sel[e]=r;hd[e]=normh(np.linalg.inv(r['H'])@ht)
 full=solve(hd);C,pp=full['C'],full['pp'];ti=full['keys'].index('target');tm=full['r'][ti];hv={}
 for e in ev:
  i=full['keys'].index(e);hv[e]=hold(sel[e],gh(tm,full['r'][i],pp))
 loo={}
 for e in ev:
  s=solve({k:v for k,v in hd.items() if k!=e},multi=False,nfev=4000);um,gr=fitview(hd[e],s['C'],s['pp']);hh=hold(sel[e],gh(s['r'][s['keys'].index('target')],um,s['pp']));loo[e]={'center_shift_cm':float(np.linalg.norm(s['C']-C)),'unseen_grid_rms_px':gr,'unseen_static_p95_px':hh['p95']}
 rz=10*FT;oh=18*IN;st=2*IN;hw=20*IN/2;P=np.array([[0,-hw,rz+oh-st],[0,hw,rz+oh-st],[0,hw,rz+st],[0,-hw,rz+st]]);O=np.array([[470.,159.],[491.,159.],[491.,174.],[470.,174.]]);Pr=proj(C,pp,tm,P);ee=np.linalg.norm(Pr-O,axis=1);ep95=float(np.percentile(ee,95))
 rng=np.random.default_rng(20260903);ps=[]
 for t in range(z.perturbation_trials):
  hh={'target':ht}
  for e in ev:
   r=sel[e];p,q=r['p'][r['tr']].copy(),r['q'][r['tr']].copy();k=terr(r['H'],p,q)<=1.5;p,q=p[k],q[k];p+=rng.uniform(-.5,.5,p.shape).astype(np.float32);q+=rng.uniform(-.5,.5,q.shape).astype(np.float32);h,_=cv2.findHomography(p,q,0);hh[e]=normh(np.linalg.inv(h)@ht)
  s=solve(hh,warm=full,nfev=2500);ps.append(float(np.linalg.norm(s['C']-C)))
 mxh=max(hv[e]['p95'] for e in ev);mnh=min(hv[e]['count'] for e in ev);mxl=max(loo[e]['center_shift_cm'] for e in ev);mxu=max(loo[e]['unseen_static_p95_px'] for e in ev);mxp=max(ps);gates={'grid_rms':full['rms']<=.25,'heldout_count':mnh>=15,'heldout_p95':mxh<=2.5,'loo_center':mxl<=25,'unseen_p95':mxu<=2.5,'perturb_center':mxp<=15,'elevated_target':ep95<=8};passed=all(gates.values())
 rep={'schema_version':1,'status':'PASS_SHARED_PHYSICAL_CENTER_V36' if passed else 'FAIL_SHARED_PHYSICAL_CENTER_V36','game_id':fp['game_id'],'camera_label':'Left Above Rim','method':'v35 metric floor + same-game background SIFT homographies -> shared centre/principal point with per-view rotation/focal','guardrail':'v26 baseline anchors and retired v1 centre are excluded; pass allows physical-centre prior only','excluded':['v26 baseline_left_lane','v26 baseline_right_lane','v1 centre [1283.715,-2.343,298.389] cm'],'camera_center_cm':C.tolist(),'principal_point_px':pp.tolist(),'shared_regulation_grid_rms_px':full['rms'],'selected_samples':{str(e):Path(sel[e]['source']).name for e in ev},'heldout_static':{str(e):{k:v for k,v in hv[e].items() if k not in ('p','q')} for e in ev},'max_heldout_static_p95_px':mxh,'leave_one_event_out':{str(e):loo[e] for e in ev},'max_leave_one_event_center_shift_cm':mxl,'max_unseen_event_static_p95_px':mxu,'half_pixel_perturbation':{'trials':len(ps),'max_center_shift_cm':mxp,'p95_center_shift_cm':float(np.percentile(ps,95))},'elevated_target_heldout':{'predicted_px':Pr.tolist(),'observed_px':O.tolist(),'p95_px':ep95,'rmse_px':float(np.sqrt(np.mean(ee**2)))},'gates':gates,'permissions':{'physical_camera_center_allowed':passed,'metric_event_camera_allowed':False,'replay_render_allowed':False}}
 (z.out/'left_above_rim_shared_center_v36.json').write_text(json.dumps(rep,indent=2)+'\n')
 im=cv2.imread(str(z.target_frame));
 for p in O:cv2.circle(im,tuple(np.round(p).astype(int)),5,(0,255,0),2)
 for p in Pr:cv2.circle(im,tuple(np.round(p).astype(int)),4,(255,0,255),2)
 cv2.polylines(im,[np.round(Pr).astype(np.int32).reshape(-1,1,2)],True,(255,0,255),2);cv2.imwrite(str(z.out/'target_elevated_holdout_overlay_v36.png'),im)
 for e in ev:
  im=cv2.imread(sel[e]['source']);i=full['keys'].index(e);hm=gh(tm,full['r'][i],pp);pred=cv2.perspectiveTransform(hv[e]['q'][:,None,:],np.linalg.inv(hm))[:,0]
  for aa,bb in zip(hv[e]['p'],pred):
   aa=tuple(np.round(aa).astype(int));bb=tuple(np.round(bb).astype(int));cv2.circle(im,aa,4,(0,255,0),1);cv2.circle(im,bb,3,(255,0,255),1);cv2.line(im,aa,bb,(255,255,255),1)
  cv2.imwrite(str(z.out/f'event_{e}_heldout_static_overlay_v36.png'),im)
 print(json.dumps({'status':rep['status'],'center_cm':rep['camera_center_cm'],'pp':rep['principal_point_px'],'grid_rms':full['rms'],'max_heldout_p95':mxh,'max_loo_center_shift_cm':mxl,'max_unseen_p95':mxu,'max_perturb_center_shift_cm':mxp,'elevated_target_p95':ep95,'gates':gates},indent=2))
 if not passed:raise SystemExit(2)
if __name__=='__main__':main()
