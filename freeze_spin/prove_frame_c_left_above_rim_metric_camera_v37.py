from __future__ import annotations
import argparse,json,math
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import least_squares
from freeze_spin.audit_game_camera_registry_preflight_v1 import sift_points
W,H=960,540; FT=30.48; IN=2.54

def k(f,pp): return np.array([[f,0,pp[0]],[0,f,pp[1]],[0,0,1.]],float)
def decomp(h,cx,cy):
 h=np.asarray(h,float); h1,h2,h3=h[:,0],h[:,1],h[:,2]; a1=np.array([h1[0]-cx*h1[2],h1[1]-cy*h1[2]]);a2=np.array([h2[0]-cx*h2[2],h2[1]-cy*h2[2]]);c=[]
 if abs(h1[2]*h2[2])>1e-12:
  q=-(a1@a2)/(h1[2]*h2[2]); c += [q] if q>0 else []
 d=h1[2]**2-h2[2]**2
 if abs(d)>1e-12:
  q=-(a1@a1-a2@a2)/d; c += [q] if q>0 else []
 f=math.sqrt(float(np.median(c)));ki=np.linalg.inv(k(f,[cx,cy]));q1,q2,q3=ki@h1,ki@h2,ki@h3;l=2/(np.linalg.norm(q1)+np.linalg.norm(q2));r0=np.c_[l*q1,l*q2,np.cross(l*q1,l*q2)];u,_,v=np.linalg.svd(r0);R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 rv,_=cv2.Rodrigues(R);return f,rv.ravel()

def curves():
 bx=15*IN;r3=23.75*FT;yc=22*FT;xt=bx+math.sqrt(r3*r3-yc*yc);fx=15*FT;rf=6*FT;n=700
 y=np.linspace(-yc,yc,n); out={'three_point_arc':np.c_[bx+np.sqrt(r3*r3-y*y),y,np.zeros(n)]}
 t=np.linspace(-math.pi/2,math.pi/2,n);out['free_throw_front_semicircle']=np.c_[fx+rf*np.cos(t),rf*np.sin(t),np.zeros(n)]
 x=np.linspace(-15*FT,xt,n);out['left_corner_three_straight']=np.c_[x,np.full(n,-yc),np.zeros(n)];out['right_corner_three_straight']=np.c_[x,np.full(n,yc),np.zeros(n)]
 y=np.linspace(-8*FT,8*FT,n);out['free_throw_line']=np.c_[np.full(n,fx),y,np.zeros(n)];return out
CURVES=curves()
def unpack(p,m):
 rv=p[:3]
 if m=='pinhole':f=np.exp(p[3]);return rv,f,f,p[4],p[5],0.
 if m=='radial':f=np.exp(p[3]);return rv,f,f,p[4],p[5],p[6]
 if m=='anisotropic':return rv,np.exp(p[3]),np.exp(p[4]),p[5],p[6],0.
 return rv,np.exp(p[3]),np.exp(p[4]),p[5],p[6],p[7]
def project(p,m,P,C):
 rv,fx,fy,cx,cy,k1=unpack(p,m);R,_=cv2.Rodrigues(rv.reshape(3,1));z=(R@(P-C).T).T;x=z[:,0]/z[:,2];y=z[:,1]/z[:,2];r2=x*x+y*y;x*=1+k1*r2;y*=1+k1*r2;return np.c_[fx*x+cx,fy*y+cy]
def bounds(m):
 if m=='pinhole':return np.r_[[-10]*3,math.log(250),100,50],np.r_[[10]*3,math.log(2500),850,520]
 if m=='radial':return np.r_[[-10]*3,math.log(250),100,50,-.5],np.r_[[10]*3,math.log(2500),850,520,.5]
 if m=='anisotropic':return np.r_[[-10]*3,math.log(250),math.log(250),100,50],np.r_[[10]*3,math.log(2500),math.log(2500),850,520]
 return np.r_[[-10]*3,math.log(250),math.log(250),100,50,-.5],np.r_[[10]*3,math.log(2500),math.log(2500),850,520,.5]
def init(m,rv,f,pp):
 if m=='pinhole':return np.r_[rv,math.log(f),pp]
 if m=='radial':return np.r_[rv,math.log(f),pp,0]
 if m=='anisotropic':return np.r_[rv,math.log(f),math.log(f),pp]
 return np.r_[rv,math.log(f),math.log(f),pp,0]
def nearest_res(pred,o):return pred[np.argmin(np.linalg.norm(pred-o,axis=1))]-o

def fit_metric(m,C,Hfloor,obs,held,targetP,targetO,warm=None,exclude=None,pert=None,targ=None):
 f0,rv0=decomp(Hfloor,456.8,249.4); seeds=[init(m,rv0,f0,[456.8,249.4]),init(m,rv0,f0,[480,360]),init(m,rv0,f0,[472,365])]
 if warm is not None:seeds=[warm]
 lo,hi=bounds(m); targ=targetO if targ is None else targ
 def res(p):
  z=[]
  for name,oo0 in obs.items():
   if name==exclude:continue
   oo=oo0 if pert is None else pert[name]; pred=project(p,m,CURVES[name],C)
   for i,x in enumerate(oo):
    if i not in held[name]:z.extend(nearest_res(pred,x))
  z.extend((project(p,m,targetP,C)-targ).ravel());return np.asarray(z)
 cand=[]
 for s in seeds:
  o=least_squares(res,s,bounds=(lo,hi),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=8000);cand.append(o)
 return min(cand,key=lambda x:x.cost).x
def metrics(p,m,C,obs,held,targetP,targetO):
 tr=[];ho=[];fam={}
 for n,oo in obs.items():
  pred=project(p,m,CURVES[n],C);e=[]
  for i,x in enumerate(oo):
   q=float(np.min(np.linalg.norm(pred-x,axis=1)));e.append(q);(ho if i in held[n] else tr).append(q)
  fam[n]=e
 te=np.linalg.norm(project(p,m,targetP,C)-targetO,axis=1)
 return {'floor_train_p95_px':float(np.percentile(tr,95)),'floor_heldout_p95_px':float(np.percentile(ho,95)),'floor_heldout_rmse_px':float(np.sqrt(np.mean(np.square(ho)))),'target_p95_px':float(np.percentile(te,95)),'target_rmse_px':float(np.sqrt(np.mean(te**2))),'families':fam}

def action_core(x):return (x[:,0]>.2*W)&(x[:,0]<.8*W)&(x[:,1]>.48*H)&(x[:,1]<.98*H)
def clip_pair(src,target):
 a=cv2.imread(str(src));b=cv2.imread(str(target));p,q=sift_points(a,b)
 if len(p)<30:return None
 tr=(((p[:,1]<.46*H)|(p[:,0]<.14*W)|(p[:,0]>.86*W))&((q[:,1]<.46*H)|(q[:,0]<.14*W)|(q[:,0]>.86*W))&~action_core(p)&~action_core(q));wh=~tr&~action_core(p)&~action_core(q)
 if tr.sum()<24 or wh.sum()<10:return None
 hm,mask=cv2.findHomography(p[tr],q[tr],cv2.RANSAC,1.5,maxIters=30000,confidence=.999)
 if hm is None:return None
 ii=mask.ravel().astype(bool);pi,qi=p[tr][ii],q[tr][ii]; pred=cv2.perspectiveTransform(pi[:,None,:],hm)[:,0];te=np.linalg.norm(pred-qi,axis=1);pred=cv2.perspectiveTransform(p[wh][:,None,:],hm)[:,0];we=np.linalg.norm(pred-q[wh],axis=1)
 if len(pi)<24 or np.percentile(te,95)>1.5 or np.median(we)>2.5 or np.percentile(we,90)>4:return None
 return {'p':pi.astype(float),'q':qi.astype(float),'pw':p[wh].astype(float),'qw':q[wh].astype(float),'H':hm}
def nearest_rot(Hm,pp,ft,fs):
 M=np.linalg.inv(k(ft,pp))@Hm@k(fs,pp);d=np.linalg.det(M);M=M/np.cbrt(abs(d));M=-M if d<0 else M;u,_,v=np.linalg.svd(M);R=u@v
 if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@v
 return cv2.Rodrigues(R)[0].ravel()
def pair_project(p,pp,ft,fs,rv):
 R,_=cv2.Rodrigues(rv.reshape(3,1));hm=k(ft,pp)@R@np.linalg.inv(k(fs,pp));z=cv2.perspectiveTransform(p.astype(np.float32)[:,None,:],hm)[:,0];return z,hm
def validate_clip(target,samples,manifest,pp,ft):
 meta={x['file']:x for x in manifest['samples']};rows=[]
 for src in sorted(samples.glob('Left_Above_Rim_target_event__*.png')):
  pr=clip_pair(src,target)
  if pr is None:continue
  fs0=ft;rv0=nearest_rot(pr['H'],pp,ft,fs0);x0=np.r_[math.log(fs0),rv0]
  def fun(x):return (pair_project(pr['p'],pp,ft,np.exp(x[0]),x[1:])[0]-pr['q']).ravel()
  o=least_squares(fun,x0,bounds=(np.r_[math.log(150),[-10]*3],np.r_[math.log(4000),[10]*3]),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=5000);fs=np.exp(o.x[0]);pt,hm=pair_project(pr['p'],pp,ft,fs,o.x[1:]);et=np.linalg.norm(pt-pr['q'],axis=1);pw,_=pair_project(pr['pw'],pp,ft,fs,o.x[1:]);ew=np.linalg.norm(pw-pr['qw'],axis=1);mm=meta[src.name]
  rows.append({'frame':src.name,'relative_seconds':mm['relative_to_freeze_seconds'],'source_focal_px':float(fs),'train_p95_px':float(np.percentile(et,95)),'withheld_p95_px':float(np.percentile(ew,95)),'withheld_median_px':float(np.median(ew))})
 return rows

def main():
 a=argparse.ArgumentParser();a.add_argument('--target-frame',type=Path,required=True);a.add_argument('--samples',type=Path,required=True);a.add_argument('--sample-manifest',type=Path,required=True);a.add_argument('--wide-court',type=Path,required=True);a.add_argument('--floor-proof',type=Path,required=True);a.add_argument('--registry',type=Path,required=True);a.add_argument('--legacy-landmarks',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--perturbation-trials',type=int,default=24);z=a.parse_args();z.out.mkdir(parents=True,exist_ok=True)
 wide=json.loads(z.wide_court.read_text());floor=json.loads(z.floor_proof.read_text());reg=json.loads(z.registry.read_text());legacy=json.loads(z.legacy_landmarks.read_text());cam=reg['cameras']['Left Above Rim'];assert cam['permissions']['physical_camera_center_allowed'] and floor['status']=='PASS_WIDE_COURT_FLOOR_HOMOGRAPHY_V35';C=np.array(cam['physical_camera_center_prior_cm'],float);Hfloor=np.array(floor['floor_homography_world_to_image'],float);obs={k:np.array(v,float) for k,v in wide['observations_px'].items()};held={k:set(v) for k,v in wide['held_out_indices'].items()}
 lv=next(v for v in legacy['views'] if v['label']=='Left Above Rim');L=lv['landmarks'];targetO=np.array([L[x] for x in ['target_inner_top_left','target_inner_top_right','target_inner_bottom_right','target_inner_bottom_left']],float);targetP=np.array([[0,-10*IN,10*FT+16*IN],[0,10*IN,10*FT+16*IN],[0,10*IN,10*FT+2*IN],[0,-10*IN,10*FT+2*IN]],float)
 comp={}
 for m in ['pinhole','radial','anisotropic','anisotropic_radial']:
  p=fit_metric(m,C,Hfloor,obs,held,targetP,targetO);comp[m]={'params':p.tolist(),'metrics':metrics(p,m,C,obs,held,targetP,targetO)}
 selected='pinhole';p=np.array(comp[selected]['params']);base=comp[selected]['metrics'];loo={}
 for fam in obs:
  q=fit_metric(selected,C,Hfloor,obs,held,targetP,targetO,warm=p,exclude=fam);pred=project(q,selected,CURVES[fam],C);ee=[float(np.min(np.linalg.norm(pred-x,axis=1))) for x in obs[fam]];loo[fam]={'p95_px':float(np.percentile(ee,95)),'max_px':float(max(ee))}
 rng=np.random.default_rng(20260903);ps=[];rvb,fb,_,cxb,cyb,_=unpack(p,selected);Rb,_=cv2.Rodrigues(rvb.reshape(3,1))
 for _ in range(z.perturbation_trials):
  po={k:v+rng.uniform(-.5,.5,v.shape) for k,v in obs.items()};to=targetO+rng.uniform(-.5,.5,targetO.shape);q=fit_metric(selected,C,Hfloor,obs,held,targetP,targetO,warm=p,pert=po,targ=to);mm=metrics(q,selected,C,obs,held,targetP,targetO);rv,f,_,cx,cy,_=unpack(q,selected);R,_=cv2.Rodrigues(rv.reshape(3,1));ang=math.degrees(math.acos(np.clip((np.trace(R@Rb.T)-1)/2,-1,1)));ps.append({'focal_fraction':abs(f-fb)/fb,'pp_shift_px':math.hypot(cx-cxb,cy-cyb),'rotation_deg':ang,'heldout_p95_px':mm['floor_heldout_p95_px'],'target_p95_px':mm['target_p95_px']})
 rv,f,_,cx,cy,_=unpack(p,selected);clip=validate_clip(z.target_frame,z.samples,json.loads(z.sample_manifest.read_text()),np.array([cx,cy]),f);before=sum(x['relative_seconds']<0 for x in clip);after=sum(x['relative_seconds']>0 for x in clip)
 gates={'metric_train_floor':base['floor_train_p95_px']<=2.5,'metric_heldout_floor':base['floor_heldout_p95_px']<=1.5,'metric_target':base['target_p95_px']<=1.0,'leave_family':max(x['p95_px'] for x in loo.values())<=2.5,'perturb_floor':max(x['heldout_p95_px'] for x in ps)<=1.5,'perturb_target':max(x['target_p95_px'] for x in ps)<=1.0,'perturb_focal':max(x['focal_fraction'] for x in ps)<=.005,'perturb_pp':max(x['pp_shift_px'] for x in ps)<=10,'perturb_rotation':max(x['rotation_deg'] for x in ps)<=.8,'clip_frames':len(clip)>=6 and before>=2 and after>=2,'clip_train':len(clip)>=6 and max(x['train_p95_px'] for x in clip)<=2.5,'clip_heldout':len(clip)>=6 and max(x['withheld_p95_px'] for x in clip)<=3.0};passed=all(gates.values())
 rep={'schema_version':1,'status':'PASS_METRIC_EVENT_CAMERA_V37' if passed else 'FAIL_METRIC_EVENT_CAMERA_V37','game_id':'0022500301','event_id':489,'camera_label':'Left Above Rim','physical_center_cm':C.tolist(),'selected_model':selected,'camera':{'rvec':rv.tolist(),'focal_px':float(f),'principal_point_px':[float(cx),float(cy)]},'model_comparison':comp,'metric_selected':base,'leave_one_floor_family_out':loo,'perturbation':{'trials':len(ps),'max_focal_fraction':max(x['focal_fraction'] for x in ps),'max_pp_shift_px':max(x['pp_shift_px'] for x in ps),'max_rotation_deg':max(x['rotation_deg'] for x in ps),'max_heldout_floor_p95_px':max(x['heldout_p95_px'] for x in ps),'max_target_p95_px':max(x['target_p95_px'] for x in ps)},'exact_clip_static_validation':{'accepted_frames':len(clip),'before':before,'after':after,'max_train_p95_px':max([x['train_p95_px'] for x in clip],default=999),'max_withheld_p95_px':max([x['withheld_p95_px'] for x in clip],default=999),'frames':clip},'gates':gates,'permissions':{'metric_event_camera_allowed':passed,'replay_render_allowed':False}}
 (z.out/'frame_c_left_above_rim_metric_camera_v37.json').write_text(json.dumps(rep,indent=2)+'\n');im=cv2.imread(str(z.target_frame));pt=project(p,selected,targetP,C)
 for x in targetO:cv2.circle(im,tuple(np.round(x).astype(int)),5,(0,255,0),2)
 for x in pt:cv2.circle(im,tuple(np.round(x).astype(int)),4,(255,0,255),2)
 cv2.polylines(im,[np.round(pt).astype(np.int32).reshape(-1,1,2)],True,(255,0,255),2);cv2.imwrite(str(z.out/'frame_c_left_above_rim_metric_overlay_v37.png'),im);print(json.dumps({'status':rep['status'],'camera':rep['camera'],'metric':base,'clip':rep['exact_clip_static_validation'],'perturbation':rep['perturbation'],'gates':gates},indent=2));
 if not passed:raise SystemExit(2)
if __name__=='__main__':main()
