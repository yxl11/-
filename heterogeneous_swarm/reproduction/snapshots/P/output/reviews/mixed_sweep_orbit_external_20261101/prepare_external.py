"""Preregister and verify finite moving targets and the two external adapters.

No closed-loop data enters target generation. Grid tests include analytic
acceleration/jerk margins, so geometry and kinematic bounds cover intervals.
"""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,time
from datetime import datetime,timezone
import numpy as np
from mission import configuration,reference_kinematics,reference,initial_state,WAYPOINT_TIMES,OFFSETS,TURNS
from shape_models import shape_axes,BASE_SIGMA,SWEEP_LONG,ORBIT_LONG
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
DT=.02
TIMES=np.arange(8901)*DT
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def ddblend(t,a,b):
    u=np.clip((t-a)/(b-a),0.,1.)
    return 60*u*(1-u)*(1-2*u)/(b-a)**2

def material_second(t):
    R,dR,s,ds=shape_axes(t)
    L=np.log(SWEEP_LONG/BASE_SIGMA[:,0]);M=np.log(ORBIT_LONG/SWEEP_LONG)
    rate=ds[:,0]/s[:,0]
    rate2=ddblend(t,8,24)*L+ddblend(t,68,86)*M
    dds=np.column_stack([s[:,0]*(rate2+rate**2),s[:,1]*(-rate2+rate**2),np.zeros(3)])
    u=np.clip((t-86)/16,0.,1.);omega=2*np.pi/72
    w=omega*u**3*(10-15*u+6*u*u)
    wd=omega*30*u*u*(1-u)**2/16
    J=np.array([[0.,-1,0],[1,0,0],[0,0,0]])
    Rdd=wd*(J@R)+w*w*(J@J@R)
    return R[None]*s[:,None,:],dR[None]*s[:,None,:]+R[None]*ds[:,None,:],Rdd[None]*s[:,None,:]+2*dR[None]*ds[:,None,:]+R[None]*dds[:,None,:]

Q,V,A=reference_kinematics(TIMES)
S,SD,SDD=(np.stack(k) for k in zip(*(material_second(t) for t in TIMES)))

def derivative_bounds(role,z):
    """Global ||x''|| and ||x'''|| upper bounds, from quintic derivative maxima.

    max|blend'|=15/8, max|blend''|=10/sqrt(3), max|blend'''|=60.
    Deformation windows are disjoint and end before the rotation ramp starts.
    """
    lengths=np.diff(WAYPOINT_TIMES)
    shift=np.linalg.norm(np.diff(OFFSETS,axis=0)[:,role],axis=1)
    qa=max((10/np.sqrt(3))*shift/lengths**2)
    qj=max(60*shift/lengths**3)
    L=abs(np.log(SWEEP_LONG[role]/BASE_SIGMA[role,0]));M=abs(np.log(ORBIT_LONG[role]/SWEEP_LONG[role]))
    d1=max(L*1.875/16,M*1.875/18)
    d2=max(L*(10/np.sqrt(3))/16**2,M*(10/np.sqrt(3))/18**2)
    d3=max(L*60/16**3,M*60/18**3)
    sigmax=max(BASE_SIGMA[role].max(),SWEEP_LONG[role],BASE_SIGMA[role,0]*BASE_SIGMA[role,1]/SWEEP_LONG[role],ORBIT_LONG[role],BASE_SIGMA[role,0]*BASE_SIGMA[role,1]/ORBIT_LONG[role])
    zn=np.linalg.norm(z)
    preA=qa+sigmax*(d2+d1*d1)*zn
    preJ=qj+sigmax*(d3+3*d1*d2+d1**3)*zn
    w=2*np.pi/72;wd=w*1.875/16;wdd=w*(10/np.sqrt(3))/16**2
    o=np.linalg.norm(TURNS[role]);r=ORBIT_LONG[role]
    ss=max(r,BASE_SIGMA[role,0]*BASE_SIGMA[role,1]/r,BASE_SIGMA[role,2])
    turnA=9*w+(wd+w*w)*(o+ss*zn)
    turnJ=9*(wd+w*w)+(wdd+3*w*wd+w**3)*(o+ss*zn)
    return max(preA,turnA),max(preJ,turnJ)

def trajectory(role,z):
    return Q[:,role]+S[:,role]@z,V[:,role]+SD[:,role]@z,A[:,role]+SDD[:,role]@z

def envelope(role,z,pos,vel,acc):
    boundA,boundJ=derivative_bounds(role,z);delta=DT/2
    speed=np.linalg.norm(vel,axis=1)
    low=float(speed.min()-boundA*delta);high=float(speed.max()+boundA*delta)
    result={'speed_lower_mps':low,'speed_upper_mps':high,'acceleration_global_bound_mps2':float(boundA),'jerk_global_bound_mps3':float(boundJ)}
    if not (np.max(np.abs(vel[:,2]))<1e-12 and np.max(np.abs(acc[:,2]))<1e-12):raise RuntimeError('Expected constant-altitude targets')
    if role==0:
        if low<=0:return False,result
        tang=np.abs(np.sum(vel*acc,axis=1)/speed)
        yaw=np.abs(vel[:,0]*acc[:,1]-vel[:,1]*acc[:,0])/speed**2
        tang_bound=float(tang.max()+(boundJ+2*boundA**2/low)*delta)
        yaw_bound=float(yaw.max()+(boundJ/low+2*boundA**2/low**2)*delta)
        result.update(tangential_acceleration_upper_mps2=tang_bound,yaw_rate_upper_deg_s=yaw_bound*180/np.pi,
                      inclination_deg=0.,inclination_rate_deg_s=0.)
        return low>=8 and high<=15 and tang_bound<=2 and yaw_bound<=np.pi/18,result
    acceleration=float(np.linalg.norm(acc,axis=1).max()+boundJ*delta)
    result['acceleration_upper_mps2']=acceleration
    return high<=[15,12,10][role] and acceleration<=[2,1.5,1][role],result

def pair_bound(p,v,pa,q,w,qa):
    d=np.linalg.norm(p-q,axis=-1)
    margin=np.linalg.norm(v-w,axis=-1)*(DT/2)+.5*(pa+qa)*(DT/2)**2
    return float(np.min(d-margin))

def generate(spec):
    rng=np.random.default_rng(spec['seed']);zs=[];roles=[];paths=[];vels=[];bounds=[];kin=[];attempts=[]
    rejects={'truncation':0,'kinematic':0,'geometry':0}
    began=time.perf_counter()
    for role,n in enumerate(configuration()['counts']):
        for j in range(n//2):
            for attempt in range(1,spec['proposals_per_pair']+1):
                z=rng.normal(size=3)
                if np.any(abs(z)>spec['normal_truncation']):rejects['truncation']+=1;continue
                trial=[];valid=True
                for sign in [1,-1]:
                    zz=sign*z;p,v,a=trajectory(role,zz);ok,k=envelope(role,zz,p,v,a)
                    if not ok:valid=False;break
                    trial.append((zz,p,v,k))
                if not valid:rejects['kinematic']+=1;continue
                for zz,p,v,k in trial:
                    aa=k['acceleration_global_bound_mps2']
                    if paths:
                        for op,ov,oa in zip(paths,vels,bounds):
                            if pair_bound(p,v,aa,op,ov,oa)<spec['minimum_m']:
                                valid=False;break
                    if not valid:break
                if valid:
                    _,p,v,k=trial[0];_,q,w,l=trial[1]
                    valid=pair_bound(p,v,k['acceleration_global_bound_mps2'],q,w,l['acceleration_global_bound_mps2'])>=spec['minimum_m']
                if not valid:rejects['geometry']+=1;continue
                for zz,p,v,k in trial:
                    zs.append(zz);paths.append(p);vels.append(v);bounds.append(k['acceleration_global_bound_mps2']);roles.append(role);kin.append(k)
                attempts.append({'role':role,'pair':j,'proposals':attempt});break
            else:raise RuntimeError(f'Predeclared geometry/kinematics budget exhausted at role {role} pair {j}')
    z=np.array(zs);roles=np.array(roles);pairs=[]
    for i in range(len(z)):
        for j in range(i):pairs.append(pair_bound(paths[i],vels[i],bounds[i],paths[j],vels[j],bounds[j]))
    np.savez_compressed(BASE/'targets.npz',material_points=z,roles=roles)
    meta={'status':'PASS','generation_seconds':time.perf_counter()-began,'rejections':rejects,'attempts':attempts,
          'minimum_continuous_target_distance_lower_m':min(pairs),'target_kinematics_by_member':kin,
          'role_material_means':[z[roles==i].mean(0).tolist() for i in range(3)],
          'target_region_fraction_by_role':[float(np.mean(np.sum(z[roles==i]**2,axis=1)<=4)) for i in range(3)],
          'targets_sha256':sha(BASE/'targets.npz'),'continuity_scope':'Nearest endpoint grid radius 0.01 s with global analytic acceleration/jerk bounds; all phases included. Targets stay at fixed individual altitudes.'}
    save(BASE/'target_checks.json',meta)
    return meta

def check_adapters():
    import external_adapters as ex
    cfg=configuration();r,v=initial_state();params={'RAS08':{'alpha':.2,'coupling':.2},'MMD19':{'gamma':10.}}
    controllers={m:ex.factory(m,cfg,r,p) for m,p in params.items()}
    ras=controllers['RAS08'];mmd=controllers['MMD19']
    ts=[0.,16.,46.,77.,94.,138.,178.]
    deriv=zero=gradient=static=transport=rotation=0.
    rng=np.random.default_rng(917)
    representation=[]
    # Existing exact common KDE metric and 64-direction shape metric.
    from p_feedback import load_frozen
    from shape_feedback import ShapeFeedback
    _,h,density=load_frozen();metrics=ShapeFeedback(h,density,cfg,reference)
    for t in ts:
        des,dv=ras.desired(t)
        zero=max(zero,float(np.max(np.abs(ras.command(t,des)-dv))))
        eps=1e-4;dp=ras.desired(t+eps)[0];dm=ras.desired(t-eps)[0]
        if t>0 and t<178:deriv=max(deriv,float(np.max(np.abs((dp-dm)/(2*eps)-dv))))
        q,qd=reference(t);R,dR,s,ds=shape_axes(t)
        S0,SD0=ex.shape_state(t);local=(des-q[ras.roles])@R
        aff=qd[ras.roles]+local@dR.T+(local*(ds/s)[ras.roles])@R.T
        transport=max(transport,float(np.max(np.abs(aff-dv))))
        x=des+.2*rng.normal(size=des.shape)
        c=mmd.command(t,x);feedback=mmd.last_fields['feedback_velocity']
        for i,g in enumerate(mmd.groups):
            direction=rng.normal(size=(len(g),3));direction/=np.linalg.norm(direction)
            xx=(x[g]-q[i])@R;dd=direction@R;e=1e-5
            loss=lambda pts:ex.mmd.mmd_squared(pts,np.zeros(3),s[i],mmd.lengths[i])
            fd=(loss(xx+e*dd)-loss(xx-e*dd))/(2*e)
            pred=-2*np.sum(feedback[g]*direction)/(len(g)*mmd.gamma)
            gradient=max(gradient,abs(fd-pred))
            # Direct full covariance Gaussian gradient, independently evaluated.
            cov=S0[i]@S0[i].T;ell=mmd.lengths[i];total=cov+ell*ell*np.eye(3)
            dx=x[g]-q[i];emb=ell**3/np.sqrt(np.linalg.det(total))*np.exp(-.5*np.einsum('ni,ij,nj->n',dx,np.linalg.inv(total),dx))
            targetgrad=-emb[:,None]*(dx@np.linalg.inv(total))
            pair=x[g,None]-x[None,g];kernel=np.exp(-np.sum(pair*pair,axis=-1)/(2*ell*ell))
            direct=mmd.gamma*((kernel[:,:,None]*pair/(ell*ell)).mean(1)+targetgrad)
            rotation=max(rotation,float(np.max(abs(direct-feedback[g]))))
        if t==0:
            original=ex.mmd.ExternalMMDController(reference,counts=cfg['counts'],sigma=cfg['sigma_m'],gamma=10)
            static=max(static,float(np.max(abs(original.command(t,x)-c))))
            fixed=ex.formation.FormationController(ras.roles,des-q[ras.roles],reference,.2,.2)
            static=max(static,float(np.max(abs(fixed.command(t,x)-ras.command(t,x)))))
        f=metrics.fast(des,t);slow=metrics.slow(des,t)
        representation.append({'time_s':t,'density_squared_by_role':slow['relative_density_l2_by_role'].tolist(),
          'centered_shape_by_role_m':slow['shape_by_role'].tolist(),'region_fraction_by_role':f['region_by_role'].tolist()})
    assert zero<1e-12 and deriv<1e-6 and gradient<1e-7 and static<1e-12 and transport<1e-12 and rotation<1e-12
    np.savez_compressed(BASE/'assignment.npz',assigned_material_points=ras.assigned_z,assignment=ras.assignment_array,roles=ras.roles)
    return {'status':'PASS','checked_times_s':ts,'RAS_zero_error_reference_max_error_mps':zero,
      'target_derivative_max_error_mps':deriv,'MMD_loss_directional_gradient_max_error':gradient,
      'static_original_core_command_max_error_mps':static,'affine_transport_at_targets_max_error_mps':transport,
      'MMD_rotated_vs_full_covariance_max_error_mps':rotation,'target_representation':representation,
      'identities':{m:c.identity for m,c in controllers.items()},'assignment_sha256':sha(BASE/'assignment.npz')}

def main():
    reg=BASE/'target_registration.json'
    if reg.exists():raise FileExistsError('Target registration already exists; inspect existing attempt before rerun')
    spec={'seed':771923,'normal_truncation':2.6,'minimum_m':2.7,'proposals_per_pair':20000,
      'kinematic_grid_dt_s':DT,'generation_rule':'Independent symmetric Gaussian pairs +/-z, reject coordinate tails, continuous conservative target envelopes and pair spacing; first passing candidate, no trajectory-dependent choice',
      'registered_utc':datetime.now(timezone.utc).isoformat(),'before_trajectory_results':True,
      'derivative_bounds':'Quintic maxima 15/8,10/sqrt(3),60; log-scale and rotating-frame chain rule; nearest grid point radius h/2. Full derivation in adaptation_notes_zh.md.',
      'gain_source_sha256':sha(BASE.parent/'mixed_airframe_paper_comparison_20261030/calibration_selection.json'),
      'gains':{'RAS08':{'alpha':.2,'coupling':.2},'MMD19':{'gamma':10}},
      'new_calibration_performed':False,'fixed_kernel_rule':'RMS initial role sigmas, physical units, constant over this trajectory',
      'sources_sha256':{p.name:sha(p) for p in [Path(__file__),BASE/'external_adapters.py',BASE/'mission.py',BASE/'shape_models.py']}}
    save(reg,spec)
    try:
        geometry=generate(spec); result=check_adapters()
        result['target_geometry']= {'minimum_continuous_target_distance_lower_m':geometry['minimum_continuous_target_distance_lower_m'],'target_region_fraction_by_role':geometry['target_region_fraction_by_role']}
        result['source_hashes']={name:sha(BASE/name) for name in ['external_adapters.py','prepare_external.py','targets.npz','target_registration.json','target_checks.json','assignment.npz']}
        save(BASE/'adapter_precheck.json',result);print(json.dumps(result,ensure_ascii=False))
    except Exception as e:
        import traceback
        save(BASE/'adapter_precheck.json',{'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
