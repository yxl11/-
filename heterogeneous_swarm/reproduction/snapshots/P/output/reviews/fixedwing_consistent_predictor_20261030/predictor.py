"""One frozen, sampled mixed-airframe predictor; no aircraft or fleet validation.

FW states interpolate in airspeed/air-path angles; position integrates that same
curve. Rotary states retain the archived vector-velocity proxy. No imported P
code is changed. All command adaptation is explicit and precedes prediction.
"""
from pathlib import Path
import json, math
import numpy as np

BASE=Path(__file__).resolve().parent
PRO=json.loads((BASE/'protocol.json').read_text(encoding='utf-8'))
H=PRO['prediction_update_s']; B=np.array(PRO['constant_common_backup_velocity_mps'])
F=PRO['fixed_wing_model']; RAD=np.pi/180
LO,HI=F['speed_range_mps']; GMAX=F['flight_path_angle_abs_max_deg']*RAD
T=np.array(list(F['time_constants_s'].values()))
RATE=np.array([F['rate_limits']['speed_mps2'],F['rate_limits']['airpath_azimuth_deg_s']*RAD,F['rate_limits']['flight_path_angle_deg_s']*RAD])
ROLES=np.r_[np.zeros(36,int),np.ones(24,int),np.full(20,2,int)]
FW=np.flatnonzero(ROLES==0); ROT=np.flatnonzero(ROLES!=0)
VMAX=np.array([15.,12.,10.])[ROLES]
TAU=np.array([.4,.6,.8])[ROLES]; ACC=np.array([2.,1.5,1.])[ROLES]
PAIR=np.triu_indices(len(ROLES),1)
PAD=PRO['position_numerical_padding_per_step_m']; TOL=PRO['certificate_tolerance_m']; SAFE=PRO['safe_distance_m']

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def angles(v):
    return np.stack([np.linalg.norm(v,axis=-1),np.arctan2(v[...,1],v[...,0]),np.arctan2(v[...,2],np.linalg.norm(v[...,:2],axis=-1))],axis=-1)

def velocity(q):
    s,c,g=np.moveaxis(q,-1,0)
    return s[...,None]*np.stack([np.cos(g)*np.cos(c),np.cos(g)*np.sin(c),np.sin(g)],axis=-1)

def map_command(c):
    c=np.asarray(c,float).copy(); vbefore=c.copy()
    if np.any(np.linalg.norm(c[FW,:2],axis=1)<=1e-9):
        raise ValueError('FW command horizontal direction undefined; no zero-vector fallback')
    q=angles(c[FW]); q[:,0]=np.clip(q[:,0],LO,HI); q[:,2]=np.clip(q[:,2],-GMAX,GMAX)
    c[FW]=velocity(q)
    c[ROT]*=np.minimum(1,VMAX[ROT]/np.maximum(np.linalg.norm(c[ROT],axis=1),1e-300))[:,None]
    change=np.linalg.norm(c-vbefore,axis=1)
    before=angles(vbefore[FW])
    return c,{'changed_FW_count':int(np.sum(change[FW]>1e-9)),
      'FW_below_floor_count':int(np.sum(before[:,0]<LO-1e-9)),
      'FW_above_speed_cap_count':int(np.sum(before[:,0]>HI+1e-9)),
      'FW_above_inclination_count':int(np.sum(np.abs(before[:,2])>GMAX+1e-9)),
      'FW_mapping_change_max_mps':float(change[FW].max()),
      'FW_mapping_change_rms_mps':float(np.sqrt(np.mean(change[FW]**2))),
      'all_command_change_max_mps':float(change.max())}

def exp_integral(s,a,theta,omega,t):
    """Integral_0^t (s+a*u) exp(i*(theta+omega*u)) du, Taylor degree 8."""
    z=1j*omega*t; term=np.ones_like(z,dtype=complex); ans=np.zeros_like(z,dtype=complex)
    for k in range(9):
        ans+=term*t*(s/(k+1)+a*t/(k+2))
        term*=z/(k+1)
    return np.exp(1j*theta)*ans

def fw_displacement(q,dq,t):
    s,c,g=q.T; a,dc,dg=dq.T
    plus=exp_integral(s,a,c+g,dc+dg,t); minus=exp_integral(s,a,c-g,dc-dg,t)
    vertical=exp_integral(s,a,g,dg,t)
    return np.stack([.5*(plus.real+minus.real),.5*(plus.imag+minus.imag),vertical.imag],axis=1)

def make_segment(v,cmd):
    q=angles(v[FW]); target=angles(cmd[FW]); err=target-q; err[:,1]=wrap(err[:,1])
    demand=err/T; dq=np.clip(demand,-RATE,RATE)
    qn=q+H*dq
    vn=v.copy(); vn[FW]=velocity(qn)
    diff=cmd[ROT]-v[ROT]; norm=np.linalg.norm(diff,axis=1)
    alpha=np.minimum(H/TAU[ROT],ACC[ROT]*H/np.maximum(norm,1e-300))
    vn[ROT]=v[ROT]+alpha[:,None]*diff
    ar=(vn[ROT]-v[ROT])/H
    amax=np.empty(len(v)); sm=np.maximum(q[:,0],qn[:,0])
    amax[FW]=np.sqrt(dq[:,0]**2+sm**2*(dq[:,1]**2+dq[:,2]**2))
    amax[ROT]=np.linalg.norm(ar,axis=1)
    angle_change=(np.abs(dq[:,1])+np.abs(dq[:,2]))*H
    remainder=3*H*(np.abs(q[:,0])+H*np.abs(dq[:,0]))*np.exp(angle_change)*angle_change**9/math.factorial(9)
    numerical=np.full(len(v),PAD);numerical[FW]+=remainder
    return {'q':q,'dq':dq,'qn':qn,'v':v,'vn':vn,'ar':ar,'amax':amax,'numerical':numerical,
      'FW_rate_saturated_counts':np.sum(np.abs(demand)>RATE+1e-12,axis=0)}

def displacement(seg,t):
    dr=np.empty_like(seg['v']); dr[FW]=fw_displacement(seg['q'],seg['dq'],t)
    dr[ROT]=t*seg['v'][ROT]+.5*t*t*seg['ar']
    return dr

def chord_min(a,b):
    d=b-a; f=np.clip(-np.sum(a*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-300),0,1)
    return np.linalg.norm(a+f[...,None]*d,axis=-1)

def scalar_area(e,tau,a):
    """Exact integrated |error| in this discrete model, including infinite tail."""
    e=np.maximum(np.asarray(e),0)
    m=np.ceil(np.maximum(0,e-a*tau)/(a*H))
    rem=np.maximum(0,e-m*a*H)
    return H*(m*e-.5*a*H*m*m)+(tau-H/2)*rem

def tail_path(v):
    q=angles(v[FW]); e=np.stack([np.abs(q[:,0]-B[0]),np.abs(wrap(q[:,1])),np.abs(q[:,2])],axis=1)
    area=scalar_area(e,T,RATE)
    out=np.empty(len(v));out[FW]=area[:,0]+B[0]*(area[:,1]+area[:,2])
    out[ROT]=scalar_area(np.linalg.norm(v[ROT]-B,axis=1),TAU[ROT],ACC[ROT])
    return out

def check_segment(r,seg,errors):
    """Continuous curve certificate; refinements don't change the dynamics."""
    rn=r+displacement(seg,H); p,j=PAIR
    d0=r[p]-r[j]; d1=rn[p]-rn[j]
    ep=errors+seg['numerical']
    allowances=ep[p]+ep[j]
    lower=chord_min(d0,d1)-(seg['amax'][p]+seg['amax'][j])*H*H/8-allowances
    dist0=np.linalg.norm(d0,axis=1);dist1=np.linalg.norm(d1,axis=1)
    sample_min=np.minimum(dist0,dist1); sample_fraction=(dist1<dist0).astype(float)
    uncertain=np.flatnonzero(lower<SAFE+TOL)
    refinement=1
    for count in (2,4,8,16,32,64):
        # A sampled violation is decisive for rejection; other ambiguous pairs get tighter bounds.
        active=uncertain[sample_min[uncertain]+allowances[uncertain]>=SAFE-TOL]
        if not len(active): break
        refinement=count; ia=p[active];ja=j[active];last=r[ia]-r[ja]
        lb=np.full(len(active),np.inf)
        for k in range(1,count+1):
            rr=r+displacement(seg,H*k/count); current=rr[ia]-rr[ja]
            lb=np.minimum(lb,chord_min(last,current)-(seg['amax'][ia]+seg['amax'][ja])*(H/count)**2/8-allowances[active])
            ds=np.linalg.norm(current,axis=1); smaller=ds<sample_min[active]
            sample_min[active]=np.minimum(sample_min[active],ds)
            sample_fraction[active[smaller]]=k/count
            last=current
        lower[active]=lb
        uncertain=active[lb<SAFE+TOL]
        if not len(uncertain):break
    k=int(np.argmin(lower));ks=int(np.argmin(sample_min))
    witness=bool(np.any(sample_min+allowances<SAFE-TOL))
    return rn,ep,{'lower_m':float(lower[k]),'lower_pair':[int(p[k]),int(j[k])],
      'sample_min_m':float(sample_min[ks]),'sample_pair':[int(p[ks]),int(j[ks])],
      'sample_time_fraction':float(sample_fraction[ks]),'sampled_violation':witness,
      'refinement':refinement,'uncertified_pairs':int(np.sum(lower<SAFE+TOL))}

def initial_check(r,v):
    q=angles(v[FW]);distance=float(np.linalg.norm(r[PAIR[0]]-r[PAIR[1]],axis=1).min())
    reasons=[]
    if distance<SAFE+TOL:reasons.append('initial_pair_distance')
    if np.any(np.linalg.norm(v,axis=1)>VMAX+1e-9):reasons.append('initial_upper_speed')
    if np.any(q[:,0]<LO-1e-9):reasons.append('initial_FW_lower_speed')
    if np.any(np.abs(q[:,2])>GMAX+1e-9):reasons.append('initial_FW_inclination')
    return {'eligible':not reasons,'reasons':reasons,'spacing_m':distance,'FW_speed_min_mps':float(q[:,0].min()),
      'FW_speed_max_mps':float(q[:,0].max()),'FW_inclination_abs_max_deg':float(np.abs(q[:,2]).max()/RAD)}

def predict(r0,v0,command,keep_profile=False):
    r0=np.asarray(r0,dtype=float);v0=np.asarray(v0,dtype=float)
    mapped,mapping=map_command(command);common,_=map_command(np.broadcast_to(B,v0.shape))
    r=r0.copy();v=v0.copy();err=np.zeros(len(v));n=1+round(PRO['explicit_backup_window_s']/H)
    lower=[];samples=[];times=[];refinements=[];tail_initial=None
    R=[r.copy()];V=[v.copy()];minimum=np.inf;minmeta=None;samplemin=np.inf;samplemeta=None
    sampled_violation=False;rate_saturation=np.zeros(3,int);maxrates=np.zeros(3);FWspeed=[np.inf,-np.inf];gammax=0.;vzmax=0.;maxa=0.; minhor=np.inf
    first_actual=None
    for k in range(n):
        seg=make_segment(v,mapped if k==0 else common)
        rn,err,cert=check_segment(r,seg,err)
        lower.append(cert['lower_m']);samples.append(cert['sample_min_m']);times.append((k+1)*H);refinements.append(cert['refinement'])
        if cert['lower_m']<minimum:
            minimum=cert['lower_m'];minmeta={'segment':k,'start_s':k*H,'end_s':(k+1)*H,'pair':cert['lower_pair'],'refinement':cert['refinement']}
        if cert['sample_min_m']<samplemin:
            samplemin=cert['sample_min_m'];samplemeta={'segment':k,'time_s':(k+cert['sample_time_fraction'])*H,'pair':cert['sample_pair']}
        sampled_violation|=cert['sampled_violation']
        rate_saturation+=seg['FW_rate_saturated_counts'];maxrates=np.maximum(maxrates,np.abs(seg['dq']).max(axis=0))
        fwq=np.concatenate([seg['q'],seg['qn']]);FWspeed[0]=min(FWspeed[0],fwq[:,0].min());FWspeed[1]=max(FWspeed[1],fwq[:,0].max());gammax=max(gammax,np.abs(fwq[:,2]).max())
        # Bound vertical speed over each segment, not just endpoints.
        vzmax=max(vzmax,float(np.maximum(seg['q'][:,0],seg['qn'][:,0]).max()*np.sin(np.abs(fwq[:,2]).max())))
        minhor=min(minhor,float(fwq[:,0].min()*np.cos(np.abs(fwq[:,2]).max())))
        maxa=max(maxa,float(seg['amax'][FW].max()))
        r=rn;v=seg['vn']
        if k==0: first_actual=v.copy();tail_initial=tail_path(v)
        if keep_profile:R.append(r.copy());V.append(v.copy())
    tail=tail_path(v);p,j=PAIR
    # Include the entire remaining backup in reported motion envelopes as well
    # as in the pair certificate. These diagnostics do not change any trajectory.
    qt=angles(v[FW]);tail_speed=np.maximum(qt[:,0],B[0])
    tail_rates=np.minimum(np.stack([np.abs(qt[:,0]-B[0]),np.abs(wrap(qt[:,1])),np.abs(qt[:,2])],axis=1)/T,RATE)
    tail_acc=np.sqrt(tail_rates[:,0]**2+tail_speed**2*(tail_rates[:,1]**2+tail_rates[:,2]**2))
    maxa=max(maxa,float(tail_acc.max()))
    vzmax=max(vzmax,float(np.max(tail_speed*np.sin(np.abs(qt[:,2])))))
    minhor=min(minhor,float(np.min(np.minimum(qt[:,0],B[0])*np.cos(np.abs(qt[:,2])))))
    tail_bounds=np.linalg.norm(r[p]-r[j],axis=1)-tail[p]-tail[j]-err[p]-err[j]
    kt=int(np.argmin(tail_bounds));tail_lower=float(tail_bounds[kt]);bound=min(minimum,tail_lower)
    status='certified' if bound>=SAFE+TOL else ('sampled_violation' if sampled_violation else 'unresolved_bound')
    # Backup scalar errors are monotonic, so all later speeds/angles remain within their last values and b.
    info={'status':status,'mapping':mapping,'continuous_distance_lower_m':minimum,'continuous_minimum_location':minmeta,
      'sampled_distance_min_m':samplemin,'sampled_minimum_location':samplemeta,'sampled_violation':bool(sampled_violation),
      'tail_distance_lower_m':tail_lower,'tail_limiting_pair':[int(p[kt]),int(j[kt])],
      'certificate_margin_m':bound-SAFE,'numerical_position_allowance_max_m':float(err.max()),
      'FW_tail_path_bound_max_m':float(tail[FW].max()),'all_tail_path_bound_max_m':float(tail.max()),
      'FW_initial_backup_path_bound_max_m':float(tail_initial[FW].max()),
      'FW_complete_speed_range_mps':[float(min(FWspeed[0],B[0])),float(max(FWspeed[1],B[0]))],
      'FW_complete_inclination_abs_max_deg':float(gammax/RAD),'FW_vertical_speed_conservative_max_mps':vzmax,
      'FW_horizontal_speed_lower_mps':minhor,'FW_max_tangential_acceleration_mps2':float(maxrates[0]),
      'FW_max_airpath_turn_rate_deg_s':float(maxrates[1]/RAD),'FW_max_inclination_rate_deg_s':float(maxrates[2]/RAD),
      'FW_total_acceleration_conservative_max_mps2':maxa,'FW_rate_saturated_agent_steps':rate_saturation.tolist(),
      'max_distance_subdivisions':max(refinements),'predicted_segments':n}
    # Preserve explicitly distinct input and actual layers.
    out={'target':mapped,'first_actual':first_actual,'lower':np.array(lower),'samples':np.array(samples),'times':np.array(times),'refinements':np.array(refinements)}
    if keep_profile:out.update(R=np.array(R),V=np.array(V))
    return info,out
