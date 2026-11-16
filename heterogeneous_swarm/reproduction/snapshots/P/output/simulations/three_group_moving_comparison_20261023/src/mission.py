"""Three-role continuous-motion mission; final time truncates observation, not motion. No simulation or filesystem side effects."""
from __future__ import annotations
import copy
import numpy as np

CFG = {
    'duration_s':120., 'dt_s':.02, 'task_period_s':.1, 'metric_period_s':.5,
    'counts':[36,24,20], 'names':['scouting','relay_support','supply_transport'],
    'sigma_m':[[5.,8.,3.],[4.,5.,3.],[4.,4.,2.5]],
    'bandwidth_m':[1.7,1.5,1.35], 'vmax_mps':[3.,2.5,2.],
    'amax_mps2':[2.,1.5,1.], 'tau_s':[.4,.6,.8],
    'D_m2ps':2.5, 'kernel_length_m':10., 'A_m2ps':[[0.,22.4,0.],[9.6,0.,12.8],[0.,19.2,0.]],
    'B':[[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],
    'weights':[3./7.,1.,2./3.], 'kappa_m2ps':10., 'role_kernel_length':1.,
    'kc_per_s':.12, 'kf_per_s':.08, 'safe_distance_m':2., 'initial_distance_m':2.7,
    'separation_buffer_m':.6, 'backup_horizon_s':8., 'numerical_margin_m':1e-7,
    'candidate_levels':8, 'initial_shape_scale':.68,
    'test_slowdown_mps':0., 'test_shape_gain_per_s':0., 'test_shape_cap_mps':.8,
    'metric_directions':64, 'metric_direction_seed':9127,
    'thresholds':{'center_m':1.,'relative_m':2.,'shape_proxy_m':2.5,'hold_s':30.,'activation_mps':.02},
    'phase_windows_s':[[0,40],[40,60],[60,100],[100,120]],
    'metric_windows_s':{'whole':[0,120],'maneuver':[60,100],'final_motion':[100,120]},
    'snapshot_times_s':[0,40,60,80,120],
}
roles=np.repeat(np.arange(3),CFG['counts'])
groups=[np.flatnonzero(roles==i) for i in range(3)]

def config(): return copy.deepcopy(CFG)

def smooth(t,a,b):
    u=np.clip((np.asarray(t,dtype=float)-a)/(b-a),0.,1.)
    return 10*u**3-15*u**4+6*u**5, (30*u**2-60*u**3+30*u**4)/(b-a), (60*u-180*u**2+120*u**3)/(b-a)**2

def integral(t,a,b):
    t=np.asarray(t,dtype=float); u=np.clip((t-a)/(b-a),0.,1.)
    return (b-a)*(2.5*u**4-3*u**5+u**6)+np.maximum(t-b,0)

def reference(t):
    t=np.asarray(t,dtype=float)
    s,d,dd=smooth(t,60,100); up,upd,_=smooth(t,40,50)
    q0=np.stack((1.4*integral(t,40,50),20*s,50+0*t),axis=-1)
    v0=np.stack((1.4*up,20*d,0*t),axis=-1)
    a0=np.stack((1.4*upd,20*dd,0*t),axis=-1)
    o=np.stack((18-6*s,12*s,0*t),axis=-1)
    od=np.stack((-6*d,12*d,0*t),axis=-1); odd=np.stack((-6*dd,12*dd,0*t),axis=-1)
    return np.stack((q0+o,q0,q0-o),axis=-2),np.stack((v0+od,v0,v0-od),axis=-2),np.stack((a0+odd,a0,a0-odd),axis=-2)

def qref(t,cfg=None):
    q,qd,_=reference(t); return q,qd
