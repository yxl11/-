"""Published feedback cores adapted to prescribed affine moving role targets.

RAS08 keeps the individual desired-velocity/error/Laplacian law. MMD19 keeps
the deterministic physical-RBF witness gradient, plus known template transport.
Neither adapter contains P's density graph, mean correction or projection.
"""
import os,sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,time
import numpy as np
from scipy.optimize import linear_sum_assignment
from mission import reference
from shape_models import shape_axes,shape_state

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
SOURCE=ROOT/'output/simulations/three_group_moving_comparison_20261023/src'
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,SOURCE/file)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj);return obj
formation=module('registered_formation_core','external_formation.py')
mmd=module('registered_mmd_core','external_mmd.py')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class AffineFormation(formation.FormationController):
    def __init__(self,cfg,initial,alpha,coupling):
        began=time.perf_counter()
        roles=np.repeat(np.arange(3),cfg['counts'])
        with np.load(BASE/'targets.npz') as f:
            z=f['material_points'].copy(); assert np.array_equal(roles,f['roles'])
        q=reference(0.)[0];S=shape_state(0.)[0]
        positions=q[roles]+np.einsum('nij,nj->ni',S[roles],z)
        mapping=np.empty(len(roles),int)
        for g in range(3):
            ids=np.flatnonzero(roles==g)
            costs=np.sum((initial[ids,None,:]-positions[None,ids,:])**2,axis=-1)
            row,col=linear_sum_assignment(costs);mapping[ids[row]]=ids[col]
        self.assigned_z=z[mapping];self.assignment_array=mapping
        self.assignment_seconds=time.perf_counter()-began
        super().__init__(roles,np.zeros_like(initial),reference,alpha,coupling)
        self.roles=roles;self.last_fields={}
        self.identity={'method':'RAS08','identity':'Ren-Sorensen2008 Eq6 velocity-law adaptation with time-varying affine member targets; all-leader exact-reference specialization; 3D component extension',
            'original_core_sha256':sha(SOURCE/'external_formation.py'),
            'parameters':{'alpha':alpha,'coupling':coupling},'targets_sha256':sha(BASE/'targets.npz'),
            'graph':self.graph_metadata,'assignment_seconds':self.assignment_seconds,
            'no_P_feedback_added':True,'theory_scope':'Affine shape change and mixed execution are adaptation conditions, not the source rigid-frame single-integrator guarantee.'}
    def desired(self,t):
        q,qd=reference(t);S,Sd=shape_state(t)
        return (q[self.roles]+np.einsum('nij,nj->ni',S[self.roles],self.assigned_z),
                qd[self.roles]+np.einsum('nij,nj->ni',Sd[self.roles],self.assigned_z))
    def command(self,t,x,v=None):
        target,td=self.desired(t);e=np.asarray(x)-target
        feedback=-self.alpha*e-self.coupling*(self.laplacian@e)
        self.last_fields={'transport_velocity':td.copy(),'feedback_velocity':feedback.copy(),
                          'desired_position':target,'desired_velocity':td}
        return td+feedback

class AffineMMD:
    def __init__(self,cfg,gamma):
        self.roles=np.repeat(np.arange(3),cfg['counts'])
        self.groups=[np.flatnonzero(self.roles==i) for i in range(3)]
        self.gamma=float(gamma)
        self.lengths=np.sqrt(np.mean(np.asarray(cfg['sigma_m'])**2,axis=1))
        self.last_fields={}
        self.identity={'method':'MMD19','identity':'Arbel2019 deterministic empirical-measure MMD flow with instantaneous Gaussian targets and prescribed affine transport',
          'original_core_sha256':sha(SOURCE/'external_mmd.py'),'parameters':{'gamma':self.gamma},
          'kernel_lengths_m':self.lengths.tolist(),'kernel_rule':'Fixed physical RBF length = RMS initial role standard deviations',
          'no_P_feedback_added':True,'theory_scope':'Time-dependent targets and mixed execution do not inherit the static-target source dissipation/convergence statement.'}
    def command(self,t,x,v=None):
        x=np.asarray(x);q,qd=reference(t);R,dR,s,ds=shape_axes(t)
        rel=(x-q[self.roles])@R
        transport=qd[self.roles]+rel@dR.T+(rel*(ds/s)[self.roles])@R.T
        feedback=np.empty_like(x)
        for i,g in enumerate(self.groups):
            feedback[g]=mmd.mmd_velocity(rel[g],np.zeros(3),s[i],self.lengths[i],self.gamma)@R.T
        self.last_fields={'transport_velocity':transport,'feedback_velocity':feedback}
        return transport+feedback
    def native_losses(self,t,x):
        q=reference(t)[0];R,_,s,_=shape_axes(t)
        values=[mmd.mmd_squared((np.asarray(x)[g]-q[i])@R,np.zeros(3),s[i],self.lengths[i]) for i,g in enumerate(self.groups)]
        return {'mmd_squared_per_role':values,'mmd_squared_equal_role_mean':float(np.mean(values))}

def factory(method,cfg,initial_positions,params):
    params=dict(params)
    if method=='RAS08':
        if set(params)!={'alpha','coupling'}:raise ValueError('Explicit RAS gains required')
        return AffineFormation(cfg,np.asarray(initial_positions),**params)
    if method=='MMD19':
        if set(params)!={'gamma'}:raise ValueError('Explicit MMD gain required')
        return AffineMMD(cfg,**params)
    raise ValueError(method)
