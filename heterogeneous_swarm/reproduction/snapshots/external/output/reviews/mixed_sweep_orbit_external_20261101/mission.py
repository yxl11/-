"""Prescribed forward sweeps followed by a complete regional survey orbit.

This module prescribes group centers and reuses the corridor pilot's launch
and initial template scales. Any active intragroup shape motion is separate.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import numpy as np
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'mixed_role_corridor_demo_20261031'
_source = PARENT/'mission.py'
_evidence = json.loads((PARENT/'artifact_manifest.json').read_text(encoding='utf8'))
if hashlib.sha256(_source.read_bytes()).hexdigest() != _evidence['files_sha256']['mission.py']:
    raise RuntimeError('Archived launch configuration changed')
_spec = importlib.util.spec_from_file_location('corridor_prior_launch', _source)
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)
ROLE_COUNTS = (10,16,20)
SCALES, ALTITUDES = prior.SCALES, prior.ALTITUDES
DEFAULT_SEED = prior.DEFAULT_SEED
DURATION = DURATION_S = 178.
CIRCLE_START = CIRCLE_START_S = 102.
PERIOD = PERIOD_S = 72.
CIRCLE_ONE_END_S = CIRCLE_START + PERIOD
RAMP_START = 86.
RAMP_LENGTH = CIRCLE_START-RAMP_START
SPEED = 9.
OMEGA = 2.*np.pi/PERIOD
RADIUS = SPEED/OMEGA
STAGE_BOUNDARIES = [0.,8.,68.,RAMP_START,CIRCLE_START,DURATION]
STAGE_NAMES = ['沿线部署','前方往返扫掠','编队重组','平滑转入','圆周巡查']
WAYPOINT_TIMES = np.asarray([8.,24.,46.,68.,RAMP_START])
# World x/y offsets from (9 t, 0) on the sweep, moving-frame offsets on the orbit.
OFFSETS = np.array([
    [[27.,-4.],[0.,0.],[-20.,0.]],
    [[39.,34.],[2.,4.],[-20.,-6.]],
    [[39.,-34.],[0.,-4.],[-17.,8.]],
    [[39.,34.],[2.,4.],[-20.,-6.]],
    [[34.,-10.],[0.,0.],[-20.,13.]]
])
TURNS = OFFSETS[-1].copy()
CIRCLE_RADII_M = np.hypot(RADIUS-TURNS[:,1], TURNS[:,0])
_GL_POINTS,_GL_WEIGHTS = leggauss(24)

def _blend(u):
    return u**3*(10.+u*(-15.+6.*u))

def _ramp_angle(u):
    return OMEGA*RAMP_LENGTH*u**4*(2.5-3.*u+u**2)

def _integral_xy(delta, nodes=_GL_POINTS, weights=_GL_WEIGHTS):
    # Integrate the prescribed unit tangent, not the aircraft model. Gaussian
    # quadrature is cross-checked against a separate order in reference_precheck.
    sample = delta[...,None]*(nodes+1.)*.5/RAMP_LENGTH
    angle = _ramp_angle(sample)
    values = np.stack((np.cos(angle),np.sin(angle)),axis=-1)
    return SPEED*delta[...,None]*.5*np.einsum('...kj,k->...j',values,weights)

_JOIN_ANGLE = float(_ramp_angle(np.array(1.)))
_ENTRY = np.array([SPEED*RAMP_START,0.])
_JOIN = _ENTRY + _integral_xy(np.array(RAMP_LENGTH))
CIRCLE_CENTER = CIRCLE_CENTER_XY = _JOIN + RADIUS*np.array([-np.sin(_JOIN_ANGLE),np.cos(_JOIN_ANGLE)])

def configuration():
    cfg = prior.configuration()
    cfg.update(counts=list(ROLE_COUNTS), duration_s=DURATION, phase_windows_s=[list(pair) for pair in zip(STAGE_BOUNDARIES[:-1],STAGE_BOUNDARIES[1:])],
               metric_windows_s={'whole':[0.,DURATION],'launch':[0.,8.],'sweep':[8.,68.],
                                 'reshape':[68.,86.],'entry':[86.,102.],'orbit':[102.,178.],
                                 'exact_circle':[102.,174.]},
               snapshot_times_s=[0.,24.,46.,68.,86.,102.,120.,138.,156.,174.,178.])
    return cfg

def initial_state(seed=DEFAULT_SEED):
    if seed != DEFAULT_SEED: raise ValueError('Only preregistered source subset is available')
    with np.load(HERE/'initial_state.npz') as data:
        return data['positions'].copy(),data['velocities'].copy()

def reference_kinematics(t):
    time=np.asarray(t,dtype=float)
    if not np.isfinite(time).all():
        raise ValueError('Nonfinite reference time')
    flat=time.reshape(-1)
    result_shape=time.shape+(3,3)
    q=np.zeros((len(flat),3,3));v=np.zeros_like(q);a=np.zeros_like(q)
    q[...,0]=SPEED*flat[:,None]
    q[...,:2]+=OFFSETS[0]
    q[...,2]=ALTITUDES
    v[...,0]=SPEED
    for start, end, delta in zip(WAYPOINT_TIMES[:-1],WAYPOINT_TIMES[1:],np.diff(OFFSETS,axis=0)):
        u=np.clip((flat-start)/(end-start),0.,1.)
        q[...,:2]+=_blend(u)[:,None,None]*delta
        v[...,:2]+=(30.*u*u*(1.-u)**2/(end-start))[:,None,None]*delta
        a[...,:2]+=(60.*u*(1.-u)*(1.-2.*u)/(end-start)**2)[:,None,None]*delta
    turning=flat>RAMP_START
    if np.any(turning):
        tx=flat[turning]
        ds=np.minimum(tx-RAMP_START,RAMP_LENGTH)
        u=ds/RAMP_LENGTH
        w=OMEGA*_blend(u)
        alpha=OMEGA*30.*u**2*(1.-u)**2/RAMP_LENGTH
        tail=np.maximum(tx-CIRCLE_START,0.)
        angle=_ramp_angle(u)+OMEGA*tail
        tangent=np.stack((np.cos(angle),np.sin(angle)),axis=-1)
        normal=np.stack((-np.sin(angle),np.cos(angle)),axis=-1)
        base=np.empty((len(tx),2))
        on_circle=tx>=CIRCLE_START
        base[on_circle]=CIRCLE_CENTER-RADIUS*normal[on_circle]
        if np.any(~on_circle):
            base[~on_circle]=_ENTRY+_integral_xy(ds[~on_circle])
        T=tangent[:,None,:];N=normal[:,None,:]
        d,e=TURNS[:,0,None],TURNS[:,1,None]
        offset=T*d+N*e
        rotated_offset=N*d-T*e
        q[turning,:,:2]=base[:,None,:]+offset
        v[turning,:,:2]=SPEED*T+w[:,None,None]*rotated_offset
        a[turning,:,:2]=SPEED*w[:,None,None]*N+alpha[:,None,None]*rotated_offset-w[:,None,None]**2*offset
    return tuple(arr.reshape(result_shape) for arr in (q,v,a))

def reference(t,cfg=None):
    q,v,_=reference_kinematics(t)
    return q,v
