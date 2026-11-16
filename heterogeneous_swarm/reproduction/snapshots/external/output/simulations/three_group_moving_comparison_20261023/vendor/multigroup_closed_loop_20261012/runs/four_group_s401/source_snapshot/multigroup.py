"""New M-population controller corresponding to the six-slide October 12 design.

This module has no dependency on the old two-population density controller.
Positions are metres, times seconds, K is dimensionless and A,D have m²/s units.
"""
from __future__ import annotations
import copy
import numpy as np
from scipy.special import ndtri


def default_config():
    return {
        'schema_version': 1, 'duration_s': 360., 'dt_s': .02,
        'task_period_s': .1, 'metric_period_s': .5,
        'counts': [36, 24, 20], 'names': ['search', 'relay', 'transport'],
        'sigma_m': [[5., 8., 3.], [4., 5., 3.], [4., 4., 2.5]],
        'bandwidth_m': [1.7, 1.5, 1.35],
        'vmax_mps': [3., 2.5, 2.], 'amax_mps2': [2., 1.5, 1.],
        'tau_s': [.4, .6, .8], 'D_m2ps': 2.5, 'kernel_length_m': 10.,
        'A_m2ps': (32.*np.array([[0., .7, 0.], [.3, 0., .4], [0., .6, 0.]])).tolist(),
        'B': [[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]],
        'kc_per_s': .12, 'kf_per_s': .08, 'cruise_mps': .8,
        'safe_distance_m': 2., 'initial_distance_m': 2.7,
        'separation_buffer_m': .6, 'backup_horizon_s': 8.,
        'numerical_margin_m': 1e-7, 'candidate_levels': 8,
        'initial_shape_scale': .68,
        'initial_centroid_error_m': [[-4.,3.,0.], [2.,-3.,1.], [-6.,-2.,-1.]],
        'test_slowdown_mps': .7, 'test_shape_gain_per_s': .16,
        'test_shape_cap_mps': .8,
        'metric_directions': 64, 'metric_direction_seed': 9127,
        'thresholds': {'center_m': 1., 'relative_m': 2., 'shape_proxy_m': 2.5,
                       'hold_s': 30., 'activation_mps': .02},
        'event_windows_s': [[0,60],[60,110],[110,140],[140,180],[180,240],[240,330],[330,360]],
        'event_names': ['deploy','accompany','transport_slowdown','relay_deformation',
                        'combined_crossing','post_excitation_correction','hold_observation'],
        'model_scope': 'Deterministic group-uniform saturated velocity loop; quadratic interpolation per inner step. No wind, sensor error, obstacle or communication model.',
        'excitation_scope': 'Controlled task-command excitation before safety; not external wind or state teleportation.',
        'density_scope': 'Analytical Gaussian KDE score with self term; K(z)=exp(-|z|²/(2 ell²)), analytical convolution with KDE and unsmoothed Gaussian target. No density floor, no Brownian increment.',
        'safety_scope': 'All pairs including zero A/B edges. Local separation candidates, common task scaling and complete zero-reference braking checks.',
    }


def smooth(t, start, stop):
    u=np.clip((t-start)/(stop-start),0.,1.)
    s=u*u*u*(10.+u*(-15.+6.*u))
    ds=(30.*u*u*(1.-u)*(1.-u)/(stop-start)) if start<t<stop else 0.
    return float(s), float(ds)


def pulse(t, start, stop, ramp=3.):
    a,_=smooth(t,start,start+ramp); b,_=smooth(t,stop-ramp,stop)
    return a-b


def references(t, cfg):
    m=len(cfg['counts'])
    q0=np.array([cfg['cruise_mps']*t,0.,50.]); v0=np.array([cfg['cruise_mps'],0.,0.])
    sy,dsy=smooth(t,60.,85.); ty,dty=smooth(t,85.,110.)
    q0[1]=8.*(sy-ty); v0[1]=8.*(dsy-dty)
    base=np.array([[18.,0.,0.],[0.,0.,0.],[-18.,0.,0.]])
    near=np.array([[5.,-9.,0.],[0.,20.,5.],[-5.,9.,0.]])
    crossed=np.array([[5.,9.,0.],[0.,20.,5.],[-5.,-9.,0.]])
    a,da=smooth(t,180.,205.); b,db=smooth(t,205.,238.); c,dc=smooth(t,240.,285.)
    offsets=base+a*(near-base)+b*(crossed-near)+c*(base-crossed)
    rate=da*(near-base)+db*(crossed-near)+dc*(base-crossed)
    if m>3:
        extras=np.array([[0.,-22.-18.*j,4.*(j%2)] for j in range(m-3)])
        offsets=np.vstack((offsets,extras)); rate=np.vstack((rate,np.zeros_like(extras)))
    return offsets+q0, rate+v0


def cap(x, radius):
    radius=np.asarray(radius)
    return x*np.minimum(1.,radius/np.maximum(np.linalg.norm(x,axis=-1),1e-300))[...,None]


class Controller:
    def __init__(self,cfg,method='full'):
        self.cfg=cfg; self.method=method
        self.counts=np.array(cfg['counts']); self.M=len(self.counts)
        self.roles=np.repeat(np.arange(self.M),self.counts)
        self.groups=[np.where(self.roles==i)[0] for i in range(self.M)]
        self.sigma=np.array(cfg['sigma_m']); self.beta=np.array(cfg['bandwidth_m'])
        self.A=np.array(cfg['A_m2ps']); self.B=np.array(cfg['B'])
        if method in ('no_A','no_AB'): self.A[:]=0.
        if method in ('no_B','no_AB'): self.B[:]=0.
        self.tau=np.array(cfg['tau_s'])[self.roles]
        self.vmax=np.array(cfg['vmax_mps'])[self.roles]
        self.amax=np.array(cfg['amax_mps2'])[self.roles]

    def means(self,x):
        return np.array([x[g].mean(axis=0) for g in self.groups])

    def center(self,x):
        return x-self.means(x)[self.roles]

    def fields(self,r,t):
        q,qd=references(t,self.cfg); centers=self.means(r)
        errors=centers-q
        pair_error=errors[:,None,:]-errors[None,:,:]
        formation=-self.cfg['kf_per_s']*np.einsum('ij,ijk->ik',self.B,pair_error)
        anchor=-self.cfg['kc_per_s']*errors
        nu=qd+anchor+formation
        D=self.cfg['D_m2ps']; ell=self.cfg['kernel_length_m']
        target=-D*(r-q[self.roles])/self.sigma[self.roles]**2
        spreading=np.zeros_like(r); F=np.empty((self.M,len(r),3))
        for j,g in enumerate(self.groups):
            d=r[g,None,:]-r[g][None,:,:]
            exponent=-np.sum(d*d,axis=2)/(2*self.beta[j]**2)
            weight=np.exp(exponent-exponent.max(axis=1,keepdims=True))
            spreading[g]=D*np.einsum('kl,kld->kd',weight,d)/(weight.sum(axis=1)[:,None]*self.beta[j]**2)
            diff=r[:,None,:]-r[g][None,:,:]
            h2=ell**2+self.beta[j]**2
            k=(ell**2/h2)**1.5*np.exp(-np.sum(diff*diff,axis=2)/(2*h2))
            actual=-np.einsum('kl,kld->kd',k,diff)/(len(g)*h2)
            target_diff=r-q[j]; var=ell**2+self.sigma[j]**2
            value=np.prod(ell/np.sqrt(var))*np.exp(-.5*np.sum(target_diff**2/var,axis=1))
            desired=-value[:,None]*target_diff/var
            F[j]=actual-desired
        # edge[k,j,:] is the j-to-i(k) density contribution AFTER sample centering.
        edge=-self.A[self.roles,:,None]*F.transpose(1,0,2)
        for g in self.groups: edge[g]-=edge[g].mean(axis=0,keepdims=True)
        target_centered=self.center(target); spread_centered=self.center(spreading)
        cross=edge.sum(axis=1)
        shape=target_centered+spread_centered+cross
        if self.method=='no_shape': shape[:]=0.; edge[:]=0.; cross[:]=0.
        raw=target+spreading-np.einsum('kj,jkd->kd',self.A[self.roles],F)
        if self.method=='no_center': shape=raw
        cnom=nu[self.roles]+shape
        dcen=np.zeros((self.M,3)); dshape=np.zeros_like(r)
        slowdown=pulse(t,110.,133.)+pulse(t,193.,216.)
        dcen[2,0]=-self.cfg['test_slowdown_mps']*slowdown
        deformation=pulse(t,143.,167.)+pulse(t,197.,223.)
        g=self.groups[1]; rel=r[g]-centers[1]
        raw_exc=np.column_stack((.35*rel[:,1],-rel[:,1],.2*rel[:,2]))*self.cfg['test_shape_gain_per_s']*deformation
        raw_exc-=raw_exc.mean(axis=0)
        raw_exc*=min(1.,self.cfg['test_shape_cap_mps']/max(float(np.linalg.norm(raw_exc,axis=1).max()),1e-300))
        dshape[g]=raw_exc
        excitation=dcen[self.roles]+dshape
        return dict(q=q,qd=qd,centers=centers,nu=nu,anchor=anchor,formation=formation,
                    target=target_centered,spreading=spread_centered,cross=cross,
                    shape=shape,edge=edge,cnom=cnom,excitation=excitation,
                    ctest=cnom+excitation,zero_mean_error=float(np.max(np.abs(self.means(cnom)-nu))),
                    shape_excitation_mean_error=float(np.max(np.abs(dshape[g].mean(axis=0)))))


def initial_state(cfg,seed):
    rng=np.random.default_rng(seed); model=Controller(cfg)
    q,_=references(0.,cfg)
    errors=np.array(cfg['initial_centroid_error_m'])
    for retry in range(100):
        points=[]; okay=True
        for i,n in enumerate(cfg['counts']):
            group=[]
            for attempts in range(20000):
                z=rng.normal(size=3)
                if np.any(np.abs(z)>2.6): continue
                x=q[i]+errors[i]+cfg['initial_shape_scale']*model.sigma[i]*z
                if points+group and np.min(np.linalg.norm(np.array(points+group)-x,axis=1))<cfg['initial_distance_m']: continue
                group.append(x)
                if len(group)==n: break
            if len(group)!=n: okay=False; break
            points+=group
        if okay: return np.array(points),np.zeros((sum(cfg['counts']),3))
    raise RuntimeError('could not construct registered separated initial state')


class Metrics:
    def __init__(self,cfg):
        self.cfg=cfg
        rng=np.random.default_rng(cfg['metric_direction_seed'])
        self.dirs=rng.normal(size=(cfg['metric_directions'],3))
        self.dirs/=np.linalg.norm(self.dirs,axis=1)[:,None]
        self.model=Controller(cfg)
        self.normal_quantiles=[ndtri((np.arange(n)+.5)/n) for n in cfg['counts']]
        self.scales=np.sqrt(np.array(cfg['sigma_m'])**2@self.dirs.T**2)

    def calculate(self,r,t):
        q,_=references(t,self.cfg); m=self.model.means(r)
        err=m-q
        relative=np.linalg.norm(err[:,None,:]-err[None,:,:],axis=-1)
        total=[]; shape=[]
        for i,g in enumerate(self.model.groups):
            projected=np.sort(r[g]@self.dirs.T,axis=0)
            template=self.normal_quantiles[i][:,None]*self.scales[i][None,:]
            total.append(np.sqrt(np.mean((projected-q[i]@self.dirs.T-template)**2)))
            shape.append(np.sqrt(np.mean((projected-m[i]@self.dirs.T-template)**2)))
        return dict(center_error=np.linalg.norm(err,axis=1),relative_error=relative,
                    density_proxy=np.array(total),shape_proxy=np.array(shape))


def four_group_config(cfg):
    c=copy.deepcopy(cfg)
    for key,value in [('counts',16),('names','support'),('sigma_m',[4.,4.,3.]),
                      ('bandwidth_m',1.4),('vmax_mps',2.3),('amax_mps2',1.2),('tau_s',.7),
                      ('initial_centroid_error_m',[3.,2.,0.])]: c[key].append(value)
    A=np.zeros((4,4)); B=np.zeros((4,4)); A[:3,:3]=c['A_m2ps']; B[:3,:3]=c['B']
    A[3,1]=12.; A[1,3]=6.; B[3,1]=B[1,3]=1.
    c['A_m2ps']=A.tolist(); c['B']=B.tolist()
    return c
