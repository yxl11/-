"""Affine transport extension of the registered P velocity feedback.

All three templates share material axes R(t). In these axes the original
diagonal Gaussian score, KDE graph gradient and mobility projection apply
with the instantaneous role scales. The isotropic physical KDE is unchanged.
"""
from pathlib import Path
import importlib.util
import numpy as np
from scipy.spatial.distance import pdist,squareform
from shape_models import shape_axes
def prior_helpers():
    p=Path(__file__).resolve().parent.parent/'mixed_role_corridor_demo_20261031/p_feedback.py'
    spec=importlib.util.spec_from_file_location('previous_p_adapter_for_shapes',p)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    return old
class TransportController:
    def __init__(self,h,cfg,reference):
        self.base=h.PController(cfg);self.groups=self.base.groups;self.roles=self.base.roles
        self.reference=reference;self.cfg=cfg;self.last_fields={}
        def local_reference(t,cfg=None):
            q,qd=reference(t,cfg);R=shape_axes(t)[0]
            # qdot is fed here as a physical vector in the instantaneous axes.
            return q@R,qd@R
        prior_helpers().bind_reference(h,local_reference)
    def command(self,t,r,v):
        R,dR,s,ds=shape_axes(t); q,qd=self.reference(t,self.cfg)
        b=self.base
        b.sigma=s
        b.Mc_diag=self.cfg['kc_per_s']*s*s/b.D
        b.K_diag=b.D*b.omega[:,None]/(s*s)
        b.K_sum_diag=b.K_diag.sum(axis=0)
        b.task_cov_diag=(b.beta[:,None]/s)**2
        local=b.fields(r@R,t)
        # x-q = R diag(s) z; material velocity is dR diag(s) z + R diag(ds) z.
        rel=(r-q[self.roles])@R
        affine=rel@dR.T + (rel*(ds/s)[self.roles])@R.T
        mean_affine=b.means(affine)
        nu=local['nu']@R.T+mean_affine
        shape=local['shape']@R.T+affine-mean_affine[self.roles]
        cmd=nu[self.roles]+shape
        self.last_fields={'nu':nu,'shape':shape,'ctest':cmd,'cnom':cmd,
            'affine_velocity':affine,'affine_mean':mean_affine,
            'projection_mean_command':local['projection_mean_command']@R.T,
            'projected_common_displacement':local['projected_common_displacement']@R.T,
            'effective_displacement':local['effective_displacement']@R.T,
            'zero_mean_error':float(np.max(np.abs(b.means(cmd)-nu)))}
        return cmd
class ShapeFeedback:
    def __init__(self,h,density,cfg,reference):
        self.cfg=cfg;self.reference=reference;self.controller=TransportController(h,cfg,reference)
        self.roles=self.controller.roles;self.groups=self.controller.groups;self.metrics=h.multigroup.Metrics(cfg)
        self.density=density.relative_density_l2;self.beta=np.asarray(cfg['bandwidth_m'])
    def fast(self,r,t):
        q,qd=self.reference(t,self.cfg);R,_,s,_=shape_axes(t)
        centers=np.stack([r[g].mean(axis=0) for g in self.groups])
        e=centers-q;z=((r-q[self.roles])@R)/s[self.roles]
        return {'reference':q,'reference_velocity':qd,'centers':centers,
                'role_center_error':np.linalg.norm(e,axis=1),'edge_error':np.linalg.norm(e[:-1]-e[1:],axis=1),
                'region_by_role':np.array([np.mean(np.sum(z[g]**2,axis=1)<=4) for g in self.groups])}
    def slow(self,r,t):
        q,_=self.reference(t,self.cfg);R,_,s,_=shape_axes(t)
        directions=self.metrics.dirs
        scales=np.sqrt((s*s)@((directions@R)**2).T)
        ans=[];density=[];nearest=[];horizontal=[]
        for i,g in enumerate(self.groups):
            pts=r[g];centered=(pts-pts.mean(axis=0))@directions.T
            expected=self.metrics.normal_quantiles[i][:,None]*scales[i]
            ans.append(np.sqrt(np.mean((np.sort(centered,axis=0)-expected)**2)))
            density.append(self.density((pts-q[i])@R,np.zeros(3),s[i],self.beta[i]))
            for dest,coordinates in ((nearest,pts),(horizontal,pts[:,:2])):
                distances=squareform(pdist(coordinates));np.fill_diagonal(distances,np.inf)
                dest.extend(distances.min(axis=1))
        return {'time':float(t),'relative_density_l2_by_role':np.asarray(density),'shape_by_role':np.asarray(ans),
                'intrarole_nn_m':np.asarray(nearest),'intrarole_horizontal_nn_m':np.asarray(horizontal)}
