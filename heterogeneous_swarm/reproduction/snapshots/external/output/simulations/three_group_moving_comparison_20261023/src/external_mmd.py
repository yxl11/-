"""Arbel et al. (NeurIPS 2019), deterministic MMD flow, Eqs.(5),(6),(9).

Three independent copies in prescribed translating role coordinates. The
empirical particle measure and exact Gaussian target are used directly; KDE
bandwidth, P graph, mean feedback, centering and projection are absent.
This is the paper's deterministic flow, not its noise-regularized algorithm.
"""
from __future__ import annotations
import numpy as np


def gaussian_target_embedding(queries, target_center, target_sigma, length):
    """E_{Y~Gaussian}[exp(-||x-Y||^2/(2 length^2))] and spatial gradient."""
    y = np.asarray(queries, dtype=float) - np.asarray(target_center, dtype=float)
    sig = np.asarray(target_sigma, dtype=float)
    ell2 = float(length)**2
    total = ell2 + sig**2
    embedding = np.prod(float(length)/np.sqrt(total)) * np.exp(-.5*np.sum(y*y/total, axis=-1))
    return embedding, -embedding[..., None]*y/total


def mmd_squared(positions, target_center, target_sigma, length):
    """Exact squared RBF MMD(empirical measure, Gaussian target), dimensionless."""
    x = np.asarray(positions, dtype=float)
    ell2 = float(length)**2
    dx = x[:, None, :] - x[None, :, :]
    xx = np.exp(-.5*np.sum(dx*dx, axis=-1)/ell2).mean()
    xy = gaussian_target_embedding(x, target_center, target_sigma, length)[0].mean()
    yy = np.prod(float(length)/np.sqrt(ell2+2*np.asarray(target_sigma, dtype=float)**2))
    ans = float(xx-2*xy+yy)
    if ans < -1e-12:
        raise FloatingPointError("negative squared MMD")
    return max(ans, 0.)


def mmd_velocity(positions, target_center, target_sigma, length, gamma):
    """Minus gamma times witness gradient; no limit or outer execution."""
    x = np.asarray(positions, dtype=float)
    ell2 = float(length)**2
    dx = x[:, None, :] - x[None, :, :]
    k = np.exp(-.5*np.sum(dx*dx, axis=-1)/ell2)
    negative_self_gradient = np.mean(k[..., None]*dx/ell2, axis=1)
    target_gradient = gaussian_target_embedding(x, target_center, target_sigma, length)[1]
    return float(gamma)*(negative_self_gradient+target_gradient)


class ExternalMMDController:
    method_id = "Arbel2019_deterministic_MMD_moving_role"

    def __init__(self, reference, counts=(36,24,20),
                 sigma=((5.,8.,3.),(4.,5.,3.),(4.,4.,2.5)), gamma=10., lengths=None):
        self.reference = reference
        self.counts = np.asarray(counts, dtype=int)
        self.sigma = np.asarray(sigma, dtype=float)
        self.gamma = float(gamma)
        self.lengths = np.sqrt(np.mean(self.sigma**2, axis=1)) if lengths is None else np.asarray(lengths, dtype=float)
        if self.sigma.shape != (len(self.counts),3) or np.any(self.counts<=0) or np.any(self.sigma<=0):
            raise ValueError("invalid counts or target sigmas")
        if self.lengths.shape != self.counts.shape or np.any(self.lengths<=0) or not np.isfinite(self.gamma) or self.gamma<=0:
            raise ValueError("positive finite lengths and gamma required")
        edges=np.r_[0,self.counts.cumsum()]
        self.groups=[slice(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]
        self.last_fields={}

    def command(self,t,x,v=None):
        x=np.asarray(x,dtype=float)
        if x.shape != (int(self.counts.sum()),3) or not np.all(np.isfinite(x)):
            raise ValueError("x must be a finite (sum(counts),3) matrix")
        q,qd=self.reference(float(t))[:2]
        q,qd=np.asarray(q,dtype=float),np.asarray(qd,dtype=float)
        if q.shape!=self.sigma.shape or qd.shape!=q.shape:
            raise ValueError("reference must return group centers and velocities")
        c=np.empty_like(x)
        for i,sl in enumerate(self.groups):
            c[sl]=qd[i]+mmd_velocity(x[sl],q[i],self.sigma[i],self.lengths[i],self.gamma)
        if not np.all(np.isfinite(c)):
            raise FloatingPointError("nonfinite MMD velocity")
        self.last_fields={"kernel_length_m":self.lengths.copy(),"gamma_m2ps":np.array(self.gamma)}
        return c

    def native_losses(self,t,x):
        q=self.reference(float(t))[0]
        values=[mmd_squared(np.asarray(x)[sl],q[i],self.sigma[i],self.lengths[i]) for i,sl in enumerate(self.groups)]
        return {"mmd_squared_per_role":values,"mmd_squared_equal_role_mean":float(np.mean(values))}


class Controller(ExternalMMDController):
    """Shared harness factory; initial_positions never selects parameters."""
    def __init__(self,cfg,initial_positions=None,gamma=10.,**kwargs):
        from mission import reference
        if kwargs:
            raise TypeError(f"Unexpected MMD parameters: {sorted(kwargs)}")
        super().__init__(reference,counts=cfg["counts"],sigma=cfg["sigma_m"],gamma=gamma)
