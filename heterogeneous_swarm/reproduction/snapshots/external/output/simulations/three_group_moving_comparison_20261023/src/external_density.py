"""Eren--Acikmese (IFAC 2017) Eq. (23), moving-role adaptation.

This implementation retains the heat-error velocity field and Gaussian KDE.
Coordinates translate by the prescribed role center; no P graph, mean control,
projection, or sample-centering operation is included.  The surrounding harness
alone applies velocity limits, safety rewriting, and the heterogeneous plant.
The original ideal continuum convergence theorem is not asserted for this
finite-particle, sampled, constrained adaptation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc


def gaussian_density_gradient(queries, centers, variance):
    """Normalized 3-D equal-weight Gaussian mixture and analytic gradient."""
    queries = np.asarray(queries, dtype=float)
    centers = np.asarray(centers, dtype=float)
    var = np.broadcast_to(np.asarray(variance, dtype=float), (3,))
    if queries.ndim != 2 or queries.shape[1] != 3 or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("queries and centers must have shape (n,3)")
    if len(centers) == 0 or np.any(var <= 0):
        raise ValueError("nonempty centers and positive variance required")
    delta = queries[:, None, :] - centers[None, :, :]
    kernels = np.exp(-0.5 * np.sum(delta**2 / var, axis=-1))
    kernels /= (2 * np.pi)**1.5 * np.sqrt(np.prod(var))
    rho = kernels.mean(axis=1)
    grad = -np.mean(kernels[:, :, None] * delta / var, axis=1)
    return rho, grad


def gaussian_overlap(delta, variance):
    """Integral of two Gaussian PDFs, with summed covariance variance."""
    var = np.broadcast_to(np.asarray(variance, dtype=float), (3,))
    return np.exp(-0.5 * np.sum(np.asarray(delta)**2 / var, axis=-1)) / (
        (2 * np.pi)**1.5 * np.sqrt(np.prod(var)))


def relative_density_l2(centers, target_center, target_sigma, bandwidth):
    """Exact ||KDE-Pi||_2^2 / ||Pi||_2^2 on R^3; no grid or tails."""
    centers = np.asarray(centers, dtype=float)
    target_center = np.asarray(target_center, dtype=float)
    sigma2 = np.asarray(target_sigma, dtype=float)**2
    beta2 = float(bandwidth)**2
    pp = gaussian_overlap(centers[:, None, :] - centers[None, :, :], 2 * beta2).mean()
    pt = gaussian_overlap(centers - target_center, beta2 + sigma2).mean()
    tt = float(gaussian_overlap(np.zeros(3), 2 * sigma2))
    result = float((pp - 2 * pt + tt) / tt)
    if result < -1e-12:
        raise FloatingPointError("negative exact density squared loss")
    return max(result, 0.0)


def native_density_l1(centers, target_center, target_sigma, bandwidth, power=14, seed=74519):
    """Deterministic scrambled-QMC estimate of source Sec.5 L1 density loss.

    Integrates on R^3 by importance sampling m=(KDE+Pi)/2, so integrand
    |KDE-Pi|/m is bounded by 2. No computational box or tail renormalization.
    power is log2 of sample count and seed is independent of trajectory seeds.
    This is a numerical diagnostic, not a certified error bound.
    """
    centers = np.asarray(centers, dtype=float)
    target_center = np.asarray(target_center, dtype=float)
    sigma = np.asarray(target_sigma, dtype=float)
    u = qmc.Sobol(d=4, scramble=True, seed=seed).random_base2(power)
    z = ndtri(u[:, 1:])
    is_kde = u[:, 0] < 0.5
    index = np.minimum((2 * u[is_kde, 0] * len(centers)).astype(int), len(centers)-1)
    queries = target_center + sigma * z
    queries[is_kde] = centers[index] + float(bandwidth) * z[is_kde]
    p = np.empty(len(queries))
    for start in range(0, len(queries), 2048):
        sl = slice(start, min(start+2048, len(queries)))
        p[sl] = gaussian_density_gradient(queries[sl], centers, float(bandwidth)**2)[0]
    target = gaussian_density_gradient(queries, target_center[None, :], sigma**2)[0]
    return float(np.mean(2 * np.abs(p-target) / (p+target)))


class ExternalDensityController:
    """Raw velocity Eq.(23) copied per role in a translating coordinate frame."""
    method_id = "Eren2017_moving_role"

    def __init__(self, reference, counts=(36, 24, 20),
                 sigma=((5., 8., 3.), (4., 5., 3.), (4., 4., 2.5)),
                 bandwidth=(1.7, 1.5, 1.35), diffusion=5.0):
        self.reference = reference
        self.counts = np.asarray(counts, dtype=int)
        self.sigma = np.asarray(sigma, dtype=float)
        self.bandwidth = np.asarray(bandwidth, dtype=float)
        self.diffusion = float(diffusion)
        self.offsets = np.r_[0, self.counts.cumsum()]
        self.groups = [slice(int(a), int(b)) for a, b in zip(self.offsets[:-1], self.offsets[1:])]
        if np.any(self.counts <= 0) or self.sigma.shape != (len(self.counts), 3):
            raise ValueError("invalid group counts or target sigmas")
        if self.bandwidth.shape != self.counts.shape or np.any(self.bandwidth <= 0):
            raise ValueError("invalid fixed physical KDE bandwidths")
        if np.any(self.sigma <= 0) or not np.isfinite(self.diffusion) or self.diffusion <= 0:
            raise ValueError("positive target sigmas and diffusion required")
        self.last_diagnostics = {}

    def command(self, t, x, v=None):
        x = np.asarray(x, dtype=float)
        if x.shape != (int(self.counts.sum()), 3) or not np.all(np.isfinite(x)):
            raise ValueError("x must be finite (sum(counts),3)")
        q, qdot = self.reference(float(t))[:2]
        q, qdot = np.asarray(q, dtype=float), np.asarray(qdot, dtype=float)
        if q.shape != self.sigma.shape or qdot.shape != q.shape:
            raise ValueError("reference must provide (groups,3) q and qdot")
        out = np.empty_like(x)
        minimum_density, maximum_feedback = [], []
        for i, sl in enumerate(self.groups):
            # Prescribed-role translation, not empirical sample centering.
            y = x[sl] - q[i]
            p, grad_p = gaussian_density_gradient(y, y, self.bandwidth[i]**2)
            _, grad_target = gaussian_density_gradient(y, np.zeros((1, 3)), self.sigma[i]**2)
            # Eq.(10) includes j=k: p(x_k) >= 1/[N*(2pi)^1.5*beta^3].
            floor = 1 / (self.counts[i] * (2*np.pi)**1.5 * self.bandwidth[i]**3)
            if np.min(p) < floor * (1 - 1e-13):
                raise FloatingPointError("positive self-kernel density bound violated")
            feedback = -self.diffusion * (grad_p-grad_target) / p[:, None]
            out[sl] = qdot[i] + feedback
            minimum_density.append(float(np.min(p)))
            maximum_feedback.append(float(np.max(np.linalg.norm(feedback, axis=1))))
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("nonfinite density velocity")
        self.last_diagnostics = {
            "minimum_particle_density_per_m3": minimum_density,
            "maximum_feedback_mps": maximum_feedback,
            "diffusion_m2ps": self.diffusion,
        }
        return out

    def native_losses(self, t, x, include_l1=False, l1_power=14):
        q = self.reference(float(t))[0]
        l2 = [relative_density_l2(x[sl], q[i], self.sigma[i], self.bandwidth[i])
              for i, sl in enumerate(self.groups)]
        result = {"relative_l2_per_role": l2, "relative_l2_equal_role_mean": float(np.mean(l2))}
        if include_l1:
            result["native_l1_per_role"] = [
                native_density_l1(x[sl], q[i], self.sigma[i], self.bandwidth[i], power=l1_power)
                for i, sl in enumerate(self.groups)]
        return result


class Controller(ExternalDensityController):
    """Harness factory; initial_positions is intentionally not used for tuning."""

    def __init__(self, cfg, initial_positions=None, diffusion=5.0, **kwargs):
        from mission import reference
        if kwargs:
            raise TypeError(f"Unexpected external density options: {sorted(kwargs)}")
        super().__init__(reference, counts=cfg["counts"], sigma=cfg["sigma_m"],
                         bandwidth=cfg["bandwidth_m"], diffusion=diffusion)
