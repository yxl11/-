"""Finite-particle Wtask controller; the frozen original controller is read-only.

Physical KDE bandwidths are unchanged.  In role coordinates z=(r-q)/sigma,
component covariance is diag((beta/sigma)**2), not an isotropic task bandwidth.

F_N = D sum_i omega_i KL(rho_i || N(0,I))
      + kappa/2 sum_{i<j} B_ij ||rho_i-rho_j||_K**2,
K(z)=exp(-|z|**2/2), with each undirected edge counted once.
The physical particle gradient interface is
    g_ik = (N_i/omega_i) dF_N/dr_ik = integral G_beta(x-r_ik) xi_i(x) dx.
Graph terms use exact Gaussian overlaps.  The self-KL score convolution uses
fixed GH5 in fields(); it is a quadrature approximation to the true gradient.
In particular it is not exactly the derivative of the GH5 energy function.
quadrature_energy_gradient() includes the finite-quadrature residual and is
the derivative of energy(..., order) for numerical consistency checks.

This module establishes no saturated, sampled, safe closed-loop convergence.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import types

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp


FROZEN_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "multigroup_closed_loop_20261012/runs/four_group_s401/source_snapshot/multigroup.py"
)
# Compile the read-only source directly: do not create __pycache__ in its archive.
_frozen = types.ModuleType("_wtask_frozen_multigroup")
_frozen.__file__ = str(FROZEN_SOURCE)
exec(compile(FROZEN_SOURCE.read_text(encoding="utf-8-sig"), str(FROZEN_SOURCE), "exec"),
     _frozen.__dict__)
FrozenController = _frozen.Controller
references = _frozen.references
pulse = _frozen.pulse


@lru_cache(maxsize=4)
def _normal_rule(order):
    """Tensor Gauss-Hermite rule for expectation under N(0,I_3)."""
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    nodes, weight = hermgauss(int(order))
    mesh = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    points = np.sqrt(2.0) * np.stack(mesh, axis=-1).reshape(-1, 3)
    weights = np.einsum("i,j,k->ijk", weight, weight, weight).reshape(-1) / np.pi**1.5
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


class Controller(FrozenController):
    """Same fields/means/capability interface as the registered controller.

    Methods are 'wtask' (kappa=10) and 'no_graph' (kappa=0).  No old A density
    term or independent B center force is added.  B supplies graph adjacency.
    """

    def __init__(self, cfg, method="wtask"):
        if method not in ("wtask", "no_graph"):
            raise ValueError("method must be 'wtask' or 'no_graph'")
        super().__init__(cfg, method="full")
        self.method = method
        if self.M != 4:
            raise ValueError("this frozen precheck controller is registered for four roles")
        self.omega = np.array([3.0 / 7.0, 1.0, 2.0 / 3.0, 1.0 / 2.0])
        self.kappa = 10.0 if method == "wtask" else 0.0
        self.ell_task = 1.0
        self.gradient_order = 5
        self.D = float(cfg["D_m2ps"])
        if self.D <= 0 or np.any(self.sigma <= 0) or np.any(self.beta <= 0):
            raise ValueError("D, sigma and physical KDE bandwidth must be positive")
        if (np.any(self.B < 0) or not np.allclose(self.B, self.B.T, rtol=0, atol=1e-14)
                or not np.allclose(np.diag(self.B), 0, rtol=0, atol=1e-14)):
            raise ValueError("B must be symmetric, nonnegative and have zero diagonal")
        # Mc is diagonal in physical coordinates; Ms=I.  Neither changes KDE.
        self.Mc_diag = float(cfg["kc_per_s"]) * self.sigma**2 / self.D
        self.task_cov_diag = (self.beta[:, None] / self.sigma)**2

    def _positions(self, r):
        r = np.asarray(r, dtype=float)
        if r.shape != (len(self.roles), 3) or not np.all(np.isfinite(r)):
            raise ValueError("positions must be a finite (N,3) array")
        return r

    def _self_terms(self, r, q, order, include_residual=False):
        nodes, weights = _normal_rule(order)
        target_g = self.D * (r - q[self.roles]) / self.sigma[self.roles]**2
        entropy_g = np.empty_like(r)
        residual_g = np.zeros_like(r)
        kl = np.empty(self.M)
        for i, group in enumerate(self.groups):
            centers = r[group]
            queries = centers[:, None, :] + self.beta[i] * nodes[None, :, :]
            diff = queries[:, :, None, :] - centers[None, None, :, :]
            exponent = -0.5 * np.sum(diff * diff, axis=-1) / self.beta[i]**2
            normalizer = logsumexp(exponent, axis=-1)
            responsibility = np.exp(exponent - normalizer[:, :, None])
            score = -np.einsum("aqk,aqkd->aqd", responsibility, diff) / self.beta[i]**2
            entropy_g[group] = self.D * np.einsum("q,aqd->ad", weights, score)
            log_density = (normalizer - np.log(len(group)) - 3.0 * np.log(self.beta[i])
                           - 1.5 * np.log(2.0 * np.pi))
            target_diff = queries - q[i]
            log_target = (-1.5 * np.log(2.0 * np.pi) - np.log(self.sigma[i]).sum()
                          - 0.5 * np.sum((target_diff / self.sigma[i])**2, axis=-1))
            kl[i] = float(np.mean((log_density - log_target) @ weights))
            if include_residual:
                # Explicit dependence of log KDE on every component center.
                # Its exact integral is zero, but a finite GH rule need not be.
                residual_g[group] = self.D * np.einsum(
                    "q,aqk,aqkd->kd", weights, responsibility, diff
                ) / self.beta[i]**2
        return target_g, entropy_g, kl, residual_g

    def _graph_terms(self, r, q):
        """Exact graph energy and its physical particle g, split by edge.

        Overlap(i,k;j,l)=prod_d ell/sqrt(H_d)*exp(-sum_d delta_d^2/H_d/2),
        H_d=ell^2+(beta_i/sigma_id)^2+(beta_j/sigma_jd)^2.
        An edge gradient at i is kappa*B_ij/(omega_i*sigma_i)
        times [mean_l grad K_ii(z_ik-z_il)-mean_l grad K_ij(z_ik-z_jl)].
        """
        edge_g = np.zeros((len(r), self.M, 3))
        if self.kappa == 0:
            return 0.0, edge_g
        z = (r - q[self.roles]) / self.sigma[self.roles]
        overlap = np.empty((self.M, self.M))
        gradients = [[None] * self.M for _ in range(self.M)]
        for i, gi in enumerate(self.groups):
            for j in range(i, self.M):
                gj = self.groups[j]
                h2 = self.ell_task**2 + self.task_cov_diag[i] + self.task_cov_diag[j]
                delta = z[gi, None, :] - z[gj][None, :, :]
                value = np.prod(self.ell_task / np.sqrt(h2)) * np.exp(
                    -0.5 * np.sum(delta * delta / h2, axis=-1)
                )
                kernel_grad = -value[:, :, None] * delta / h2
                overlap[i, j] = overlap[j, i] = float(value.mean())
                gradients[i][j] = kernel_grad.mean(axis=1)
                if j != i:
                    gradients[j][i] = -kernel_grad.mean(axis=0)
        energy = 0.0
        for i, gi in enumerate(self.groups):
            for j in range(self.M):
                if self.B[i, j] == 0:
                    continue
                edge_g[gi, j] = (
                    self.kappa * self.B[i, j] / (self.omega[i] * self.sigma[i])
                    * (gradients[i][i] - gradients[i][j])
                )
                if i < j:
                    energy += 0.5 * self.kappa * self.B[i, j] * (
                        overlap[i, i] + overlap[j, j] - 2.0 * overlap[i, j]
                    )
        return float(energy), edge_g

    def particle_gradient(self, r, t, order=5):
        """Return g=(N_i/omega_i)*dF_N/dr: GH self term, exact graph term."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        target_g, entropy_g, _, _ = self._self_terms(r, q, order)
        _, edge_g = self._graph_terms(r, q)
        return target_g + entropy_g + edge_g.sum(axis=1)

    def gradient(self, r, t, order=5):
        """Compatibility alias: returns g, with scale N_i/omega_i."""
        return self.particle_gradient(r, t, order)

    def graph_gradient(self, r, t):
        """Exact graph-only g, also scaled by N_i/omega_i."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        _, edge_g = self._graph_terms(r, q)
        return edge_g.sum(axis=1)

    def command_from_gradient(self, r, t, g):
        """Uncapped, unexcited nominal command; no new quadrature evaluation."""
        r = self._positions(r)
        g = np.asarray(g, dtype=float)
        if g.shape != r.shape or not np.all(np.isfinite(g)):
            raise ValueError("gradient must be a finite (N,3) array")
        _, qd = references(t, self.cfg)
        mean_g = self.means(g)
        return (qd - self.Mc_diag * mean_g)[self.roles] - (g - mean_g[self.roles])

    def energy_gradient(self, r, t, order=5):
        """Approximation to the true F_N derivative, not exact GH-objective derivative."""
        return (self.omega[self.roles] / self.counts[self.roles])[:, None] * self.particle_gradient(r, t, order)

    def quadrature_energy_gradient(self, r, t, order=5):
        """Exact derivative of this module's energy(...,order), up to roundoff."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        target_g, entropy_g, _, residual_g = self._self_terms(r, q, order, True)
        _, edge_g = self._graph_terms(r, q)
        g = target_g + entropy_g + residual_g + edge_g.sum(axis=1)
        return (self.omega[self.roles] / self.counts[self.roles])[:, None] * g

    def energy_components(self, r, t, order=5):
        """KL quadrature and exact graph overlap; no quadrature error certificate."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        _, _, kl, _ = self._self_terms(r, q, order)
        graph, _ = self._graph_terms(r, q)
        weighted_kl = self.D * self.omega * kl
        return {"total": float(weighted_kl.sum() + graph), "self_kl": kl,
                "weighted_self_kl": weighted_kl, "graph": graph, "order": int(order)}

    def energy(self, r, t, order=5, kernel_only=False):
        """Numerical F_N: GH-order self KL plus exact Gaussian-overlap graph energy."""
        if kernel_only:
            r = self._positions(r)
            q, _ = references(t, self.cfg)
            return self._graph_terms(r, q)[0]
        return self.energy_components(r, t, order)["total"]

    def fields(self, r, t):
        r = self._positions(r)
        q, qd = references(t, self.cfg)
        centers = self.means(r)
        target_g, entropy_g, kl, _ = self._self_terms(r, q, self.gradient_order)
        graph_energy, edge_g = self._graph_terms(r, q)
        graph_g = edge_g.sum(axis=1)
        g = target_g + entropy_g + graph_g
        # Both channels are derived from this same physical gradient.
        anchor = -self.Mc_diag * self.means(target_g + entropy_g)
        formation = -self.Mc_diag * self.means(graph_g)
        nu = qd + anchor + formation
        target = -self.center(target_g)
        spreading = -self.center(entropy_g)
        edge = -edge_g.copy()
        for group in self.groups:
            edge[group] -= edge[group].mean(axis=0, keepdims=True)
        cross = edge.sum(axis=1)
        shape = target + spreading + cross
        cnom = nu[self.roles] + shape

        # Registered pre-safety test excitations, copied without changes.
        dcen = np.zeros((self.M, 3))
        dshape = np.zeros_like(r)
        slowdown = pulse(t, 110.0, 133.0) + pulse(t, 193.0, 216.0)
        dcen[2, 0] = -self.cfg["test_slowdown_mps"] * slowdown
        deformation = pulse(t, 143.0, 167.0) + pulse(t, 197.0, 223.0)
        group = self.groups[1]
        rel = r[group] - centers[1]
        raw_exc = np.column_stack((0.35 * rel[:, 1], -rel[:, 1], 0.2 * rel[:, 2]))
        raw_exc *= self.cfg["test_shape_gain_per_s"] * deformation
        raw_exc -= raw_exc.mean(axis=0)
        raw_exc *= min(1.0, self.cfg["test_shape_cap_mps"]
                       / max(float(np.linalg.norm(raw_exc, axis=1).max()), 1e-300))
        dshape[group] = raw_exc
        excitation = dcen[self.roles] + dshape
        return dict(
            q=q, qd=qd, centers=centers, nu=nu, anchor=anchor, formation=formation,
            target=target, spreading=spreading, cross=cross, shape=shape, edge=edge,
            cnom=cnom, excitation=excitation, ctest=cnom + excitation,
            zero_mean_error=float(np.max(np.abs(self.means(cnom) - nu))),
            shape_excitation_mean_error=float(np.max(np.abs(dshape[group].mean(axis=0)))),
            particle_gradient=g, graph_particle_gradient=graph_g,
            self_score_mean_residual=self.means(entropy_g),
            self_kl_gh=kl, graph_energy=graph_energy,
            total_energy_gh=float(self.D * np.dot(self.omega, kl) + graph_energy),
            gradient_quadrature_order=self.gradient_order,
        )


TaskDensityController = Controller
