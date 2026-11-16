"""Point-query self KL plus the exact Wtask particle graph gradient.

Online fields() does not call the inherited GH routines or inherited fields().
For each role, s=D*score_KDE(r)+D*Sigma^{-1}(r-q) and
    tilde_self = center(s)+D*Sigma^{-1}(mean(r)-q).
The shared implemented field is tilde_g=tilde_self+g_graph.  Its mean and
centered part are mapped with Mc=(kc/D)*Sigma and Ms=I.  The self mean command
is exactly -kc*(mean(r)-q), while the same graph gradient drives both channels.

The point-query field is not asserted to be the exact KDE KL particle gradient.
Its difference from that gradient is a shape-channel implementation residual.
Inherited gradient()/energy()/quadrature_energy_gradient() are OFFLINE GH
diagnostics only; approx_gradient() and fields() use no Gaussian quadrature.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import types

import numpy as np


GRAPH_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "unified_density_cooperation_20261014/src/task_density_controller.py"
)
_graph_source_bytes = GRAPH_SOURCE.read_bytes()
_graph = types.ModuleType("_point_precheck_task_density_controller")
_graph.__file__ = str(GRAPH_SOURCE)
# Execute read-only source without writing bytecode beside either dependency.
exec(compile(_graph_source_bytes.decode("utf-8-sig"), str(GRAPH_SOURCE), "exec"),
     _graph.__dict__)
TaskDensityController = _graph.TaskDensityController
FrozenController = _graph.FrozenController
references = _graph.references
pulse = _graph.pulse

DEPENDENCY_HASHES = {
    "task_density_controller.py": hashlib.sha256(_graph_source_bytes).hexdigest(),
    "multigroup.py": hashlib.sha256(_graph.FROZEN_SOURCE.read_bytes()).hexdigest(),
}
DEPENDENCY_PATHS = {
    "task_density_controller.py": str(GRAPH_SOURCE),
    "multigroup.py": str(_graph.FROZEN_SOURCE),
}


class PointController(TaskDensityController):
    """Registered four-role point implementation: point or no_graph (kappa=0).

    All gradient arrays use g-scale (N_i/omega_i times the physical derivative
    when a true derivative exists).  The graph array is an exact derivative;
    the self array and total approx_gradient retain the point-query residual.
    """

    def __init__(self, cfg, method="point"):
        if method not in ("point", "no_graph"):
            raise ValueError("method must be 'point' or 'no_graph'")
        super().__init__(cfg, method="wtask" if method == "point" else "no_graph")
        self.method = method
        self.dependency_hashes = dict(DEPENDENCY_HASHES)
        self.dependency_paths = dict(DEPENDENCY_PATHS)
        self.implementation_metadata = {
            "method": method,
            "online_self_gradient": "sample-centered KDE point score plus analytic Gaussian mean gradient",
            "online_graph_gradient": "exact inherited heterogeneous Gaussian-overlap particle gradient",
            "online_quadrature": False,
            "kappa": self.kappa,
            "ell_task": self.ell_task,
            "omega": self.omega.tolist(),
            "dependency_paths": dict(self.dependency_paths),
            "dependency_sha256": dict(self.dependency_hashes),
            "field_semantics": {
                "anchor": "exact self-KL mean command -kc*(centers-q)",
                "formation": "-Mc times mean graph gradient; not the old independent B controller",
                "target": "sample-centered negative Gaussian target gradient",
                "spreading": "sample-centered negative point KDE score times D",
                "cross": "sample-centered negative graph gradient",
                "edge": "centered negative graph-edge gradients, including each edge's self-density term",
                "nu": "qd+anchor+formation; group average of cnom before execution modifications",
                "shape": "target+spreading+cross; negative centered approx_gradient up to roundoff",
                "approx_gradient": "tilde_self+graph_gradient; not claimed to be the exact full KDE gradient",
                "graph_gradient": "exact graph-only g=(N_i/omega_i)*dF_graph/dr",
                "point_score_sample_mean": "mean of D*point score before centering; not a bound on query residual",
            },
        }

    def _point_self_terms(self, r, q):
        """Analytic Gaussian anchor and original finite-sample point score."""
        target_g = self.D * (r - q[self.roles]) / self.sigma[self.roles]**2
        entropy_g = np.zeros_like(r)
        for i, group in enumerate(self.groups):
            diff = r[group, None, :] - r[group][None, :, :]
            exponent = -np.sum(diff * diff, axis=2) / (2.0 * self.beta[i]**2)
            weights = np.exp(exponent - exponent.max(axis=1, keepdims=True))
            entropy_g[group] = -self.D * np.einsum("kl,kld->kd", weights, diff) / (
                weights.sum(axis=1)[:, None] * self.beta[i]**2
            )
        analytic_mean = self.D * (self.means(r) - q) / self.sigma**2
        tilde_self = self.center(target_g + entropy_g) + analytic_mean[self.roles]
        return target_g, entropy_g, analytic_mean, tilde_self

    def point_self_gradient(self, r, t):
        """Return center(s)+D*Sigma^{-1}a, without graph or GH evaluation."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        return self._point_self_terms(r, q)[3]

    def approx_gradient(self, r, t):
        """Implemented common field: point self term plus exact graph term."""
        r = self._positions(r)
        q, _ = references(t, self.cfg)
        tilde_self = self._point_self_terms(r, q)[3]
        _, edge_g = self._graph_terms(r, q)
        return tilde_self + edge_g.sum(axis=1)

    def fields(self, r, t):
        """Online fields compatible with the registered execution interface.

        At kappa=0, cnom/ctest reproduce frozen no_AB up to floating roundoff.
        No inherited GH self term or old independent A/B controller is called.
        """
        r = self._positions(r)
        q, qd = references(t, self.cfg)
        centers = self.means(r)
        target_g, entropy_g, analytic_mean, tilde_self = self._point_self_terms(r, q)
        graph_energy, edge_g = self._graph_terms(r, q)
        graph_g = edge_g.sum(axis=1)
        approx_g = tilde_self + graph_g

        anchor = -self.cfg["kc_per_s"] * (centers - q)
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

        # Original registered pre-safety command excitations, unchanged.
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
            approx_gradient=approx_g, graph_gradient=graph_g,
            point_self_gradient=tilde_self, analytic_self_mean_gradient=analytic_mean,
            point_score_sample_mean=self.means(entropy_g), graph_energy=graph_energy,
        )


Controller = PointController
