"""Ren & Sorensen (RAS 2008), Eq. (6), velocity-law adaptation.

This is NOT Hua et al. TAC 2019, nor a heterogeneous-plant theorem.
Original: doi:10.1016/j.robot.2007.08.005, pp. 327--328, Eq. (6).
For desired particle positions r_d and exact desired velocities r_d_dot:
    u = r_d_dot - alpha * (x-r_d) - coupling * L_normalized @ (x-r_d).

All vehicles receive the exact common reference. The original Eq. (4)
formation-state estimator is initialized on its all-leader exact-reference
invariant subspace, so it has zero estimation error and is evaluated exactly
instead of introducing numerical estimator error. Known role offsets and
fixed density-representing member offsets determine r_d(t). Three dimensions
are the componentwise extension of the original planar equation.

The plant lag, magnitude saturation, sample-and-hold, and safety acceptance
are outside this function and common to every compared method. The original
single-integrator convergence guarantee does not transfer through them.
No P density feedback, graph-density energy, or projection is included.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def role_graph_laplacian(group_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Complete within-role links and links between adjacent role blocks.

    All enabled edges have equal positive weight 1/lambda_max(L_binary).
    This is a graph/gain parameter choice allowed by the source, not a new
    normalization of each controller state or a density-field projection.
    """
    groups = np.asarray(group_ids, dtype=int)
    if groups.ndim != 1 or len(groups) < 2:
        raise ValueError("group_ids must be a one-dimensional membership array")
    if set(np.unique(groups)) != {0, 1, 2}:
        raise ValueError("this registered role graph requires groups 0, 1, 2")
    adjacency = (np.abs(groups[:, None] - groups[None, :]) <= 1).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    lap = np.diag(adjacency.sum(axis=1)) - adjacency
    spectrum = np.linalg.eigvalsh(lap)
    scale = float(spectrum[-1])
    if scale <= 0 or spectrum[1] <= 1e-10:
        raise ValueError("formation graph must be connected")
    normalized = lap / scale
    return normalized, {
        "node_count": len(groups),
        "undirected_edge_count": int(adjacency.sum() // 2),
        "binary_laplacian_max_eigenvalue": scale,
        "binary_algebraic_connectivity": float(spectrum[1]),
        "normalized_max_eigenvalue": float(np.linalg.eigvalsh(normalized)[-1]),
        "enabled_edge_weight_before_gain": 1.0 / scale,
        "definition": "complete within role and across role pairs (0,1),(1,2); no (0,2) edges",
    }


@dataclass
class FormationController:
    """Pure, memoryless task-level controller with fixed member assignment.

    reference(t) must return (role_positions, role_velocities, ...), each first
    pair shape (3,3). slot_offsets shape (N,3) is already assigned to member
    indices by the caller. Store the assignment separately for reproducibility.
    alpha/coupling are per-second gains; this module does not select gains.
    """

    group_ids: np.ndarray
    slot_offsets: np.ndarray
    reference: Callable
    alpha: float
    coupling: float

    def __post_init__(self) -> None:
        self.group_ids = np.array(self.group_ids, dtype=int, copy=True)
        self.slot_offsets = np.array(self.slot_offsets, dtype=float, copy=True)
        if self.slot_offsets.shape != (len(self.group_ids), 3):
            raise ValueError("slot_offsets must have shape (N,3)")
        if not np.isfinite(self.slot_offsets).all():
            raise ValueError("slot offsets must be finite")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if not np.isfinite(self.coupling) or self.coupling <= 0:
            raise ValueError("coupling must be positive and finite")
        self.laplacian, self.graph_metadata = role_graph_laplacian(self.group_ids)
        self.group_ids.flags.writeable = False
        self.slot_offsets.flags.writeable = False
        self.laplacian.flags.writeable = False

    def desired(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        values = self.reference(float(t))
        q, qdot = np.asarray(values[0], dtype=float), np.asarray(values[1], dtype=float)
        if q.shape != (3, 3) or qdot.shape != (3, 3):
            raise ValueError("reference must supply role position/velocity arrays (3,3)")
        return q[self.group_ids] + self.slot_offsets, qdot[self.group_ids]

    def command(self, t: float, x: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """Return unsaturated 3D velocity reference in world coordinates.

        v is accepted for a common harness API; Eq. (6) uses position tracking
        errors and known reference velocity, not measured vehicle velocity.
        """
        positions = np.asarray(x, dtype=float)
        target, target_velocity = self.desired(t)
        if positions.shape != target.shape or not np.isfinite(positions).all():
            raise ValueError("x must be a finite (N,3) array")
        error = positions - target
        return target_velocity - self.alpha * error - self.coupling * (self.laplacian @ error)

    def native_errors(self, t: float, x: np.ndarray) -> dict:
        target, _ = self.desired(t)
        error = np.asarray(x) - target
        return {
            "assigned_position_error_m": error,
            "assigned_position_rms_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "assigned_position_rms_by_role_m": [
                float(np.sqrt(np.mean(np.sum(error[self.group_ids == g]**2, axis=1))))
                for g in range(3)
            ],
        }
