"""Proposal equations (1), (6)-(9), discretized on the actual UAV topology.

The independent coordinate is a topology parameter, NOT world x or cable arclength.
Only the three measured UAV nodes exist online. No hidden virtual UAVs are used.
The first-order PDE supplies a velocity field to a finite-bandwidth flight loop.
"""
import numpy as np


def backstepping_kernel(x, y, sigma):
    """-sigma*y*I1(sqrt(sigma*(x*x-y*y)))/sqrt(...).

    Evaluate the entire power series to retain the diagonal limit -sigma*y/2.
    This avoids a new SciPy dependency and an artificial epsilon at x == y.
    """
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if not np.isfinite(sigma) or sigma < 0 or np.any(y < 0) or np.any(x < y):
        raise ValueError("Kernel requires finite sigma >= 0 and 0 <= y <= x")
    s2 = sigma*(x*x-y*y)
    term = np.full_like(s2, 0.5)
    value = term.copy()
    for n in range(1, 150):
        term = term*s2/(4*n*(n+1))
        value += term
        if np.all(np.abs(term) <= 1e-15*np.maximum(1, np.abs(value))):
            return -sigma*y*value
    raise ValueError("Kernel series did not converge; reduce parameters")


class PDEFormation:
    """Anchor node 0, interior follower 1, controlled right boundary 2.

    e_t = time_scale*(e_xx + 2*b*e_x + reaction*e).
    e_t(0) = -time_scale*anchor_c*e(0).
    e(alpha) = integral k(alpha,y)*exp(b*(y-alpha))*e(y) dy.
    The right boundary is realized through a velocity servo, never teleported.
    """
    def __init__(self, parameters):
        self.c = dict(parameters)
        p = self.c
        for name in ("alpha", "time_scale", "anchor_c", "mu", "boundary_gain",
                     "velocity_gain", "acceleration_limit", "velocity_limit"):
            if not np.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f"pde.{name} must be finite and positive")
        self.x = np.linspace(0, p["alpha"], 3)
        self.h = p["alpha"]/2
        self.sigma = p["mu"]+p["reaction"]-p["b"]**2
        # Left rectangle sum: h * sum_{j=0}^{N-2}. It matches proposal (9)
        # at alpha=1 and excludes the unknown right boundary from the integral.
        self.weights = self.h*backstepping_kernel(p["alpha"], self.x[:-1], self.sigma)
        self.weights *= np.exp(p["b"]*(self.x[:-1]-p["alpha"]))
        self.operator = np.zeros((2, 3))
        self.operator[0, 0] = -p["time_scale"]*p["anchor_c"]
        self.operator[1] = p["time_scale"]*np.array([
            1/self.h**2-p["b"]/self.h,
            -2/self.h**2+p["reaction"],
            1/self.h**2+p["b"]/self.h])
        # Ideal first-order closed loop, including dynamic enforcement of the
        # boundary. This spectrum is not a proof for the nonlinear flight loop.
        self.ideal_matrix = np.vstack((self.operator,
            self.weights @ self.operator+p["boundary_gain"]*np.r_[self.weights, -1.0]))
        self.ideal_eigenvalues = np.linalg.eigvals(self.ideal_matrix)
        if np.max(self.ideal_eigenvalues.real) >= 0:
            raise ValueError("The chosen three-node ideal PDE discretization is unstable")

    def feedback(self, error, error_velocity):
        """Return the PDE velocity correction using current measured errors."""
        e, ev = np.asarray(error), np.asarray(error_velocity)
        boundary = self.weights @ e[:2]
        residual = e[2]-boundary
        correction = np.empty((3, 3))
        correction[:2] = self.operator @ e
        # Differentiate e_leader = weights @ e_sources. The finite boundary
        # servo adds -k_b*residual; only measured neighbor velocities are needed.
        correction[2] = self.weights @ ev[:2]-self.c["boundary_gain"]*residual
        return correction, boundary, residual

    def volterra_nodes(self, error):
        """Quadrature samples of z = exp(b*x)e - integral k*exp(b*y)e dy."""
        weighted = np.exp(self.c["b"]*self.x)[:, None]*error
        z = weighted.copy()
        for i in range(1, 3):
            row = self.h*backstepping_kernel(self.x[i], self.x[:i], self.sigma)
            z[i] -= row @ weighted[:i]
        return z

    def metadata(self):
        return {"source_equations": [1, 6, 7, 8, 9],
                "topology_coordinates": self.x.tolist(),
                "node_roles": ["anchor", "follower", "leader"],
                "quadrature": "left_rectangle_on_three_actual_nodes",
                "boundary_weights": self.weights.tolist(),
                "kernel_sigma": self.sigma,
                "ideal_first_order_eigenvalues": [
                    {"real": float(v.real), "imag": float(v.imag)} for v in self.ideal_eigenvalues],
                "tracking_reference_lifting": True,
                "dynamic_boundary_realization": True,
                "continuum_stability_proof_for_physical_plant": False}
