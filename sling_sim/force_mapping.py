"""Vector cable-force allocation and differentiable inverse geometry.

Allocation is over 9 vector components, not 3 scalar tensions at fixed directions.
The desired payload attachment frame is fixed/world aligned in this experiment.
"""
import numpy as np
import osqp
from scipy import sparse
from .model import ModelDomainError


def skew(r):
    x, y, z = r
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


class CableAllocator:
    def __init__(self, config, anchors, radial):
        self.c, self.p = config, config["force_mapping"]
        self.anchors, self.radial = anchors.copy(), radial.copy()
        self.G = np.vstack((np.tile(np.eye(3), (1, 3)), np.hstack([skew(r) for r in anchors])))
        if np.linalg.matrix_rank(self.G) != 6:
            raise ValueError("Three noncollinear payload attachment points are required")
        self.previous = self.spread_reference(np.array([0., 0., config["payload_mass"]*config["gravity"]]))
        # Convex inner approximation: |fx|+|fy| <= tan(theta)*fz,
        # Tmin <= fz <= Tmax*cos(theta). Hence Tmin <= ||f|| <= Tmax.
        rows, lower, upper = [], [], []
        angle = np.deg2rad(self.p["cone_angle_deg"])
        for i in range(3):
            row = np.zeros(9); row[3*i+2] = 1
            rows.append(row); lower.append(self.p["tension_min"])
            upper.append(self.p["tension_max"]*np.cos(angle))
            for sx, sy in ((1,1), (1,-1), (-1,1), (-1,-1)):
                row = np.zeros(9); row[3*i:3*i+3] = [sx, sy, -np.tan(angle)]
                rows.append(row); lower.append(-np.inf); upper.append(0.)
        self.cone_A = np.asarray(rows)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        weight = 1+self.p["allocation_smooth_weight"]
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.eye(9, format="csc")*2*weight, q=np.zeros(9),
            A=sparse.csc_matrix(np.vstack((self.G, self.cone_A))),
            l=np.r_[np.zeros(6), self.lower], u=np.r_[np.zeros(6), self.upper],
            verbose=False, eps_abs=1e-8, eps_rel=1e-8, max_iter=8000, polishing=False)
        self.count = 0

    def spread_reference(self, force):
        # A pure minimum-norm allocation collapses the three UAVs over their
        # nearby anchors. Nominal radial self-stress preserves the sling spread.
        out = np.tile(force/3, (3,1))
        out += force[2]/3*np.tan(np.deg2rad(self.c["hover_rope_angle_deg"]))*self.radial
        return out

    def solve(self, force, moment):
        wrench = np.r_[force, moment]
        q = -2*(self.spread_reference(force)+self.p["allocation_smooth_weight"]*self.previous).ravel()
        self.solver.update(q=q, l=np.r_[wrench, self.lower], u=np.r_[wrench, self.upper])
        result = self.solver.solve(raise_error=False)
        if result.info.status_val != 1:
            raise ModelDomainError(f"Cable allocation QP failed: {result.info.status}")
        out = result.x.reshape(3,3)
        self.residual = self.G @ result.x-wrench
        cone = self.cone_A @ result.x
        if np.max(np.abs(self.residual)) > 2e-5 or np.any(cone < self.lower-2e-5) or np.any(cone > self.upper+2e-5):
            raise ModelDomainError("Cable allocation solution failed explicit feasibility check")
        self.previous = out.copy()
        self.count += 1
        self.solve_ms = result.info.run_time*1000
        return out


class SmoothForce:
    """Exact ZOH integration of (D+omega)^3 f = omega^3 f_input.

    Initial derivatives are zero. This positive impulse-response filter preserves
    fixed convex cone constraints and gives C2 force/formation references.
    """
    def __init__(self, initial, omega):
        self.value = np.array(initial, float)
        self.velocity = np.zeros_like(self.value)
        self.acceleration = np.zeros_like(self.value)
        self.target = self.value.copy()
        self.omega = omega

    def advance(self, dt):
        w = self.omega
        d = self.value-self.target
        b = self.velocity+w*d
        cc = .5*(self.acceleration+2*w*self.velocity+w*w*d)
        poly = d+b*dt+cc*dt*dt
        pd = b+2*cc*dt
        decay = np.exp(-w*dt)
        self.value = self.target+poly*decay
        self.velocity = (pd-w*poly)*decay
        self.acceleration = (2*cc-2*w*pd+w*w*poly)*decay


def inverse_geometry(reference, anchors, length, force, force_rate, force_acceleration):
    """p_i,d = p_L,d + R_L,d*r_i + l*q_i,d, with R_L,d = I.

    Normalize allocated force VECTORS, not scalar tensions; compute analytic
    first/second derivatives so reference velocity is consistent with position.
    """
    tension = np.linalg.norm(force, axis=1, keepdims=True)
    if np.any(tension < 1e-8):
        raise ModelDomainError("Inverse geometry cannot normalize zero cable force")
    direction = force/tension
    td = np.sum(direction*force_rate, axis=1, keepdims=True)
    qd = (force_rate-direction*td)/tension
    tdd = np.sum(qd*force_rate+direction*force_acceleration, axis=1, keepdims=True)
    qdd = (force_acceleration-direction*tdd-2*qd*td)/tension
    return (reference.position+anchors+length*direction,
            reference.velocity+length*qd, reference.acceleration+length*qdd, direction)


class MomentumESO:
    """LESO using ideal simulated velocity and cable-wrench measurements.

    No injected wind/drag truth is accepted. The disturbance output has force or
    torque units. A diagonal inertia acts as the rotational 'mass' per axis.
    """
    def __init__(self, inertia_or_mass, bandwidth):
        self.mass = np.broadcast_to(inertia_or_mass, (3,)).copy()
        self.bandwidth = bandwidth
        self.state = None

    def update(self, velocity, known_acceleration, dt):
        if self.state is None:
            self.state = np.r_[velocity, np.zeros(3)]
        elif dt > 0:
            # Interpolate two measurements across the completed sample interval.
            def rhs(s, fraction):
                measured = self.last_velocity+(velocity-self.last_velocity)*fraction
                known = self.last_acceleration+(known_acceleration-self.last_acceleration)*fraction
                error = measured-s[:3]
                return np.r_[known+s[3:]/self.mass+2*self.bandwidth*error,
                             self.mass*self.bandwidth**2*error]
            s = self.state
            k1 = rhs(s, 0); k2 = rhs(s+dt*k1/2, .5)
            k3 = rhs(s+dt*k2/2, .5); k4 = rhs(s+dt*k3, 1)
            self.state = s+dt*(k1+2*k2+2*k3+k4)/6
        self.last_velocity, self.last_acceleration = velocity.copy(), known_acceleration.copy()
        return self.state[3:].copy()
