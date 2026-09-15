"""Linear translational tube-MPC design model; no nonlinear stability claim.

Nominal state is propagated (not reset to the measurement each solve). A
Lyapunov ellipsoid gives an RPI error tube for the stated additive box W.
Its support functions tighten nominal state and input constraints by Z and KZ.
"""
import itertools
import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, eigvalsh
from .model import ModelDomainError


class TubeMPC:
    def __init__(self, config, trajectory):
        self.c, self.p, self.trajectory = config, config["force_mapping"], trajectory
        p = self.p
        self.dt, self.N = p["prediction_period"], p["horizon"]
        h = self.dt
        self.A = np.array([[1., h], [0., 1.]])
        self.B = np.array([h*h/2, h])
        self.K = -np.array(p["tube_feedback"])
        self.closed = self.A+np.outer(self.B, self.K)
        if np.max(np.abs(np.linalg.eigvals(self.closed))) >= 1:
            raise ValueError("Tube local feedback must be Schur stable")
        self.P = solve_discrete_lyapunov(self.closed.T, np.eye(2))
        self.Pinv = np.linalg.inv(self.P)
        self.gamma = np.sqrt(eigvalsh(self.closed.T@self.P@self.closed, self.P).max())
        self.w_bound = np.asarray(p["acceleration_residual_bound"])
        corners = np.array(list(itertools.product((-1., 1.), repeat=2)))*self.B
        unit_radius = np.sqrt(np.einsum("ni,ij,nj->n", corners, self.P, corners).max())
        self.rho = self.w_bound*unit_radius/(1-self.gamma)
        self.state_margin = self.rho[:, None]*np.sqrt(np.diag(self.Pinv))[None, :]
        self.input_margin = self.rho*np.sqrt(self.K@self.Pinv@self.K)
        self.input_bound = np.asarray(p["acceleration_bounds"])-self.input_margin
        self.low = np.column_stack((p["position_min"], -np.asarray(p["velocity_bounds"])))+self.state_margin
        self.high = np.column_stack((p["position_max"], p["velocity_bounds"]))-self.state_margin
        if np.any(self.input_bound <= 0) or np.any(self.low >= self.high):
            raise ValueError("Tube tightening makes the nominal constraints empty")
        N = self.N
        self.Sx = np.vstack([np.linalg.matrix_power(self.A, k+1) for k in range(N)])
        self.Su = np.zeros((2*N, N))
        for k in range(N):
            for j in range(k+1):
                self.Su[2*k:2*k+2, j] = np.linalg.matrix_power(self.A, k-j)@self.B
        Q = np.diag(p["state_cost"])
        R, S = p["input_cost"], p["input_change_cost"]
        terminal = solve_discrete_are(self.A, self.B[:, None], Q, np.array([[R]]))
        self.Q = sparse.block_diag([sparse.csc_matrix(Q)]*(N-1)+[sparse.csc_matrix(terminal)]).toarray()
        self.D = np.eye(N)-np.eye(N, k=-1)
        H = self.Su.T@self.Q@self.Su+R*np.eye(N)+S*self.D.T@self.D
        self.constraint = np.vstack((self.Su, np.eye(N)))
        self.solvers = []
        for axis in range(3):
            solver = osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(2*H)), q=np.zeros(N),
                A=sparse.csc_matrix(self.constraint), l=np.full(3*N, -np.inf),
                u=np.full(3*N, np.inf), verbose=False, eps_abs=1e-7, eps_rel=1e-7,
                max_iter=8000, polishing=False)
            self.solvers.append(solver)
        self.nominal = None
        self.previous_input = np.zeros(3)
        self.prediction = np.zeros((N+1,3,2))
        self.sequence = np.zeros((N,3))
        self.last_time = None
        self.count = 0

    def nominal_at(self, time):
        elapsed = time-self.last_time
        A = np.array([[1., elapsed], [0., 1.]])
        B = np.array([elapsed**2/2, elapsed])
        return self.nominal@A.T+self.previous_input[:, None]*B

    def update(self, time, position, velocity):
        measured = np.column_stack((position, velocity))
        due = self.last_time is None or time-self.last_time >= self.dt-1e-9
        self.updated = due
        if due:
            self.nominal = measured.copy() if self.last_time is None else self.nominal_at(time)
            self.last_time = time
            future = [self.trajectory.at(time+(j+1)*self.dt) for j in range(self.N)]
            refs = np.stack([np.column_stack((r.position, r.velocity)) for r in future])
            acceleration = np.stack([self.trajectory.at(time+j*self.dt).acceleration for j in range(self.N)])
            self.solve_ms, self.constraint_violation = 0., 0.
            for axis, solver in enumerate(self.solvers):
                free = self.Sx@self.nominal[axis]
                offset = np.zeros(self.N); offset[0] = self.previous_input[axis]
                q = 2*(self.Su.T@self.Q@(free-refs[:, axis].ravel())
                       -self.p["input_cost"]*acceleration[:, axis]
                       -self.p["input_change_cost"]*self.D.T@offset)
                lower = np.r_[np.tile(self.low[axis], self.N)-free, np.full(self.N, -self.input_bound[axis])]
                upper = np.r_[np.tile(self.high[axis], self.N)-free, np.full(self.N, self.input_bound[axis])]
                solver.update(q=q, l=lower, u=upper)
                result = solver.solve(raise_error=False)
                if result.info.status_val != 1:
                    raise ModelDomainError(f"Tube nominal QP axis {axis} failed: {result.info.status}")
                check = self.constraint@result.x
                violation = max(0., float(np.max(lower-check)), float(np.max(check-upper)))
                if violation > 2e-5:
                    raise ModelDomainError("Tube QP failed explicit constraint check")
                self.constraint_violation = max(self.constraint_violation, violation)
                self.sequence[:, axis] = result.x
                self.prediction[0, axis] = self.nominal[axis]
                self.prediction[1:, axis] = (free+self.Su@result.x).reshape(-1, 2)
                self.solve_ms += result.info.run_time*1000
            self.previous_input = self.sequence[0].copy()
            self.count += 1
        self.error = measured-self.nominal_at(time)
        self.utilization = np.sqrt(np.einsum("ni,ij,nj->n", self.error, self.P, self.error))/self.rho
        self.input = self.previous_input+self.error@self.K
        return self.input.copy()

    def metadata(self):
        return {"model": "three_independent_double_integrators", "horizon_steps": self.N,
                "prediction_period_s": self.dt, "horizon_s": self.N*self.dt,
                "nominal_state_reset_each_solve": False,
                "tube_type": "product_of_Lyapunov_RPI_ellipsoids_for_additive_W",
                "K": self.K.tolist(), "P": self.P.tolist(), "gamma": float(self.gamma),
                "rho": self.rho.tolist(), "state_tightening": self.state_margin.tolist(),
                "input_tightening": self.input_margin.tolist(),
                "assumed_acceleration_residual_bound_m_s2": self.w_bound.tolist(),
                "terminal_cost": "DARE", "terminal_invariant_constraint": False,
                "nonlinear_plant_robust_stability_proven": False}
