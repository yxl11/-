"""Proposal section 2.3(3) force prediction -> inverse formation -> PDE loop."""
import numpy as np
from .pde_controller import PDEController
from .tube_mpc import TubeMPC
from .force_mapping import CableAllocator, SmoothForce, MomentumESO, inverse_geometry
from .model import unpack
from .rotations import quaternion_matrix, vee


class ForceMappedPDEController(PDEController):
    def __init__(self, config, plant, trajectory):
        super().__init__(config, plant)
        self.mapping = config["force_mapping"]
        self.anchors = plant.anchors.copy()
        self.load_inertia = plant.inertia[0].copy()
        self.mpc = TubeMPC(config, trajectory)
        self.allocator = CableAllocator(config, self.anchors, self.radial)
        self.filter = SmoothForce(self.allocator.previous, self.mapping["force_filter_bandwidth"])
        self.eso_force = MomentumESO(config["payload_mass"], self.mapping["eso_bandwidth"])
        self.eso_moment = MomentumESO(self.load_inertia, self.mapping["eso_bandwidth"])
        self.last_time = None

    def update(self, time, y, reference, measured_wrench):
        c = self.c
        p, v, quat, omega, _ = unpack(y)
        rotation = quaternion_matrix(quat[0])
        dt = 0. if self.last_time is None else time-self.last_time
        self.last_time = time
        self.filter.advance(dt)  # Advance the PREVIOUS held target causally.
        force_measured, moment_measured = measured_wrench[:3], measured_wrench[3:]
        known_acc = force_measured/c["payload_mass"]-np.array([0., 0., c["gravity"]])
        self.disturbance_force = self.eso_force.update(v[0], known_acc, dt)
        torque_body = rotation.T@moment_measured
        gyro = np.cross(omega[0], self.load_inertia*omega[0])
        self.disturbance_moment = self.eso_moment.update(
            omega[0], (torque_body-gyro)/self.load_inertia, dt)
        self.load_acceleration_command = self.mpc.update(time, p[0], v[0])
        self.base_force = c["payload_mass"]*(reference.acceleration+np.array([0.,0.,c["gravity"]]))
        self.eso_feedforward = self.base_force-self.disturbance_force
        self.requested_force = c["payload_mass"]*(self.load_acceleration_command+
                                   np.array([0.,0.,c["gravity"]]))-self.disturbance_force
        error_R = .5*vee(rotation-rotation.T)  # R_load,desired == I
        self.requested_moment = rotation@(-self.mapping["attitude_gain"]*error_R
            -self.mapping["angular_velocity_gain"]*omega[0]+gyro-self.disturbance_moment)
        self.raw_cable_force = self.allocator.solve(self.requested_force, self.requested_moment)
        self.filter.target = self.raw_cable_force.copy()
        position, velocity, acceleration, self.desired_direction = inverse_geometry(
            reference, self.anchors, c["rope_length"], self.filter.value,
            self.filter.velocity, self.filter.acceleration)
        # Current allocated force provides prompt load/attitude damping. Only the
        # geometric reference is filtered: delaying torque feedback through the
        # third-order reference filter can destabilize the suspended-load mode.
        self.track_formation(y, reference, position, velocity, acceleration, self.raw_cable_force)

    def diagnostics(self, y, reference):
        out = super().diagnostics(y, reference)
        smooth_wrench = self.allocator.G@self.filter.value.ravel()
        out.update({"force_base": self.base_force, "force_eso_feedforward": self.eso_feedforward,
            "force_requested": self.requested_force, "moment_requested": self.requested_moment,
            "eso_force_estimate": self.disturbance_force, "eso_moment_estimate": self.disturbance_moment,
            "force_allocated_raw": self.raw_cable_force, "force_allocated_smooth": self.filter.value,
            "force_allocated_rate": self.filter.velocity, "force_allocated_acceleration": self.filter.acceleration,
            "force_allocation_residual": self.allocator.residual, "force_smoothed_wrench": smooth_wrench,
            "force_direction_reference": self.desired_direction,
            "load_acceleration_command": self.load_acceleration_command,
            "mpc_nominal_state": self.mpc.nominal_at(self.last_time),
            "mpc_prediction": self.mpc.prediction,
            "mpc_acceleration_prediction": self.mpc.sequence,
            "mpc_updated": np.array(self.mpc.updated), "tube_utilization": self.mpc.utilization,
            "mpc_constraint_violation": np.array(self.mpc.constraint_violation),
            "optimization_solve_ms": np.array([self.mpc.solve_ms, self.allocator.solve_ms])})
        return {k: np.asarray(value).copy() for k, value in out.items()}

    def metadata(self):
        return {"source_section": "2.3(3)", "source_equations": list(range(17, 30)),
            "tube_mpc": self.mpc.metadata(), "mpc_solves": self.mpc.count,
            "allocation_solves": self.allocator.count,
            "allocation_variables": "9 cable force vector components",
            "allocation_wrench_equalities": 6, "attachment_offset_in_inverse": True,
            "reference_smoothing": "causal_third_order_critical_force_filter_C2",
            "actuator_force_feedforward": "current_QP_force_without_geometry_filter_delay",
            "measurement_assumption": "ideal_load_pose_velocity_angular_velocity_and_cable_wrench",
            "injected_disturbance_truth_used_by_controller": False,
            "scope": "translation_tube_MPC_rotation_ESO_feedback_vector_allocation_PDE"}
