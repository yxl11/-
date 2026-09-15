"""PDE/Backstepping outer loop connected to the independent rigid-body plant."""
import numpy as np
from .controller import BaselineController
from .model import unpack
from .pde import PDEFormation


def limit_vectors(values, limit):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    factors = np.minimum(1.0, limit/np.maximum(norms, 1e-12))
    return values*factors, bool(np.any(factors < 1-1e-12))


class PDEController(BaselineController):
    """Reuse only the actuator/attitude implementation of the old controller.

    Position PD from BaselineController.position_update is NOT called.
    Velocity commands depend on the discretized PDE and measured neighbor states.
    Neither load truth nor solved cable tension is fed to this controller.
    """
    def __init__(self, config, plant_geometry):
        super().__init__(config, plant_geometry)
        self.pde = PDEFormation(config["pde"])
        self.velocity_limited = self.acceleration_limited = False

    def position_update(self, y, reference):
        c = self.c
        share = c["payload_mass"]/3
        cable_force = np.tile(share*reference.acceleration, (3, 1))
        cable_force[:, 2] += share*c["gravity"]
        cable_force += share*c["gravity"]*np.tan(np.deg2rad(c["hover_rope_angle_deg"]))*self.radial
        self.track_formation(y, reference, reference.position+self.offsets,
                             np.tile(reference.velocity, (3,1)),
                             np.tile(reference.acceleration, (3,1)), cable_force)

    def track_formation(self, y, reference, positions, velocities, accelerations, cable_force):
        c, g = self.c, self.c["pde"]
        p, v, _, _, _ = unpack(y)
        self.last_reference = reference
        self.uav_reference = positions.copy()
        self.uav_reference_velocity = velocities.copy()
        self.uav_reference_acceleration = accelerations.copy()
        self.error = p[1:]-self.uav_reference
        self.error_velocity = v[1:]-velocities
        correction, boundary, residual = self.pde.feedback(self.error, self.error_velocity)
        self.boundary_target = self.uav_reference[2]+boundary
        self.boundary_residual = residual
        self.raw_velocity_command = velocities+correction
        self.velocity_command, self.velocity_limited = limit_vectors(
            self.raw_velocity_command, g["velocity_limit"])
        self.velocity_defect = v[1:]-self.velocity_command
        self.acceleration_command, self.acceleration_limited = limit_vectors(
            accelerations-g["velocity_gain"]*self.velocity_defect,
            g["acceleration_limit"])
        self.force = c["uav_mass"]*self.acceleration_command
        self.force[:, 2] += c["uav_mass"]*c["gravity"]
        self.force += cable_force

    def diagnostics(self, y, reference):
        p, v, _, _, _ = unpack(y)
        error = p[1:]-self.uav_reference
        ev = v[1:]-self.uav_reference_velocity
        correction, boundary, residual = self.pde.feedback(error, ev)
        return {"pde_error": error, "pde_error_velocity": ev,
                "pde_transformed_error": self.pde.volterra_nodes(error),
                "pde_velocity_command": self.velocity_command.copy(),
                "pde_raw_velocity_command": self.raw_velocity_command.copy(),
                "pde_acceleration_command": self.acceleration_command.copy(),
                "pde_boundary_target": self.uav_reference[2]+boundary,
                "pde_boundary_residual": residual,
                "pde_interior_equation_defect": ev[1]-correction[1],
                "pde_velocity_defect": v[1:]-self.velocity_command,
                "pde_limit_flags": np.array([self.velocity_limited, self.acceleration_limited])}
