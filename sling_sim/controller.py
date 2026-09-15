"""Baseline: individual position PD + nominal load sharing + attitude PD.

No payload state feedback, tension truth feedback, PDE, observer, or MPC is used.
Nominal inertial feedforward and hover pretension are only controller assumptions.
"""
import numpy as np
from .model import unpack
from .rotations import quaternion_matrix, desired_rotation, vee


class BaselineController:
    def __init__(self, config, plant_geometry):
        self.c = dict(config)
        self.offsets = plant_geometry.offsets.copy()
        self.radial = plant_geometry.radial.copy()
        self.mixer_inverse = plant_geometry.mixer_inverse.copy()
        self.inertia = np.array(config["uav_inertia"])
        self.force = np.zeros((3, 3))
        self.last_reference = None
        self.saturated = np.zeros((3, 4), dtype=bool)

    def position_update(self, y, reference):
        c = self.c
        p, v, _, _, _ = unpack(y)
        self.last_reference = reference
        pref = reference.position + self.offsets
        nominal_share = c["payload_mass"]/3
        self.force = np.tile((c["uav_mass"]+nominal_share)*reference.acceleration, (3, 1))
        self.force[:, 2] += (c["uav_mass"]+nominal_share)*c["gravity"]
        self.force += nominal_share*c["gravity"]*np.tan(np.deg2rad(c["hover_rope_angle_deg"]))*self.radial
        self.force -= c["uav_mass"]*(np.array(c["position_kp"])*(p[1:]-pref)
                          + np.array(c["position_kd"])*(v[1:]-reference.velocity))

    def attitude_update(self, y):
        _, _, quat, omega, _ = unpack(y)
        r = quaternion_matrix(quat[1:])
        commands = np.empty((3, 4))
        for i in range(3):
            rd = desired_rotation(self.force[i])
            error = 0.5*vee(rd.T @ r[i] - r[i].T @ rd)
            moment = (-np.array(self.c["attitude_kp"])*error
                      - np.array(self.c["attitude_kd"])*omega[i+1]
                      + np.cross(omega[i+1], self.inertia*omega[i+1]))
            thrust = max(0.0, float(self.force[i] @ r[i, :, 2]))
            commands[i] = self.mixer_inverse @ np.r_[thrust, moment]
        clipped = np.clip(commands, 0, self.c["max_rotor_thrust"])
        self.saturated = np.abs(commands-clipped) > 1e-10
        return clipped
