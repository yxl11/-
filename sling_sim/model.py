"""Constrained Newton-Euler plant; it does not know the reference or controller.

Body 0 is the payload; bodies 1..3 are quadrotors. World z is up. Cable forces
act at each quadrotor COM and at three noncollinear payload attachment points.
The massless inextensible-cable taut mode is enforced. A negative solved
tension raises ModelDomainError, rather than silently treating a rope as a rod.
"""
import numpy as np
from .rotations import quaternion_matrix, quaternion_derivative, matrix_quaternion, desired_rotation


class ModelDomainError(RuntimeError):
    pass


def unpack(y):
    return (y[:12].reshape(4, 3), y[12:24].reshape(4, 3),
            y[24:40].reshape(4, 4), y[40:52].reshape(4, 3), y[52:64].reshape(3, 4))


class SlingPlant:
    def __init__(self, config):
        self.c = dict(config)
        self.masses = np.array([config["payload_mass"]] + [config["uav_mass"]]*3)
        x, y, z = config["payload_dimensions"]
        il = self.masses[0]/12*np.array([y*y+z*z, x*x+z*z, x*x+y*y])
        self.inertia = np.vstack((il, np.tile(config["uav_inertia"], (3, 1))))
        phi = np.arange(3)*2*np.pi/3
        self.radial = np.column_stack((np.cos(phi), np.sin(phi), np.zeros(3)))
        self.anchors = config["anchor_radius"]*self.radial
        self.anchors[:, 2] = config["anchor_height"]
        alpha = np.deg2rad(config["hover_rope_angle_deg"])
        self.rest_directions = self.radial*np.sin(alpha)
        self.rest_directions[:, 2] = np.cos(alpha)
        self.offsets = self.anchors + config["rope_length"]*self.rest_directions
        arm = config["arm_length"]/np.sqrt(2)
        rotor_x, rotor_y = np.array([1, -1, -1, 1])*arm, np.array([1, 1, -1, -1])*arm
        self.rotor_points = np.column_stack((rotor_x, rotor_y, np.zeros(4)))
        self.mixer = np.vstack((np.ones(4), rotor_y, -rotor_x,
                               config["rotor_yaw_coefficient"]*np.array([1, -1, 1, -1])))
        self.mixer_inverse = np.linalg.inv(self.mixer)
        self.linear_drag = np.array([config["payload_linear_drag"]] + [config["uav_linear_drag"]]*3)
        self.angular_drag = np.array([config["payload_angular_drag"]] + [config["uav_angular_drag"]]*3)

    def equilibrium(self, position):
        y = np.zeros(64)
        p, v, q, w, rotors = unpack(y)
        p[0] = position
        p[1:] = position + self.offsets
        q[:, 0] = 1.0
        tension = self.masses[0]*self.c["gravity"]/(3*self.rest_directions[0, 2])
        forces = tension*self.rest_directions.copy()
        forces[:, 2] += self.masses[1:]*self.c["gravity"]
        for i in range(3):
            q[i+1] = matrix_quaternion(desired_rotation(forces[i]))
            rotors[i] = np.linalg.norm(forces[i])/4
        if np.max(rotors) >= self.c["max_rotor_thrust"]:
            raise ModelDomainError("The assumed vehicle cannot sustain the chosen hover load")
        return y

    def evaluate(self, time, y, rotor_command, payload_force=None, check_domain=True):
        c = self.c
        p, v, quat, omega, rotors = unpack(y)
        rotation = quaternion_matrix(quat)
        wrench = rotors @ self.mixer.T
        external = -self.linear_drag[:, None]*v
        external[1:] += wrench[:, :1]*rotation[1:, :, 2]
        if payload_force is not None:
            external[0] += payload_force(time)
        free_acceleration = external/self.masses[:, None]
        free_acceleration[:, 2] -= c["gravity"]
        moment = -self.angular_drag[:, None]*omega
        moment[1:] += wrench[:, 1:]
        angular_acceleration = (moment - np.cross(omega, self.inertia*omega))/self.inertia

        rl = rotation[0]
        arms = self.anchors @ rl.T
        attachment = p[0] + arms
        delta = p[1:] - attachment
        length = np.linalg.norm(delta, axis=1)
        if np.any(length < 1e-8):
            raise ModelDomainError("Degenerate zero-length cable geometry")
        direction = delta/length[:, None]
        omega_world = rl @ omega[0]
        relative_velocity = v[1:] - v[0] - np.cross(omega_world, arms)
        alpha_free_world = rl @ angular_acceleration[0]
        relative_free_acceleration = (free_acceleration[1:] - free_acceleration[0]
             - np.cross(alpha_free_world, arms) - np.cross(omega_world, np.cross(omega_world, arms)))
        torque_arms_body = np.cross(self.anchors, direction @ rl)
        matrix = (direction @ direction.T/self.masses[0]
                 + (torque_arms_body/self.inertia[0]) @ torque_arms_body.T
                 + np.diag(1/self.masses[1:]))
        constraint = 0.5*(length**2 - c["rope_length"]**2)
        radial_speed = np.sum(direction*relative_velocity, axis=1)
        rhs = (np.sum(direction*relative_free_acceleration, axis=1)
               + np.sum(relative_velocity**2, axis=1)/length
               + 2*c["constraint_zeta"]*c["constraint_omega"]*radial_speed
               + c["constraint_omega"]**2*constraint/length)
        tension = np.linalg.solve(matrix, rhs)
        if check_domain and np.min(tension) < -1e-7:
            raise ModelDomainError(f"t={time:.6f}s: negative tension {tension.tolist()}; slack mode is not implemented")
        cable_force = tension[:, None]*direction
        acceleration = free_acceleration.copy()
        acceleration[0] += cable_force.sum(axis=0)/self.masses[0]
        acceleration[1:] -= cable_force/self.masses[1:, None]
        angular_acceleration[0] += (torque_arms_body.T @ tension)/self.inertia[0]
        dy = np.concatenate((v.ravel(), acceleration.ravel(),
              quaternion_derivative(quat, omega).ravel(), angular_acceleration.ravel(),
              ((rotor_command-rotors)/c["motor_time_constant"]).ravel()))
        diagnostics = {"tension": tension, "length_error": length-c["rope_length"],
            "directions": direction, "attachments": attachment,
            "relative_velocity": relative_velocity, "constraint_matrix": matrix,
            "acceleration": acceleration, "angular_acceleration": angular_acceleration,
            "external_force": external, "rotation": rotation,
            "cable_power": -float(np.dot(tension, radial_speed)), "rotor_wrench": wrench}
        return dy, diagnostics

    def step(self, time, y, rotor_command, dt, payload_force=None):
        def rhs(t, state):
            return self.evaluate(t, state, rotor_command, payload_force)[0]
        k1 = rhs(time, y)
        k2 = rhs(time+dt/2, y+dt*k1/2)
        k3 = rhs(time+dt/2, y+dt*k2/2)
        k4 = rhs(time+dt, y+dt*k3)
        out = y+dt*(k1+2*k2+2*k3+k4)/6
        q = out[24:40].reshape(4, 4)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        if not np.all(np.isfinite(out)):
            raise ModelDomainError("Non-finite state")
        return out

    def energy(self, y):
        p, v, q, omega, rotors = unpack(y)
        return float(0.5*np.sum(self.masses[:, None]*v*v)
                     + 0.5*np.sum(self.inertia*omega*omega)
                     + np.dot(self.masses, p[:, 2])*self.c["gravity"])

