import unittest
import numpy as np
from sling_sim.config import load_config
from sling_sim.model import SlingPlant, unpack, ModelDomainError
from sling_sim.trajectory import TransportTrajectory
from sling_sim.controller import BaselineController
from sling_sim.rotations import quaternion_matrix, matrix_quaternion, desired_rotation


class PhysicsChecks(unittest.TestCase):
    def setUp(self):
        self.c = load_config()
        self.plant = SlingPlant(self.c)
        self.traj = TransportTrajectory(self.c)
        self.y = self.plant.equilibrium(self.traj.at(0).position)
        self.command = unpack(self.y)[4].copy()

    def test_static_balance_matches_hand_calculation(self):
        dy, d = self.plant.evaluate(0, self.y, self.command)
        expected = 3*9.81/(3*np.cos(np.deg2rad(20)))
        np.testing.assert_allclose(d["tension"], expected, atol=1e-11)
        self.assertLess(np.linalg.norm(dy), 1e-10)
        self.assertLess(np.abs(d["length_error"]).max(), 1e-14)

    def test_internal_forces_cancel_in_linear_momentum(self):
        y = self.y.copy()
        p, v, q, w, f = unpack(y)
        v[:] = [.2, -.1, .05]
        f[0] *= 1.04
        _, d = self.plant.evaluate(0, y, self.command)
        gravity = np.array([0, 0, -self.c["gravity"]])*self.plant.masses.sum()
        actual = (self.plant.masses[:, None]*d["acceleration"]).sum(axis=0)
        np.testing.assert_allclose(actual, d["external_force"].sum(axis=0)+gravity, atol=1e-12)

    def test_constraint_acceleration_is_zero_for_tangent_velocity(self):
        y = self.y.copy()
        p, v, q, w, f = unpack(y)
        # Relative velocities tangential to each rope, with no initial length error.
        v[1:] = np.cross(self.plant.rest_directions, np.array([0, 0, .2]))
        _, d = self.plant.evaluate(0, y, self.command)
        arms = self.plant.anchors
        relative_a = d["acceleration"][1:] - d["acceleration"][0] - np.cross(d["angular_acceleration"][0], arms)
        gdd = self.c["rope_length"]*np.sum(d["directions"]*relative_a, axis=1) + np.sum(d["relative_velocity"]**2, axis=1)
        np.testing.assert_allclose(gdd, 0, atol=1e-10)
        self.assertLess(abs(d["cable_power"]), 1e-12)

    def test_internal_cable_moments_cancel_about_world_origin(self):
        y = self.y.copy()
        unpack(y)[4][1] *= 1.05
        _, d = self.plant.evaluate(0, y, self.command)
        p = unpack(y)[0]
        forces = d["tension"][:, None]*d["directions"]
        total = np.cross(p[0], forces.sum(axis=0)) + np.cross(p[1:], -forces).sum(axis=0)
        total += np.cross(d["attachments"]-p[0], forces).sum(axis=0)
        np.testing.assert_allclose(total, 0, atol=1e-12)

    def test_negative_tension_is_rejected_not_clamped(self):
        y = self.y.copy()
        unpack(y)[4][:] = 0.0
        with self.assertRaises(ModelDomainError):
            self.plant.evaluate(0, y, np.zeros((3,4)), lambda t: np.array([0,0,5.0]))

    def test_quaternion_rotation_round_trip(self):
        r = desired_rotation(np.array([2.0, -3.0, 9.0]), .7)
        np.testing.assert_allclose(quaternion_matrix(matrix_quaternion(r)), r, atol=1e-14)

    def test_controller_does_not_read_payload_feedback(self):
        first = BaselineController(self.c, self.plant)
        second = BaselineController(self.c, self.plant)
        changed = self.y.copy()
        changed[:3] += [1, 2, 3]
        first.position_update(self.y, self.traj.at(0))
        second.position_update(changed, self.traj.at(0))
        np.testing.assert_allclose(first.attitude_update(self.y), second.attitude_update(changed), atol=0)

    def test_reference_is_smooth_at_segment_boundaries(self):
        h, d = self.traj.hold, self.traj.leg
        for t in (h, h+d, 2*h+d, 2*h+2*d):
            left, right = self.traj.at(t-1e-5), self.traj.at(t+1e-5)
            for attr in ("position", "velocity", "acceleration", "jerk"):
                np.testing.assert_allclose(getattr(left, attr), getattr(right, attr), atol=1e-7)

    def test_rotor_allocation_obeys_realizable_bounds(self):
        ctrl = BaselineController(self.c, self.plant)
        y = self.y.copy()
        unpack(y)[0][1:] += [50, 50, -50]
        ctrl.position_update(y, self.traj.at(0))
        cmd = ctrl.attitude_update(y)
        self.assertTrue(np.any(ctrl.saturated))
        self.assertGreaterEqual(cmd.min(), 0)
        self.assertLessEqual(cmd.max(), self.c["max_rotor_thrust"])


if __name__ == "__main__":
    unittest.main()
