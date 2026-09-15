import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import numpy as np
from sling_sim.config import load_config, FORCE_CONFIG
from sling_sim.model import SlingPlant
from sling_sim.replay import (ReplayTrajectory, ReplayController, ReplayCase, run_replay,
                              export_replay, evaluate_acceptance)


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.c = load_config(FORCE_CONFIG)
        self.plant = SlingPlant(self.c)

    def hover(self, duration=.4):
        t = np.linspace(0, duration, 21)
        p = np.tile(np.array([-3, 0, self.c["payload_com_height"]])+self.plant.offsets, (len(t), 1, 1))
        return ReplayTrajectory(t, p, np.zeros_like(p), np.zeros_like(p))

    def test_Hermite_reproduces_quintic_and_analytic_derivatives(self):
        t = np.array([0., .3, .7, 1.])
        coefficients = np.random.default_rng(3).normal(size=(6, 3, 3))
        def evaluate(x, order):
            import math
            return sum(coefficients[k]*math.factorial(k)/math.factorial(k-order)*x**(k-order)
                       for k in range(order, 6))
        fields = [np.array([evaluate(x, order) for x in t]) for order in range(3)]
        replay = ReplayTrajectory(t, *fields)
        for x in np.linspace(0, 1, 27):
            for order, value in enumerate(replay.at(x)):
                np.testing.assert_allclose(value, evaluate(x, order), atol=2e-12)
        # Three clocks are evaluated independently, with consistent derivatives.
        for order, value in enumerate(replay.at([.1, .4, .9])):
            for i, x in enumerate([.1, .4, .9]):
                np.testing.assert_allclose(value[i], evaluate(x, order)[i], atol=2e-12)

    def test_C2_across_knots_and_endpoint_hold(self):
        t = np.arange(4.)
        rng = np.random.default_rng(8)
        p, v, a = [rng.normal(size=(4, 3, 3)) for _ in range(3)]
        replay = ReplayTrajectory(t, p, v, a)
        for i in range(1, 3):
            for order, expected in enumerate((p[i], v[i], a[i])):
                np.testing.assert_allclose(replay.at(t[i])[order], expected, atol=1e-12)
                np.testing.assert_allclose(replay.at(t[i]-1e-9)[order], expected, atol=2e-6)
        np.testing.assert_allclose(replay.at(-1)[0], p[0])
        np.testing.assert_array_equal(replay.at(4)[1], np.zeros((3, 3)))

    def test_table_rejects_bad_time_nonfinite_data_and_hidden_channels(self):
        p = np.zeros((2, 3, 3))
        with self.assertRaises(ValueError):
            ReplayTrajectory([0, 0], p, p, p)
        bad = p.copy(); bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            ReplayTrajectory([0, 1], bad, p, p)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"bad.npz"
            np.savez(path, time=[0, 1], position=p, velocity=p, acceleration=p,
                     yaw=np.zeros((2, 3)), tension_truth=np.ones(3))
            with self.assertRaises(ValueError):
                ReplayTrajectory.load(path)

    def test_controller_input_boundary_excludes_payload_and_tension(self):
        controller = ReplayController(self.c)
        ref = self.hover().at(0)
        with self.assertRaises(ValueError):
            controller.position_update(np.zeros((4, 3)), np.zeros((4, 3)), ref)
        controller.position_update(ref[0], ref[1], ref)
        expected = self.plant.equilibrium([-3, 0, self.c["payload_com_height"]])
        q = expected[24:40].reshape(4, 4)
        motors = controller.attitude_update(q[1:], np.zeros((3, 3)))
        np.testing.assert_allclose(motors, expected[52:64].reshape(3, 4), atol=1e-12)

    def test_replay_runs_without_planner_and_response_is_not_assigned(self):
        # Any attempt to instantiate an online planner makes this test fail.
        with patch("sling_sim.force_controller.ForceMappedPDEController", side_effect=AssertionError("online planner")):
            arrays, metrics, actual = run_replay(self.c, self.hover(),
                ReplayCase("mass_mismatch", payload_mass_factor=1.1), duration=.4)
        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["nominal_payload_mass_kg"], 3.)
        self.assertAlmostEqual(actual["payload_mass"], 3.3)
        self.assertFalse(metrics["payload_feedback"])
        self.assertGreater(metrics["payload_position_peak_m"], 1e-4)
        for values in arrays.values():
            self.assertEqual(len(values), len(arrays["time"]))

    def test_measurement_latency_uses_past_not_current_state(self):
        arrays, _, _ = run_replay(self.c, self.hover(),
            ReplayCase("delay", payload_mass_factor=1.1, measurement_delay_s=.04), duration=.4)
        pos = arrays["state"][:, :12].reshape(-1, 4, 3)[:, 1:]
        np.testing.assert_allclose(arrays["measured_uav_position"][2:], pos[:-2], atol=1e-14)
        self.assertGreater(np.max(np.abs(pos[2:]-arrays["measured_uav_position"][2:])), 1e-5)

    def test_noise_is_seeded_and_affects_measurements(self):
        case = ReplayCase("noise", position_noise_std_m=.005, velocity_noise_std_m_s=.01)
        a, _, _ = run_replay(self.c, self.hover(), case, .1)
        b, _, _ = run_replay(self.c, self.hover(), case, .1)
        np.testing.assert_array_equal(a["state"], b["state"])
        self.assertGreater(np.max(np.abs(a["measured_uav_position"][0]-a["uav_reference"][0])), 1e-4)

    def test_export_contains_reference_not_actual_response(self):
        trajectory = self.hover()
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp)/"source", Path(temp)/"bundle"
            source.mkdir()
            (source/"metrics.json").write_text(json.dumps({"status": "completed"}))
            (source/"config_snapshot.json").write_text("{}")
            np.savez(source/"state_log.npz", time=trajectory.time, uav_reference=trajectory.p,
                uav_reference_velocity=trajectory.v, uav_reference_acceleration=trajectory.a,
                state=np.ones((len(trajectory.time), 64))*999)
            manifest = export_replay(source, destination)
            loaded = ReplayTrajectory.load(destination/"trajectory.npz")
            np.testing.assert_array_equal(loaded.p, trajectory.p)
            self.assertFalse(manifest["payload_feedback"])
            self.assertEqual(manifest["samples_per_uav"], len(trajectory.time))

    def test_local_integrator_does_not_wind_up_under_unreachable_position(self):
        controller = ReplayController(self.c)
        ref = self.hover().at(0)
        for _ in range(500):
            controller.position_update(ref[0]-100, ref[1], ref)
        self.assertTrue(controller.limited)
        np.testing.assert_array_equal(controller.integral_acceleration, np.zeros((3, 3)))
        controller.position_update(ref[0], ref[1], ref)
        self.assertFalse(controller.limited)
        np.testing.assert_allclose(controller.acceleration_command, 0, atol=1e-14)

    def test_per_UAV_reference_ages_are_applied_in_physical_replay(self):
        base = self.hover()
        t = base.time
        p, v, a = base.p.copy(), base.v.copy(), base.a.copy()
        p[:, :, 0] += .02*t[:, None]**3
        v[:, :, 0] = .06*t[:, None]**2
        a[:, :, 0] = .12*t[:, None]
        trajectory = ReplayTrajectory(t, p, v, a)
        case = ReplayCase("aged", command_delay_s=(.02, .04, .06), clock_offset_s=(0, .01, -.01))
        data, _, _ = run_replay(self.c, trajectory, case, .2)
        times = np.array([.18, .17, .13])
        np.testing.assert_allclose(data["playback_time_per_uav"][-1], times)
        np.testing.assert_allclose(data["received_reference_position"][-1], trajectory.at(times)[0], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
