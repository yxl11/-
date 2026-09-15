import tempfile
import unittest
import numpy as np
from sling_sim.config import load_config, FORCE_CONFIG
from sling_sim.model import SlingPlant
from sling_sim.trajectory import TransportTrajectory
from sling_sim.force_mapping import CableAllocator, SmoothForce, MomentumESO, inverse_geometry
from sling_sim.tube_mpc import TubeMPC
from sling_sim.simulation import run_simulation, save_run


class ForceMappingChecks(unittest.TestCase):
    def setUp(self):
        self.c = load_config(FORCE_CONFIG)
        self.plant = SlingPlant(self.c)
        self.trajectory = TransportTrajectory(self.c)

    def test_vector_allocation_satisfies_force_moment_and_positive_cones(self):
        allocator = CableAllocator(self.c, self.plant.anchors, self.plant.radial)
        force, moment = np.array([.8, -.5, 29.43]), np.array([.01, -.02, .004])
        f = allocator.solve(force, moment)
        np.testing.assert_allclose(f.sum(axis=0), force, atol=1e-6)
        np.testing.assert_allclose(np.cross(self.plant.anchors, f).sum(axis=0), moment, atol=1e-6)
        norms = np.linalg.norm(f, axis=1)
        self.assertTrue(np.all(norms > 4) and np.all(norms < 18))
        self.assertLess(np.max(np.arccos(f[:,2]/norms)), np.deg2rad(35))

    def test_inverse_mapping_uses_attachment_offsets_and_changes_shape(self):
        allocator = CableAllocator(self.c, self.plant.anchors, self.plant.radial)
        f = allocator.solve(np.array([.8, 0., 29.43]), np.zeros(3))
        ref = self.trajectory.at(0)
        p, _, _, q = inverse_geometry(ref, self.plant.anchors, 1.8, f, np.zeros((3,3)), np.zeros((3,3)))
        np.testing.assert_allclose(np.linalg.norm(p-ref.position-self.plant.anchors, axis=1), 1.8, atol=1e-14)
        self.assertGreater(np.linalg.norm(p-(ref.position+self.plant.offsets)), .01)
        # Force directions reconstruct the same optimized resultant.
        np.testing.assert_allclose((q*np.linalg.norm(f,axis=1)[:,None]).sum(axis=0), [.8,0,29.43], atol=1e-6)

    def test_inverse_analytic_velocity_and_acceleration(self):
        ref = self.trajectory.at(0)
        f = np.array([[3., 0., 10.], [-1.,3.,11.], [-2.,-3.,9.]])
        fd = np.array([[.1,-.2,.03], [.2,.1,-.02], [-.1,.2,.01]])
        fdd = np.ones((3,3))*.02
        def position(t):
            return inverse_geometry(ref, self.plant.anchors, 1.8, f+t*fd+.5*t*t*fdd, fd+t*fdd, fdd)[0]
        p, v, acc, _ = inverse_geometry(ref, self.plant.anchors, 1.8, f, fd, fdd)
        h = 1e-3
        np.testing.assert_allclose(v, (position(h)-position(-h))/(2*h), atol=1e-9)
        np.testing.assert_allclose(acc, (position(h)-2*p+position(-h))/h**2, atol=3e-9)

    def test_force_filter_preserves_convex_segment_and_C2_at_target_change(self):
        f = np.array([[1., 2., 10.], [-1.,-2.,10.], [0.,0.,10.]])
        filt = SmoothForce(f, 5.)
        target = f*1.1
        before = (filt.value.copy(), filt.velocity.copy(), filt.acceleration.copy())
        filt.target = target
        for current, old in zip((filt.value, filt.velocity, filt.acceleration), before):
            np.testing.assert_array_equal(current, old)
        for _ in range(100):
            filt.advance(.02)
            low, high = np.minimum(f,target), np.maximum(f,target)
            self.assertTrue(np.all(filt.value >= low-1e-12) and np.all(filt.value <= high+1e-12))

    def test_momentum_ESO_estimates_unprovided_disturbance(self):
        eso = MomentumESO(3., 3.)
        true_acceleration = np.array([.1, -.03, .01])
        for t in np.arange(0, 8.001, .02):
            estimated = eso.update(t*true_acceleration, np.zeros(3), .02 if t else 0.)
        np.testing.assert_allclose(estimated, 3*true_acceleration, atol=1e-7)

    def test_tube_invariant_boundary_under_corner_disturbances(self):
        import itertools
        mpc = TubeMPC(self.c, self.trajectory)
        root = np.linalg.cholesky(mpc.P)
        angles = np.linspace(0, 2*np.pi, 101)
        for axis in range(3):
            boundary = np.linalg.solve(root.T, np.array([np.cos(angles), np.sin(angles)]))*mpc.rho[axis]
            for corner in itertools.product((-1.,1.), repeat=2):
                w = np.array(corner)*mpc.B*mpc.w_bound[axis]
                nxt = mpc.closed@boundary+w[:,None]
                norm = np.sqrt(np.einsum('in,ij,jn->n', nxt, mpc.P, nxt))
                self.assertLessEqual(norm.max(), mpc.rho[axis]*(1+1e-12))
        self.assertTrue(np.all(mpc.input_margin > 0))

    def test_MPC_uses_future_reference_and_keeps_nominal_separate(self):
        mpc = TubeMPC(self.c, self.trajectory)
        ref = self.trajectory.at(7.5)
        command = mpc.update(7.5, ref.position, ref.velocity)
        self.assertGreater(np.linalg.norm(command), 1e-5)  # current reference acceleration is zero
        expected_nominal = mpc.nominal_at(7.6)
        measured = ref.position+np.array([.1, 0., 0.])
        mpc.update(7.6, measured, ref.velocity)
        np.testing.assert_allclose(mpc.nominal, expected_nominal, atol=1e-13)
        self.assertGreater(np.linalg.norm(mpc.error), .05)

    def test_integrated_hover_export_uses_per_UAV_derivatives(self):
        import csv
        a, m = run_simulation(self.c, "hover", .2)
        self.assertEqual(m["status"], "completed")
        self.assertLess(m["payload_position_peak_m"], 1e-7)
        for value in a.values():
            self.assertEqual(len(value), len(a["time"]))
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            save_run(folder, a, m, self.c)
            with (Path(folder)/"reference_only_NOT_FLIGHT_READY.csv").open(encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            np.testing.assert_allclose(np.array(rows[-1][5:8],float), a["uav_reference_velocity"][-1,2])

    def test_rotational_damping_reaches_force_before_geometry_moves(self):
        from sling_sim.force_controller import ForceMappedPDEController
        from sling_sim.model import unpack
        controller = ForceMappedPDEController(self.c, self.plant, self.trajectory)
        ref = self.trajectory.at(0)
        state = self.plant.equilibrium(ref.position)
        unpack(state)[3][0, 0] = .1
        hover = controller.filter.value.copy()
        measured = np.r_[np.array([0., 0., self.c["payload_mass"]*self.c["gravity"]]), np.zeros(3)]
        controller.update(0, state, ref, measured)
        np.testing.assert_allclose(controller.uav_reference, ref.position+self.plant.offsets, atol=1e-14)
        hover_thrust = hover.copy(); hover_thrust[:,2] += self.c["uav_mass"]*self.c["gravity"]
        self.assertGreater(np.linalg.norm(controller.force-hover_thrust), 1e-4)
        self.assertLess(np.linalg.norm(controller.allocator.G[3:]@controller.filter.value.ravel()), 1e-12)


if __name__ == "__main__":
    unittest.main()
