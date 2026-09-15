import tempfile
import unittest
from pathlib import Path
import numpy as np
from sling_sim.config import load_config, PDE_CONFIG
from sling_sim.pde import PDEFormation, backstepping_kernel
from sling_sim.pde_controller import PDEController
from sling_sim.model import SlingPlant, unpack
from sling_sim.trajectory import TransportTrajectory
from sling_sim.simulation import run_simulation, save_run


class PDEChecks(unittest.TestCase):
    def setUp(self):
        self.c = load_config(PDE_CONFIG)
        self.pde = PDEFormation(self.c["pde"])
        self.plant = SlingPlant(self.c)
        self.reference = TransportTrajectory(self.c).at(0)
        self.state = self.plant.equilibrium(self.reference.position)

    def test_kernel_diagonal_and_origin_limits(self):
        x = np.linspace(0, 1, 20)
        np.testing.assert_allclose(backstepping_kernel(x, x, self.pde.sigma), -self.pde.sigma*x/2)
        np.testing.assert_allclose(backstepping_kernel(x, 0*x, self.pde.sigma), 0)
        np.testing.assert_allclose(backstepping_kernel(x, x/2, 0), 0)

    def test_kernel_satisfies_hyperbolic_kernel_equation(self):
        # Independent finite-difference residual: k_xx-k_yy = sigma*k.
        s, d = self.pde.sigma, 1e-4
        for x, y in ((.8, .3), (.6, .4), (.9, .1)):
            k = backstepping_kernel(x, y, s)
            xx = (backstepping_kernel(x+d,y,s)-2*k+backstepping_kernel(x-d,y,s))/d**2
            yy = (backstepping_kernel(x,y+d,s)-2*k+backstepping_kernel(x,y-d,s))/d**2
            self.assertAlmostEqual(float(xx-yy), float(s*k), delta=1e-6)

    def test_convection_transform_has_required_sign(self):
        # For e=exp(a*x), exp(b*x)*(e_xx+2b e_x+lambda e)
        # equals q_xx+(lambda-b^2)q with q=exp(b*x)e.
        b, lam, a, x = .2, 1., .7, .4
        left = (a*a+2*b*a+lam)*np.exp((a+b)*x)
        right = ((a+b)**2+lam-b*b)*np.exp((a+b)*x)
        self.assertAlmostEqual(left, right)
        expected = self.pde.h*backstepping_kernel(1, .5, self.pde.sigma)*np.exp(-b*.5)
        self.assertAlmostEqual(self.pde.weights[1], float(expected))

    def test_dynamic_boundary_enforces_dirichlet_residual(self):
        e = np.array([[.03, -.04, .01], [.01, .02, 0], [-.02, .04, 0]])
        ev = np.zeros((3,3))
        ev[:2] = self.pde.operator @ e
        correction, _, residual = self.pde.feedback(e, ev)
        bdot = correction[2]-self.pde.weights @ correction[:2]
        np.testing.assert_allclose(bdot, -self.c["pde"]["boundary_gain"]*residual, atol=1e-14)
        self.assertLess(self.pde.ideal_eigenvalues.real.max(), 0)

    def test_neighbor_feedback_changes_follower_and_leader(self):
        ctrl = PDEController(self.c, self.plant)
        ctrl.position_update(self.state, self.reference)
        first = ctrl.force.copy()
        changed = self.state.copy()
        unpack(changed)[0][2, 1] += .01  # follower measurement only
        ctrl.position_update(changed, self.reference)
        self.assertGreater(np.linalg.norm(ctrl.force[2]-first[2]), 1e-3)
        self.assertGreater(np.linalg.norm(ctrl.force[1]-first[1]), 1e-3)
        np.testing.assert_allclose(ctrl.force[0], first[0])

    def test_reference_lifting_preserves_translation_invariance(self):
        from copy import deepcopy
        ctrl1, ctrl2 = PDEController(self.c, self.plant), PDEController(self.c, self.plant)
        shifted, reference = self.state.copy(), deepcopy(self.reference)
        shift = np.array([100., -30., 20.])
        unpack(shifted)[0][:] += shift
        reference.position += shift
        ctrl1.position_update(self.state, self.reference)
        ctrl2.position_update(shifted, reference)
        np.testing.assert_allclose(ctrl1.force, ctrl2.force, atol=1e-11)

    def test_controller_does_not_read_load_truth(self):
        ctrl = PDEController(self.c, self.plant)
        ctrl.position_update(self.state, self.reference)
        first = ctrl.force.copy()
        changed = self.state.copy()
        unpack(changed)[0][0] += [1, 2, 3]
        unpack(changed)[1][0] += [4, 5, 6]
        ctrl.position_update(changed, self.reference)
        np.testing.assert_array_equal(ctrl.force, first)

    def test_pde_hover_and_log_lengths(self):
        arrays, metrics = run_simulation(self.c, "hover", duration=.2)
        self.assertEqual(metrics["status"], "completed")
        self.assertLess(metrics["payload_position_peak_m"], 1e-12)
        for values in arrays.values():
            self.assertEqual(len(values), len(arrays["time"]))
        with tempfile.TemporaryDirectory() as folder:
            save_run(folder, arrays, metrics, self.c)
            self.assertTrue((Path(folder)/"pde_control_commands.csv").exists())


if __name__ == "__main__":
    unittest.main()
