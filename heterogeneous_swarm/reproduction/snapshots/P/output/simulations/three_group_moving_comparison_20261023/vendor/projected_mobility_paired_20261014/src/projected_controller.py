"""Fixed K-orthogonal group projection for the SAME point-density energy."""
from __future__ import annotations
import numpy as np
from point_controller import PointController


class ProjectedController(PointController):
    """projected uses original kf; point is the exact zero-projection ablation."""

    def __init__(self, cfg, method="projected", projection_gain=None):
        if method not in ("projected", "point"):
            raise ValueError("method must be projected or point")
        super().__init__(cfg, "point")
        selected = float(cfg["kf_per_s"]) if method == "projected" else 0.
        if projection_gain is not None and float(projection_gain) != selected:
            raise ValueError("Only the registered full or zero projection gain is allowed")
        self.method = method
        self.projection_gain = selected
        self.K_diag = self.D * self.omega[:, None] / self.sigma**2
        self.K_sum_diag = self.K_diag.sum(axis=0)
        if np.any(self.K_diag <= 0):
            raise ValueError("Positive fixed Gaussian task metric required")
        self.implementation_metadata.update({
            "method": method, "projection_gain_per_s": selected,
            "mean_metric_K_diag_per_s": self.K_diag.tolist(),
            "projection": "nu=qd-kc*x-kf_proj*(x-z), x=Sigma/D*mean(g), z=sum(h)/sum(K), h=omega*mean(g)",
            "ablation": "kf_proj=0 restores point exactly; kappa=0 is NOT old no_AB",
            "communication": "All-group mean-gradient aggregate; original B remains in graph energy",
            "execution_capability_adaptive": False,
        })
        self.implementation_metadata["field_semantics"].update({
            "formation": "graph mean command plus projected density-induced displacement command",
            "nu": "qd+analytic self anchor+graph mean+projection mean",
            "projection_mean_command": "-kf_proj*(effective_displacement-common_weighted_displacement)",
            "true_mean_energy_gradient": "h=omega*mean(tilde_g)=omega*mean(true g); not exact full point gradient",
            "shape": "Unchanged point raw shape command at the same state",
        })

    def fields(self, r, t):
        f = super().fields(r, t)
        mean_g = self.means(f["approx_gradient"])
        h = self.omega[:, None] * mean_g
        x = self.sigma**2 / self.D * mean_g
        z = h.sum(axis=0) / self.K_sum_diag
        delta = -self.projection_gain * (x - z)
        if self.projection_gain:
            f["formation"] = f["formation"] + delta
            f["nu"] = f["nu"] + delta
            f["cnom"] = f["nu"][self.roles] + f["shape"]
            f["ctest"] = f["cnom"] + f["excitation"]
            f["zero_mean_error"] = float(np.max(abs(self.means(f["cnom"]) - f["nu"])))
        f.update({
            "projection_mean_command": delta,
            "effective_displacement": x,
            "projected_common_displacement": z,
            "true_mean_energy_gradient": h,
            "projection_extra_nominal_power": float(-self.projection_gain * np.sum(self.K_diag * (x - z)**2)),
        })
        return f


Controller = ProjectedController
