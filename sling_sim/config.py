from pathlib import Path
import json
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
PDE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pde.json"
FORCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "force_pde.json"


def load_config(path=DEFAULT_CONFIG):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    positive = ("gravity", "payload_mass", "uav_mass", "rope_length",
                "motor_time_constant", "max_rotor_thrust", "physics_dt",
                "leg_duration", "hold_duration", "arm_length", "rotor_yaw_coefficient")
    for name in positive:
        if not np.isfinite(c[name]) or c[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("uav_inertia", "payload_dimensions"):
        if np.shape(c[name]) != (3,) or np.any(np.asarray(c[name]) <= 0):
            raise ValueError(f"{name} must contain three positive values")
    for name in ("position_period", "attitude_period", "log_period"):
        ratio = c[name]/c["physics_dt"]
        if ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-9):
            raise ValueError(f"{name} must be an integer multiple of physics_dt")
    if not 0 < c["hover_rope_angle_deg"] < 70:
        raise ValueError("hover_rope_angle_deg must be in (0, 70)")
    if c.get("controller") in ("pde", "force_pde"):
        from .pde import PDEFormation
        PDEFormation(c["pde"])
        shift = np.asarray(c["pde"]["initial_translation_m"])
        if shift.shape != (3,) or not np.all(np.isfinite(shift)):
            raise ValueError("initial_translation_m must have three finite values")
        if not np.isfinite(c["pde"]["initial_payload_tilt_deg"]):
            raise ValueError("initial_payload_tilt_deg must be finite")
    if c.get("controller") == "force_pde":
        f = c["force_mapping"]
        if f["horizon"] < 2 or int(f["horizon"]) != f["horizon"]:
            raise ValueError("horizon must be an integer >= 2")
        for name in ("prediction_period", "eso_bandwidth", "force_filter_bandwidth", "input_cost",
                     "input_change_cost", "tension_min", "tension_max"):
            if not np.isfinite(f[name]) or f[name] <= 0:
                raise ValueError(f"force_mapping.{name} must be positive")
        ratio = f["prediction_period"]/c["position_period"]
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("prediction_period must be a multiple of position_period")
        if not 0 < f["cone_angle_deg"] < 70 or f["tension_min"] >= f["tension_max"]:
            raise ValueError("Invalid cable cone/tension constraints")
    return c
