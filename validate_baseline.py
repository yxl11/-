"""Integration checks and a same-controller, same-log-time step convergence study."""
from pathlib import Path
import json
import numpy as np
from sling_sim.config import load_config
from sling_sim.simulation import run_simulation, save_run
from sling_sim.visualization import plot_run


def main():
    c = load_config()
    out = Path("results/validation")
    checks, runs = {}, {}
    for name, duration in (("hover", 10), ("perturbed_hover", 20), ("pulse", 40)):
        print(f"Running validation scenario: {name}", flush=True)
        arrays, metrics = run_simulation(c, name, duration, progress=True)
        save_run(out/name, arrays, metrics, c)
        plot_run(arrays, metrics, c, out/name)
        runs[name] = metrics
        checks[name+"_completed"] = metrics["status"] == "completed"
        checks[name+"_positive_tension"] = metrics["tension_min_N"] > 0
        checks[name+"_constraint_residual"] = metrics["rope_length_error_peak_m"] < 1e-6
        checks[name+"_quaternion_norm"] = metrics["quaternion_norm_error_peak"] < 1e-10
        checks[name+"_actuator_bound"] = metrics["rotor_thrust_peak_N"] <= c["max_rotor_thrust"]+1e-10
    checks["hover_equilibrium_preserved"] = runs["hover"]["payload_position_peak_m"] < 1e-8
    states, convergence_metrics = {}, {}
    for dt in (.004, .002, .001):
        local = dict(c, physics_dt=dt)
        print(f"Step convergence: dt={dt}", flush=True)
        data, metrics = run_simulation(local, "perturbed_hover", 8)
        states[dt] = data
        convergence_metrics[str(dt)] = metrics
    fine = states[.001]
    errors = {}
    for dt in (.004, .002):
        np.testing.assert_allclose(states[dt]["time"], fine["time"], atol=1e-12)
        difference = states[dt]["state"]-fine["state"]
        errors[str(dt)] = {
            "max_position_difference_m": float(np.linalg.norm(difference[:,:12].reshape(-1,4,3), axis=2).max()),
            "max_tension_difference_N": float(np.abs(states[dt]["tension"]-fine["tension"]).max()),
        }
    checks["convergence_position_at_2ms"] = errors["0.002"]["max_position_difference_m"] < 1e-5
    checks["convergence_tension_at_2ms"] = errors["0.002"]["max_tension_difference_N"] < 1e-3
    checks["refinement_reduces_error"] = errors["0.002"]["max_position_difference_m"] < errors["0.004"]["max_position_difference_m"]
    report = {"all_passed": all(checks.values()), "checks": checks, "scenario_metrics": runs,
              "step_convergence_errors_vs_1ms": errors, "step_convergence_metrics": convergence_metrics}
    out.mkdir(parents=True, exist_ok=True)
    (out/"validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"all_passed": report["all_passed"], "checks": checks,
                      "step_convergence_errors_vs_1ms": errors}, indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
