import argparse
import json
from pathlib import Path
from sling_sim.config import load_config, FORCE_CONFIG
from sling_sim.simulation import run_simulation, save_run


def main():
    parser = argparse.ArgumentParser(description="Three-UAV PDE formation simulation; not a PX4 flight tool")
    parser.add_argument("--config", type=Path, default=FORCE_CONFIG)
    parser.add_argument("--scenario", choices=["nominal", "hover", "perturbed_hover", "pulse", "pde_transport", "force_transport"], default=None)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--leg-duration", type=float)
    parser.add_argument("--payload-mass", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--output", type=Path, default=Path("results/force_mapping"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--animation", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for field, value in (("payload_mass", args.payload_mass), ("physics_dt", args.dt), ("leg_duration", args.leg_duration)):
        if value is not None:
            if value <= 0:
                parser.error(f"{field} must be positive")
            config[field] = value
    for field in ("position_period", "attitude_period", "log_period"):
        ratio = config[field]/config["physics_dt"]
        if ratio < 1 or abs(ratio-round(ratio)) > 1e-8:
            parser.error(f"{field} must be an integer multiple of physics dt")
    scenario = args.scenario or {"force_pde": "force_transport", "pde": "pde_transport"}.get(config.get("controller"), "nominal")
    arrays, metrics = run_simulation(config, scenario, args.duration, progress=True)
    save_run(args.output, arrays, metrics, config)
    if args.plots or args.animation:
        from sling_sim.visualization import plot_run, animate_run
        if args.plots:
            plot_run(arrays, metrics, config, args.output)
        if args.animation:
            animate_run(arrays, metrics, config, args.output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if metrics["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
