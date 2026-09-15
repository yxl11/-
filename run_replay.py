"""Export and execute a frozen three-UAV trajectory, with no online load feedback."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from pathlib import Path
from sling_sim.replay import ReplayTrajectory, ReplayCase, default_cases, export_replay, run_replay, save_replay


def execute(source, output, case, duration=None, progress=False):
    config = json.loads((Path(source)/"config_snapshot.json").read_text(encoding="utf-8"))
    bundle = Path(output)/"trajectory"/"trajectory.npz"
    trajectory = ReplayTrajectory.load(bundle)
    arrays, metrics, actual = run_replay(config, trajectory, case, duration, progress)
    save_replay(Path(output)/"replay"/case.name, arrays, metrics, actual, config, bundle)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/experiment_v1/planning"))
    parser.add_argument("--output", type=Path, default=Path("results/experiment_v1"))
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--cases", type=Path, help="JSON list of explicit ReplayCase settings")
    parser.add_argument("--append", action="store_true", help="Keep previous cases and reuse the identical frozen trajectory")
    args = parser.parse_args()
    if args.suite and args.duration:
        parser.error("The acceptance suite must run the complete recorded trajectory")
    if args.append and not args.cases:
        parser.error("--append requires an explicit --cases file")
    if not args.append:
        export_replay(args.source, args.output/"trajectory")
    else:
        from sling_sim.replay import file_hash
        manifest = json.loads((args.output/"trajectory/manifest.json").read_text(encoding="utf-8"))
        if (manifest["source_log_sha256"] != file_hash(args.source/"state_log.npz")
                or manifest["source_config_sha256"] != file_hash(args.source/"config_snapshot.json")
                or manifest["trajectory_sha256"] != file_hash(args.output/"trajectory/trajectory.npz")):
            parser.error("Appended cases must use the unchanged source and frozen trajectory")
    cases = [ReplayCase(**row) for row in json.loads(args.cases.read_text(encoding="utf-8"))] if args.cases else (default_cases() if args.suite else [ReplayCase()])
    if len({c.name for c in cases}) != len(cases) or any(not re.fullmatch(r"[a-zA-Z0-9_]+", c.name) for c in cases):
        parser.error("Case names must be unique plain directory names")
    previous = json.loads((args.output/"replay_suite_metrics.json").read_text(encoding="utf-8")) if args.append else []
    if any(c.name in {m['scenario'] for m in previous} for c in cases):
        parser.error("Appended case names already exist")
    # Store assumptions/thresholds before any case is run.
    from dataclasses import asdict
    (args.output/"replay_cases.json").write_text(json.dumps([m['case'] for m in previous]+[asdict(c) for c in cases], indent=2), encoding="utf-8")
    results = []
    if args.suite or args.cases:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(execute, args.source, args.output, case): case.name for case in cases}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"{result['scenario']}: status={result['status']}, screen={result['engineering_screen_passed']}, "
                      f"payload RMSE={result['payload_position_rmse_m']:.4f} m, tilt={result['payload_tilt_peak_deg']:.3f} deg", flush=True)
    else:
        results.append(execute(args.source, args.output, cases[0], args.duration, True))
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    order = {c.name: i for i, c in enumerate(cases)}
    results.sort(key=lambda m: order[m["scenario"]])
    (args.output/"replay_suite_metrics.json").write_text(json.dumps(previous+results, indent=2, ensure_ascii=False), encoding="utf-8")
    if any(m["status"] != "completed" for m in results) or ((args.suite or args.cases) and any(
            not m["engineering_screen_passed"] for m in results if m["case"]["category"] == "acceptance")):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
