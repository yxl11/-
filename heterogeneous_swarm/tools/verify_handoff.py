#!/usr/bin/env python3
"""Audit relocated handoff inputs and recompute saved evidence; no trajectories.

Run with Python 3.11 and reproduction/requirements.txt installed. --out must be
a new directory outside reproduction/. This tool never invokes run_full.py or
validate_delivery.py, and never writes into the frozen evidence tree.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reproduction"


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="New audit output directory")
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error("--out must not exist; previous evidence is never overwritten")
    if out == PACKAGE or out.is_relative_to(PACKAGE):
        parser.error("--out cannot be inside the frozen reproduction package")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    # Child processes deliberately run outside the reproduction package.
    probe_cwd = out / "foreign_cwd"
    probe_cwd.mkdir()
    summary = {
        "status": "RUNNING", "started_utc": datetime.now(timezone.utc).isoformat(),
        "package": str(PACKAGE), "verification_output": str(out),
        "child_working_directory": str(probe_cwd),
        "environment": {
            "platform": platform.platform(), "python": sys.version,
            "python_executable": sys.executable, "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("numpy", "scipy", "matplotlib")},
        },
        "commands": [], "preflights": [],
        "new_closed_loop_trajectories": 0, "total_model_propagation_steps": 0,
        "scope": "Hash verification; saved-trajectory statistics and plot-data "
                 "recomputation; initial controller construction and nominal "
                 "reference evaluation. No trajectory rerun or timing rerun.",
        "bitwise_policy": "Initial-reference SHA256 comparisons are informational. "
                          "A platform-dependent bit difference is not automatically "
                          "a numerical or physical failure. Investigate numeric "
                          "differences separately before drawing physical conclusions.",
        "numerical_policy": "Frozen recomputation tolerances are retained: "
                            "statistics <1e-9 and plot arrays atol=3e-10, rtol=0. "
                            "A failed check diagnoses reproducibility, not physical safety.",
    }
    write_json(out / "summary.json", summary)

    def run(name, arguments):
        command = [sys.executable, "-X", "utf8", "-B", *map(str, arguments)]
        start = time.perf_counter()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(command, cwd=probe_cwd, capture_output=True,
                                text=True, encoding="utf-8", env=env)
        stdout = logs / f"{name}.stdout.txt"
        stderr = logs / f"{name}.stderr.txt"
        stdout.write_text(result.stdout, encoding="utf-8")
        stderr.write_text(result.stderr, encoding="utf-8")
        record = {"name": name, "argv": command, "cwd": str(probe_cwd),
                  "exit_code": result.returncode,
                  "elapsed_wall_s": time.perf_counter() - start,
                  "stdout": str(stdout.relative_to(out)),
                  "stderr": str(stderr.relative_to(out))}
        summary["commands"].append(record)
        write_json(out / "summary.json", summary)
        if result.returncode:
            raise RuntimeError(f"{name} returned {result.returncode}; see saved logs")
        parsed = parse_json(result.stdout)
        if parsed.get("status") != "PASS":
            raise RuntimeError(f"{name} did not report PASS")
        return parsed

    exit_code = 0
    try:
        summary["integrity_before"] = run("integrity_before", [PACKAGE / "verify_package.py"])
        recomputed = out / "recomputed"  # Must not exist before reproduce.py runs.
        summary["recomputed_stdout"] = run("recomputed", [PACKAGE / "reproduce.py", "--out", recomputed])
        report = json.loads((recomputed / "report.json").read_text(encoding="utf-8"))
        checks = report["figure_array_checks"]
        summary["recomputed"] = {
            "status": report["status"],
            "per_seed_metric_checks": report["per_seed_metric_checks"],
            "maximum_statistical_error": report["maximum_statistical_error"],
            "figure_array_checks": len(checks),
            "maximum_figure_array_error": max(row["max_absolute_error"] for row in checks),
            "detailed_report": "recomputed/report.json",
            "statistical_checks": "recomputed/numerical_checks.csv",
        }
        if report["per_seed_metric_checks"] != 351 or len(checks) != 37:
            raise RuntimeError("Unexpected recomputation coverage")
        if report["new_closed_loop_trajectories"] != 0:
            raise RuntimeError("Recomputation scope unexpectedly changed")

        for method in ("P", "RAS04", "MMD19"):
            snapshot = PACKAGE / "snapshots" / ("P" if method == "P" else "external")
            for seed in (33001, 33002, 33003):
                name = f"preflight_{method}_{seed}"
                answer = run(name, [PACKAGE / "preflight_context.py", "--snapshot", snapshot,
                                    "--method", method, "--seed", seed])
                history = PACKAGE / "data" / f"{method}_s{seed}" / "history.npz"
                with np.load(history, allow_pickle=False) as archive:
                    initial_held = archive["held_reference"][0]
                    saved_hash = hashlib.sha256(initial_held.tobytes()).hexdigest()
                    shape = list(initial_held.shape)
                    dtype = str(initial_held.dtype)
                answer.update({
                    "historical_held_reference_0_sha256": saved_hash,
                    "historical_held_reference_0_shape": shape,
                    "historical_held_reference_0_dtype": dtype,
                    "raw_command_bitwise_matches_saved_held_reference":
                        answer["raw_command_sha256"] == saved_hash,
                    "bitwise_comparison_is_informational": True,
                    "history_path": str(history.relative_to(PACKAGE)),
                })
                if answer["state_propagation_steps"] != 0 or answer["closed_loop_executed"]:
                    raise RuntimeError("Preflight unexpectedly propagated the model")
                summary["preflights"].append(answer)
                write_json(out / "summary.json", summary)
                print(f"PASS {name}", flush=True)
        summary["integrity_after"] = run("integrity_after", [PACKAGE / "verify_package.py"])
        matches = sum(x["raw_command_bitwise_matches_saved_held_reference"]
                      for x in summary["preflights"])
        summary["preflight_count"] = len(summary["preflights"])
        summary["initial_command_bitwise_matches"] = matches
        summary["initial_command_bitwise_differences"] = len(summary["preflights"]) - matches
        summary["status"] = "PASS" if matches == 9 else "PASS_WITH_BITWISE_DIFFERENCES"
    except Exception as error:
        summary["status"] = "FAILED_REPRODUCIBILITY_CHECK"
        summary["error"] = str(error)
        (out / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        exit_code = 1
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["exit_code"] = exit_code
    write_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out / "summary.json"),
                      "preflights": len(summary["preflights"]),
                      "new_closed_loop_trajectories": 0}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
