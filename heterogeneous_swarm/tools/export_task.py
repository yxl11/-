"""Export the frozen task and initial states, without simulating any aircraft.

Run from any directory with Python 3.11+ and the reproduction requirements:
    python -X utf8 -B tools/export_task.py --out work/task_export_01

The destination must not exist. Outputs are derived reading aids: online
feedback still requires actual state and continuous reference evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproduction"
SNAPSHOT = REPRO / "snapshots/P"
SEEDS = (33001, 33002, 33003)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwritten")
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    require(not out.exists(), f"Output already exists: {out}")
    require(REPRO not in out.parents, "Keep derived output outside the frozen reproduction directory")
    sys.path.insert(0, str(REPRO))
    from verify_package import verify
    integrity = verify()

    folder = SNAPSHOT / "output/reviews/mixed_sweep_orbit_external_20261101"
    mission_path, shape_path = folder / "mission.py", folder / "shape_models.py"
    manifest_path = SNAPSHOT / "output/reviews/mixed_sweep_orbit_paired_20261102/initial_manifest.json"
    config_path = REPRO / "data/P_s33001/configuration.json"
    protocol_path = SNAPSHOT / "output/reviews/p_gain_calibration_20261108/protocol.json"
    protocol, cfg, initial_manifest = read(protocol_path), read(config_path), read(manifest_path)
    require(tuple(protocol["evaluation_seeds"]) == SEEDS, "Unexpected selected evaluation seed set")
    require(cfg["counts"] == [10, 16, 20], "Unexpected role counts")
    require(cfg["kc_per_s"] == .24 and cfg["kf_per_s"] == .16, "Unexpected P gains")
    require(cfg["duration_s"] == 178. and cfg["task_period_s"] == .1, "Unexpected task clock")

    mission = load_module("handoff_task_mission", mission_path)
    shape = load_module("handoff_task_shape", shape_path)
    t = np.arange(1781, dtype=np.float64) * .1
    q, qd, qdd = mission.reference_kinematics(t)
    S, dS = shape.shape_state(t)
    require(q.shape == qd.shape == qdd.shape == (1781, 3, 3), "Unexpected reference array shape")
    require(S.shape == dS.shape == (1781, 3, 3, 3), "Unexpected shape-transform array shape")
    require(all(np.isfinite(a).all() for a in (q, qd, qdd, S, dS)), "Nonfinite task samples")
    require(np.all(np.linalg.det(S) > 0.), "Nonpositive role transform determinant")

    # Preserve archived ordering. Dense current member IDs are NOT historic IDs.
    current_ids = np.arange(46, dtype=np.int64)
    historical_ids = np.asarray(initial_manifest["globally_retained_ids"], dtype=np.int64)
    expected_roles = np.repeat(np.arange(3), cfg["counts"])
    require(historical_ids.shape == (46,), "Unexpected historical member-ID mapping")
    initials = {}
    source_files = [mission_path, shape_path, manifest_path, config_path, protocol_path,
                    SNAPSHOT / "output/reviews/mixed_role_corridor_demo_20261031/mission.py",
                    SNAPSHOT / "output/reviews/mixed_role_corridor_demo_20261031/artifact_manifest.json"]
    for seed in SEEDS:
        record = initial_manifest["files"][str(seed)]
        path = manifest_path.parent / record["path"]
        require(sha(path) == record["sha256"], f"Initial-state hash mismatch: {seed}")
        source_files.append(path)
        with np.load(path, allow_pickle=False) as saved:
            r, v, roles = (saved[k].copy() for k in ("positions", "velocities", "roles"))
        require(r.shape == v.shape == (46, 3), f"Unexpected state shape: {seed}")
        require(np.array_equal(roles, expected_roles), f"Unexpected role order: {seed}")
        require(np.isfinite(r).all() and np.isfinite(v).all(), f"Nonfinite initial state: {seed}")
        initials[seed] = (r, v, roles)

    # Validate all inputs before creating the destination. No existing files are replaced.
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "task_samples_10hz.npz", time_s=t,
                        q_enu_m=q, qd_enu_mps=qd, qdd_enu_mps2=qdd,
                        S_world_m=S, dS_world_mps=dS)
    write_json(out / "configuration_P.json", cfg)
    for seed, (r, v, roles) in initials.items():
        np.savez_compressed(out / f"initial_s{seed}.npz", positions=r, velocities=v, roles=roles,
                            member_ids=current_ids, historical_member_ids=historical_ids)
        write_json(out / f"initial_s{seed}.json", {
            "seed": seed, "coordinates": "world ENU: x east, y north, z up; metres and metres/second",
            "source_npz_sha256": sha(manifest_path.parent / initial_manifest["files"][str(seed)]["path"]),
            "members": [{"member_id": int(k), "historical_member_id": int(historical_ids[k]),
                         "role": int(roles[k]), "role_name": cfg["names"][roles[k]],
                         "position_enu_m": r[k].tolist(), "velocity_enu_mps": v[k].tolist()}
                        for k in range(46)]})

    write_json(out / "vehicle_map.template.json", {
        "configured": False,
        "note": "Populate from the actual PX4/Gazebo installation. Null values are intentional; no ports or model IDs are guessed.",
        "members": [{"member_id": int(k), "role": int(expected_roles[k]),
                     "role_name": cfg["names"][expected_roles[k]], "gazebo_model_name": None,
                     "px4_instance": None, "mavros_namespace": None, "mav_sys_id": None,
                     "mavlink_udp_port": None, "shared_map_transform": None}
                    for k in range(46)]})
    (out / "README.md").write_text(
        "# Frozen task exports\n\n"
        "These are reading/adapter inputs, not new simulation results or executable PX4 commands.\n\n"
        "- `task_samples_10hz.npz`: 1781 samples including 0 and 178 s. "
        "q/qd/qdd axes are [time, role, world xyz]; S/dS axes are [time, role, world row, material column]. "
        "The task map is x = q_i + S_i z with dimensionless z. S has metres, dS metres/second. "
        "World coordinates are ENU by migration convention; confirm the shared physical map before use.\n"
        "- Online task references must call the source analytic functions at the actual simulation time. "
        "Do not infer a continuous controller or its feedforward from linear interpolation of these samples.\n"
        "- `initial_s*.npz/json`: the three selected archived states, preserving row order. "
        "member_id is the present 0--45 index; historical_member_id is provenance only. "
        "They start airborne with nonzero speed; they are not takeoff commands.\n"
        "- `configuration_P.json`: selected P run configuration; use the source-map guide to distinguish active and inherited fields.\n"
        "- `vehicle_map.template.json`: fill and validate actual instance/namespace/port/frame mappings before ROS use.\n"
        "- `export_manifest.json`: input hashes, schemas, and output hashes. All frozen inputs were verified before export.\n",
        encoding="utf-8")
    outputs = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    result = {"status": "PASS", "purpose": "task and initial-state export; no trajectory propagation",
              "frozen_package_integrity": integrity, "script_sha256": sha(Path(__file__)),
              "source_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
              "time_samples": len(t), "time_step_s": .1, "time_range_s": [0., 178.],
              "role_counts": cfg["counts"], "seeds": list(SEEDS), "model_propagation_steps": 0,
              "outputs": outputs}
    write_json(out / "export_manifest.json", result)
    print(json.dumps({"status": "PASS", "out": str(out), "time_samples": len(t),
                      "initial_states": len(initials), "members_per_state": 46,
                      "model_propagation_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
