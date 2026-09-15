from pathlib import Path
import csv
import hashlib
import json
import platform
import time as wallclock
import numpy as np
from .model import SlingPlant, ModelDomainError, unpack
from .controller import BaselineController
from .pde_controller import PDEController
from .trajectory import TransportTrajectory
from .rotations import quaternion_matrix


def disturbance(time):
    """Known test input: 2 N sin-squared pulse in world y, from t=20 to 22 s."""
    value = 2*np.sin(np.pi*(time-20)/2)**2 if 20 < time < 22 else 0.0
    return np.array([0.0, value, 0.0])


def run_simulation(config, scenario="nominal", duration=None, progress=False):
    c = dict(config)
    plant = SlingPlant(c)
    trajectory = TransportTrajectory(c, "hover" if scenario in ("hover", "perturbed_hover") else "s_curve")
    duration = trajectory.duration if duration is None else float(duration)
    dt = c["physics_dt"]
    steps = int(round(duration/dt))
    if steps < 1 or not np.isclose(steps*dt, duration):
        raise ValueError("duration must be a positive integer multiple of physics_dt")
    position_stride = int(round(c["position_period"]/dt))
    attitude_stride = int(round(c["attitude_period"]/dt))
    log_stride = int(round(c["log_period"]/dt))
    if min(position_stride, attitude_stride, log_stride) < 1:
        raise ValueError("Control/log periods must not be shorter than physics_dt")
    use_force = c.get("controller") == "force_pde"
    use_pde = c.get("controller", "baseline") in ("pde", "force_pde")
    if c.get("controller", "baseline") not in ("baseline", "pde", "force_pde"):
        raise ValueError("Unknown controller")
    if use_force:
        from .force_controller import ForceMappedPDEController
        controller = ForceMappedPDEController(c, plant, trajectory)
    else:
        controller = PDEController(c, plant) if use_pde else BaselineController(c, plant)
    state = plant.equilibrium(trajectory.at(0).position)
    if scenario == "pde_transport":
        if not use_pde:
            raise ValueError("pde_transport requires controller=pde")
        p, _, quat, _, _ = unpack(state)
        p[:] += np.asarray(c["pde"]["initial_translation_m"])
        angle = np.deg2rad(c["pde"]["initial_payload_tilt_deg"])
        quat[0] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        p[1:] = p[0]+plant.anchors @ quaternion_matrix(quat[0]).T+c["rope_length"]*plant.rest_directions
    if scenario == "perturbed_hover":
        p, _, quat, _, _ = unpack(state)
        angle = np.deg2rad(3)
        quat[0] = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        p[1:] = p[0] + plant.anchors @ quaternion_matrix(quat[0]).T + c["rope_length"]*plant.rest_directions
    load_force = disturbance if scenario == "pulse" else None
    if scenario == "force_transport":
        if not use_force:
            raise ValueError("force_transport requires controller=force_pde")
        f = c["force_mapping"]
        def load_force(t):
            s = (t-f["wind_start_s"])/f["wind_duration_s"]
            return np.array([0., f["wind_peak_N"]*np.sin(np.pi*s)**2 if 0 < s < 1 else 0., 0.])
    command = unpack(state)[4].copy()
    records = {k: [] for k in ("time", "state", "reference_position", "reference_velocity",
              "reference_acceleration", "uav_reference", "rotor_command", "tension",
              "length_error", "cable_direction", "load_force", "cable_power")}
    physical_keys = tuple(records)
    status, reason = "completed", None
    saturated_updates = total_updates = 0
    velocity_limited_updates = acceleration_limited_updates = position_updates = 0
    start_wall = wallclock.perf_counter()
    for step in range(steps+1):
        time = step*dt
        reference = trajectory.at(time)
        try:
            if step % position_stride == 0:
                if use_force:
                    # Ideal simulated load-cell + cable-direction sensor. Do not
                    # expose injected wind, drag truth, or true acceleration to ESO.
                    _, sensed = plant.evaluate(time, state, command, load_force)
                    cable_vectors = sensed["tension"][:, None]*sensed["directions"]
                    p_load = unpack(state)[0][0]
                    measured_wrench = np.r_[cable_vectors.sum(axis=0),
                        np.cross(sensed["attachments"]-p_load, cable_vectors).sum(axis=0)]
                    controller.update(time, state, reference, measured_wrench)
                else:
                    controller.position_update(state, reference)
                position_updates += 1
                if use_pde:
                    velocity_limited_updates += int(controller.velocity_limited)
                    acceleration_limited_updates += int(controller.acceleration_limited)
            if step % attitude_stride == 0:
                command = controller.attitude_update(state)
                saturated_updates += int(np.any(controller.saturated))
                total_updates += 1
            if step % log_stride == 0 or step == steps:
                _, diag = plant.evaluate(time, state, command, load_force)
                values = (time, state.copy(), reference.position, reference.velocity,
                    reference.acceleration, controller.uav_reference.copy() if use_pde else reference.position+plant.offsets, command.copy(),
                    diag["tension"], diag["length_error"], diag["directions"],
                    load_force(time) if load_force else np.zeros(3), diag["cable_power"])
                for key, value in zip(physical_keys, values):
                    records[key].append(value)
                if use_pde:
                    for key, value in controller.diagnostics(state, reference).items():
                        records.setdefault(key, []).append(value.copy())
                if use_force:
                    cv = diag["tension"][:, None]*diag["directions"]
                    p, v, quat, omega, _ = unpack(state)
                    extra = {"uav_reference_velocity": controller.uav_reference_velocity,
                             "uav_reference_acceleration": controller.uav_reference_acceleration,
                             "force_actual": cv.sum(axis=0),
                             "moment_actual": np.cross(diag["attachments"]-p[0], cv).sum(axis=0),
                             "model_acceleration_residual": diag["acceleration"][0]-controller.load_acceleration_command,
                             "disturbance_force_truth_for_audit": (load_force(time) if load_force else np.zeros(3))-c["payload_linear_drag"]*v[0],
                             "disturbance_moment_truth_for_audit": -c["payload_angular_drag"]*omega[0]}
                    for key, value in extra.items():
                        records.setdefault(key, []).append(np.asarray(value).copy())
            if step == steps:
                break
            state = plant.step(time, state, command, dt, load_force)
        except ModelDomainError as exc:
            status, reason = "model_domain_exceeded", str(exc)
            break
        if progress and step and step % max(1, int(20/dt)) == 0:
            print(f"{scenario}: simulated {time:.0f}/{duration:.0f} s", flush=True)
    arrays = {k: np.asarray(v) for k, v in records.items()}
    if len(arrays["time"]) == 0:
        raise ModelDomainError(reason or "Simulation produced no samples")
    metrics = compute_metrics(arrays, plant, trajectory)
    metrics.update({"status": status, "failure_reason": reason,
        "scenario": scenario, "requested_duration_s": duration,
        "logged_end_s": float(arrays["time"][-1]),
        "wall_time_s": wallclock.perf_counter()-start_wall,
        "saturated_attitude_update_fraction": saturated_updates/max(1, total_updates),
        "state_extrema_sampling_hz": 1/c["log_period"],
        "flight_ready": False, "px4_validated": False,
        "controller": "proposal_PDE_backstepping_three_nodes_velocity_realization" if use_pde
                      else "individual_position_PD_nominal_load_share_geometric_attitude_PD"})
    if use_pde:
        e = arrays["pde_error"]
        weights = np.array([.5, 1, .5])*controller.pde.h
        l2 = np.sqrt(np.sum(weights[None, :, None]*e**2, axis=(1, 2)))
        transport = arrays["time"] >= trajectory.hold
        payload_error = arrays["state"][:, :3]-arrays["reference_position"]
        metrics.update({"pde": controller.pde.metadata(),
            "formation_error_L2_initial_m": float(l2[0]),
            "formation_error_L2_final_m": float(l2[-1]),
            "formation_error_L2_peak_m": float(l2.max()),
            "boundary_residual_peak_m": float(np.linalg.norm(arrays["pde_boundary_residual"], axis=1).max()),
            "boundary_residual_final_m": float(np.linalg.norm(arrays["pde_boundary_residual"][-1])),
            "interior_PDE_defect_rms_m_s": float(np.sqrt(np.mean(arrays["pde_interior_equation_defect"]**2)*3)),
            "pde_velocity_command_peak_m_s": float(np.linalg.norm(arrays["pde_velocity_command"], axis=2).max()),
            "pde_acceleration_command_peak_m_s2": float(np.linalg.norm(arrays["pde_acceleration_command"], axis=2).max()),
            "pde_velocity_limit_fraction": velocity_limited_updates/max(position_updates, 1),
            "pde_acceleration_limit_fraction": acceleration_limited_updates/max(position_updates, 1),
            "post_initial_hold_payload_rmse_m": float(np.sqrt(np.mean(np.sum(payload_error[transport]**2, axis=1))))
                 if np.any(transport) else None})
    if use_force:
        from .force_results import force_metrics
        metrics["controller"] = "ESO_linear_tube_MPC_vector_force_inverse_geometry_PDE"
        metrics["force_mapping"] = controller.metadata()
        metrics.update(force_metrics(arrays, c, plant))
    return arrays, metrics


def compute_metrics(a, plant, trajectory):
    states = a["state"]
    pos = states[:, :12].reshape(-1, 4, 3)
    quat = states[:, 24:40].reshape(-1, 4, 4)
    rotation = quaternion_matrix(quat)
    payload_error = pos[:, 0]-a["reference_position"]
    uav_error = pos[:, 1:]-a["uav_reference"]
    pairwise = np.stack([np.linalg.norm(pos[:, i]-pos[:, j], axis=1)
                        for i, j in ((1, 2), (1, 3), (2, 3))], axis=1)
    payload_tilt = np.rad2deg(np.arccos(np.clip(rotation[:, 0, 2, 2], -1, 1)))
    cable_deviation = np.rad2deg(np.arccos(np.clip(
        np.sum(a["cable_direction"]*plant.rest_directions, axis=2), -1, 1)))
    time = a["time"]
    rate = np.gradient(a["tension"], time, axis=0) if len(time)>1 else np.zeros_like(a["tension"])
    final = time >= 2*(trajectory.hold+trajectory.leg)
    final_tilt = float(payload_tilt[final].max()) if np.any(final) else None
    half = np.asarray(plant.c["workspace_half_size"])
    radius = plant.c["arm_length"]+plant.c["propeller_radius"]
    uav_clearance = float(pairwise.min()-2*radius)
    # Bounding sphere for UAVs; payload box bounded conservatively by its radius.
    object_radius = np.array([np.linalg.norm(plant.c["payload_dimensions"])/2]+[radius]*3)
    x_clearance = half[0]-np.abs(pos[:, :, 0])-object_radius
    y_clearance = half[1]-np.abs(pos[:, :, 1])-object_radius
    floor_clearance = pos[:, :, 2]-object_radius
    ceiling_clearance = half[2]-pos[:, :, 2]-object_radius
    return {
        "payload_position_rmse_m": float(np.sqrt(np.mean(np.sum(payload_error**2, axis=1)))),
        "payload_position_peak_m": float(np.linalg.norm(payload_error, axis=1).max()),
        "uav_position_rmse_m": np.sqrt(np.mean(np.sum(uav_error**2, axis=2), axis=0)).tolist(),
        "payload_tilt_peak_deg": float(payload_tilt.max()),
        "final_hold_payload_tilt_peak_deg": final_tilt,
        "cable_direction_deviation_peak_deg": float(cable_deviation.max()),
        "tension_min_N": float(a["tension"].min()),
        "tension_max_N": float(a["tension"].max()),
        "tension_rate_rms_N_s": np.sqrt(np.mean(rate**2, axis=0)).tolist(),
        "rope_length_error_peak_m": float(np.abs(a["length_error"]).max()),
        "cable_constraint_power_peak_W": float(np.abs(a["cable_power"]).max()),
        "quaternion_norm_error_peak": float(np.abs(np.linalg.norm(quat, axis=2)-1).max()),
        "minimum_uav_center_distance_m": float(pairwise.min()),
        "minimum_uav_bounding_sphere_clearance_m": uav_clearance,
        "minimum_workspace_bounding_sphere_clearance_m": float(min(
            x_clearance.min(), y_clearance.min(), floor_clearance.min(), ceiling_clearance.min())),
        "rotor_thrust_peak_N": float(states[:, 52:64].max()),
    }


def save_run(output, arrays, metrics, config):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path/"state_log.npz", **arrays)
    (path/"metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (path/"config_snapshot.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    source_dir = Path(__file__).resolve().parent
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(source_dir.glob("*.py"))}
    meta = {"python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": hashes, "frame": "world_z_up_body_z_thrust_positive",
        "quaternion": "wxyz_body_to_world", "body_order": ["payload", "uav1", "uav2", "uav3"],
        "state_layout": {"position": [0,12], "velocity": [12,24], "quaternion": [24,40],
                         "body_angular_velocity": [40,52], "rotor_thrust": [52,64]},
        "limitations": ["uncalibrated_parameters", "taut_cables_only", "no_ground_contact",
            "no_sensor_noise_or_delay", "no_PX4_or_ROS_execution", "no_downwash_or_cable_collision"]}
    if "pde" in metrics:
        meta["pde"] = metrics["pde"]
        meta["reference_csv_meaning"] = "nominal_moving_formation_geometry_not_applied_PDE_control"
        meta["limitations"] += ["three_node_coarse_PDE_discretization", "finite_bandwidth_boundary_realization",
                                "no_continuum_stability_guarantee_for_nonlinear_plant"]
    if "force_mapping" in metrics:
        import scipy, osqp
        meta.update({"scipy": scipy.__version__, "osqp": osqp.__version__,
                     "force_mapping": metrics["force_mapping"],
                     "reference_csv_meaning": "force_inverted_C2_formation_reference_with_analytic_per_UAV_derivatives"})
        meta["limitations"] += ["ideal_load_state_and_cable_wrench_sensors",
            "translation_only_linear_MPC_with_no_terminal_invariant_constraint",
            "tube_guarantee_conditional_on_assumed_W_not_proven_for_nonlinear_plant"]
    (path/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Audit-friendly trajectory references; deliberately separate from actual states.
    with (path/"reference_only_NOT_FLIGHT_READY.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "uav_id", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                         "ax_m_s2", "ay_m_s2", "az_m_s2", "yaw_rad"])
        for n, time in enumerate(arrays["time"]):
            for i in range(3):
                velocity = arrays["uav_reference_velocity"][n, i] if "uav_reference_velocity" in arrays else arrays["reference_velocity"][n]
                acceleration = arrays["uav_reference_acceleration"][n, i] if "uav_reference_acceleration" in arrays else arrays["reference_acceleration"][n]
                writer.writerow([time, i+1, *arrays["uav_reference"][n, i],
                                 *velocity, *acceleration, 0])
    if "force_mapping" in metrics:
        from .force_results import export_force_chain
        export_force_chain(path, arrays)
    if "pde" in metrics:
        with (path/"pde_control_commands.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "uav_id", "role", "vx_command_m_s", "vy_command_m_s", "vz_command_m_s",
                             "ax_command_m_s2", "ay_command_m_s2", "az_command_m_s2"])
            for n, time in enumerate(arrays["time"]):
                for i, role in enumerate(("anchor", "follower", "leader")):
                    writer.writerow([time, i+1, role, *arrays["pde_velocity_command"][n, i],
                                     *arrays["pde_acceleration_command"][n, i]])
    with (path/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "load_x_m", "load_y_m", "load_z_m", "ref_x_m", "ref_y_m", "ref_z_m",
                         "T1_N", "T2_N", "T3_N", "rope1_error_m", "rope2_error_m", "rope3_error_m"])
        for n, time in enumerate(arrays["time"]):
            writer.writerow([time, *arrays["state"][n,:3], *arrays["reference_position"][n],
                             *arrays["tension"][n], *arrays["length_error"][n]])
