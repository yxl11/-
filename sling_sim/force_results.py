"""Auditable metrics for the force-to-formation chain, computed from state logs."""
import numpy as np


def export_force_chain(path, arrays):
    import csv
    keys = ["force_base", "force_eso_feedforward", "force_requested", "force_actual",
            "moment_requested", "moment_actual", "eso_force_estimate", "tube_utilization"]
    with (path/"force_prediction_mapping_chain.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s"]+[f"{key}_{axis}" for key in keys for axis in "xyz"]+
                   [f"f_des_uav{i}_{axis}_N" for i in (1,2,3) for axis in "xyz"])
        for n, t in enumerate(arrays["time"]):
            w.writerow([t]+np.concatenate([arrays[key][n] for key in keys]+
                                         [arrays["force_allocated_smooth"][n].ravel()]).tolist())


def force_metrics(a, config, plant):
    def rms_norm(value):
        return float(np.sqrt(np.mean(np.sum(value**2, axis=-1))))
    raw, smooth = a["force_allocated_raw"], a["force_allocated_smooth"]
    actual = a["force_actual"]
    anchor = a["reference_position"][:, None, :]+plant.anchors
    inverse_residual = np.linalg.norm(a["uav_reference"]-anchor, axis=2)-config["rope_length"]
    fixed = a["reference_position"][:, None, :]+plant.offsets
    angles = np.rad2deg(np.arccos(np.clip(a["force_direction_reference"][:, :, 2], -1, 1)))
    bounds = np.asarray(config["force_mapping"]["acceleration_residual_bound"])
    model_ratio = np.max(np.abs(a["model_acceleration_residual"])/bounds, axis=1)
    return {
        "requested_to_actual_force_rmse_N": rms_norm(actual-a["force_requested"]),
        "smoothed_to_actual_force_rmse_N": rms_norm(actual-smooth.sum(axis=1)),
        "requested_to_actual_moment_rmse_Nm": rms_norm(a["moment_actual"]-a["moment_requested"]),
        "allocation_wrench_residual_peak": float(np.abs(a["force_allocation_residual"]).max()),
        "allocation_force_residual_peak_N": float(np.linalg.norm(a["force_allocation_residual"][:, :3], axis=1).max()),
        "allocation_moment_residual_peak_Nm": float(np.linalg.norm(a["force_allocation_residual"][:, 3:], axis=1).max()),
        "inverse_geometry_rope_residual_peak_m": float(np.abs(inverse_residual).max()),
        "force_induced_reference_offset_peak_m": float(np.linalg.norm(a["uav_reference"]-fixed, axis=2).max()),
        "desired_cable_angle_range_deg": [float(angles.min()), float(angles.max())],
        "allocated_tension_range_N": [float(np.linalg.norm(raw, axis=2).min()), float(np.linalg.norm(raw, axis=2).max())],
        "geometry_filtered_tension_range_N": [float(np.linalg.norm(smooth, axis=2).min()), float(np.linalg.norm(smooth, axis=2).max())],
        "geometry_filter_force_lag_rmse_N": rms_norm(raw.sum(axis=1)-smooth.sum(axis=1)),
        "tube_utilization_peak": float(a["tube_utilization"].max()),
        "sampled_acceleration_residual_bound_ratio_peak": float(model_ratio.max()),
        "sampled_acceleration_residual_bound_violation_fraction": float(np.mean(model_ratio > 1+1e-8)),
        "eso_force_estimation_rmse_N": rms_norm(a["eso_force_estimate"]-a["disturbance_force_truth_for_audit"]),
        "mpc_constraint_violation_peak": float(a["mpc_constraint_violation"].max()),
        "mpc_solve_time_ms_median": float(np.median(a["optimization_solve_ms"][a["mpc_updated"], 0])),
        "mpc_solve_time_ms_peak": float(a["optimization_solve_ms"][:, 0].max()),
        "allocation_solve_time_ms_peak": float(a["optimization_solve_ms"][:, 1].max()),
    }
