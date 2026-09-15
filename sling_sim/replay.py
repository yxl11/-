"""Offline trajectory execution. The executor never receives a payload state.

This is an engineering execution model, not PX4 SITL. Only a saved clock and
three UAV p/v/a references cross the planning/execution boundary. All payload,
tension, wind and clearance quantities are computed outside the controller.
"""
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import hashlib
import json
import time as wallclock
import numpy as np
from .model import SlingPlant, ModelDomainError, unpack
from .rotations import quaternion_matrix, desired_rotation, vee
from .simulation import compute_metrics
from .trajectory import TransportTrajectory


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReplayTrajectory:
    """C2 quintic Hermite interpolation of an immutable prerecorded UAV table."""
    def __init__(self, times, position, velocity, acceleration):
        self.time = np.array(times, dtype=float, copy=True)
        self.p, self.v, self.a = [np.array(x, dtype=float, copy=True)
                                 for x in (position, velocity, acceleration)]
        if (len(self.time) < 2 or self.time.ndim != 1 or
                self.time[0] != 0 or np.any(np.diff(self.time) <= 0)):
            raise ValueError("Replay timestamps must start at zero and strictly increase")
        for x in (self.p, self.v, self.a):
            if x.shape != (len(self.time), 3, 3) or not np.all(np.isfinite(x)):
                raise ValueError("Each reference must be a finite N x 3 UAV x 3 axis array")
        if not np.all(np.isfinite(self.time)):
            raise ValueError("Nonfinite timestamp")
        self.duration = float(self.time[-1])
        self.h = np.diff(self.time)
        h = self.h[:, None, None]
        c0, c1, c2 = self.p[:-1], h*self.v[:-1], .5*h*h*self.a[:-1]
        dp = self.p[1:]-c0-c1-c2
        dv = h*self.v[1:]-c1-2*c2
        da = h*h*self.a[1:]-2*c2
        self.coefficients = np.stack((c0, c1, c2, 10*dp-4*dv+.5*da,
                                     -15*dp+7*dv-da, 6*dp-3*dv+.5*da), axis=1)
        for x in (self.time, self.p, self.v, self.a, self.h, self.coefficients):
            x.setflags(write=False)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            required = {"time", "position", "velocity", "acceleration", "yaw"}
            if set(data.files) != required or np.any(data["yaw"] != 0):
                raise ValueError("Expected p/v/a/time and zero yaw only; no force or load oracle")
            return cls(data["time"], data["position"], data["velocity"], data["acceleration"])

    def at(self, times):
        t = np.broadcast_to(np.asarray(times, dtype=float), (3,))
        clipped = np.clip(t, 0, self.duration)
        index = np.minimum(np.searchsorted(self.time, clipped, side="right")-1, len(self.h)-1)
        h = self.h[index]
        u = (clipped-self.time[index])/h
        c = self.coefficients[index, :, np.arange(3), :]
        p = sum(c[:, k]*u[:, None]**k for k in range(6))
        v = sum(k*c[:, k]*u[:, None]**(k-1) for k in range(1, 6))/h[:, None]
        a = sum(k*(k-1)*c[:, k]*u[:, None]**(k-2) for k in range(2, 6))/h[:, None]**2
        outside = (t < 0) | (t > self.duration)
        v[outside], a[outside] = 0, 0
        return p, v, a


def export_replay(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source/"metrics.json").read_text(encoding="utf-8"))
    if metrics["status"] != "completed":
        raise ValueError("Cannot export an incomplete planning run")
    with np.load(source/"state_log.npz", allow_pickle=False) as data:
        t = data["time"].copy()
        p, v, a = [data[k].copy() for k in ("uav_reference", "uav_reference_velocity", "uav_reference_acceleration")]
    trajectory = ReplayTrajectory(t, p, v, a)
    np.savez_compressed(destination/"trajectory.npz", time=t, position=p, velocity=v,
                        acceleration=a, yaw=np.zeros((len(t), 3)))
    with (destination/"three_uav_trajectory_NOT_FLIGHT_READY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "uav_id", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
                         "ax_m_s2", "ay_m_s2", "az_m_s2", "yaw_rad"])
        for n, time in enumerate(t):
            for i in range(3):
                writer.writerow([time, i+1, *p[n, i], *v[n, i], *a[n, i], 0])
    manifest = {"source_directory": str(source.resolve()), "source_log_sha256": file_hash(source/"state_log.npz"),
        "source_config_sha256": file_hash(source/"config_snapshot.json"),
        "trajectory_sha256": file_hash(destination/"trajectory.npz"),
        "duration_s": trajectory.duration, "samples_per_uav": len(t), "sample_period_s": float(np.median(np.diff(t))),
        "coordinate_frame": "shared_local_world_z_up; heading fixed; ROS/PX4 transforms not implemented",
        "interpolation": "quintic Hermite, analytical p/v/a, C2 within recorded interval; outside endpoint hold",
        "start_velocity_peak_m_s": float(np.linalg.norm(v[0], axis=1).max()),
        "end_velocity_peak_m_s": float(np.linalg.norm(v[-1], axis=1).max()),
        "start_acceleration_peak_m_s2": float(np.linalg.norm(a[0], axis=1).max()),
        "end_acceleration_peak_m_s2": float(np.linalg.norm(a[-1], axis=1).max()),
        "execution_inputs": ["time", "three_UAV_position_velocity_acceleration", "zero_yaw"],
        "payload_feedback": False, "tension_feedback": False, "online_PDE_MPC_ESO": False,
        "flight_ready": False}
    (destination/"manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


class ReplayController:
    """Local UAV PID and attitude loop with fixed, nominal load-share compensation.

    The assumed load mass is configured once. Even the feedforward is obtained
    from this assumption and recorded UAV acceleration, never recorded QP forces.
    """
    def __init__(self, nominal_config):
        self.c = dict(nominal_config)
        geometry = SlingPlant(nominal_config)
        self.radial, self.mixer_inverse = geometry.radial, geometry.mixer_inverse
        self.inertia = np.asarray(nominal_config["uav_inertia"])
        self.force = np.zeros((3, 3))
        self.acceleration_command = np.zeros((3, 3))
        self.integral_acceleration = np.zeros((3, 3))
        self.integral_gain = np.array([.3, .3, .8])
        self.saturated = np.zeros((3, 4), dtype=bool)
        self.limited = False

    def position_update(self, measured_position, measured_velocity, reference):
        p, v, a = reference
        if np.shape(measured_position) != (3, 3) or np.shape(measured_velocity) != (3, 3):
            raise ValueError("Executor accepts only the three UAV position/velocity measurements")
        c = self.c
        feedback = (np.asarray(c["position_kp"])*(p-measured_position)
                    + np.asarray(c["position_kd"])*(v-measured_velocity))
        candidate = np.clip(self.integral_acceleration+c["position_period"]*self.integral_gain*(p-measured_position), -.6, .6)
        candidate_command = a+feedback+candidate
        integrate = np.linalg.norm(candidate_command, axis=1) <= 1.
        self.integral_acceleration[integrate] = candidate[integrate]
        command = a+feedback+self.integral_acceleration
        norm = np.linalg.norm(command, axis=1)
        scale = np.minimum(1., 1./np.maximum(norm, 1e-12))
        self.limited = bool(np.any(scale < 1))
        self.acceleration_command = command*scale[:, None]
        share = c["payload_mass"]/3
        self.force = c["uav_mass"]*self.acceleration_command+share*a
        self.force[:, 2] += (c["uav_mass"]+share)*c["gravity"]
        self.force += share*c["gravity"]*np.tan(np.deg2rad(c["hover_rope_angle_deg"]))*self.radial

    def attitude_update(self, uav_quaternion, uav_omega):
        if np.shape(uav_quaternion) != (3, 4) or np.shape(uav_omega) != (3, 3):
            raise ValueError("Executor accepts only the three UAV attitude/gyro measurements")
        rotations = quaternion_matrix(uav_quaternion)
        commands = np.empty((3, 4))
        for i, r in enumerate(rotations):
            rd = desired_rotation(self.force[i])
            error = .5*vee(rd.T@r-r.T@rd)
            moment = (-np.asarray(self.c["attitude_kp"])*error
                      -np.asarray(self.c["attitude_kd"])*uav_omega[i]
                      +np.cross(uav_omega[i], self.inertia*uav_omega[i]))
            commands[i] = self.mixer_inverse@np.r_[max(0., float(self.force[i]@r[:, 2])), moment]
        clipped = np.clip(commands, 0, self.c["max_rotor_thrust"])
        self.saturated = np.abs(clipped-commands) > 1e-10
        return clipped


@dataclass(frozen=True)
class ReplayCase:
    name: str = "ideal"
    category: str = "acceptance"
    payload_mass_factor: float = 1.
    position_noise_std_m: float = 0.
    velocity_noise_std_m_s: float = 0.
    noise_correlation_s: float = .15
    measurement_delay_s: float = 0.
    position_bias_m: tuple = ((0., 0., 0.),)*3
    command_delay_s: tuple = (0., 0., 0.)
    clock_offset_s: tuple = (0., 0., 0.)
    wind_peak_N: float = 0.
    wind_start_s: float = 30.
    wind_duration_s: float = 8.
    seed: int = 7

    def __post_init__(self):
        if self.category not in ("acceptance", "boundary"):
            raise ValueError("Unknown case category")
        if self.payload_mass_factor <= 0 or self.measurement_delay_s < 0:
            raise ValueError("Invalid mass/delay")
        if min(self.position_noise_std_m, self.velocity_noise_std_m_s) < 0 or self.noise_correlation_s <= 0:
            raise ValueError("Invalid noise")
        if np.shape(self.position_bias_m) != (3, 3) or np.shape(self.command_delay_s) != (3,) or np.shape(self.clock_offset_s) != (3,):
            raise ValueError("Expected per-UAV bias/delay/clock parameters")
        if np.any(np.asarray(self.command_delay_s) < 0) or self.wind_duration_s <= 0:
            raise ValueError("Invalid communication delay or pulse duration")


def default_cases():
    large_bias = ((.01, -.005, .004), (-.008, .01, -.003), (.005, -.008, .005))
    sensing = dict(position_noise_std_m=.005, velocity_noise_std_m_s=.01, measurement_delay_s=.04,
        position_bias_m=tuple(tuple(.3*x for x in row) for row in large_bias))
    link = dict(command_delay_s=(.02, .04, .06), clock_offset_s=(0., .01, -.01))
    cases = [ReplayCase(), ReplayCase("localization", **sensing), ReplayCase("communication", **link),
             ReplayCase("mass_plus_10pct", payload_mass_factor=1.1),
             ReplayCase("mass_minus_10pct", payload_mass_factor=.9)]
    cases.append(ReplayCase("combined_light_seed7", payload_mass_factor=.9, wind_peak_N=.3, **sensing, **link))
    for seed in (7, 19, 41):
        cases.append(ReplayCase(f"combined_heavy_seed{seed}", payload_mass_factor=1.1, wind_peak_N=.3,
                                seed=seed, **sensing, **link))
    coarse = dict(sensing, position_bias_m=large_bias)
    cases += [ReplayCase("boundary_bias_1cm", category="boundary", **coarse),
              ReplayCase("boundary_combined_bias_1cm", category="boundary", payload_mass_factor=1.1,
                         wind_peak_N=.3, **coarse, **link)]
    return cases


def evaluate_acceptance(metrics):
    """Chosen engineering screening criteria, not measured vehicle limits."""
    checks = {
        "full_duration": metrics["status"] == "completed" and abs(metrics["logged_end_s"]-metrics["requested_duration_s"]) < 1e-8,
        "payload_peak_error_at_most_15cm": metrics["payload_position_peak_m"] <= .15,
        "uav_peak_error_at_most_10cm": metrics["uav_position_peak_m"] <= .10,
        "payload_tilt_at_most_5deg": metrics["payload_tilt_peak_deg"] <= 5.,
        "cable_deviation_at_most_5deg": metrics["cable_direction_deviation_peak_deg"] <= 5.,
        "tension_within_4_to_18N": 4 <= metrics["tension_min_N"] <= metrics["tension_max_N"] <= 18,
        "uav_surface_clearance_at_least_25cm": metrics["minimum_uav_bounding_sphere_clearance_m"] >= .25,
        "workspace_clearance_at_least_30cm": metrics["minimum_workspace_bounding_sphere_clearance_m"] >= .30,
        "rotor_at_most_80pct_assumed_limit": metrics["rotor_limit_utilization_peak"] <= .8,
        "no_motor_saturation": metrics["saturated_attitude_update_fraction"] == 0,
        "no_acceleration_clipping": metrics["acceleration_limit_fraction"] == 0,
        "final_hold_RMSE_at_most_5cm": metrics["final_hold_payload_rmse_m"] <= .05,
        "final_hold_tilt_at_most_1deg": (metrics["final_hold_payload_tilt_peak_deg"] or 0) <= 1.,
        "rope_length_residual_at_most_10um": metrics["rope_length_error_peak_m"] <= 1e-5,
    }
    return checks


def run_replay(nominal_config, trajectory, case, duration=None, progress=False):
    c = dict(nominal_config)
    c["payload_mass"] *= case.payload_mass_factor
    plant = SlingPlant(c)
    executor = ReplayController(nominal_config)
    # This desired load path is available to evaluation only, not the executor.
    evaluation_path = TransportTrajectory(nominal_config)
    state = plant.equilibrium(evaluation_path.at(0).position)
    duration = trajectory.duration if duration is None else float(duration)
    dt = c["physics_dt"]
    steps = int(round(duration/dt))
    if steps < 1 or duration > trajectory.duration+1e-9 or not np.isclose(steps*dt, duration):
        raise ValueError("Replay duration must be positive, on the physics grid, and within the recording")
    pos_stride, att_stride, log_stride = [int(round(c[k]/dt)) for k in ("position_period", "attitude_period", "log_period")]
    if any(n < 1 for n in (pos_stride, att_stride, log_stride)):
        raise ValueError("Control/log period below physics timestep")
    n_delay = int(np.floor(case.measurement_delay_s/dt))
    fraction = case.measurement_delay_s/dt-n_delay
    history = deque(maxlen=n_delay+2)
    rng = np.random.default_rng(case.seed)
    noise = np.zeros((2, 3, 3))
    decay = np.exp(-c["position_period"]/case.noise_correlation_s)
    noise_std = np.array([case.position_noise_std_m, case.velocity_noise_std_m_s])[:, None, None]
    measurement = np.stack((unpack(state)[0][1:], unpack(state)[1][1:])).copy()
    desired = trajectory.at(0)
    replay_times = np.zeros(3)
    command = unpack(state)[4].copy()
    logs = {}
    status, reason, failure_time = "completed", None, None
    limited = saturated = updates = att_updates = 0
    start = wallclock.perf_counter()

    def wind(t):
        s = (t-case.wind_start_s)/case.wind_duration_s
        return np.array([0., case.wind_peak_N*np.sin(np.pi*s)**2 if 0 < s < 1 else 0., 0.])

    for step in range(steps+1):
        time = step*dt
        p, v, quat, omega, _ = unpack(state)
        history.append(np.stack((p[1:], v[1:])).copy())
        try:
            if step % pos_stride == 0:
                recent = history[max(0, len(history)-1-n_delay)]
                older = history[max(0, len(history)-2-n_delay)]
                delayed = (1-fraction)*recent+fraction*older
                noise = decay*noise+np.sqrt(1-decay*decay)*noise_std*rng.standard_normal((2, 3, 3))
                measurement = delayed+noise
                measurement[0] += np.asarray(case.position_bias_m)
                replay_times = time-np.asarray(case.command_delay_s)+np.asarray(case.clock_offset_s)
                desired = trajectory.at(replay_times)
                executor.position_update(measurement[0], measurement[1], desired)
                limited += int(executor.limited)
                updates += 1
            if step % att_stride == 0:
                command = executor.attitude_update(quat[1:], omega[1:])
                saturated += int(np.any(executor.saturated))
                att_updates += 1
            if step % log_stride == 0 or step == steps:
                _, diag = plant.evaluate(time, state, command, wind)
                ref = evaluation_path.at(time)
                common = trajectory.at(time)
                cv = diag["tension"][:, None]*diag["directions"]
                values = {"time": time, "state": state.copy(), "reference_position": ref.position,
                    "reference_velocity": ref.velocity, "reference_acceleration": ref.acceleration,
                    "uav_reference": common[0], "uav_reference_velocity": common[1], "uav_reference_acceleration": common[2],
                    "received_reference_position": desired[0], "received_reference_velocity": desired[1],
                    "received_reference_acceleration": desired[2], "playback_time_per_uav": replay_times,
                    "measured_uav_position": measurement[0], "measured_uav_velocity": measurement[1],
                    "executor_acceleration_command": executor.acceleration_command,
                    "executor_integral_acceleration": executor.integral_acceleration,
                    "rotor_command": command, "tension": diag["tension"], "length_error": diag["length_error"],
                    "cable_direction": diag["directions"], "cable_power": diag["cable_power"],
                    "load_force": wind(time), "force_actual": cv.sum(axis=0),
                    "moment_actual": np.cross(diag["attachments"]-p[0], cv).sum(axis=0)}
                for key, value in values.items():
                    logs.setdefault(key, []).append(np.asarray(value).copy())
            if step == steps:
                break
            state = plant.step(time, state, command, dt, wind)
        except ModelDomainError as exc:
            status, reason, failure_time = "model_domain_exceeded", str(exc), time
            break
        if progress and step and step % int(20/dt) == 0:
            print(f"{case.name}: simulated {time:.0f}/{duration:g} s", flush=True)
    arrays = {key: np.asarray(value) for key, value in logs.items()}
    if not len(arrays.get("time", [])):
        raise ModelDomainError(reason or "No replay log")
    metrics = compute_metrics(arrays, plant, evaluation_path)
    error = arrays["state"][:, :12].reshape(-1, 4, 3)[:, 1:]-arrays["uav_reference"]
    final = arrays["time"] >= trajectory.duration-nominal_config["hold_duration"]
    final_error = arrays["state"][final, :3]-arrays["reference_position"][final]
    metrics.update({"status": status, "failure_reason": reason, "failure_step_time_s": failure_time,
        "scenario": case.name, "case": asdict(case), "controller": "offline_pva_replay_local_UAV_PID_nominal_load_share",
        "requested_duration_s": duration, "logged_end_s": float(arrays["time"][-1]),
        "wall_time_s": wallclock.perf_counter()-start, "flight_ready": False, "px4_validated": False,
        "payload_feedback": False, "tension_feedback": False, "online_PDE_MPC_ESO": False,
        "nominal_payload_mass_kg": nominal_config["payload_mass"], "actual_payload_mass_kg": c["payload_mass"],
        "uav_position_peak_m": float(np.linalg.norm(error, axis=2).max()),
        "final_hold_payload_rmse_m": float(np.sqrt(np.mean(np.sum(final_error**2, axis=1)))) if final.any() else 1e9,
        "acceleration_limit_fraction": limited/max(updates, 1),
        "saturated_attitude_update_fraction": saturated/max(att_updates, 1),
        "rotor_limit_utilization_peak": float(arrays["state"][:, 52:64].max()/c["max_rotor_thrust"]),
        "state_extrema_sampling_hz": 1/c["log_period"]})
    metrics["acceptance_checks"] = evaluate_acceptance(metrics)
    metrics["engineering_screen_passed"] = all(metrics["acceptance_checks"].values())
    return arrays, metrics, c


def save_replay(output, arrays, metrics, actual_config, nominal_config, bundle_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/"state_log.npz", **arrays)
    for filename, value in (("metrics.json", metrics), ("config_snapshot.json", actual_config),
                            ("nominal_controller_config.json", nominal_config)):
        (output/filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    source = Path(__file__).resolve().parent
    metadata = {"numpy": np.__version__, "trajectory_sha256": file_hash(bundle_path),
        "source_sha256": {f.name: file_hash(f) for f in sorted(source.glob("*.py"))},
        "control_inputs": ["recorded_UAV_pva_and_time", "delayed_noisy_UAV_position_velocity", "ideal_UAV_attitude_gyro"],
        "evaluation_only": ["payload_state", "tension", "external_force", "force_actual"],
        "nominal_load_share_compensation": "constant nominal mass, radial hover preload, recorded acceleration inertial share",
        "local_position_integral": {"gain_per_second": [.3, .3, .8], "component_acceleration_limit_m_s2": .6,
                                    "antiwindup": "conditional_integration_when_candidate_acceleration_norm_at_most_1"},
        "assumptions": ["taut_cables_only", "no_PX4_ROS", "no_ground_contact", "no_downwash", "ideal_attitude_gyro",
            "fixed_link_latency_and_clock_offsets_not_packet_loss", "position_velocity_noise_are_correlated_Gaussian_models",
            "mass_variation_changes_payload_inertia_but_not_geometry", "sampled_clearance_not_continuous_collision_proof"],
        "flight_ready": False}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
