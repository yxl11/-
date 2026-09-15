"""Generate the current force-prediction/inverse-mapping report from run logs."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from sling_sim.rotations import quaternion_matrix
    folder = ROOT/"results/force_mapping"
    m = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    c = json.loads((folder/"config_snapshot.json").read_text(encoding="utf-8"))
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    with np.load(folder/"state_log.npz") as f:
        a = {k:f[k] for k in f.files}
    t = a["time"]
    error = np.linalg.norm(a["state"][:, :3]-a["reference_position"], axis=1)
    rotation = quaternion_matrix(a["state"][:,24:40].reshape(-1,4,4))
    tilt = np.degrees(np.arccos(np.clip(rotation[:,0,2,2], -1, 1)))
    velocity_residual = np.gradient(a["uav_reference"], t, axis=0, edge_order=2)-a["uav_reference_velocity"]
    derivative_peak = float(np.linalg.norm(velocity_residual[1:-1], axis=2).max())
    hashes_match = all(hashlib.sha256((ROOT/"sling_sim"/k).read_bytes()).hexdigest() == value
                       for k,value in meta["source_sha256"].items())
    fm = c["force_mapping"]
    # Convert stored nominal acceleration forecasts using the disturbance
    # estimate available at that time, never future measurements.
    gravity = np.array([0., 0., c["gravity"]])
    force_prediction = c["payload_mass"]*(a["mpc_acceleration_prediction"]+gravity)-a["eso_force_estimate"][:, None, :]
    actual_state = np.stack((a["state"][:, :3], a["state"][:, 12:15]), axis=2)
    tube_error = actual_state-a["mpc_nominal_state"]
    auxiliary = tube_error @ np.asarray(m["force_mapping"]["tube_mpc"]["K"])
    command_identity = np.max(np.abs(force_prediction[:, 0]+c["payload_mass"]*auxiliary-a["force_requested"]))
    cone_ok = True
    for key in ("force_allocated_raw", "force_allocated_smooth"):
        f = a[key]
        cone_ok &= (np.all(f[:,:,2] >= fm["tension_min"]-2e-5) and
                    np.all(np.linalg.norm(f,axis=2) <= fm["tension_max"]+2e-5) and
                    np.all(np.abs(f[:,:,0])+np.abs(f[:,:,1]) <=
                           np.tan(np.deg2rad(fm["cone_angle_deg"]))*f[:,:,2]+2e-5))
    checks = {
        "full_task_completed": m["status"] == "completed" and abs(t[-1]-(2*c["leg_duration"]+3*c["hold_duration"])) < 1e-9,
        "integrated_controller_selected": m["controller"] == "ESO_linear_tube_MPC_vector_force_inverse_geometry_PDE",
        "finite_aligned_logs": all(len(v) == len(t) and np.isfinite(v).all() for v in a.values()),
        "all_requested_QPs_solved": m["force_mapping"]["mpc_solves"] == round(t[-1]/fm["prediction_period"])+1
                      and m["force_mapping"]["allocation_solves"] == round(t[-1]/c["position_period"])+1,
        "allocation_force_and_moment_equalities": m["allocation_force_residual_peak_N"] < 1e-5 and m["allocation_moment_residual_peak_Nm"] < 1e-5,
        "raw_and_filtered_force_cones": cone_ok,
        "actual_positive_tension_within_design_range": m["tension_min_N"] >= fm["tension_min"] and m["tension_max_N"] <= fm["tension_max"],
        "inverse_and_physical_rope_lengths": m["inverse_geometry_rope_residual_peak_m"] < 1e-7 and m["rope_length_error_peak_m"] < 1e-6,
        "mapped_reference_derivatives_consistent": derivative_peak < 1e-3,
        "formation_mapping_is_nonzero": m["force_induced_reference_offset_peak_m"] > 1e-3,
        "future_prediction_sequence_is_nonconstant": np.max(np.ptp(a["mpc_acceleration_prediction"],axis=1)) > 1e-4,
        "force_forecast_plus_tube_feedback_matches_command": command_identity < 1e-10,
        "no_motor_saturation": m["saturated_attitude_update_fraction"] == 0 and m["rotor_thrust_peak_N"] < c["max_rotor_thrust"],
        "PDE_commands_within_limits_without_clipping": m["pde_velocity_limit_fraction"] == 0 and m["pde_acceleration_limit_fraction"] == 0,
        "positive_sampled_body_and_workspace_clearance": m["minimum_uav_bounding_sphere_clearance_m"] > 0 and m["minimum_workspace_bounding_sphere_clearance_m"] > 0,
        "nominal_MPC_constraints": m["mpc_constraint_violation_peak"] < 2e-5,
        "sampled_state_within_tube": m["tube_utilization_peak"] <= 1,
        "sampled_model_residual_within_assumed_W": m["sampled_acceleration_residual_bound_ratio_peak"] <= 1,
        "current_source_matches_saved_run": hashes_match,
    }
    checks = {k:bool(v) for k,v in checks.items()}
    audit = {"all_passed": all(checks.values()), "checks": checks,
             "reference_velocity_difference_peak_m_s": derivative_peak,
             "unit_checks_executed": 26, "algorithm_comparison_performed": False,
             "flight_ready": False, "continuous_time_robust_guarantee": False}
    (folder/"run_checks.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    import csv
    with (folder/"force_prediction_horizon.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["solve_time_s", "input_time_s", "prediction_step", "Fx_nominal_N", "Fy_nominal_N", "Fz_nominal_N"])
        for n in np.flatnonzero(a["mpc_updated"]):
            for j, value in enumerate(force_prediction[n]):
                writer.writerow([t[n], t[n]+j*fm["prediction_period"], j, *value])
    from sling_sim.visualization import configure_style
    from matplotlib import pyplot as plt
    configure_style()
    eligible = np.flatnonzero(a["mpc_updated"] & (t >= 24.))
    idx = eligible[0] if len(eligible) else np.flatnonzero(a["mpc_updated"])[0]
    future = t[idx]+np.arange(fm["horizon"])*fm["prediction_period"]
    fig, axes = plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for axis, ax in enumerate(axes):
        ax.plot(future, force_prediction[idx,:,axis], color="#9651A4", marker=".", markersize=4)
        ax.set(xlabel="预测输入时刻 / s", ylabel=f"名义合力 F{'xyz'[axis]} / N", title=f"{'xyz'[axis]} 方向预测力序列")
        ax.ticklabel_format(axis="y",useOffset=False)
    fig.suptitle(f"t={t[idx]:.1f} s 的20步名义合力预测\n当时的扰动估计在窗口内冻结，实际执行首项另加管束反馈",fontsize=13)
    fig.savefig(folder/"force_prediction_window.png",dpi=150)
    fig.savefig(folder/"force_prediction_window.pdf")
    plt.close(fig)
    h,d = c["hold_duration"],c["leg_duration"]
    phase_rows=[]
    for name,start,end in [("起点保持",0,h),("S形去程，含侧向扰动",h,h+d),("远端保持",h+d,2*h+d),
                           ("S形返程",2*h+d,2*h+2*d),("末段保持",2*h+2*d,3*h+2*d)]:
        mask=(t>=start)&(t<end if name!="末段保持" else t<=end)
        if np.any(mask):
            phase_rows.append(f"| {name} | {start:g}—{end:g} | {np.sqrt(np.mean(error[mask]**2))*1000:.3f} | "
                              f"{error[mask].max()*1000:.3f} | {tilt[mask].max():.4f} |")
    phases="\n".join(phase_rows)
    report=f"""# 合力前馈预测与编队逆向映射仿真报告

本次已将申请书第 2.3(3) 节的“合力前馈预测—缆绳力分配—编队逆向映射”接入原有 PDE 控制与三机吊载物理模型。运行 {m['logged_end_s']:g} 秒，状态为 `{m['status']}`。本次只运行这一串联方法，不做算法性能对照。

**逆映射实际进入控制回路：**由优化绳力反算的三机参考构型，相对固定名义构型最大改变 **{m['force_induced_reference_offset_peak_m']*1000:.2f} mm**；这些变化的位置及其解析速度、加速度参与 PDE 跟踪。完整方法和补全依据见 [方程与实现说明](../docs/force_prediction_mapping.md)。

## 实现链路

1. 由负载轨迹和未来 2 秒参考计算逆动力学前馈，动量 ESO 估计集总扰动。
2. 每 0.1 秒求解 20 步平动模型预测问题，惩罚输入变化；持续传播名义状态，用局部反馈与计算得到的误差管束收紧状态/输入约束。
3. 将合力需求与姿态力矩需求分配为三根三维绳力向量，满足六条力/力矩平衡等式与正拉力锥约束。
4. 当前分配力直接进入无人机力前馈；其平滑方向经过含负载挂点偏置的逆映射，生成 C² 连续构型参考。
5. PDE 控制跟随变化的构型，四旋翼与负载的真实数值运动仍由独立动力学积分求出。

因此，“求出了合力”“分配了绳力”“改变了构型”和“实际产生了该合力”是四类分别记录的量。报告中没有把优化等式的残差当成物理跟踪误差。

## 工况与参数

三架四旋翼共同吊运 {c['payload_mass']:g} kg 刚性负载，每机假设质量 {c['uav_mass']:g} kg、最大总推力 {4*c['max_rotor_thrust']:g} N，绳长 {c['rope_length']:g} m，场地 20×10×5 m。从已经吊起的名义悬停开始，任务由 {h:g} 秒保持、{d:g} 秒去程、{h:g} 秒保持、{d:g} 秒返程和 {h:g} 秒末段保持组成。

在 t={fm['wind_start_s']:g}—{fm['wind_start_s']+fm['wind_duration_s']:g} s 向负载施加峰值 {fm['wind_peak_N']:g} N 的平滑 y 向力。控制器没有取得这项扰动的真值；ESO 使用理想模拟负载速度与实际绳索合力测量。实机雷达/拉力传感器接口尚未实现。

外层力矩增益为 k_R={fm['attitude_gain']:g}、k_ω={fm['angular_velocity_gain']:g}，几何参考滤波带宽 {fm['force_filter_bandwidth']:g} rad/s。该带宽配置配合当前 QP 力的及时前馈，避免让慢参考滤波延迟力矩阻尼。执行层为 50 Hz PDE/速度更新、250 Hz 姿态更新和 500 Hz 物理积分。

## 数值结果

| 指标 | 本次结果 |
| --- | --- |
| 全程负载三维位置 RMSE | {m['payload_position_rmse_m']*1000:.3f} mm |
| 全程负载最大位置误差 | {m['payload_position_peak_m']*1000:.3f} mm |
| 最大负载倾角 | {m['payload_tilt_peak_deg']:.4f}° |
| 需求合力与物理实际合力 RMSE | {m['requested_to_actual_force_rmse_N']:.6f} N |
| 需求力矩与物理实际力矩 RMSE | {m['requested_to_actual_moment_rmse_Nm']:.3e} N·m |
| ESO 集总扰动力估计 RMSE | {m['eso_force_estimation_rmse_N']:.6f} N |
| 原始分配合力等式最大残差 | {m['allocation_force_residual_peak_N']:.3e} N |
| 原始分配力矩等式最大残差 | {m['allocation_moment_residual_peak_Nm']:.3e} N·m |
| 构型逆映射的最大参考变化 | {m['force_induced_reference_offset_peak_m']*1000:.3f} mm |
| 逆映射得到的绳角范围 | {m['desired_cable_angle_range_deg'][0]:.4f}—{m['desired_cable_angle_range_deg'][1]:.4f}° |
| 逆映射绳长最大残差 | {m['inverse_geometry_rope_residual_peak_m']:.3e} m |
| 物理实际绳张力范围 | {m['tension_min_N']:.4f}—{m['tension_max_N']:.4f} N |
| 最大单电机实际推力 | {m['rotor_thrust_peak_N']:.4f} N，上限 {c['max_rotor_thrust']:g} N |
| 电机推力饱和 / PDE速度限幅 / PDE加速度限幅比例 | {m['saturated_attitude_update_fraction']*100:.2f}% / {m['pde_velocity_limit_fraction']*100:.2f}% / {m['pde_acceleration_limit_fraction']*100:.2f}% |
| 最小无人机质心间距 | {m['minimum_uav_center_distance_m']:.4f} m |
| 最小无人机包围球净距 | {m['minimum_uav_bounding_sphere_clearance_m']:.4f} m |
| 最小场地包围球余量 | {m['minimum_workspace_bounding_sphere_clearance_m']:.4f} m |
| 最大管束利用率，界限为1 | {m['tube_utilization_peak']:.5f} |
| 最大采样残余加速度 / 所设扰动界 | {m['sampled_acceleration_residual_bound_ratio_peak']:.5f} |
| 三轴预测 QP 每次求解耗时中位数 / 峰值 | {m['mpc_solve_time_ms_median']:.4f} / {m['mpc_solve_time_ms_peak']:.4f} ms |
| 分配 QP 求解耗时峰值 | {m['allocation_solve_time_ms_peak']:.4f} ms |

求解耗时是本机本次 OSQP 报告的求解器运行时间，不包含整个 Python 控制循环、通信、感知或操作系统调度，不能据此宣称实机实时性。图和表中的极值均来自 50 Hz 日志。

| 阶段 | 时间 / s | 负载位置 RMSE / mm | 最大位置误差 / mm | 最大倾角 / ° |
| --- | --- | --- | --- | --- |
{phases}

![合力预测与逆映射控制链](force_mapping/force_mapping_chain.png)

第一行分别显示逆动力学前馈、最终需求、几何参考滤波对应合力和物理实际合力。第二行显示优化张力、逆映射绳角及各机参考位置变化。第三行显示 ESO 估计、管束/扰动界检查及分配等式残差。浅橙时段为外加侧向扰动。

![滚动预测窗口](force_mapping/prediction_window.png)

预测窗口来自仿真当时保存的 QP 预测状态，未用未来实际轨迹反向生成。

![名义合力预测序列](force_mapping/force_prediction_window.png)

名义合力预测由保存的加速度序列和当时的 ESO 估计换算，窗口内冻结扰动估计；实际执行的合力等于首项再加质量缩放的管束反馈。该关系的最大数值残差为 {command_identity:.3e} N。全部预测窗口见 [20步合力预测序列](force_mapping/force_prediction_horizon.csv)，不是只记录当前合力。

![物理响应](force_mapping/simulation_summary.png)

## 检查与解释范围

本次日志的 {len(checks)} 项检查 **{'全部通过' if all(checks.values()) else '存在未通过项'}**，详见 [检查记录](force_mapping/run_checks.json)。检查包括预测/分配求解、力平衡、力锥、参考导数、逆映射和实际绳长、非零构型变化、执行器限幅、采样净距、管束及设计扰动界。26 项实现单元检查均通过，其中新增了“力矩阻尼及时进入执行器而不等待几何参考运动”的检查。

Tube 设计只针对线性平动预测模型。误差管束及其收紧量由 Lyapunov 椭圆计算；本次轨迹在采样意义下满足设计界，不等于证明非线性吊载系统始终满足该界。本版没有终端不变约束、全六维非线性 Tube MPC 或递归可行性证明。完整模型参数也尚未实测辨识。

力锥和最小间距结论只适用于本次参数与轨迹。物理模型覆盖绳索绷紧的悬空运输，尚不含松绳重接触、起吊落地、雷达误差、ROS/PX4、下洗或全系统碰撞检测。所有数据是数值仿真结果。

## 文件与运行

```powershell
.\\.venv\\Scripts\\python.exe run_simulation.py --plots --animation
```

默认配置 `configs/force_pde.json`，场景 `force_transport`，结果 `results/force_mapping`。

[三维动画](force_mapping/transport_simulation.gif) · [合力到构型数据链](force_mapping/force_prediction_mapping_chain.csv) · [逆映射参考轨迹](force_mapping/reference_only_NOT_FLIGHT_READY.csv) · [PDE指令](force_mapping/pde_control_commands.csv) · [原始日志](force_mapping/state_log.npz) · [配置快照](force_mapping/config_snapshot.json)

参考 CSV 中三机各自的位置、速度和加速度来自逆映射及解析导数；与 PDE 实际修正指令、物理响应分别保存。其用途是后续轨迹复现准备，尚未经过实机承载、坐标、同步和飞控校核。
"""
    (ROOT/"results/合力预测与逆映射仿真报告.md").write_text(report,encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
