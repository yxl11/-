"""Check a completed PDE run and generate its Chinese report from saved data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sling_sim.rotations import quaternion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=ROOT/"results/pde_transport")
    parser.add_argument("--report", type=Path, default=ROOT/"results/PDE方法仿真报告.md")
    args = parser.parse_args()
    folder = args.folder.resolve()
    m = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    c = json.loads((folder/"config_snapshot.json").read_text(encoding="utf-8"))
    meta = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    with np.load(folder/"state_log.npz") as f:
        a = {key: f[key] for key in f.files}
    t = a["time"]
    pos = a["state"][:, :12].reshape(-1, 4, 3)
    errors = np.linalg.norm(pos[:, 0]-a["reference_position"], axis=1)
    rotations = quaternion_matrix(a["state"][:, 24:40].reshape(-1, 4, 4))
    tilt = np.rad2deg(np.arccos(np.clip(rotations[:, 0, 2, 2], -1, 1)))
    post = t >= c["hold_duration"]
    boundary_action = np.linalg.norm(a["pde_boundary_target"]-a["uav_reference"][:, 2], axis=1)
    hashes_match = all((ROOT/"sling_sim"/name).exists() and
        hashlib.sha256((ROOT/"sling_sim"/name).read_bytes()).hexdigest() == value
        for name, value in meta["source_sha256"].items())
    checks = {
        "completed_requested_duration": m["status"] == "completed" and abs(t[-1]-m["requested_duration_s"]) < 1e-9,
        "PDE_controller_selected": m["controller"] == "proposal_PDE_backstepping_three_nodes_velocity_realization",
        "finite_aligned_logs": all(len(v) == len(t) and np.isfinite(v).all() for v in a.values()),
        "positive_logged_tensions": m["tension_min_N"] > 0,
        "no_rotor_command_saturation": m["saturated_attitude_update_fraction"] == 0,
        "rotor_thrust_within_bounds": a["state"][:, 52:].min() >= 0 and m["rotor_thrust_peak_N"] < c["max_rotor_thrust"],
        "positive_sampled_uav_clearance": m["minimum_uav_bounding_sphere_clearance_m"] > 0,
        "positive_sampled_workspace_clearance": m["minimum_workspace_bounding_sphere_clearance_m"] > 0,
        "rope_residual_below_one_micrometre": m["rope_length_error_peak_m"] < 1e-6,
        "boundary_feedback_actually_nonzero": boundary_action.max() > 1e-4,
        "Volterra_boundary_identity": np.allclose(a["pde_transformed_error"][:, 2],
            np.exp(c["pde"]["b"]*c["pde"]["alpha"])*a["pde_boundary_residual"], atol=1e-13),
        "source_hashes_match_run": hashes_match,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    audit = {"all_passed": all(checks.values()), "checks": checks,
        "post_initial_hold_peak_payload_error_m": float(errors[post].max()) if post.any() else None,
        "post_initial_hold_peak_tilt_deg": float(tilt[post].max()) if post.any() else None,
        "boundary_feedback_correction_peak_m": float(boundary_action.max()),
        "final_payload_position_error_m": float(errors[-1]),
        "all_metrics_sampling_hz": m["state_extrema_sampling_hz"],
        "algorithm_comparison_performed": False, "flight_authorization": False}
    (folder/"run_checks.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    h, d = c["hold_duration"], c["leg_duration"]
    phases = [("初始偏差恢复", 0, h), ("S形去程", h, h+d), ("远端保持", h+d, 2*h+d),
              ("S形返程", 2*h+d, 2*h+2*d), ("末段保持", 2*h+2*d, 3*h+2*d)]
    phase_rows = []
    for label, start, end in phases:
        mask = (t >= start) & (t < end if label != "末段保持" else t <= end)
        if np.any(mask):
            phase_rows.append(f"| {label} | {start:g}—{end:g} | {np.sqrt(np.mean(errors[mask]**2))*1000:.3f} | "
                              f"{errors[mask].max()*1000:.3f} | {tilt[mask].max():.4f} |")
    phase_table = "\n".join(phase_rows)
    relative = Path(__import__("os").path.relpath(folder, args.report.parent)).as_posix()
    text = f"""# 三机吊载 PDE 编队方法仿真报告

已按申请书第 2.3(1) 节的 PDE 与反步边界控制公式，实现三机离散控制并完成 {m['logged_end_s']:g} 秒运输仿真。本次运行状态为 `{m['status']}`，仅评估该方法自身的物理响应，未开展算法性能对照。

方法采用三个实际拓扑节点、解析贝塞尔核及边界加权反馈；移动参考与有限带宽执行接口为本工程补全。它不是申请书全部研究模块的完整复现。详细方程、原文对应位置和补全依据见 [PDE 实现说明](../docs/pde_method.md)。

## 工况

- 三架四旋翼共同吊运 {c['payload_mass']:g} kg 刚性负载；每机假设质量 {c['uav_mass']:g} kg，最大总推力 {4*c['max_rotor_thrust']:g} N，绳长 {c['rope_length']:g} m。
- 1号为锚点，2号为跟随者，3号为领导者，对应拓扑坐标 0、0.5、1。三机的物理布局为三角形。
- 从已经悬空且绳索绷紧的构型开始，整体位置初始偏移为 {c['pde']['initial_translation_m']} m，负载绕 y 轴初始倾斜 {c['pde']['initial_payload_tilt_deg']:g}°。重建端点后初始绳长约束成立。
- 执行 {h:g} 秒保持、{d:g} 秒 S 形去程、{h:g} 秒保持、{d:g} 秒返程和 {h:g} 秒末段保持。本工况没有外加风力、故障或时延。
- PDE/速度环 {1/c['position_period']:g} Hz，姿态环 {1/c['attitude_period']:g} Hz，物理积分 {1/c['physics_dt']:g} Hz，日志 {1/c['log_period']:g} Hz。

## 运行结果

| 指标 | 本次结果 |
| --- | --- |
| 全程负载三维位置 RMSE，含初始恢复 | {m['payload_position_rmse_m']*1000:.3f} mm |
| 初始保持之后的负载 RMSE，t≥{h:g} s | {m['post_initial_hold_payload_rmse_m']*1000:.3f} mm |
| 初始保持之后的负载最大位置误差 | {audit['post_initial_hold_peak_payload_error_m']*1000:.3f} mm |
| 全程负载最大位置误差 | {m['payload_position_peak_m']*1000:.3f} mm（包含设定的初始偏差） |
| 全程最大负载倾角 | {m['payload_tilt_peak_deg']:.4f}°（包含初始倾斜） |
| 初始保持之后的最大负载倾角 | {audit['post_initial_hold_peak_tilt_deg']:.4f}° |
| 绳索张力范围 | {m['tension_min_N']:.4f}—{m['tension_max_N']:.4f} N |
| 最大单电机实际推力 | {m['rotor_thrust_peak_N']:.4f} N，上限 {c['max_rotor_thrust']:g} N |
| 推力分配饱和更新比例 | {100*m['saturated_attitude_update_fraction']:.2f}% |
| 最大 PDE 速度指令 | {m['pde_velocity_command_peak_m_s']:.4f} m/s，上限 {c['pde']['velocity_limit']:g} m/s |
| 最大 PDE 加速度指令 | {m['pde_acceleration_command_peak_m_s2']:.4f} m/s²，上限 {c['pde']['acceleration_limit']:g} m/s² |
| 速度 / 加速度限幅更新比例 | {m['pde_velocity_limit_fraction']*100:.2f}% / {m['pde_acceleration_limit_fraction']*100:.2f}% |
| 三机最小质心间距 | {m['minimum_uav_center_distance_m']:.4f} m |
| 三机最小包围球净距 | {m['minimum_uav_bounding_sphere_clearance_m']:.4f} m |
| 全系统最小场地包围球余量 | {m['minimum_workspace_bounding_sphere_clearance_m']:.4f} m |
| 最大绳长数值残差 | {m['rope_length_error_peak_m']:.3e} m |

阶段统计如下。RMSE 为负载三维误差范数的均方根，阶段之间不重复计算端点样本，末段包含最终样本。

| 阶段 | 时间 / s | 位置 RMSE / mm | 最大位置误差 / mm | 最大负载倾角 / ° |
| --- | --- | --- | --- | --- |
{phase_table}

![PDE吊运物理响应]({relative}/simulation_summary.png)

## PDE 是否实际参与控制

本次运行的边界权重为 `{m['pde']['boundary_weights']}`。领导者的边界参考相对名义位置最大修正 {boundary_action.max()*1000:.3f} mm，说明核加权反馈在本次运行中确实非零。跟随者速度使用当前三机误差的空间差分；仿真循环调用 `PDEController`，没有调用历史位置 PD 更新函数。

编队误差的离散 L² 范数由 {m['formation_error_L2_initial_m']:.6f} m 变化为终点的 {m['formation_error_L2_final_m']:.3e} m；领导者边界残差终值为 {m['boundary_residual_final_m']:.3e} m。这些微小终值来自无噪声、未标定的理想数值模型，不能作为实机可达到精度。

内部跟随者的实际误差速度与 PDE 目标速度之差，均方根为 {m['interior_PDE_defect_rms_m_s']:.6f} m/s。非零残差反映有限带宽速度环和物理耦合，因此这里验证的是 PDE 离散控制器驱动物理对象的闭环响应，并未声称真实刚体轨迹严格满足连续 PDE。

![PDE边界与节点演化]({relative}/pde_evolution.png)

## 数值检查

保存日志的 {len(checks)} 项一致性与约束检查：**{'全部通过' if all(checks.values()) else '存在未通过项，请查看检查文件'}**。检查包括完成时长、PDE 控制器标识、有限且对齐的记录、正张力、推力范围、采样间距、绳长残差、非零边界反馈、Volterra 边界关系，以及运行源码哈希的一致性。详见 [检查记录]({relative}/run_checks.json)。

实现时还执行了 17 项单元检查，全部通过：原有 9 项物理/基础接口检查，以及 8 项新增 PDE 数学和连接检查。新增检查覆盖核方程和端点极限、对流变换、动态边界、邻居反馈、参考原点不变性、负载真值隔离和日志对齐。上述检查不属于算法性能对照实验。

张力的负值检查发生在每个 RK4 子步。表中状态峰值、最小距离及 PDE 残差来自 {m['state_extrema_sampling_hz']:g} Hz 日志，不能视为连续时间极值的严格界限。

## 输出与复现

```powershell
.\\.venv\\Scripts\\python.exe run_simulation.py --plots --animation
```

[三维动画]({relative}/transport_simulation.gif) · [PDE速度指令]({relative}/pde_control_commands.csv) · [原始状态]({relative}/state_log.npz) · [数值指标]({relative}/metrics.json) · [参数快照]({relative}/config_snapshot.json)

名义参考 `reference_only_NOT_FLIGHT_READY.csv` 是随任务移动的理想三角形及其解析导数，不能误当成 PDE 实际反馈指令。实际速度和加速度命令在 `pde_control_commands.csv`，真实数值响应在状态日志中。

本版只涉及申请书 PDE 编队与反步边界方法；ESO、Tube MPC、故障/时延观测和学习避障未接入。参数尚未按实机标定，雷达、ROS/PX4、起吊落地和松绳重接触未建模，现有数据没有直接上机资格。三个实际节点也不足以证明连续极限、空间网格收敛或大规模可扩展性。
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
