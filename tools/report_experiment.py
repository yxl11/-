"""Audit saved replay runs and write the feasibility report / filming plan."""
import argparse
import csv
import json
import re
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sling_sim.replay import ReplayTrajectory, file_hash, evaluate_acceptance
from sling_sim.model import SlingPlant
from sling_sim.experiment_visualization import plot_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiment_v1"))
    args = parser.parse_args()
    root = args.root
    (root/"media").mkdir(parents=True, exist_ok=True)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    metrics = read(root/"replay_suite_metrics.json")
    config = read(root/"planning/config_snapshot.json")
    planning_metrics = read(root/"planning/metrics.json")
    planning = dict(np.load(root/"planning/state_log.npz", allow_pickle=False))
    manifest = read(root/"trajectory/manifest.json")
    trajectory = ReplayTrajectory.load(root/"trajectory/trajectory.npz")
    plant = SlingPlant(config)
    t = trajectory.time
    delta = trajectory.p-planning["reference_position"][:, None]-plant.offsets
    common_shift = delta.mean(axis=1)
    relative_shape = delta-common_shift[:, None]
    speed = np.linalg.norm(trajectory.v, axis=2).max()
    acceleration = np.linalg.norm(trajectory.a, axis=2).max()
    jerk_peak = 0.
    for u in (0., .5, 1.):
        c = trajectory.coefficients
        jerk = (6*c[:, 3]+24*c[:, 4]*u+60*c[:, 5]*u*u)/trajectory.h[:, None, None]**3
        jerk_peak = max(jerk_peak, float(np.linalg.norm(jerk, axis=2).max()))
    peak_delta = float(np.linalg.norm(delta, axis=2).max())
    accepted = [m for m in metrics if m["engineering_screen_passed"]]
    capture = [m for m in metrics if m["scenario"].startswith("capture_")]
    ideal = next(m for m in metrics if m["scenario"] == "ideal")
    audit = {
        "planning_completed": planning_metrics["status"] == "completed",
        "planning_no_external_wind": bool(np.all(planning["load_force"] == 0)),
        "source_log_hash_matches": manifest["source_log_sha256"] == file_hash(root/"planning/state_log.npz"),
        "source_config_hash_matches": manifest["source_config_sha256"] == file_hash(root/"planning/config_snapshot.json"),
        "trajectory_hash_matches": manifest["trajectory_sha256"] == file_hash(root/"trajectory/trajectory.npz"),
        "exported_reference_is_exact_planning_pva": all(np.array_equal(x, planning[k]) for x, k in (
            (trajectory.p, "uav_reference"), (trajectory.v, "uav_reference_velocity"), (trajectory.a, "uav_reference_acceleration"))),
        "duration_88s_and_50Hz": bool(abs(t[-1]-88) < 1e-10 and np.allclose(np.diff(t), .02)),
        "reference_speed_below_08m_s": bool(speed < .8),
        "reference_acceleration_below_1m_s2": bool(acceleration < 1.),
        "inverse_reference_is_dynamic": peak_delta > .005,
        "terminal_reference_velocity_acceleration_negligible": bool(np.max(abs(trajectory.v[-1])) < 1e-5 and np.max(abs(trajectory.a[-1])) < 1e-4),
        "planning_physical_margins": planning_metrics["minimum_uav_bounding_sphere_clearance_m"] >= .25 and planning_metrics["tension_min_N"] >= 4 and planning_metrics["tension_max_N"] <= 18,
        "planning_QPs_feasible": planning_metrics["mpc_constraint_violation_peak"] < 1e-5 and planning_metrics["allocation_force_residual_peak_N"] < 1e-5,
    }
    source = Path(__file__).resolve().parents[1]/"sling_sim"
    case_audits = {}
    for m in metrics:
        directory = root/"replay"/m["scenario"]
        stored = read(directory/"metrics.json")
        meta = read(directory/"metadata.json")
        actual = read(directory/"config_snapshot.json")
        nominal = read(directory/"nominal_controller_config.json")
        data = dict(np.load(directory/"state_log.npz", allow_pickle=False))
        checks = {
            "completed_full_88s": m["status"] == "completed" and abs(data["time"][-1]-88) < 1e-10,
            "finite_aligned_logs": all(len(value) == len(data["time"]) and np.all(np.isfinite(value)) for value in data.values()),
            "same_frozen_trajectory": meta["trajectory_sha256"] == manifest["trajectory_sha256"],
            "controller_source_matches": all(meta["source_sha256"][name] == file_hash(source/name)
                for name in ("replay.py", "model.py", "rotations.py", "simulation.py", "trajectory.py")),
            "nominal_mass_not_replaced_by_truth": nominal["payload_mass"] == 3 and abs(actual["payload_mass"]-3*m["case"]["payload_mass_factor"]) < 1e-10,
            "no_online_load_or_planner_feedback": all(m[key] is False for key in ("payload_feedback", "tension_feedback", "online_PDE_MPC_ESO")),
            "recomputed_threshold_flags_match": evaluate_acceptance(m) == m["acceptance_checks"],
            "suite_metric_matches_saved_case": m == stored,
        }
        case_audits[m["scenario"]] = {key: bool(value) for key, value in checks.items()}
    audit["case_audits_all_pass"] = all(all(checks.values()) for checks in case_audits.values())
    # These are integrity checks. They do NOT relabel failed engineering criteria as passed.
    (root/"artifact_audit.json").write_text(json.dumps({"integrity_checks": audit, "case_integrity": case_audits,
        "integrity_passed": all(audit.values()), "engineering_passed_cases": len(accepted), "engineering_total_cases": len(metrics),
        "all_capture_cases_passed": bool(capture) and all(m["engineering_screen_passed"] for m in capture)}, indent=2), encoding="utf-8")
    if not all(audit.values()):
        raise RuntimeError(f"Artifact integrity failed: {audit}")

    with (root/"replay_case_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["scenario", "engineering_screen_passed", "actual_payload_mass_kg", "payload_position_rmse_m",
                  "payload_position_peak_m", "payload_tilt_peak_deg", "final_hold_payload_tilt_peak_deg",
                  "final_hold_payload_rmse_m", "tension_min_N", "tension_max_N", "minimum_uav_bounding_sphere_clearance_m"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(metrics)
    shot_rows = [
        [0, 88, "固定全景", "完整保持三机、绳索、负载在画面内，禁止把不同次飞行剪成一次连续验证", "编号、统一时间、轨迹叠加"],
        [0, 8, "起点悬停", "展示初始相对构型与负载状态", "起点与编号"],
        [16, 32, "去程转弯片段", "展示加减速和转弯触发的离线合力、逆映射方向变化", "预测合力、参考差量、实际张力（仿真）"],
        [40, 48, "远端定点", "展示三机停止运输后的负载姿态与定位保持", "负载误差、倾角"],
        [58, 74, "返回机动", "展示反向路径的同步轨迹执行", "三机路径与统一时间"],
        [80, 88, "末段悬停", "按末段误差和倾角指标评估静止画面质量", "末段误差、倾角、录像编号"],
    ]
    with (root/"shot_list.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_s", "end_s", "shot", "purpose", "overlay"])
        writer.writerows(shot_rows)
    case_rows = []
    for m in metrics:
        failed = [key for key, value in m["acceptance_checks"].items() if not value]
        note = "全部通过" if not failed else "；".join("末段倾角>1°" if key == "final_hold_tilt_at_most_1deg" else key for key in failed)
        case_rows.append(f"| {m['scenario']} | {m['actual_payload_mass_kg']:.1f} | {m['payload_position_rmse_m']*1000:.2f} | {m['payload_position_peak_m']*1000:.2f} | {m['payload_tilt_peak_deg']:.3f} | {m['final_hold_payload_tilt_peak_deg']:.3f} | {note} |")
    capture_passed = bool(capture) and all(m["engineering_screen_passed"] for m in capture)
    unit_log = (root/"unit_tests.txt").read_text(encoding="utf-8")
    unit_match = re.search(r"Ran (\d+) tests", unit_log)
    unit_count = int(unit_match.group(1)) if unit_match else 0
    if not unit_count or not re.search(r"\nOK\s*$", unit_log):
        raise RuntimeError("A successful saved unit test run is required")
    summary = {"frozen_trajectory_duration_s": float(t[-1]), "uav_reference_speed_peak_m_s": float(speed),
        "uav_reference_acceleration_peak_m_s2": float(acceleration), "Hermite_jerk_peak_sampled_m_s3": jerk_peak,
        "relative_to_payload_mapping_offset_peak_m": peak_delta,
        "common_formation_shift_peak_m": float(np.linalg.norm(common_shift, axis=1).max()),
        "shape_change_after_common_shift_peak_m": float(np.linalg.norm(relative_shape, axis=2).max()),
        "all_capture_cases_passed": capture_passed, "engineering_passed_cases": len(accepted), "total_cases": len(metrics),
        "all_cases_full_duration": all(m["status"] == "completed" for m in metrics),
        "all_case_min_tension_N": min(m["tension_min_N"] for m in metrics), "all_case_max_tension_N": max(m["tension_max_N"] for m in metrics),
        "all_case_min_uav_clearance_m": min(m["minimum_uav_bounding_sphere_clearance_m"] for m in metrics),
        "ideal_replay": ideal, "flight_ready": False, "PX4_validated": False}
    (root/"experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# 离线轨迹回放可行性报告

已完成离线轨迹回放模块和88秒展示轨迹定稿。{len(metrics)} 个最终执行器工况均记录在本报告中；其中 {len(accepted)} 个通过全部14项预设工程检查。低噪声拍摄候选工况全部通过：**{'是' if capture_passed else '否'}**。未通过的工况没有删去，也没有调宽1°末段倾角检查值。

理想回放：负载位置 RMSE **{ideal['payload_position_rmse_m']*1000:.2f} mm**，最大误差 **{ideal['payload_position_peak_m']*1000:.2f} mm**，最大负载倾角 **{ideal['payload_tilt_peak_deg']:.3f}°**。

## 回放到底执行了什么

离线阶段继续使用申请书方法：ESO → 2秒平动合力预测 → 三绳向量力分配 → 编队逆映射 → PDE跟踪。将生成的三机各自位置、速度、加速度保存为独立数据包。所选轨迹来自无外加风扰的规划，不把一次特定风扰的反馈修正硬编码进正式拍摄轨迹。

回放从新的悬空平衡状态重新积分四刚体和三根绳索。数据包只有时间、三机p/v/a和零偏航，不包含绳力、负载状态、ESO输出或电机指令。执行器只取得各机自身位置/速度测量及姿态/角速度；PDE、MPC和ESO都不在回放期间在线运行。负载状态、真实张力及外加风力只进入评价日志。

执行层采用本机位置PID与几何姿态PD，**不是PX4的实际代码**。其力指令为

`F_i = m_U·sat_1[a_i,d + Kp(p_i,d−p_i,meas) + Kd(v_i,d−v_i,meas) + I_i] + (m_L,nom/3)·a_i,d + (m_U+m_L,nom/3)g·e_z + (m_L,nom/3)g tan20°·r_i`。

`Kp=(4,4,6)`，`Kd=(3.5,3.5,4)`；积分增益 `(0.3,0.3,0.8)`，积分加速度分量限于±0.6 m/s²，候选总加速度超过1 m/s²时暂停对应无人机的积分。标称负载始终为3 kg，质量偏差只改变物理负载及惯量。该静态分担补偿和本机积分是回放可行性的执行器假设，后续接PX4仍需校核是否等效实现。

数据包为50 Hz、每机 {len(t)} 个时刻，采用五次Hermite插值，p/v/a在记录区间内C²连续。区间外保持端点位置并置零速度/加速度；末端残余速度约 {manifest['end_velocity_peak_m_s']:.3e} m/s、加速度约 {manifest['end_acceleration_peak_m_s2']:.3e} m/s²，已在检查中确认可忽略。

## 最终展示轨迹

8秒起点悬停 → 32秒S形去程 → 8秒远端定点 → 32秒返回 → 8秒末段悬停。负载路径 x∈[−3,3] m、y∈[−1,1] m，负载质心高度0.875 m，名义无人机高度约2.691 m，绳长1.8 m；场地20×10×5 m。模型从已吊起状态开始。

三机参考速度峰值 {speed:.4f} m/s，加速度峰值 {acceleration:.4f} m/s²。Hermite区间端点/中点采样的jerk峰值为 {jerk_peak:.4f} m/s³。相对负载固定构型的参考变化最大 **{peak_delta*1000:.2f} mm**；其中编队共同位移峰值 {summary['common_formation_shift_peak_m']*1000:.2f} mm，扣除共同位移后的单机相对变形峰值 {summary['shape_change_after_common_shift_peak_m']*1000:.2f} mm。这些量不等同于机间距离变化。

![轨迹与回放结果](media/replay_overview.png)

## 误差与失配设置

- `ideal`：理想本机状态测量和同步；`communication`：三机参考通信延迟20/40/60 ms，时钟偏移0/+10/−10 ms，相当于参考滞后20/30/70 ms。这是固定陈旧参考模型，不含丢包和网络抖动。
- `mass_plus/minus_10pct`：物理负载3.3/2.7 kg，控制器仍假设3 kg。
- `localization`、`combined_*`：位置每轴噪声标准差5 mm，速度每轴0.01 m/s，观测延迟40 ms，偏置向量各分量最大3 mm。
- `capture_*`：位置每轴噪声标准差2 mm，速度每轴0.005 m/s，观测延迟40 ms，偏置同上。组合工况同时加入质量偏差、通信/时钟误差，以及30—38秒峰值0.3 N的平滑侧向风力。重载组合记录种子7、19、41；轻载组合记录种子7。
- `boundary_*`：位置5 mm、速度0.01 m/s随机噪声，固定位置偏置各分量最大1 cm。

噪声是相关时间0.15 s的高斯AR(1)序列，从零噪声状态启动，参数为设定的渐近标准差；三机、三轴、位置与速度通道独立驱动。姿态和陀螺测量仍理想。固定偏置完整向量见 `replay_cases.json`。这些是假设工况，尚未确认实际雷达性能。少量随机种子不能给出成功概率或连续参数域保证。

## 全部结果

| 工况 | 实际负载/kg | 位置RMSE/mm | 最大位置误差/mm | 最大倾角/° | 末段最大倾角/° | 14项检查 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(case_rows)}

全过程张力总范围 {summary['all_case_min_tension_N']:.2f}—{summary['all_case_max_tension_N']:.2f} N，最小无人机包围球净距 {summary['all_case_min_uav_clearance_m']:.3f} m。每个工况的推力、限幅、误差和间距数值均保存在 `replay/<工况>/metrics.json`。

![各工况工程检查](media/replay_case_checks.png)

预设检查值：负载最大位置误差≤15 cm、无人机最大位置误差≤10 cm、负载倾角≤5°、绳方向偏差≤5°、张力4—18 N、机间包围球净距≥25 cm、场地包围球余量≥30 cm、单电机推力≤假设上限的80%、无电机饱和、无加速度限幅、末段位置RMSE≤5 cm、末段最大倾角≤1°、绳长残差≤10 μm，并完成全程。它们是本次选择的工程筛选值，不是实机认证指标。间距及极值来自50 Hz采样；负张力在所有RK4子步检查，出现即停止，未实现松绳模式。

较大测量误差下，完成运输并不自动满足末段静止画面的1°要求。拍摄候选设置的通过仅支持已运行的具体组合。若实测定位达不到该设置，应保留原始误差并重新运行相应工况，不能引用理想回放精度。

## 可复核数据与限制

{unit_count}项单元测试全部通过，见[最终单元测试记录](unit_tests.txt)；另见[文件与运行数据审计](artifact_audit.json)。文件审计检查源日志、配置、冻结轨迹哈希、实际运行源码、载重失配隔离、全时段有限日志及判据重算；审计通过不表示未达标的工程工况通过。

初始执行器调试记录保留在 `debug_pd_initial/`，不混入上表最终执行器统计。研究方法本身没有引入其他算法做性能对照。

当前仅支持绷紧绳索的悬空运输，不含起吊落地、松绳重接触、下洗、全系统连续碰撞、ROS/PX4或实机传感器。飞机质量/推力、执行带宽、挂点和定位尚未实测。实机轨迹复现可支持“该轨迹能执行、能完成吊载演示”的结论；不能据此宣称申请书完整控制器已经在线实机验证。

## 文件与复现

[冻结三机轨迹CSV](trajectory/three_uav_trajectory_NOT_FLIGHT_READY.csv) · [轨迹清单](trajectory/manifest.json) · [各工况指标CSV](replay_case_metrics.csv) · [实验与拍摄方案](实验轨迹与拍摄方案.md) · [MP4演示](media/experiment_replay.mp4) · [GIF动画](media/experiment_replay.gif)

在项目根目录运行：

```powershell
.\\.venv\\Scripts\\python.exe run_simulation.py --scenario nominal --leg-duration 32 --output results/experiment_v1/planning
.\\.venv\\Scripts\\python.exe run_replay.py --suite --workers 3
.\\.venv\\Scripts\\python.exe run_replay.py --cases configs/replay_capture_cases.json --append --workers 3
.\\.venv\\Scripts\\python.exe tools/render_experiment.py --animation
.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath results/experiment_v1/unit_tests.txt
.\\.venv\\Scripts\\python.exe tools/report_experiment.py
```

广范围 `--suite` 中包含未达标工况，因此会以非零状态结束并保留全部结果；按顺序继续拍摄候选工况即可。`--append`保留既有结果并复用同一冻结轨迹，不允许重复工况名。更新源轨迹后应重新执行整套回放，不能混合不同轨迹的结果。
"""
    (root/"轨迹回放可行性报告.md").write_text(report, encoding="utf-8")
    plan = f"""# 三机协同吊运：实验轨迹与拍摄方案

本版完成轨迹和仿真演示定稿，实机部分为后续实施设计。载重标称3 kg，可在重新规划和回放后减轻。三机采用统一局部坐标系，z向上，偏航保持0。控制与执行分为：离线生成含申请书方法的轨迹、在线由各机跟踪冻结轨迹。

## 时间与路径

| 时间/s | 动作 | 负载参考位置 | 需要呈现的内容 |
| --- | --- | --- | --- |
| 0—8 | 起点悬停 | (−3, 0, 0.875) m | 三机相对构型、三绳连接和编号 |
| 8—40 | 平滑S形去程 | x由−3到3 m，y为一次正弦 | 加速、转弯、减速驱动预测合力和逆映射变化 |
| 40—48 | 远端定点 | (3, 0, 0.875) m | 三机定点、负载姿态和张力保持 |
| 48—80 | 沿同一路径返回 | 原路反向 | 反向机动与三机同步 |
| 80—88 | 末段悬停 | (−3, 0, 0.875) m | 静止画面、末段误差和倾角 |

单程采用七次时间律 `s=35u^4−84u^5+70u^6−20u^7`，`u=(t−t_start)/32`；负载几何路径 `p_L,d=(−3+6s, sin(2πs), 0.875)`，返程将s反向。三机轨迹由合力逆映射给出，不能用固定三角队形简单平移替代导出文件。

每机50 Hz位置、速度、加速度，共 {len(t)} 个时间点。最大参考速度 {speed:.3f} m/s，最大参考加速度 {acceleration:.3f} m/s²。负载尺寸0.6×0.5×0.25 m，挂点径向半径0.25 m，挂点高于质心0.125 m，绳长1.8 m，名义绳角20°；名义无人机高度约2.691 m。不同硬件不能只换CSV文件名，应修改参数后重新验证。

![场地与分镜](media/trajectory_storyboard.png)

## 说明链与证据对应

1. **预测合力**：仿真动画右上角显示当时保存的未来2秒名义Fx/Fy序列；这不是未来实际数据回填。
2. **逆向映射**：右中图显示三机相对负载的实际参考位置；灰叉为固定名义构型，彩点为逆映射结果。位移箭头明确放大8倍，机位坐标仍按真比例。最大参考变化 {peak_delta*1000:.2f} mm。
3. **可执行轨迹**：左侧三维场景由独立回放的物理状态生成，不把参考位置直接画成实际位置；橙色实际水平合力箭头按1 N=1 m显示，Fz以数字显示。
4. **运输响应**：显示回放求出的张力与负载位置误差。离线图中的预测/逆映射量和回放中的实际响应分开标明来源。

毫米至厘米量级的构型调整主要通过仿真曲线和标明比例的插图解释，实机录像以三机同步运输和轨迹复现为主要内容。

## 拍摄安排

- 主机位建议固定于场地侧边 `(0,−4.6,约2.0) m`，朝向+y，镜头覆盖三机、三根绳和完整运输路径。该位置是布置示意，现场需按镜头视角复核。
- 如有顶部固定安装条件，可在场地上方约4.7 m处设置俯视机位，中心对准原点，记录三机路径投影和相对构型。安装位置与飞行高度的实际余量另行检查。
- 一次任务连续录制全景；后期可用同一次录像裁切16—32秒去程转弯、40—48秒定点、58—74秒返回和80—88秒末段悬停。分镜清单见 [shot_list.csv](shot_list.csv)。
- 画面保留无人机1/2/3编号、统一时间和录像批次。仿真叠加层标注“离线规划”或“仿真响应”；实机轨迹叠加来自实测日志，并保留原始未叠加视频。
- 主机位保持定焦和连续视角便于对照空间运动；正式记录建议同时保存飞控、定位、指令时间戳和原始录像，后续依据共同时间事件对齐。

## 当前候选条件与交付范围

理想回放以及质量±10%单项检查已给出量化结果。拍摄候选设置采用位置每轴2 mm、速度0.005 m/s随机噪声，固定偏置各分量最大3 mm、定位延迟40 ms；组合检查还包含固定通信延迟、时钟偏移、质量偏差与0.3 N侧向脉冲。五个拍摄候选工况是否全部通过：**{'是' if capture_passed else '否'}**。

这些仅是已检验的数值样本，尚未测定雷达是否满足；更大的误差可能仍能完成运输，但静止画面倾角会超过本方案1°目标，详见 [可行性报告](轨迹回放可行性报告.md)。本版冻结悬空运输段，不含起吊/落地或ROS/PX4接入。

[MP4仿真演示](media/experiment_replay.mp4) · [GIF动画](media/experiment_replay.gif) · [三机轨迹CSV](trajectory/three_uav_trajectory_NOT_FLIGHT_READY.csv)

![动画截图](media/experiment_preview.png)
"""
    (root/"实验轨迹与拍摄方案.md").write_text(plan, encoding="utf-8")
    plot_cases(metrics, root/"media")
    print(json.dumps({"integrity_passed": all(audit.values()), "engineering_passed": len(accepted),
                      "total_cases": len(metrics), "capture_all_passed": capture_passed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
