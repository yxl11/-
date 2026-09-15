# 三机吊载合力前馈预测与编队逆映射仿真

**迁移到已有Ubuntu 20.04 / ROS 1环境：**见[Git上传与Ubuntu迁移步骤](docs/MIGRATION_UBUNTU20_ROS1.md)。代码使用独立Python 3.12环境，现有ROS/Gazebo/PX4系统保持原环境；Windows虚拟环境和中间调试结果不进入Git。正式结果、轨迹、研究内容和方法说明保留。

**最新交付：88秒离线轨迹回放与拍摄方案。** 已将申请书方法生成的三机p/v/a冻结为独立文件，用仅接收本机状态的PID执行器重新运行吊载物理模型；回放期间不运行负载ESO、MPC或PDE。16个最终工况全部完成，10个通过14项预设检查，另外6个的末段倾角超过1°；5个低噪声拍摄候选工况全部通过。这些是有限数值样本，未接入ROS/PX4或实机。

- [轨迹回放可行性报告](results/experiment_v1/轨迹回放可行性报告.md)：数据隔离、执行器假设、全部误差与载重失配结果。
- [实验轨迹与拍摄方案](results/experiment_v1/实验轨迹与拍摄方案.md)：88秒动作时序、20×10×5m场地安排、分镜与证据来源。
- [MP4演示](results/experiment_v1/media/experiment_replay.mp4) / [GIF动画](results/experiment_v1/media/experiment_replay.gif) / [三机轨迹CSV](results/experiment_v1/trajectory/three_uav_trajectory_NOT_FLIGHT_READY.csv)。

回放和图表入口：

```powershell
.\.venv\Scripts\python.exe run_simulation.py --scenario nominal --leg-duration 32 --output results/experiment_v1/planning
.\.venv\Scripts\python.exe run_replay.py --suite --workers 3
.\.venv\Scripts\python.exe run_replay.py --cases configs/replay_capture_cases.json --append --workers 3
.\.venv\Scripts\python.exe tools/render_experiment.py --animation
.\.venv\Scripts\python.exe -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath results/experiment_v1/unit_tests.txt
.\.venv\Scripts\python.exe tools/report_experiment.py
```

广范围`--suite`包含未达标工况，会返回非零状态并保存结果；继续运行后面的拍摄候选工况即可。`--append`不会覆盖已有工况。轨迹源更新后应重新跑完整流程。36项单元测试覆盖物理/PDE/合力方法及回放插值、输入隔离、观测延迟、参考时差和本机积分抗饱和。

下文保留原104秒在线研究控制器的运行说明；其结果与最新88秒轨迹回放属于不同执行设置。

当前实现申请书第 2.3(3) 节的合力前馈预测与集群编队逆向映射机制，并与第 2.3(1) 节 PDE 编队控制串联。默认运行链为：轨迹前馈与动量 ESO → 平动 Tube 结构 MPC → 三根缆绳的向量力分配 → 构型逆映射 → PDE、姿态和电机控制 → 独立三机吊载物理模型。本次没有算法性能对照。

已完成 104 秒、3 kg 负载的 S 形往返任务，包含峰值 0.3 N 的平滑侧向扰动。当前报告为 [合力预测与逆映射仿真报告](results/合力预测与逆映射仿真报告.md)，方程和工程补全见 [方法说明](docs/force_prediction_mapping.md)。

## 运行

在本项目目录的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe run_simulation.py --plots --animation
```

默认配置为 `configs/force_pde.json`，默认场景为 `force_transport`，输出到 `results/force_mapping`。显式指定的等价命令：

```powershell
.\.venv\Scripts\python.exe run_simulation.py --config configs/force_pde.json --scenario force_transport --output results/force_mapping --plots --animation
```

图表重绘和日志检查：

```powershell
.\.venv\Scripts\python.exe render_results.py results/force_mapping --animation
.\.venv\Scripts\python.exe tools/report_force_mapping.py
```

本工况从已吊起的名义悬停开始，执行 8 秒保持、40 秒去程、8 秒保持、40 秒返程及 8 秒末段保持。20—28 秒对负载施加峰值 0.3 N 的平滑 y 向力，以显示 ESO 和合力修正的作用。扰动真值仅供事后评价，不输入控制器。

`--scenario nominal` 可运行同一控制器的无外加扰动运输。`--payload-mass 2` 将候选负载改为 2 kg，`--leg-duration 30` 将单程时长改为 30 秒；新的参数组合应查看求解状态和物理约束，不能直接视为已校核工况。质量变化时控制器标称质量和物理质量同时改变，不是质量失配试验。

## 仿真体现了什么

- 预测器读取未来 2 秒参考，每 0.1 秒求解 20 步 QP，并记录当时的预测序列。
- 动量 ESO 使用理想模拟负载速度及实际绳索合力测量，输出扰动补偿。
- 分配器优化 9 个缆绳力向量分量，满足 6 条合力/力矩等式和正拉力锥，避免把三根固定方向绳误当成任意六维控制输入。
- 含挂点偏置的逆映射生成每架机不同的位置、速度、加速度参考，最大构型修正约 31.8 mm，实际进入 PDE 闭环。
- 几何参考采用 C² 滤波；当前分配力及时进入执行器前馈。负载及绳索响应由独立物理模型积分，不由参考赋值。
- 名义模型采用计算得到的误差管束和约束收紧；报告同时检查实际响应是否处于设计界内。

当前单次运行负载位置 RMSE 约 2.73 mm，最大误差 11.07 mm，最大负载倾角约 0.13°。26 项实现单元检查和 19 项本次日志检查均通过。以上是未标定数值模型的结果，不是实测精度。

## 结果文件

| 文件 | 内容 |
| --- | --- |
| `results/合力预测与逆映射仿真报告.md` | 本次工况、分层证据、数值结果与限制 |
| `results/force_mapping/force_mapping_chain.png` | 合力、张力、绳角、构型变化、ESO 与管束的九图汇总 |
| `results/force_mapping/prediction_window.png` | 仿真当时保存的未来 2 秒预测窗口 |
| `results/force_mapping/force_prediction_window.png` / `force_prediction_horizon.csv` | 由当时预测状态和ESO估计换算的20步名义合力序列，运行报告脚本生成 |
| `results/force_mapping/simulation_summary.png` | 物理轨迹、误差、张力、倾角、推力及约束残差 |
| `results/force_mapping/pde_evolution.png` | 动态构型下的 PDE 误差与边界反馈 |
| 同名 PDF | 可用于导出的矢量曲线图 |
| `results/force_mapping/transport_simulation.gif` | 由物理状态生成的 8 倍速动画 |
| `results/force_mapping/reference_only_NOT_FLIGHT_READY.csv` | 逆映射生成的三机位置、速度和加速度参考 |
| `results/force_mapping/pde_control_commands.csv` | 实际进入仿真速度/姿态控制链的 PDE 指令 |
| `results/force_mapping/force_prediction_mapping_chain.csv` | 前馈、需求、实际合力与各机分配力的数据链 |
| `results/force_mapping/state_log.npz` | 原始物理状态、ESO、QP预测、分配、滤波及逆映射日志 |
| `results/force_mapping/metrics.json` / `run_checks.json` | 数值指标与逐项检查结果 |
| `results/force_mapping/config_snapshot.json` / `metadata.json` | 运行参数、测量假设、管束定义和源码哈希 |

## 代码与依赖

| 文件 | 用途 |
| --- | --- |
| `sling_sim/force_controller.py` | 合力前馈、观测、预测、分配与 PDE 的串联 |
| `sling_sim/tube_mpc.py` | 平动 QP、持续名义状态、局部反馈、椭圆管束和约束收紧 |
| `sling_sim/force_mapping.py` | 动量 ESO、向量力分配、C² 参考滤波与解析逆映射 |
| `sling_sim/pde.py` / `pde_controller.py` | PDE 核、边界控制与动态构型跟踪 |
| `sling_sim/model.py` | 四刚体、绳索约束、四电机推力及 RK4 物理积分 |
| `sling_sim/force_results.py` / `force_visualization.py` | 分层指标、导出和图表 |

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

在其他电脑建立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

当前为 Python 3.12，依赖 NumPy、SciPy、OSQP、Matplotlib 和 Pillow。Windows 优先使用微软雅黑绘图。

## 实现边界

Tube 设计用于线性平动预测模型，旋转采用 ESO 与姿态反馈。程序没有终端不变约束、全六自由度非线性 Tube MPC 或递归可行性证明。报告将管束的模型条件和本次采样检查结果分开陈述。

研究外层假设能取得无噪声负载位姿、速度、角速度及绳力测量；这不同于此前只读取无人机状态的 PDE 单层控制。实机雷达、拉力传感器和 ROS/PX4 接口尚未接入，机体质量、推力、惯量、阻尼和带宽也未按实机标定。

当前物理模型只覆盖绳索绷紧的悬空运输，不包含松绳重接触、起吊落地、下洗或全系统碰撞检测。机间和场地余量是包围球采样诊断。输出轨迹尚未经过承载、坐标转换和时间同步校核；实机若后续只回放轨迹，应将其表述为仿真轨迹复现。

先前 PDE 单层及基础模型的结果保留为历史记录，不参与本次统计；初始串联调试结果保存在 `results/force_mapping_initial_unstable` 等明确命名的目录，不作为当前可行方案。当前执行依据是本页和最新合力预测与逆映射报告。
