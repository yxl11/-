# Ubuntu / ROS / Gazebo / PX4 迁移说明

核查日期：2026-11-16。本文件用于下一台 Ubuntu 计算机上的复现与改写交接；当前交付的是既有数值仿真及其证据，尚未完成 PX4 SITL 接入。先恢复数值基准，再建立独立物理执行接口，最后扩展到多机。

**主路线沿用用户已有 Ubuntu 20.04 / ROS 1 / Gazebo / PX4 环境，不主动升级。**本次新查到的协同吊运项目文档记录了 Ubuntu 20.04、ROS Noetic、Gazebo Classic 11、PX4 1.13.2；Ubuntu 机器的实际安装版本和本地改动仍需现场采集。后文 ROS 2 / PX4 v1.17 只作为明确版本条件下的备选接口，不能直接套到当前 ROS 1 项目。

## 1. 首先恢复哪一个实验

以交接包内 `reproduction/README.md`、`manifest.json`、`entrypoint_hashes.json` 和 `environment.json` 为入口。当前实验为 **10 架固定翼、16 架 FPV、20 架四旋翼，178 s**；正式初态种子为 33001–33003。本文方法的冻结增益为 `kc=0.24, kf=0.16`。保持任务、成员、参考和统计窗口的身份，不把历史 80/96 机或 120/240 s 任务混入本次复现。

建议在新目录中依次执行下列操作，命令参数以包内 README 为准：

```bash
# 在 heterogeneous_swarm/ 目录；独立环境的安装方法见 README
python3.11 -m venv .venv-swarm
.venv-swarm/bin/python -m pip install -r reproduction/requirements.txt
.venv-swarm/bin/python -X utf8 -B reproduction/verify_package.py
.venv-swarm/bin/python -X utf8 -B reproduction/reproduce.py --out ../work/linux_reanalysis_01
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py --method P --seed 33001
# 上一行仅进行初态和控制接口预检。
# 需要重跑原数值仿真时才执行；work-dir 必须是新目录：
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py --method P --seed 33001 --execute --work-dir work/linux_full_P_33001
```

已登记的原环境为 Windows / Python 3.11.6，依赖为 NumPy 2.4.6、SciPy 1.17.1、Matplotlib 3.10.9。Linux 上的数值一致性仍需实际运行确认。`reproduce.py` 从保存轨迹重新计算统计，不启动控制器；`run_full.py --execute` 才重新推进旧数值模型。两个操作均不能称为 Gazebo/PX4 仿真。

既有 Ubuntu 20.04 / Noetic 接口使用系统 Python 3.8；不要把论文仿真的 Python 3.11 依赖强行装进 ROS 环境。第一步保留两个环境：独立 Python 3.11 复现原基准，系统 Python/ROS 工作空间承担桥接，可先通过进程间接口交换明确的状态与参考。后续若将控制核嵌入 ROS 节点，应在适配分支登记兼容依赖并做同状态数值回归，不修改冻结快照和哈希清单。

## 2. 先复用现有 ROS 1 环境，再决定是否需要新版本

本次已读本机协同吊运项目的 `docs/MIGRATION_UBUNTU20_ROS1.md` 和 `docs/GAZEBO_ROS1_PX4.md`。其中已有三套 MAVROS / PX4 SITL、Gazebo ODE、同步播放器、独立定位输入、端口分配和日志方案。这些基础设施可以参考；吊载模型、三机轨迹、Python 3.12 依赖和项目私有的载荷参数不属于本论文控制任务，不应整体移植。

在 Ubuntu 原 ROS 终端先保存版本清单：

```bash
lsb_release -ds
rosversion -d
gazebo --version
rosversion mavros
# 进入实际 PX4 目录后：
git describe --tags --always
git rev-parse HEAD
git status --short
```

检查现有 MAVROS `setpoint_raw.cpp` 的 `local_cb`，确认 ROS 输入坐标与 MAVLink 转换；检查当前 PX4 模型、插件路径、端口和启动参数，保持已有吊运工程不变，在新目录创建集群任务运行文件。现有文档记录的三机端口只作参考，46 机需要重新生成唯一端口表，避免与另一实验同跑时冲突。既有 ROS 1 架构不需要 uXRCE-DDS；使用 MAVROS/MAVLink。

先复用当前安装版本的官方旋翼与固定翼模型，核对质量、惯量、气动、推进和飞行包线。仅将现有旋翼模型速度上限改成 12 m/s，不能使它自动成为可信的 FPV 型号。新的模型和飞控参数需要单独归档，不能替换原数值基准。

## 3. 原代码中应保留与替换的接口

下面所有 `output/...` 均为**快照根相对路径**：本文方法快照根是 `reproduction/snapshots/P/`，外部方法快照根是 `reproduction/snapshots/external/`。已实查两个快照，所列 `shape_feedback.py`、`shape_models.py`、`engine.py` 和 `mission_engine.py` 均在 `output/reviews/mixed_sweep_orbit_external_20261101/`；目录名中的 external 不表示这些文件只服务外部控制器。

| 快照根相对路径或对象 | 作用与迁移处理 |
|---|---|
| `output/reviews/mixed_sweep_orbit_external_20261101/shape_feedback.py`，`TransportController.command(t,r,v)` | 产生逐机世界坐标速度参考；保留密度反馈、群级投影、旋转/尺度变化前馈和成员顺序 |
| `output/reviews/mixed_sweep_orbit_external_20261101/shape_models.py`，`shape_axes` | 任务形状随时间变化，物理 KDE 带宽固定；不可只移植群心路径 |
| `output/reviews/mixed_sweep_orbit_external_20261101/engine.py` | 任务切向巡航封装；`Governor.select` 委托既有接口，`horizontal_backup` 生成 9 m/s 水平后备 |
| `output/reviews/mixed_sweep_orbit_external_20261101/mission_engine.py` | 对任务总时长配置数值累积余量；它调用上行 `make_engine`，不是独立的飞行动力学文件 |
| `output/reviews/mixed_role_corridor_demo_20261031/engine.py` | 上述 engine 明确导入的底层执行入口，需继续跟踪其导入链以读取动力学、候选与认证实现 |
| `output/reviews/p_gain_calibration_20261108/code/run_candidate.py`（P 快照） | 本轮校准后的 P 正式入口，含实际参数与执行装配；不能只凭较早文件名选择运行版本 |

交接根下 `reproduction/run_full.py` 负责选择完整快照和正式入口，属于包级文件，不在上述快照根内。完整共同执行还包含装配时启用的局部修正，需由该入口的导入链核实；迁移时不能因只阅读单个 wrapper 而遗漏它。

建议新适配层保留接口 `command(t, positions_enu, velocities_enu)`，输入数组按固定成员 ID 排序。状态来自 PX4 估计器；Gazebo 真值另行记录，用于量化估计误差，不能混用两者生成一条未说明来源的反馈轨迹。日志分别保存名义参考、保持参考、限幅参考、原筛选输出、实际发送给飞控的量以及测得速度。

## 4. 固定翼不能直接套用多旋翼速度通道

多旋翼可通过 MAVROS `setpoint_raw/local` 的 `mavros_msgs/PositionTarget` 发送速度，明确 `type_mask` 只启用需要的量；从现有三机播放器拆出通用发送、状态采集、预发送和模式检查。**ROS 字段保持 ENU，由 MAVROS 转换一次；即使 `coordinate_frame=FRAME_LOCAL_NED`，也不要预先再转换为 NED。**已核对 [MAVROS 1.20.0 源码 `local_cb`](https://github.com/mavlink/mavros/blob/1.20.0/mavros/src/plugins/setpoint_raw.cpp)，Ubuntu 实际版本还须对应核对。

PX4 v1.13 官方文档明确：固定翼 `SET_POSITION_TARGET_LOCAL_NED` 支持位置目标，速度和加速度字段被忽略。因此现有旋翼 MAVROS 速度发送器不能原封不动控制固定翼。可先使用官方位置/航路接口检查固定翼模型；若把速度场转换为移动航路目标，前视距离、速度/高度指令、转弯与后备语义都是需要登记并验证的适配设计。也可以走姿态/推力接口，但那需要额外的制导与速度控制，不能视为简单话题转换。本次不替用户发明或预先宣称这些接口已完成。[PX4 v1.13 官方 Offboard 支持表](https://docs.px4.io/v1.13/en/flight_modes/offboard)

固定翼是本次迁移的第一项决定性接口检查：先采集 Ubuntu 的真实版本、检查其固定翼 Offboard 源码，再选能保留速度场意义的适配方式。如果现有版本需要较多新控制开发，再评估文末版本备选；不因文档有新接口就立即升级整个已运行系统。

## 5. 坐标、时间和多机 ID

1. **统一物理坐标。**以原仿真的水平 x/y、向上 z 建立共享 ENU 世界系。ROS 1/MAVROS 保持 ENU 字段，由插件完成一次转换；机体 FLU/FRD、姿态四元数另行核对。逐机 PX4 局部原点可能不同，应保存共享地图变换并处理估计器重置，不能把各机出生点都当同一个零点。转换依据见上一节 MAVROS 源码。
2. **以仿真时间为准。**原任务每 0.1 s 计算并保持参考；原安全与执行每 0.02 s 更新，指标每 0.5 s 采样。Gazebo 积分和 PX4 内部控制频率另行记录。ROS 1/Gazebo Classic 沿用已有 `/clock` 与 `use_sim_time`；对暂停、倒退、超时和缺帧显式处理，不能用循环次数替代物理时间。
3. **逐机身份表。**保存成员 ID → 角色 → Gazebo model name → PX4 实例 → MAVROS namespace → MAVLink 端口 → `MAV_SYS_ID`，核对每条状态和指令确实对应目标成员。ROS 1 主路线不使用 DDS namespace 或 uXRCE key。
4. **通信与耗时。**检查队列、消息时间戳、MAVROS 连接、状态年龄和丢包；记录反馈耗时与截止期错失。原 20 ms 更新不是 46 个 SITL 实例的实时保证。发送线程与心跳不能被耗时的全机安全搜索阻塞。预发送和失联动作按所固定 PX4 版本登记，持续发送频率须满足其 Offboard 要求。[PX4 v1.13 Offboard](https://docs.px4.io/v1.13/en/flight_modes/offboard)

## 6. 分阶段推进与验收

| 阶段 | 应完成的工作 | 可检查的结果 |
|---|---|---|
| A. Linux 数值基准 | 按第 1 节校验哈希、重算统计、预检初态；需要时重跑一条原模型轨迹 | 保存环境、命令、退出码和差异报告；浮点阈值按原复现包，不擅改 |
| B. 单四旋翼 | 官方模型起飞/进入工作高度后，测试东、北、上、转向和零参考；接入一段冻结速度输入 | 坐标与符号正确，输入层级和状态来源清楚，速度响应/超调/饱和/延迟有日志 |
| C. 单固定翼 | 官方固定翼模型先建立稳定飞行；按实际版本验证固定翼接口、8–15 m/s 原包线的可实现范围和巡航后备 | 核对空速、地速、转弯率、爬升、失速/饱和及实际后备路径；不要求结果必须吻合原代理 |
| D. 三角色小规模闭环 | 每角色 1–2 架验证命名、共享坐标、任务重算和接管，使用真实返回状态 | 明确这是迁移检查而非论文 46 机正式比较；检查状态完整性、失联动作与运行预算 |
| E. 原规模闭环 | 资源和执行接口合格后，再加载原 46 机、原任务、原初态与冻结控制参数 | 先完成本文方法，再按同一执行对象和配置进行外部对照；所有新结果保存到新目录 |

原实验从空中状态开始，固定翼具有非零初速。PX4 正式闭环需要明确起飞/初始化与任务计时起点：可先完成独立起飞和稳定飞行，再在满足登记状态条件时启动任务时钟。不能从地面直接发 9 m/s 参考，却把起飞过渡当成原有初始状态；也不能为匹配曲线偷偷重置动力学状态。

单机重放检验输入响应；多机闭环必须以独立对象当前状态重新计算 P、候选与安全决定。旧认证依赖旧预测和执行一致，不能原封不动继承到 PX4。尤其保留后备证书依赖其预测状态，PX4 的状态偏离后须重新检验。先保留旧筛选输出作为可追溯参考，量化预测残差和后备可执行性；若未建立覆盖这些偏差的条件，新仿真只能报告实际最小间距及接管行为，不能声称原安全保证已被延伸。

建议每次运行保存：软件与模型提交、全部 PX4 参数、任务/角色/初态文件、随机种子、ROS bag、PX4 ULog、Gazebo 真值、逐层指令、控制模式切换、后备事件、时钟与资源统计。这样可区分算法反馈、飞控执行、模型变化与通信实现造成的差异。

## 可选版本附注：当前无需升级

只有在 ROS 1 固定翼接口评估后确认值得另建环境，才考虑官方支持的 Ubuntu 22.04 / ROS 2 Humble / Gazebo Harmonic / PX4 **v1.17.0 稳定标签**（提交 `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`）。`main` 是开发分支。该版本具有 `FixedWingLateralSetpoint` 与 `FixedWingLongitudinalSetpoint` 接口；其中 `course` 是相对北的地面航迹方向，`height_rate` 向上为正，`equivalent_airspeed` 是等效空速，不能直接当作地速。多旋翼 `TrajectorySetpoint` 的速度模式不能直接照搬给固定翼。新环境需要匹配 `px4_msgs`、uXRCE-DDS、时钟和控制模式，属于单独适配工作。

核查入口：[稳定发布](https://github.com/PX4/PX4-Autopilot/releases/tag/v1.17.0)、[ROS 2 版本/桥接说明](https://docs.px4.io/v1.17/en/ros2/user_guide)、[固定翼控制接口](https://docs.px4.io/v1.17/en/ros2/px4_ros2_control_interface)、[v1.17 固定翼模式源码](https://github.com/PX4/PX4-Autopilot/blob/v1.17.0/src/modules/fw_mode_manager/FixedWingModeManager.cpp)、[纵向字段定义](https://github.com/PX4/PX4-Autopilot/blob/v1.17.0/msg/versioned/FixedWingLongitudinalSetpoint.msg)。这里只提供核查入口，不提供升级命令。

## 本地资源依据

- 同仓库的[协同吊运迁移说明](../../docs/MIGRATION_UBUNTU20_ROS1.md)。
- 同仓库的[已有 Gazebo / ROS 1 / PX4 接口记录](../../docs/GAZEBO_ROS1_PX4.md)，记录 Noetic / Classic 11 / PX4 1.13.2。
- 本文原始复现档案：`output/reviews/two_round_revision_20261108/reproducibility/`；**此次 Git 交付目录名为 `reproduction/`**。论文 Python 3.11 环境与吊运工程 Python 3.12 环境分开。

本文件只核查文档、源码接口与迁移关系，未启动 Ubuntu、Gazebo 或 PX4 实例；实际版本兼容性及物理闭环表现留待 Ubuntu 端验证。
