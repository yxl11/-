# 异构无人机群：原仿真复现与 ROS / Gazebo / PX4 迁移交接

本目录保存论文当前采用的 **10 架固定翼 + 16 架 FPV + 20 架四旋翼、178 s 扫掠—重组—巡检任务**。包含控制代码、冻结参数、初态、三种方法的九条原始轨迹、统计与图形数据，以及 Ubuntu 接入说明。

**交付状态：原 Python 数值模型及结果可复核；本论文的 ROS / Gazebo / PX4 适配尚未实现。** 同仓库原有的协同吊运项目已经提供可参考的 ROS / MAVROS / PX4 基础设施，两项实验各自保留模型、参数和数据身份。

## 1. 在 Ubuntu 获取材料

上传分支：`codex/heterogeneous-swarm-handoff-20261116`。没有合入原 `master`，也没有改写吊运项目。

```bash
git clone --single-branch --branch codex/heterogeneous-swarm-handoff-20261116 \
  https://github.com/yxl11/-.git heterogeneous-uav-handoff
cd heterogeneous-uav-handoff/heterogeneous_swarm
```

仓库名称确实为 `-`，因此命令显式给出本地目录名。已有吊运工作树时，建议另行克隆，避免在有未提交改动的工作树切换分支。认证沿用自己的 GitHub 授权；仓库不保存凭证。

## 2. 按这个顺序阅读

| 入口 | 回答的问题 |
|---|---|
| [代码与参数地图](docs/CODE_AND_PARAMETERS.md) | 本文控制器在哪里、实际增益是什么、任务和执行如何装配 |
| [PX4 迁移说明](docs/PX4_MIGRATION.md) | 保留什么、替换什么、固定翼与旋翼接口有什么差别 |
| [交给 Ubuntu 端 Codex 的提示词](docs/NEXT_MACHINE_TASK.md) | 下一台电脑如何从环境核实推进到单机接口与真实状态闭环 |
| [冻结包原说明](reproduction/README.md) | 原始仿真和结果复算的正式入口与来源 |
| [当前论文 PDF](paper_context/execution_error_derivation.pdf) | 模型、控制结构、实验解释与公式；是既有 PDF 的复制件 |
| [任务动画](media/P_B_s33001_sweep_orbit.mp4) | 当前 P 参数、初态 33001 的实际保存轨迹；动画主体为 5 倍速 |
| [交付验证](verification/summary.json) | 本次迁移后的实际检查结果和运行环境 |

`reproduction/snapshots/` 保持历史相对目录，是有意保留的依赖闭包。不要只复制某一个 controller 文件、扁平化目录或覆盖其中的历史论文。原入口会核对源码、初态和历史稿件哈希。

## 3. 核实已有 Ubuntu 环境

协同吊运项目记录了 **Ubuntu 20.04 / ROS Noetic / Gazebo Classic 11 / PX4 1.13.2**。这是一份已存在的项目记录，另一台电脑实际版本以现场采集为准。先加载原 ROS 环境，再执行：

```bash
mkdir -p work
PX4_DIR="$HOME/PX4-Autopilot" bash tools/inspect_environment.sh > work/ubuntu_environment.txt
```

将 `PX4_DIR` 换成实际路径。此脚本只读取版本、源码状态与硬件信息，不安装、升级或启动飞控。保留现有 ROS、Gazebo、PX4 和吊运工程，先检查其可复用接口。

## 4. 建立独立的数值复现环境

原包验证环境是 Python 3.11.6，依赖固定为 NumPy 2.4.6、SciPy 1.17.1、Matplotlib 3.10.9。入口使用 `hashlib.file_digest`，需要 Python 3.11 或以上；建议先使用独立 Python 3.11。**不要使用吊运工程的 Python 3.12 虚拟环境，也不要把依赖装进 ROS Noetic 的系统 Python 3.8。**

如果已安装独立 Python 3.11：

```bash
python3.11 -m venv .venv-swarm
.venv-swarm/bin/python -m pip install -r reproduction/requirements.txt
```

否则可用 uv 在用户目录安装独立解释器。已有 uv 时跳过安装两行：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/swarm-uv-install.sh
sh /tmp/swarm-uv-install.sh
"$HOME/.local/bin/uv" python install 3.11
"$HOME/.local/bin/uv" venv --python 3.11 .venv-swarm
"$HOME/.local/bin/uv" pip install --python .venv-swarm/bin/python -r reproduction/requirements.txt
```

以上不替换 `/usr/bin/python3`；后续命令显式调用虚拟环境解释器。安装依据：[uv 安装](https://docs.astral.sh/uv/getting-started/installation/)、[独立 Python 安装](https://docs.astral.sh/uv/guides/install-python/)。目标机器的 Linux 依赖安装和运行仍需实测。

## 5. 先验证材料，再决定是否重跑

```bash
# 从本 README 所在目录运行；输出目录必须尚不存在
.venv-swarm/bin/python -X utf8 -B tools/verify_handoff.py --out work/ubuntu_check_01
```

该命令校验冻结清单，复算保存轨迹的统计与图数据，并对三方法 × 三初态分别构造控制器、计算初始参考。它不推进平台、不运行九条新轨迹。详细日志和数值检查保存在指定目录；`verification/` 是本次 Windows 交付检查的历史结果，不是 Ubuntu 已通过的证据。

也可以逐项执行原入口：

```bash
.venv-swarm/bin/python -X utf8 -B reproduction/verify_package.py
.venv-swarm/bin/python -X utf8 -B reproduction/reproduce.py --out ../work/ubuntu_metrics_01
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py --method P --seed 33001
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py --method RAS04 --seed 33001
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py --method MMD19 --seed 33001
```

`reproduce.py` 的相对输出路径以 `reproduction/` 为基准。统计复算保留原指标定义：其中密度/形状采用原已保存记录，不能把复算表格称为重新估计了全部 KDE。

重新运行一条完整 **原数值模型** 的 178 s 仿真才使用：

```bash
.venv-swarm/bin/python -X utf8 -B reproduction/run_full.py \
  --method P --seed 33001 --execute --work-dir "$HOME/uav_runs/P_s33001_01"
```

新目录不能预先存在。外部方法替换 `--method` 为 `RAS04` 或 `MMD19`；正式初态为 33001、33002、33003。运行时间由计算机决定，任务时间 178 s 不等于墙钟耗时。原执行入口会复制冻结快照再运行，避免覆盖已保存证据。

## 6. 导出任务和初态，供 ROS 适配读取

```bash
.venv-swarm/bin/python -X utf8 -B tools/export_task.py --out work/task_export_01
```

导出由原解析函数计算的角色参考及形状矩阵、固定成员顺序和三组初态，附来源哈希与字段说明。它不是逐机开环飞行任务，也不生成 PX4 参数文件。闭环适配仍须将实时状态送入控制器，持续重算逐机指令；参考路径重放只适用于接口响应检查。

本次已生成一份[任务与初态读取样例](adaptation_inputs/README.md)，可直接查看 JSON 或 NPZ；实际 PX4 实例、端口和坐标变换在 `vehicle_map.template.json` 中留空，须按另一台电脑的环境填写。

## 7. 当前有效参数和数据身份

| 项目 | 当前基线 |
|---|---|
| 本文方法 | `P`，校准候选 `B`：`kc=0.24 s^-1`，`kf=0.16 s^-1` |
| 点位对照 | `RAS04` 数据别名，内部方法名 `RAS08`；有效 `alpha=coupling=0.4 s^-1` |
| 密度对照 | `MMD19`，有效 `gamma=20` |
| 成员数 | 固定翼 10、FPV 16、四旋翼 20 |
| 速度上限 | 15、12、10 m/s；固定翼另有 8 m/s 下限与转弯/爬升约束 |
| 时钟 | 名义反馈 0.1 s；原安全/执行 0.02 s；密度/形状指标 0.5 s |
| 评价集合 | 3 个正式初态，共 9 条轨迹；独立校准种子与规则随快照保存 |
| 主要分析窗口 | 0–8 s 初始部署；8–86 s 扫掠和重组；86–178 s 巡检 |

`RAS08` 中 08 是原文年份；`RAS04` 中 04 是本次增益别名，均不是新的算法。外部配置继承的 `kc=.12/kf=.08` 不进入其反馈，应读各运行的 `controller_identity.json`。某些历史 scope 字符串提及 ten-seed，交付评价集合以三个正式种子及九个 `data/` 目录为准。新说明澄清这些历史字段，原件字节保持不变。

## 8. 移植边界与文件保护

原控制器输出逐机世界坐标速度参考。ROS 1 / MAVROS 接口通常接收 ENU 字段并转换到 PX4 使用的坐标；固定翼的控制通道与多旋翼不同。PX4 1.13 的固定翼位置目标通道忽略速度/加速度字段，需先验证相应制导适配，不能只改消息名就接入。[PX4 1.13 官方接口说明](https://docs.px4.io/v1.13/en/flight_modes/offboard)

原安全筛选依赖代理模型与执行一致。PX4 和 Gazebo 的执行、估计、时延和后备轨迹必须重新核查；原 2 m 距离参数不能直接作为高保真对象已通过认证的证据。详见迁移说明中的分阶段检查。

- `reproduction/`：冻结输入及历史证据，保持只读；修改算法或包装时在新工作副本进行。
- `paper_context/`：2026-11-16 从当前论文及其已有 PDF、两幅图复制的阅读上下文，未重新编译。
- `media/`：当前 P/33001 已保存轨迹动画及原来源清单；视频不是物理引擎录像。
- `provenance/`：逐文件 SHA256、来源和排除项；未复制冗余 `migration_probe/` 或缓存。
- `tools/`、`docs/`：本次新增的迁移辅助和说明。
- `verification/`：本次交付检查，保留与未来 Ubuntu 检查的身份差别。
- `work/`：新导出、日志及开发产物，Git 默认忽略。未来 ROS 包放独立目录，Gazebo/PX4 依赖用精确版本记录，不提交整个工具链或虚拟环境。

各次新试验保存任务、控制配置、PX4 参数、模型版本、完整状态和逐层指令；比较中保持同一执行对象和时钟。Linux 中发现的浮点差异、运行失败或物理执行偏差按事实记录，不覆盖冻结基线。
