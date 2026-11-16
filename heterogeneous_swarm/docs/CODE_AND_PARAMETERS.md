# 源码阅读与参数对照

本文档说明仓库中现有仿真实际执行了什么，以及将它接入 Ubuntu、ROS、Gazebo 和 PX4 时，应从哪些接口开始。冻结材料位于 `reproduction/`；该目录保存原实现和证据，新适配应放在其外部。

现有基线是 **46 架、178 s 的三角色数值闭环**，包括 P、点位编队和核分布匹配三种反馈。它使用固定翼航迹运动学和旋翼速度响应模型，没有包含 ROS 节点、Gazebo 模型或 PX4 固件配置。这里的源码地图是迁移依据，并不表示物理引擎适配已经完成。

## 1. 从哪里开始读

按以下顺序阅读，可以先获得可运行入口，再理解反馈、执行和统计。

| 顺序 | 文件或目录，相对 `heterogeneous_swarm/` | 用途 |
|---|---|---|
| 1 | `reproduction/README.md` | 原复现包的运行约定与证据范围 |
| 2 | `reproduction/run_full.py`、`preflight_context.py` | 方法身份、参数选择、初态读取、预检与完整重跑入口 |
| 3 | `reproduction/data/P_s33001/configuration.json`、`controller_identity.json` | 当前 P 的实际配置与控制身份 |
| 4 | `reproduction/snapshots/P/output/reviews/p_gain_calibration_20261108/code/worker.py` | 当前 P 配置注入与完整闭环调用顺序 |
| 5 | `reproduction/snapshots/P/output/reviews/mixed_sweep_orbit_external_20261101/mission.py`、`shape_models.py` | 移动参考和群内时变形状 |
| 6 | 同目录 `shape_feedback.py`、`p_feedback.py` | 实际使用的 P 控制接口及历史核心加载方式 |
| 7 | 同目录 `mission_engine.py`、`engine.py` | 46 架共同执行、安全搜索与共同巡航后备的组装 |
| 8 | `reproduction/frozen_analysis/metrics.py`、`reproduce.py` | 现有论文指标和图形数据的复算 |

后文以两个相对路径作为简称：

- **R**：`reproduction/snapshots/P/output/reviews/`
- **S**：`reproduction/snapshots/P/output/simulations/three_group_moving_comparison_20261023/`

这些是阅读简称，不是程序所需的环境变量。外部方法完整重跑使用 `snapshots/external/` 中的对应文件；两份快照分别保留其历史来源哈希，不应合并覆盖。

`build_package.py`、`complete_dependencies.py`、`add_density_reference.py`、`finalize_delivery.py` 等是作者端打包记录，不是接入控制器的入口。部分历史文件仍有四群、80 架、旧任务或旧增益的文字，当前入口、实际配置和控制身份记录优先于这些历史注释。

## 2. 当前控制链及可迁移接口

### 2.1 P 的公开调用边界

当前 P 的在线边界是 `R/mixed_sweep_orbit_external_20261101/shape_feedback.py` 中：

```python
TransportController(h, cfg, reference)
controller.command(t, r, v)  # 返回 shape=(46, 3) 的世界坐标名义速度参考
```

| 输入/输出 | 含义 | 形状与单位 |
|---|---|---|
| `t` | 自任务开始计时的时间，不是 Unix 时间戳 | 标量，s |
| `r` | 全体无人机当前实际位置，按固定成员顺序排列 | `(46,3)`，m |
| `v` | 同一时刻的实际速度；P 当前名义位置反馈接口保留此输入，安全和执行使用它 | `(46,3)`，m/s |
| `cfg` | 当前配置，尤其包含校准后的 P 增益 | Python 字典 |
| `reference(t,cfg)` | 三个角色的位置与速度参考 `q, qd` | 各 `(3,3)`，m 和 m/s |
| 返回值 | 限速、飞行包线映射和安全改写之前的名义速度 | `(46,3)`，m/s |
| `controller.last_fields` | 群均值、形状、投影和仿射输运等诊断量 | 随字段定义 |

`h` 由 `p_feedback.load_frozen()` 返回的历史控制加载器提供。实际构造器为 **`S/src/harness.py:PController`**，它构造三群数据结构，再使用 vendor 中的原在线反馈函数。不能直接把历史 `ProjectedController` 的构造器当作当前三群入口：其父类构造过程含四群限制。

调用关系为：

```text
当前 worker.make_configuration（注入 kc=.24, kf=.16）
  -> p_feedback.load_frozen()
  -> ShapeFeedback(...).controller
  -> TransportController
  -> harness.PController
  -> vendor/ProjectedController.fields
  -> vendor/PointController.fields + 精确密度图梯度
```

以下三个文件解释 P 的核心计算：

- `S/vendor/projected_mobility_paired_20261014/src/projected_controller.py`：群级加权投影及相对修正。
- 同目录 `point_controller.py`：高斯 KDE 点查询、样本均值校正与群内形状反馈。
- `S/vendor/unified_density_cooperation_20261014/src/task_density_controller.py`：角色密度图的高斯核重叠和梯度。

当前在线控制调用点查询反馈，不调用父类中的 Gauss–Hermite 离线诊断梯度。不要因看到父类中 `gradient_order=5`，就把在线算法写成每步做五阶数值积分。

### 2.2 任务与形状接口

`mission.reference_kinematics(t)` 返回 `q, qd, qdd`；`mission.reference(t,cfg)` 返回前两项。标量时间对应三角色的 `(3,3)` 数组，也支持时间数组。

`shape_models.shape_axes(t)` 对标量时间返回：

- `R`：`(3,3)` 正交旋转矩阵，列向量为形状坐标轴在世界坐标中的方向；
- `dR`：其时间导数，单位 s⁻¹；
- `s`：`(3,3)`，每角色三个目标标准差，单位 m；
- `ds`：标准差变化率，单位 m/s。

`shape_state(t)` 给出每角色 `S_i=R diag(s_i)` 及其导数。目标协方差为 `S_i S_iᵀ`，物理坐标与角色坐标的关系为 `x=q_i+S_i z`。代码采用行向量，因此 `(r-q) @ R` 是进入旋转坐标的操作，乘 `R.T` 是返回世界坐标；移植时不要将转置方向颠倒。

`TransportController` 每次调用都更新瞬时尺度和任务度量，并加入由 `dR`、`ds` 形成的仿射输运速度。只复制静态 `ProjectedController` 而遗漏此包装，会丢失现有实验中的群内形变和随航向旋转。

### 2.3 一次闭环更新的顺序

当前 P 运行器为 `R/p_gain_calibration_20261108/code/worker.py`；外部方法运行器为 `R/mixed_sweep_orbit_paired_20261102/worker.py`。

1. 每 **0.1 s** 用当前实际 `r,v` 计算 `controller.command(t,r,v)`，随后保持名义参考。
2. 每 **0.02 s** 对保持参考做三维合成速度限幅，得到 `speed_capped`。
3. 固定翼包线映射产生 `envelope_mapped`；安全候选中的局部变化同样经过映射。
4. `engine.set_direction(...)` 设置本次候选的共同巡航方向；`gov.select(step,r,v,capped)` 搜索并返回最终接受输入。
5. `engine.advance_model(r,v,command)` 按与认证一致的模型推进；`gov.commit(rn,vn)` 核对实际推进结果与保留证书的一致性。

最终接受输入可在每个 0.02 s 周期改变；不能把名义参考的 0.1 s 保持等同于最终输入保持。接受输入在原模型推进前没有再次改写。

历史 `history.npz` 保留 `held_reference`、`speed_capped`、`envelope_mapped`、`accepted`、位置和速度等层级。原运行器还生成 0.1 s 的 `task_history.npz`，其中含 `raw_reference` 及反馈分解；当前交付的九条 `data/` 轨迹未复制此额外文件，名义更新时刻的原始参考可从 `history.npz` 的 `held_reference` 对应行读取。将 ROS 实际速度与 `accepted` 比较是在检查执行接口；将它与原始名义参考比较还包含限幅和安全改写，两个误差含义不同。

## 3. 坐标、成员和时钟

世界坐标采用右手笛卡尔系、**z 向上**；x 为初始前进方向，y 为水平横向。源码没有经纬度原点或真北定义，也没有预设它已是某个地理 ENU 坐标系。固定翼模型假设零风，因而当前地速与相对空气速度相同。

| 角色编号 | 数组成员下标（Python） | 数量 | 当前角色名称 |
|---|---|---:|---|
| 0 | 0–9 | 10 | `fixed_wing_scout`，固定翼侦察 |
| 1 | 10–25 | 16 | `fpv_relay`，FPV 中继 |
| 2 | 26–45 | 20 | `quadrotor_support`，四旋翼支援 |

任务图为角色 **0–1–2**。初态读取冻结的 `positions`、`velocities`、`roles`，不要在 ROS 中按消息到达顺序重排成员，也不要自行重新抽样代替已登记初态。正式初态为 33001、33002、33003；它们是同一任务的三组初始位置和速度，不是三个任务场景。

原任务已从运动状态开始，不含真实地面起飞、固定翼弹射、解锁或降落程序。PX4 需要另行处理初始化与接入任务的方式，并明确何时定义任务时间零点。

接入 PX4 前须明确选择 ROS/world 坐标与 PX4 所用接口坐标的转换。若适配器将源世界坐标明确指定为 ENU，则通常的 ENU→NED 分量变换为 `[x,y,z]_NED=[y,x,-z]_ENU`，速度同样变换；这是待落实的适配约定，不是原代码已经做过的转换。航向、四元数和机体坐标还需单独统一。原 `chi` 为速度方向的航迹方位角，`gamma` 为航迹倾角，**不是机体 yaw/pitch/roll**。

时钟含义如下：

| 时钟 | 周期 | 作用 |
|---|---:|---|
| 名义任务反馈 | 0.1 s | 读取同步状态，生成并保持名义参考 |
| 原安全和数值执行 | 0.02 s | 候选检查、接受输入、原模型推进 |
| 慢指标采样 | 0.5 s | 密度与中心化形状等统计 |

0.5 s 不是密度控制更新周期。Gazebo/PX4 的物理积分、飞控更新和 Offboard 消息发送周期须另行登记；不能将它们假定为上述三个时钟的同义词。

## 4. 当前任务规格

任务由 `R/mixed_sweep_orbit_external_20261101/mission.py` 精确定义：基准沿线速度 **9 m/s**，角色参考高度为 **60、53、50 m**，总时长 **178 s**。

| 原始时间段 | 任务动作 |
|---|---|
| 0–8 s | 沿线初始部署 |
| 8–68 s | 前方左右往返扫掠 |
| 68–86 s | 编队重组 |
| 86–102 s | 平滑转入环巡 |
| 102–178 s | 圆周巡查，截断时仍在运动 |

完整圆周的解析区间为 **102–174 s**，周期 72 s。基准圆半径为 `9/(2π/72)`，约 103.13 m；各角色叠加纵横偏置，其实际参考圆半径不同。主结果另按 **0–8 s、8–86 s、86–178 s** 汇总，后两个是结果比较窗口，不应误写成原任务只有两个运动阶段。

各时刻相对 `(9t,0)` 的水平角色偏置如下，顺序为固定翼、FPV、四旋翼：

| 时刻 s | 固定翼偏置 m | FPV 偏置 m | 四旋翼偏置 m |
|---:|---|---|---|
| 8 | `(27,-4)` | `(0,0)` | `(-20,0)` |
| 24 | `(39,34)` | `(2,4)` | `(-20,-6)` |
| 46 | `(39,-34)` | `(0,-4)` | `(-17,8)` |
| 68 | `(39,34)` | `(2,4)` | `(-20,-6)` |
| 86 | `(34,-10)` | `(0,0)` | `(-20,13)` |

相邻规定时刻用五次平滑函数连接。86 s 后最后一组偏置进入随基准航迹旋转的切向—法向坐标，不能继续作为固定世界偏置使用。

形状标准差的初值为 `(13,9,3.5)`、`(5.5,4.5,3)`、`(4.8,4.5,2.8)` m。8–24 s，纵向标准差平滑变为 `7.5、4.0、4.3` m；68–86 s 再变为 `14.5、6.0、5.5` m。水平另一轴反向缩放以保持两轴乘积，竖直标准差不变；86 s 后形状坐标轴随航迹方向旋转。初值表不能代替完整时变协方差。

## 5. 有效参数及容易误读的字段

### 5.1 P 反馈

以 `data/P_s33001/configuration.json` 和当前 worker 的 `make_configuration` 为准：

| 参数 | 当前值 | 实际作用 |
|---|---|---|
| `kc_per_s` | 0.24 s⁻¹ | 公共群级反馈增益 |
| `kf_per_s` | 0.16 s⁻¹ | 相对群级反馈的附加增益；对应相对系数为 0.40 s⁻¹ |
| `D_m2ps` | 2.5 m²/s | 密度反馈系数 |
| `kappa_m2ps` | 10 m²/s | 角色密度偏差图耦合强度 |
| `weights` | `[3/7,1,2/3]` | 角色任务权重，不是机数或平台能力 |
| `role_kernel_length` | 1，无量纲 | 角色坐标中图核的长度参数 |
| `bandwidth_m` | `[1.7,1.5,1.35]` m | 物理空间固定 Gaussian KDE 带宽 |
| `B` | 0–1–2 对称邻接矩阵 | P 的角色密度任务图 |

`sigma_m` 是初始目标标准差；运行时 `shape_axes(t)` 更新尺度。固定物理 KDE 带宽与时变目标标准差是两个不同对象。当前在线实现是高斯目标和固定高斯核的具体实现，不应把论文的一般目标密度理论误读为这份代码已经实现任意目标模板。

配置仍保留 `A_m2ps`、`kernel_length_m=10` 等继承字段。P 的实际在线图反馈由 `B`、`kappa_m2ps`、`role_kernel_length` 及角色尺度构成，不叠加旧的独立 A/B 控制力。`kernel_length_m=10` 不是当前 P 的角色图核宽度，也不是 MMD 的核宽度。

当前激励字段 `test_slowdown_mps=0`、`test_shape_gain_per_s=0`。已有群内形变来自规定的时变模板，不能把旧的零激励分支解释为正在人为施加落后或变形事件。

### 5.2 真实使用的执行参数

三维合成速度上限为固定翼 **15 m/s**、FPV **12 m/s**、四旋翼 **10 m/s**。完整执行参数来自 `R/fixedwing_consistent_predictor_20261030/protocol.json` 和 `predictor.py`，由 `mixed_role_corridor_demo_20261031/engine.py` 重绑为 46 架。

| 对象 | 当前执行模型与参数 |
|---|---|
| 固定翼速度 | 8–15 m/s；速度时间常数 0.4 s；纵向速度变化率上限 2 m/s² |
| 固定翼航迹方位角 | 时间常数 1.0 s；变化率上限 10°/s；方向误差 wrap 至 `[-π,π)` |
| 固定翼航迹倾角 | ±10°；时间常数 0.6 s；变化率上限 5°/s |
| FPV | 向量速度响应时间常数 0.6 s；三维合成加速度上限 1.5 m/s² |
| 四旋翼 | 向量速度响应时间常数 0.8 s；三维合成加速度上限 1.0 m/s² |
| 机间距离阈值 | 2 m；46 架全配对共 1035 对 |
| 显式后备预测窗口 | 8 s；另有无限尾部检查 |

固定翼的 2 m/s² 是**速率大小的变化率界**，不是全部三维加速度界；转向还产生横向加速度。`configuration.json` 中固定翼 `tau_s=.4` 也不代表其三个通道都用 .4 s。

这些是研究执行模型参数；协议明确其不是 Bixler、QAV250 或 X500 的实测参数。固定翼模型没有气动力、推力、机体滚转、风场或失速/载荷因子认证。PX4 移植可以据此设定任务期望和检查响应，但不能把这些值直接认作选定真实机型的参数辨识结果。

此外，旧总配置中的 `candidate_levels=8`、`separation_buffer_m=.6`、`numerical_margin_m=1e-7` 不能作为当前混合机型安全器的完整配置。当前安全器由下节列出的独立协议装配；例如其段认证容差为 `1e-6 m`，数值积分垫量从 `1e-10 m/step` 加余项和任务时长累计项构成，均不是物理抗扰裕量。

## 6. 两个外部方法的准确身份

| 数据/入口别名 | 原实现身份 | 本次有效参数 |
|---|---|---|
| `P` | 群级投影密度反馈，加规定仿射模板输运 | `kc=.24, kf=.16` |
| `RAS04` | Ren–Sorensen 2008 点位速度律适配，内部 `method` 为 `RAS08` | `alpha=.4, coupling=.4` |
| `MMD19` | Arbel 2019 确定性经验测度 MMD 流适配 | `gamma=20` |

`RAS08` 中 08 是文献年份，不是增益 0.8；`RAS04` 是当前 0.4 增益的数据别名。外部 `configuration.json` 继承了旧 P 的 `kc=.12,kf=.08` 字段，但外部控制器不使用 P 反馈；有效参数读取各自 `controller_identity.json.parameters`。不要据此把当前 P 改回旧值。

外部适配在 `R/mixed_sweep_orbit_external_20261101/external_adapters.py`：

- `AffineFormation.command(t,x,v=None)` 使用逐机目标位置、目标速度及个体误差/Laplacian 反馈。`targets.npz` 是冻结的角色材料点；每个初态做一次角色内最小平方距离分配。当前图在群内及角色边 0–1、1–2 之间全连接，46 节点、835 条无向边，Laplacian 按最大特征值 46 归一化。不能删掉个体目标分配后仍称为同一适配。
- `AffineMMD.command(t,x,v=None)` 保留物理 RBF 的 MMD 梯度，并加入相同已知模板输运。每角色核宽度为初始标准差的 RMS，约 `[9.349688,4.453463,4.128357]` m；它不等于 KDE 带宽，也不随目标形变重选。

两个适配均未加入 P 的密度图、均值校正或投影。三种方法使用相同平台和安全接口，包括局部修正与保留后备。外部身份文件中的旧 `ten-seed` scope 是历史登记文字；本次交付实际比较集合是三个种子、九条轨迹。

## 7. 安全预测与 PX4 替换边界

当前执行引擎由以下文件组成：

| 文件，均相对 R | 作用 |
|---|---|
| `fixedwing_consistent_predictor_20261030/predictor.py` 及 `protocol.json` | 混合机型包线、离散状态更新、连续段轨迹、段距离界和尾部界 |
| `mixed_airframe_paper_comparison_20261030/safety_interface.py` 及协议 | 当前段与共同巡航后备认证、证书与推进 |
| `fixedwing_local_repair_20261030/local_interface.py` 及协议 | 局部成员巡航替换 |
| `fixedwing_local_escape_20261031/local_interface.py` 及协议 | 局部脱困候选和保留后备接口 |
| `mixed_role_corridor_demo_20261031/engine.py` | 校验历史源码哈希、重绑角色人数、组装私有模块 |
| `mixed_sweep_orbit_external_20261101/engine.py` | 共同巡航方向随任务切向变化；已接受证书保留自己的后备方向 |
| 同目录 `mission_engine.py` | 按 178 s 累计数值误差额度 |

当前分支包括原映射候选、局部脱困、局部共同巡航和保留后备；历史 `policy_branch` 为 `nominal`、`pair_escape`、`local_cruise`、`retained_backup`。共同后备是沿认证时方向的 **9 m/s 水平巡航**，不是零参考制动。新候选方向变化不改写旧证书的后备方向。

`gov.commit` 在原模型下检查实际执行与预测的一致性。这一性质依赖执行器、预测器、输入和参数一致。将 `advance_model` 换成 Gazebo/PX4 后，真实状态一般不会逐位等于旧预测；直接删掉一致性检查或加宽容差不能恢复原保证。

迁移时可复用的部分是任务参考、形状输运、P/外部名义反馈、成员映射和评价口径。需要重新落实的是：

1. 状态采集与时间同步，保证每架位置和速度使用相同坐标与任务时刻。
2. 名义参考到飞控支持 setpoint 的转换；固定翼速度方向不能自动当作机体姿态或任意可跟踪的三维速度指令。
3. Gazebo/PX4 实际执行对象与安全预测的对应，包括时延、风、加速度和航迹变化能力。
4. 共同巡航后备的可执行性、触发流程和重新认证。原有数值距离裕量不能替代上述模型误差。
5. 记录同步的名义、限幅、映射、最终下发参考与实测状态，以区分反馈收益和执行变化。

若先做单机指令重放，应明确它检查速度执行响应；真正的集群闭环必须使用 PX4/Gazebo 当前实际状态重新调用控制与安全接口，不能将九条历史轨迹重放称为新的闭环验证。

## 8. 数据和复核材料

- `reproduction/data/{P,RAS04,MMD19}_s3300{1,2,3}/`：当前九条轨迹、配置、初态、控制身份、汇总及终端证书。
- `reproduction/reference/figures/`：既有图的绘图数据，可与重算输出比较。
- `reproduction/recomputed/`：已完成的统计复算记录，不是新的仿真。
- `reproduction/reference/audit/`：既有完整评价审计；包内为控制体积未复制全部逐候选事件和证书 ledger。
- `reproduction/manifest.json`、`entrypoint_hashes.json`：复制来源、字节数和哈希。

`run_full.py` 默认只做预检；明确指定 `--execute --work-dir <新目录>` 才跑原 178 s 数值闭环。先复算原统计、再运行基线、最后新增物理引擎适配，有助于判断差异来自移植错误还是执行对象变化。冻结目录的源码和配置保持不变，适配输出另存。

文档中的配置和接口来自本仓库冻结材料的代码审读；本次交付核查通过了文件哈希及三种控制器的初态预检，没有声称已在 Ubuntu 或 PX4/Gazebo 完成验证。
