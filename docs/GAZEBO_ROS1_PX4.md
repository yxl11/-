# 三机吊运 Gazebo / ROS 1 / PX4 运行说明

此接口沿用 Ubuntu 20.04、ROS Noetic、Gazebo Classic 11 和 PX4 1.13.2。
Python 研究控制器继续在独立 Python 3.12 环境运行，ROS 节点使用系统 Python 3.8。

## 架构与物理模型

冻结三机 p/v/a → ROS 50 Hz 同步播放器 → 三套 MAVROS → 三套真实 PX4 SITL
位置/速度/姿态控制器 → Gazebo 旋翼插件 → ODE 刚体与吊载。

`gazebo_sling/src/sling_cables.cpp` 在每个 1 ms 物理步计算三根绳的力：

`T = max(0, k (distance − length) + c radial_speed)`，仅在 `distance > length` 时施力。

力作用在负载真实挂点和无人机质心，大小相等、方向相反。负载挂点施力
自然产生负载力矩。松绳时不传力，既不把绳当作受压杆，也不直接设置位置
或速度。绳线图形只显示当前几何，物理效果来自独立施力。

这是无质量、单段直线、仅受拉的弹性阻尼近似，默认 k=5000 N/m、c=35 N·s/m；
约 10 N 张力产生约 2 mm 伸长。没有分段绳自重、绳间碰撞、下洗模型，绳参数
尚未实测。与原 Python 的理想不可伸长约束模型不同。负载与无人机具有
真实 Gazebo 碰撞几何，起吊与接地由 ODE 处理。

无人机模型从本机 Iris SDF 模板生成，修改质量、惯量和旋翼系数以匹配
本项目 2.5 kg 单机、65 N 总推力的假设。**这不是原装 Iris 的承载参数。**
使用本机已有传感器/旋翼/MAVLink插件和 PX4 二进制，不修改原 PX4 目录。
运行时 ROMFS、参数、ULog、ROS 日志保存在项目指定输出目录。

## 定位与坐标

`localization.py` 是独立的理想无人机定位传感器：只读取每架无人机自身
Gazebo 位姿，以 50 Hz 通过 MAVROS `vision_pose/pose` 输入 PX4 EKF2。
它近似共同坐标雷达/LIO 输出，不提供负载状态、绳张力或规划器输出。
当前没有模拟雷达点云、噪声、遮挡或重定位。

ROS 消息字段使用共同地图 ENU/FLU，世界 z 向上，yaw=0 指向地图 x。
`PositionTarget.coordinate_frame=FRAME_LOCAL_NED` 是 MAVLink 目标帧标识；
ROS 字段仍为 ENU，由 MAVROS 执行一次转换。不要预先把 CSV 换成 NED 后
再送入该接口。EKF2 的视觉定位采用共同绝对地图坐标，因此模拟坐标偏移
为零；实机必须另行提供每架飞控原点和航向的标定变换。

## 编译与运行

在项目根目录、已加载 ROS Noetic 的终端执行：

```bash
cmake -S gazebo_sling -B build/gazebo_sling \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/ros/noetic \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
cmake --build build/gazebo_sling -j2
/usr/bin/python3 gazebo_sling/prepare.py --output results/gazebo_run_new
/usr/bin/python3 gazebo_sling/run.py --gui --root results/gazebo_run_new
MPLCONFIGDIR=/tmp/sling-matplotlib .venv/bin/python gazebo_sling/analyze.py \
  --root results/gazebo_run_new
```

每次使用新的输出目录，避免覆盖参数、录制视频和日志。`prepare.py --px4`
可指定其他 PX4 目录；模板和插件应为兼容的 PX4/Gazebo Classic 版本。
FFmpeg 路径可通过 `run.py --ffmpeg` 指定。

ROS master 使用端口 11321，Gazebo master 使用 11351；三机 PX4 仿真 TCP
端口为 4560/4561/4562，MAVROS 为 14540/14541/14542，PX4 Offboard 发端为
14580/14581/14582。不要同时运行占用这些端口的其他 SITL。

自动流程包括连接与定位等待、预发送、三机 Offboard 解锁、16 秒平滑
起吊、12 秒稳定、88 秒往返、4 秒保持和 PX4 自动降落。三机均就绪后才
开始运输，使用公共仿真时间。中途失去 Offboard 或定位陈旧会结束任务。
这是仿真联调流程，实机的起吊、降落与故障协同需要专门验证。

## 输出

实际任务速度试验可使用时间缩放工具，位置路径保持不变，速度乘缩放系数，
加速度乘系数平方，再按 50 Hz 重采样。以 2 倍速度为例：

```bash
.venv/bin/python gazebo_sling/scale_trajectory.py --source results/experiment_v1/trajectory/trajectory.npz --output results/gazebo_transport_2x_reference/trajectory.npz --factor 2
/usr/bin/python3 gazebo_sling/prepare.py --output results/gazebo_transport_2x_verified --speed-factor 2 --trajectory results/gazebo_transport_2x_reference/trajectory.npz
/usr/bin/python3 gazebo_sling/run.py --gui --root results/gazebo_transport_2x_verified
MPLCONFIGDIR=/tmp/sling-matplotlib .venv/bin/python gazebo_sling/analyze.py --root results/gazebo_transport_2x_verified
```

先采用相同编队位置路径做动力学试验，不重新求解加速后的合力逆映射。
运输与其内部保持缩短为 44 秒，起吊和定位稳定时长保留。准备器同步调整
运输参考时间与水平速度限制；面板时间轴和分析末段区间随参考自动变化。
物理步长与控制发送频率保持不变，视频不是后期倍速播放。

最新数据面板版位于 `results/gazebo_transport_dashboard/`，见 [往返视频](../results/gazebo_transport_dashboard/gazebo_roundtrip.mp4) 与 [数据面板说明](../results/gazebo_transport_dashboard/实时数据面板说明.md)。创建新的独立输出目录即可复现新版渲染与叠加，不会覆盖之前的录制。

- `gazebo_transport.mp4`：Gazebo 原生相机 RGB 图像编码的 10 fps 视频；
  帧率采用仿真时间，不用 Matplotlib 重建画面。
  新版保留完整场景，删除放大特写，右侧同步显示三绳张力、负载位置误差、
  负载倾角的累计曲线；底部显示实时数值、XYZ 误差分量、三机状态。
  保留任务阶段、公共仿真时间、运输计时、三机编号及运输进度条。
  编号位置由已知相机参数投影 Gazebo 真值得到，张力来自绳索遥测，仅用于
  展示。分辨率仍为 1280×720，10 fps，原编码参数不变。
- `gazebo_states.csv`：Gazebo 无人机与负载的真实位姿/速度，仅供评价。
- `cables.csv`：各绳张力、实际长度和相对径向速度。
- `commands.csv`、`estimates.csv`：发送的参考及 PX4 定位估计、模式、解锁状态。
- `px4_*/log/`：三套 PX4 ULog；`*.log`：进程诊断。
- `execution.json`、`gazebo_metrics.json`：运行状态与工程筛选。
- `gazebo_summary.png/pdf`：物理轨迹、负载误差、张力及倾角。
- `hardware_candidate/uav1_enu_50hz.csv` 等三文件：共同时间轴的运输参考，
  每架 4401 行数据，位置/速度/加速度与原研究轨迹一致。
- `hardware_candidate/manifest.json`：坐标、单位、同步、物理假设、数据哈希
  和当前仿真筛选状态。它明确记录 `hardware_calibrated=false`、
  `flight_ready=false`，不会把 Gazebo 通过等同于实机验证。

## 实机使用前需要补齐

已验证结果位于 `results/gazebo_transport_final/`：[88 秒往返视频](../results/gazebo_transport_final/gazebo_roundtrip.mp4)、[包含起吊与降落的原始视频](../results/gazebo_transport_final/gazebo_transport.mp4)、[验证报告](../results/gazebo_transport_final/Gazebo物理仿真验证报告.md)。背景为浅色网格试验场、蓝色 S 路径、黄色端点和低矮围栏。三根吊索在负载模型下预创建，插件持续按真实挂点更新可视圆柱的位置、方向和长度，原生相机已确认可见。

各机局部坐标导出器使用 `p_map = Rz(yaw_map_from_local) p_local_ENU + origin_map_ENU` 标定关系，位置、速度、加速度与 yaw 一起变换。复制并填写 `configs/hardware_frames.example.json`，完成真实标定后才将 `hardware_calibrated` 设为 true，再运行：

```bash
.venv/bin/python gazebo_sling/export_local_frames.py --candidate results/gazebo_transport_final/hardware_candidate --frames configs/hardware_frames.calibrated.json --output results/hardware_local_frames
```

`local_frame_example/` 是通过 `--allow-example` 导出的零变换示例，仅用于检查文件格式，不代表实际三机坐标。导出器不会连接飞控或解锁，输出仍记录 `flight_ready=false`。

确认实际机型与满配质量、持续推力/载重、负载含吊具总质量、挂点和绳长；
对实际定位坐标与各 PX4 局部帧、yaw、时间戳进行标定。测量位置控制带宽、
定位误差与延迟后，在对应参数下重跑；使用真实飞控参数验证三机同步和
协同退出。当前候选 CSV 只覆盖悬空运输，不包含实机起吊/降落。

参考接口依据：[PX4 1.13 Offboard](https://docs.px4.io/v1.13/en/flight_modes/offboard)、
[Gazebo WorldPlugin](https://classic.gazebosim.org/tutorials?cat=write_plugin&tut=plugins_world)。
ENU/NED 转换另外核对了本机 MAVROS `setpoint_raw.cpp` 的 `local_cb` 实现。
