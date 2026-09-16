# Gazebo 三机协同吊运往返验证

本次在 Ubuntu 20.04 中完成 Gazebo Classic 11 / ODE、ROS Noetic / MAVROS 与三套 PX4 1.13.2 SITL 联合仿真。运输使用原项目冻结的 88 秒三机位置、速度和加速度参考，公共仿真时钟同步执行。包含平滑起吊、稳定、S 形往返、保持与自动降落，三机最后均解除武装。

## 视频与背景

- [88 秒完整往返视频](gazebo_roundtrip.mp4)。
- [含起吊、稳定与降落的原始视频](gazebo_transport.mp4)：1320 帧，1280×720，10 fps，约 132 秒。
- [Gazebo 原生相机截图](camera_snapshot.png)。

采用浅色网格试验场、蓝色 S 形路径、黄色端点与低矮围栏。三根金色吊索清晰可见，其显示随真实负载挂点和无人机位置变化。视频来自 Gazebo 相机 RGB 数据，原始视频已通过 FFmpeg 全片解码检查。

## 运输段实测结果

| 指标 | 测得值 |
|---|---:|
| 运输时间 | 88.000 s |
| 负载位置 RMSE / 最大误差 | 1.19 / 2.79 cm |
| 三机最大位置误差 | 3.35 / 2.86 / 3.20 cm |
| 负载最大倾角 | 0.345° |
| 末段保持最大倾角 | 0.251° |
| 绳索张力范围 | 10.223–10.659 N |
| 最大绳索伸长 | 2.13 mm |
| 三机最小中心间距 | 1.494 m |
| 按机体包络半径 0.48 m 估算的最小净距 | 0.534 m |
| 负载牛顿力平衡独立核验 RMS | 0.00492 N |

全部 12 项预设仿真筛选通过。详细结果、代码及 PX4 二进制哈希见 [gazebo_metrics.json](gazebo_metrics.json)，曲线见 [gazebo_summary.png](gazebo_summary.png)。项目 38 项单元测试通过。

## 物理与定位边界

负载为 3 kg，每架参数化无人机为 2.5 kg，最大总推力 65 N；这些参数需要与真实机型核对。吊索为 1.8 m、仅受拉的无质量弹簧阻尼模型，刚度 5000 N/m、阻尼 35 N·s/m；松绳不传递力，拉力以等大反向作用于无人机与负载。ODE 积分步长 1 ms，未通过设置位姿强制拖动负载。模型不包含分段绳索接触、自碰撞或气动下洗。

各无人机真实位姿作为理想外部定位，经 MAVROS vision_pose 输入 PX4 EKF2；没有模拟真实定位噪声与延迟。控制器只接收本机定位估计，负载真值与绳索张力用于事后评价。一次理想定位仿真通过不代表真实环境中具有相同精度。

## 三机轨迹交付

- [uav1 ENU 50 Hz](hardware_candidate/uav1_enu_50hz.csv)
- [uav2 ENU 50 Hz](hardware_candidate/uav2_enu_50hz.csv)
- [uav3 ENU 50 Hz](hardware_candidate/uav3_enu_50hz.csv)
- [坐标、同步、单位与哈希清单](hardware_candidate/manifest.json)

每架 4401 个时间点，覆盖 0–88 s，提供位置、速度、加速度及 yaw。它们是经过上述物理仿真筛选的实机候选运输参考，不包含实机起吊与降落。实际机型、推力余量、负载与挂点、绳长、定位误差和三机局部坐标尚未标定，因此清单明确保留 `hardware_calibrated=false`、`flight_ready=false`。

项目提供各机局部 ENU 坐标变换工具及两个测试。`local_frame_example/` 仅是零变换格式示例。真实标定与重跑步骤见 [运行说明](../../docs/GAZEBO_ROS1_PX4.md)。本次运行使用独立 PX4 运行目录，未修改原 PX4 源码或原 Python 正式结果。
