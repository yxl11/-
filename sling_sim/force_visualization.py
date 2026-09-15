"""Scientific figures exposing each stage of the force-to-formation pipeline."""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


def plot_force_mapping(a, metrics, config, output):
    from .visualization import configure_style, COLORS
    configure_style()
    out, t = Path(output), a["time"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for j, label in enumerate(("x", "y", "z")):
        ax = axes[0, j]
        ax.plot(t, a["force_base"][:, j], color=".5", ls=":", lw=1.2, label="轨迹逆动力学前馈")
        ax.plot(t, a["force_requested"][:, j], color="#9651A4", lw=1.1, label="ESO与预测反馈后的需求")
        ax.plot(t, a["force_smoothed_wrench"][:, j], color="#E88B28", ls="--", lw=1.2, label="几何参考滤波对应合力")
        ax.plot(t, a["force_actual"][:, j], color="#007F9F", lw=1.1, label="动力学实际绳索合力")
        ax.set(xlabel="时间 / s", ylabel=f"合力 F{label} / N", title=f"{label} 方向合力生成与执行")
        if j == 0: ax.legend(fontsize=7.5)
    tension = np.linalg.norm(a["force_allocated_raw"], axis=2)
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(t, tension[:, i], color=COLORS[i], label=f"绳{i+1}期望")
        ax.plot(t, a["tension"][:, i], color=COLORS[i], ls="--", lw=1, alpha=.8)
    ax.set(xlabel="时间 / s", ylabel="张力 / N", title="张力分配：实线期望，虚线实际")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    angles = np.rad2deg(np.arccos(np.clip(a["force_direction_reference"][:, :, 2], -1, 1)))
    for i in range(3): ax.plot(t, angles[:, i], color=COLORS[i], label=f"绳{i+1}")
    ax.set(xlabel="时间 / s", ylabel="绳方向与竖直夹角 / °", title="力方向反算得到的绳角")
    ax.legend(fontsize=8)
    from .model import SlingPlant
    fixed = a["reference_position"][:, None, :]+SlingPlant(config).offsets
    change = np.linalg.norm(a["uav_reference"]-fixed, axis=2)
    ax = axes[1, 2]
    for i in range(3): ax.plot(t, change[:, i]*1000, color=COLORS[i], label=f"无人机{i+1}")
    ax.set(xlabel="时间 / s", ylabel="参考位置变化 / mm", title="逆映射实际改变了编队构型")
    ax.legend(fontsize=8)
    ax = axes[2, 0]
    ax.plot(t, a["eso_force_estimate"][:, 1], color="#9651A4", label="ESO估计")
    ax.plot(t, a["disturbance_force_truth_for_audit"][:, 1], color="#444444", ls="--", label="注入风力与阻尼（仅事后核对）")
    ax.set(xlabel="时间 / s", ylabel="y 方向集总扰动 / N", title="扰动观测进入合力计算")
    ax.legend(fontsize=8)
    ax = axes[2, 1]
    ax.semilogy(t, np.maximum(a["tube_utilization"].max(axis=1), 1e-8), color="#1760AA", label="最大管束利用率")
    bounds = np.asarray(config["force_mapping"]["acceleration_residual_bound"])
    ratio = np.max(np.abs(a["model_acceleration_residual"])/bounds, axis=1)
    ax.semilogy(t, np.maximum(ratio, 1e-8), color="#E88B28", label="采样残差 / 设计扰动界")
    ax.axhline(1, color="#AB2D2D", ls="--", label="设计边界")
    ax.set(xlabel="时间 / s", ylabel="比值（对数轴）", title="管束与模型误差的数值检查")
    ax.legend(fontsize=8)
    ax = axes[2, 2]
    ax.semilogy(t, np.maximum(np.linalg.norm(a["force_allocation_residual"][:, :3], axis=1), 1e-14), color="#1760AA")
    ax.set(xlabel="时间 / s", ylabel="分配合力等式残差 / N", title="优化解满足合力约束")
    wind = config["force_mapping"]
    for ax in axes.flat:
        ax.axvspan(wind["wind_start_s"], wind["wind_start_s"]+wind["wind_duration_s"], color="#E88B28", alpha=.06)
    fig.suptitle("合力前馈预测 → 缆绳力分配 → 编队逆映射 → PDE物理闭环\n"
                 "浅橙区域为平滑侧向扰动；需求、分配与物理响应分别记录", fontsize=15)
    fig.savefig(out/"force_mapping_chain.png", dpi=155)
    fig.savefig(out/"force_mapping_chain.pdf")
    plt.close(fig)

    # Predictions are saved at the actual QP update time, not reconstructed from
    # future measured data. This snapshot is chosen during the known trajectory.
    eligible = np.flatnonzero(a["mpc_updated"] & (t >= 18.))
    idx = eligible[0] if len(eligible) else np.flatnonzero(a["mpc_updated"])[0]
    future_t = t[idx]+np.arange(config["force_mapping"]["horizon"]+1)*config["force_mapping"]["prediction_period"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    from .trajectory import TransportTrajectory
    trajectory = TransportTrajectory(config)
    refs = np.stack([trajectory.at(s).position for s in future_t])
    for axis, ax in enumerate(axes):
        ax.plot(future_t, refs[:, axis], "--", color=".4", label="未来任务参考")
        ax.plot(future_t, a["mpc_prediction"][idx, :, axis, 0], color="#9651A4", label="当时计算的名义位置预测")
        ax.set(xlabel="时间 / s", ylabel=f"{'xyz'[axis]} / m", title=f"{'xyz'[axis]} 方向的20步预测")
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="y", useOffset=False)
    fig.suptitle(f"t={t[idx]:.1f} s 时保存的滚动预测窗口（2 s）", fontsize=14)
    fig.savefig(out/"prediction_window.png", dpi=150)
    fig.savefig(out/"prediction_window.pdf")
    plt.close(fig)
