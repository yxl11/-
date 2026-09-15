"""Research figures and an annotated movie from saved planning/replay logs."""
from pathlib import Path
import subprocess
import numpy as np
from matplotlib import pyplot as plt, animation
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .model import SlingPlant
from .visualization import configure_style, unpack_log, box_vertices, FACES, COLORS


def save_figure(fig, output, name):
    fig.savefig(Path(output)/(name+".png"), dpi=155)
    fig.savefig(Path(output)/(name+".pdf"))
    plt.close(fig)


def plot_experiment(planning, replay, config, output):
    configure_style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    t = replay["time"]
    pos, rotation = unpack_log(replay)
    plant = SlingPlant(config)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), constrained_layout=True)
    ax = axs[0, 0]
    ax.plot(planning["reference_position"][:, 0], planning["reference_position"][:, 1], "--", color="0.55", label="负载期望路径")
    ax.plot(pos[:, 0, 0], pos[:, 0, 1], color="#1760AA", label="回放物理响应")
    for i in range(3):
        ax.plot(pos[:, i+1, 0], pos[:, i+1, 1], color=COLORS[i], lw=.8, alpha=.5)
    ax.set(xlabel="x / m", ylabel="y / m", title="冻结轨迹后的独立吊载运动")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    for i in range(3):
        ax.plot(t, np.linalg.norm(replay["uav_reference_velocity"][:, i], axis=1), color=COLORS[i], label=f"无人机{i+1}")
    ax.set(xlabel="时间 / s", ylabel="参考速度 / (m/s)", title="回放三机各自的速度参考")
    ax.legend(fontsize=8)
    ax = axs[0, 2]
    delta = planning["uav_reference"]-planning["reference_position"][:, None]-plant.offsets
    for i in range(3):
        ax.plot(planning["time"], np.linalg.norm(delta[:, i], axis=1)*1000, color=COLORS[i], label=f"无人机{i+1}")
    ax.set(xlabel="时间 / s", ylabel="相对固定构型的参考变化 / mm", title="离线逆映射已写入轨迹文件")
    ax.legend(fontsize=8)
    ax = axs[1, 0]
    ax.plot(t, np.linalg.norm(pos[:, 0]-replay["reference_position"], axis=1)*100, color="#1760AA", label="负载")
    for i in range(3):
        ax.plot(t, np.linalg.norm(pos[:, i+1]-replay["uav_reference"][:, i], axis=1)*100, color=COLORS[i], lw=.9, label=f"无人机{i+1}")
    ax.set(xlabel="时间 / s", ylabel="位置误差 / cm", title="理想回放的实际跟踪误差")
    ax.legend(fontsize=8)
    ax = axs[1, 1]
    for i in range(3):
        ax.plot(t, replay["tension"][:, i], color=COLORS[i], label=f"绳索{i+1}")
    ax.set(xlabel="时间 / s", ylabel="张力 / N", title="物理模型求得的张力")
    ax.legend(fontsize=8)
    ax = axs[1, 2]
    tilt = np.rad2deg(np.arccos(np.clip(rotation[:, 0, 2, 2], -1, 1)))
    ax.plot(t, tilt, color="#1760AA")
    ax.set(xlabel="时间 / s", ylabel="负载倾角 / °", title="回放时未使用负载姿态反馈")
    for ax in axs.flat[1:]:
        for phase in (8, 40, 48, 80):
            ax.axvline(phase, color="0.65", lw=.6, alpha=.5)
    fig.suptitle("88秒三机吊载轨迹复现  |  离线合力预测 / 逆映射 / PDE → 冻结p/v/a → 本机PID执行\n"
                 "3 kg 负载 · 纯数值仿真 · 尚未接入 ROS/PX4", fontsize=14)
    save_figure(fig, output, "replay_overview")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6.3), constrained_layout=True)
    ax = axs[0]
    ax.add_patch(Rectangle((-10, -5), 20, 10, fill=False, ec="#203F55", lw=1.4))
    ax.plot(planning["reference_position"][:, 0], planning["reference_position"][:, 1], color="#1760AA", lw=2, label="负载 S 形路径")
    for i in range(3):
        ax.plot(planning["uav_reference"][:, i, 0], planning["uav_reference"][:, i, 1], color=COLORS[i], lw=.9, alpha=.7)
    for time, label, shift in ((8, "起点 / 最终保持", (-2, 1.9)), (24, "中点通过", (-1, -2.2)), (40, "远端保持", (3.3, 1.9))):
        n = np.argmin(abs(planning["time"]-time))
        p = planning["reference_position"][n]
        ax.scatter(*p[:2], color="#1760AA", s=35)
        ax.annotate(label, p[:2], xytext=shift, arrowprops=dict(arrowstyle="-", color="0.5"), fontsize=9)
    ax.scatter([0], [-4.6], marker="^", s=95, color="#D46B27", label="固定主机位建议")
    ax.annotate("朝向 +y，安装高约 2 m", (0, -4.6), xytext=(1, -4.6), fontsize=9)
    ax.set(xlim=(-10.7, 10.7), ylim=(-5.6, 5.6), xlabel="x / m", ylabel="y / m", title="20 × 10 m 场地内的轨迹与固定机位")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax = axs[1]
    phases = [(0, 8, "悬停"), (8, 32, "S形去程"), (40, 8, "远端定点"), (48, 32, "S形返回"), (80, 8, "末段悬停")]
    for i, (start, length, name) in enumerate(phases):
        ax.broken_barh([(start, length)], (3.2, .7), facecolors=["#DDEAF0", "#69AEC0", "#DDEAF0", "#A39DC6", "#DDEAF0"][i])
        ax.text(start+length/2, 3.55, name, ha="center", va="center", fontsize=8, rotation=90 if length == 8 else 0)
    shoots = [(0, 88, 2.25, "固定全景：完整任务"), (16, 16, 1.45, "去程转弯：合力与构型"),
              (40, 8, .65, "定点：张力和负载姿态"), (58, 16, -.15, "返回：反向机动")]
    for start, length, row, name in shoots:
        ax.broken_barh([(start, length)], (row, .45), facecolors="#E6BA8B")
        ax.text(start+.7, row+.22, name, va="center", fontsize=9)
    ax.set(xlim=(0, 88), ylim=(-.4, 4.4), xlabel="统一任务时间 / s", yticks=[], title="实机拍摄分镜与统一时标")
    fig.suptitle("实验轨迹定稿与拍摄安排  |  示意机位需按现场视野复核", fontsize=14)
    save_figure(fig, output, "trajectory_storyboard")


def plot_cases(metrics, output):
    configure_style()
    labels = [m["scenario"] for m in metrics]
    rows = np.arange(len(labels))
    colors = ["#217A68" if m["engineering_screen_passed"] else "#BA6D2D" for m in metrics]
    fig, axs = plt.subplots(1, 3, figsize=(16, max(6, len(labels)*.34)), constrained_layout=True, sharey=True)
    for ax, key, scale, title, bound in zip(axs,
            ("payload_position_peak_m", "payload_tilt_peak_deg", "final_hold_payload_tilt_peak_deg"),
            (100, 1, 1), ("负载最大位置误差 / cm", "全过程最大倾角 / °", "末段保持最大倾角 / °"), (15, 5, 1)):
        values = [m[key]*scale for m in metrics]
        ax.barh(rows, values, color=colors, alpha=.85)
        ax.axvline(bound, color="#AA3939", ls="--", lw=1, label="预设检查值")
        for row, value in zip(rows, values):
            ax.text(value+.015*bound, row, f"{value:.2f}", va="center", fontsize=8)
        ax.set(xlabel=title, xlim=(0, max(bound*1.17, max(values)*1.25)))
        ax.legend(fontsize=8, loc="lower right")
    axs[0].set(yticks=rows, yticklabels=labels)
    axs[0].invert_yaxis()
    fig.suptitle("同一冻结轨迹的误差与失配检查（不是算法对照）\n绿色：全部14项通过；棕色：存在未达标项，完整结果保留", fontsize=13)
    save_figure(fig, output, "replay_case_checks")


def animate_experiment(planning, replay, config, output, speedup=4, fps=10):
    configure_style()
    output = Path(output)
    plant = SlingPlant(config)
    pos, rotation = unpack_log(replay)
    fig = plt.figure(figsize=(14.4, 8.4))
    grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.15], left=.035, right=.97,
                          bottom=.16, top=.87, wspace=.3, hspace=.65)
    ax = fig.add_subplot(grid[:, :2], projection="3d")
    ax.set(xlim=(-4.2, 4.4), ylim=(-2.3, 2.3), zlim=(0, 3.5), xlabel="x / m", ylabel="y / m", zlabel="z / m")
    ax.set_box_aspect((8.6, 4.6, 3.5))
    ax.view_init(elev=25, azim=-68)
    ax.plot(*planning["reference_position"].T, "--", color=".7", lw=1, label="负载期望路径")
    trace, = ax.plot([], [], [], color="#1760AA", lw=1.4, label="回放物理响应")
    ropes, arms, dots, labels = [], [], [], []
    for i, color in enumerate(COLORS):
        ropes.append(ax.plot([], [], [], color=color, lw=1.8)[0])
        arms.append([ax.plot([], [], [], color=color, lw=2.4)[0] for _ in range(2)])
        dots.append(ax.plot([], [], [], ls="", marker="o", color=color, markersize=4.5)[0])
        labels.append(ax.text(0, 0, 0, f"无人机{i+1}", color=color, fontsize=8))
    poly = Poly3DCollection([], facecolors="#5686B1", edgecolors="#23425D", alpha=.85)
    ax.add_collection3d(poly)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("物理回放：执行层只接收三机轨迹和本机状态", fontsize=11, pad=0)
    vertex = box_vertices(config["payload_dimensions"])

    forecast_ax = fig.add_subplot(grid[0, 2])
    tau = np.arange(config["force_mapping"]["horizon"])*config["force_mapping"]["prediction_period"]
    forecast_lines = [forecast_ax.plot(tau, np.zeros_like(tau), color=color, label=label)[0]
                      for color, label in (("#D06E2D", "Fx"), ("#1760AA", "Fy"))]
    forecast_ax.set(xlim=(0, 2), ylim=(-.6, .6), xlabel="预测时域 τ / s", ylabel="名义合力 / N", title="离线规划时保存的2秒预测", yticks=[-.5, 0, .5])
    forecast_ax.legend(fontsize=8, loc="upper right", ncols=2)
    geometry_ax = fig.add_subplot(grid[1, 2])
    geometry_ax.scatter(*plant.offsets[:, :2].T, marker="x", color=".6", s=25)
    geometry_ax.scatter(*plant.anchors[:, :2].T, marker="s", color=".6", s=10)
    geometry_ax.set(xlim=(-.9, 1.2), ylim=(-1.1, 1.1), xlabel="相对负载 x / m", ylabel="y / m",
                    title="逆映射构型：位移箭头放大8倍")
    geometry_ax.set_aspect("equal", adjustable="box")
    shape_points, shape_arrows = [], []
    for i, color in enumerate(COLORS):
        shape_points.append(geometry_ax.plot([], [], "o", color=color, markersize=4)[0])
        arrow = FancyArrowPatch(plant.offsets[i, :2], plant.offsets[i, :2], arrowstyle="-|>", mutation_scale=11, color=color, lw=1.2)
        geometry_ax.add_patch(arrow)
        shape_arrows.append(arrow)
    tension_ax = fig.add_subplot(grid[2, 2])
    for i in range(3):
        tension_ax.plot(replay["time"], replay["tension"][:, i], color=COLORS[i], lw=.9, label=f"T{i+1}")
    cursor = tension_ax.axvline(0, color=".3", lw=1)
    tension_ax.set(xlim=(0, replay["time"][-1]), xlabel="回放时间 / s", ylabel="实际张力 / N", title="独立动力学求出的绳索张力")
    tension_ax.legend(fontsize=8, ncols=3, loc="upper right")
    title = fig.suptitle("", fontsize=15, y=.96)
    caption = fig.text(.045, .055, "", fontsize=10, linespacing=1.6)
    planning_updates = np.flatnonzero(planning["mpc_updated"])
    indices = np.unique(np.r_[np.searchsorted(replay["time"], np.arange(0, replay["time"][-1], speedup/fps)), len(replay["time"])-1])
    force_arrow = [None]

    def update(frame):
        n = indices[frame]
        t = replay["time"][n]
        j = min(np.searchsorted(planning["time"], t), len(planning["time"])-1)
        point = vertex@rotation[n, 0].T+pos[n, 0]
        poly.set_verts([point[face] for face in FACES])
        attachment = plant.anchors@rotation[n, 0].T+pos[n, 0]
        trace.set_data_3d(*pos[:n+1, 0].T)
        for i in range(3):
            ropes[i].set_data_3d(*np.vstack((attachment[i], pos[n, i+1])).T)
            rotor = plant.rotor_points@rotation[n, i+1].T+pos[n, i+1]
            for k, ends in enumerate(((0, 2), (1, 3))):
                arms[i][k].set_data_3d(*rotor[list(ends)].T)
            dots[i].set_data_3d(*rotor.T)
            labels[i].set_position(pos[n, i+1, :2])
            labels[i].set_3d_properties(pos[n, i+1, 2]+.18, zdir=None)
            mapped = planning["uav_reference"][j, i]-planning["reference_position"][j]
            shape_points[i].set_data([mapped[0]], [mapped[1]])
            shape_arrows[i].set_positions(plant.offsets[i, :2], plant.offsets[i, :2]+8*(mapped-plant.offsets[i])[:2])
        qidx = planning_updates[max(0, np.searchsorted(planning["time"][planning_updates], t, side="right")-1)]
        predicted_force = config["payload_mass"]*planning["mpc_acceleration_prediction"][qidx]-planning["eso_force_estimate"][qidx]
        for i in range(2):
            forecast_lines[i].set_ydata(predicted_force[:, i])
        if force_arrow[0] is not None:
            force_arrow[0].remove()
        actual = replay["force_actual"][n]
        force_arrow[0] = ax.quiver(*pos[n, 0], actual[0], actual[1], 0, color="#D06E2D", arrow_length_ratio=.25, linewidth=2)
        cursor.set_xdata([t, t])
        err = np.linalg.norm(pos[n, 0]-replay["reference_position"][n])*1000
        delta = np.linalg.norm(planning["uav_reference"][j]-planning["reference_position"][j]-plant.offsets, axis=1).max()*1000
        phase = "起点悬停" if t < 8 else "S形去程" if t < 40 else "远端定点" if t < 48 else "S形返回" if t < 80 else "末段悬停"
        title.set_text(f"三机协同吊运 · 离线规划与独立轨迹回放  |  {phase}  |  t = {t:.1f} s  |  {speedup}倍速")
        caption.set_text(f"回放负载误差 {err:.1f} mm    逆映射参考变化 {delta:.1f} mm    实际张力 "
            + " / ".join(f"{value:.2f}" for value in replay["tension"][n])+" N\n"
            + f"橙色箭头为实际水平合力：1 N = 1 m；实际 Fz = {actual[2]:.2f} N。右侧预测和构型来自离线规划记录。\n"
            + "数值仿真 · 回放期间无在线负载反馈 / ESO / MPC / PDE · 尚未验证 ROS/PX4 或实机")
        return []

    movie = animation.FuncAnimation(fig, update, frames=len(indices), interval=1000/fps, blit=False)
    movie.save(output/"experiment_replay.gif", writer=animation.PillowWriter(fps=fps), dpi=85)
    update(int(np.argmin(abs(replay["time"][indices]-21.2))))
    fig.savefig(output/"experiment_preview.png", dpi=145)
    plt.close(fig)
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(output/"experiment_replay.gif"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output/"experiment_replay.mp4")], check=True)
