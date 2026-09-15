"""Figures and animation are generated only from saved simulated states."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager, animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .rotations import quaternion_matrix
from .model import SlingPlant

COLORS = ["#007F9F", "#E88B28", "#9651A4"]


def configure_style():
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.titleweight": "bold"})


def unpack_log(a):
    y = a["state"]
    return y[:, :12].reshape(-1,4,3), quaternion_matrix(y[:,24:40].reshape(-1,4,4))


def plot_run(a, metrics, config, output):
    configure_style()
    out = Path(output)
    pos, rotation = unpack_log(a)
    t = a["time"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    ax = axes[0,0]
    ax.plot(a["reference_position"][:,0], a["reference_position"][:,1], "--", color="#444444", label="负载参考")
    ax.plot(pos[:,0,0], pos[:,0,1], color="#1760AA", label="负载响应", linewidth=1.6)
    ax.set(xlabel="x / m", ylabel="y / m", title="负载水平轨迹")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9)
    error = np.linalg.norm(pos[:,0]-a["reference_position"], axis=1)
    ax = axes[0,1]
    ax.plot(t, error, color="#1760AA", label="负载")
    for i in range(3):
        ax.plot(t, np.linalg.norm(pos[:,i+1]-a["uav_reference"][:,i], axis=1),
                color=COLORS[i], alpha=.8, linewidth=1, label=f"无人机{i+1}")
    ax.set(xlabel="时间 / s", ylabel="位置误差范数 / m", title="参考与物理响应的差异")
    ax.legend(fontsize=8)
    ax = axes[0,2]
    for i in range(3):
        ax.plot(t, a["tension"][:,i], color=COLORS[i], label=f"绳索{i+1}")
    ax.set(xlabel="时间 / s", ylabel="张力 / N", title="动力学求解的绳索张力")
    ax.legend(fontsize=8)
    ax = axes[1,0]
    tilt = np.rad2deg(np.arccos(np.clip(rotation[:,0,2,2], -1, 1)))
    ax.plot(t, tilt, color="#1760AA", label="负载倾角")
    directions = SlingPlant(config).rest_directions
    cable = np.rad2deg(np.arccos(np.clip(np.sum(a["cable_direction"]*directions, axis=2), -1, 1)))
    ax.plot(t, cable.max(axis=1), color="#E88B28", label="最大绳方向偏差")
    ax.set(xlabel="时间 / s", ylabel="角度 / °", title="姿态与绳方向变化")
    ax.legend(fontsize=8)
    ax = axes[1,1]
    rotor = a["state"][:,52:64].reshape(-1,3,4)
    for i in range(3):
        ax.plot(t, rotor[:,i].max(axis=1)/config["max_rotor_thrust"]*100,
                color=COLORS[i], label=f"无人机{i+1}")
    ax.axhline(100, color="#AB2D2D", linestyle="--", linewidth=1)
    ax.set(xlabel="时间 / s", ylabel="最大单电机推力 / 上限 %", title="执行器负荷")
    ax.legend(fontsize=8)
    ax = axes[1,2]
    ax.plot(t, np.max(np.abs(a["length_error"]), axis=1)*1e6, color="#1760AA")
    ax.set(xlabel="时间 / s", ylabel="最大绳长残差 / μm", title="数值约束残差")
    for ax in axes.flat:
        if ax is not axes[0,0]:
            for x in [config["hold_duration"], config["hold_duration"]+config["leg_duration"],
                      2*config["hold_duration"]+config["leg_duration"],
                      2*(config["hold_duration"]+config["leg_duration"])]:
                if x < t[-1]: ax.axvline(x, color="0.6", linewidth=.6, alpha=.5)
    method = "合力预测与逆映射PDE控制" if "force_mapping" in metrics else ("PDE 反步边界控制" if "pde" in metrics else "基础控制")
    fig.suptitle(f"三机吊载{method}数值仿真  |  {metrics['scenario']}\n"
        f"负载 {config['payload_mass']:g} kg · 每机假设 {config['uav_mass']:g} kg · "
        f"负载位置 RMSE {metrics['payload_position_rmse_m']:.4f} m", fontsize=15)
    fig.savefig(out/"simulation_summary.png", dpi=160)
    fig.savefig(out/"simulation_summary.pdf")
    plt.close(fig)
    if "pde" in metrics:
        plot_pde(a, metrics, config, out)
    if "force_mapping" in metrics:
        from .force_visualization import plot_force_mapping
        plot_force_mapping(a, metrics, config, out)


def plot_pde(a, metrics, config, output):
    from .pde import PDEFormation, backstepping_kernel
    pde = PDEFormation(config["pde"])
    t = a["time"]
    error = np.linalg.norm(a["pde_error"], axis=2)
    roles = ["1号：锚点", "2号：跟随者", "3号：领导者"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), constrained_layout=True)
    ax = axes[0, 0]
    for i in range(3):
        ax.plot(t, error[:, i]*100, color=COLORS[i], label=roles[i])
    ax.set(xlabel="时间 / s", ylabel="节点位置误差 / cm",
           title="逆映射构型的PDE跟踪" if "force_mapping" in metrics else "初始偏差下的编队收敛",
           xlim=(0, t[-1] if "force_mapping" in metrics else min(12, t[-1])))
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.grid(False)
    mesh = ax.pcolormesh(t, np.arange(3), error.T*100, shading="nearest", cmap="Blues")
    ax.set(yticks=np.arange(3), yticklabels=["锚点  ξ=0", "跟随者  ξ=0.5", "领导者  ξ=1"],
           xlabel="时间 / s", title="三个实际节点的误差演化")
    fig.colorbar(mesh, ax=ax, label="误差 / cm", shrink=.85)
    ax = axes[0, 2]
    xi = np.linspace(0, pde.c["alpha"], 300)
    density = backstepping_kernel(pde.c["alpha"], xi, pde.sigma)*np.exp(pde.c["b"]*(xi-pde.c["alpha"]))
    ax.plot(xi, density, color="#1760AA", label="连续积分核（含对流因子）")
    ax.scatter(pde.x[:-1], pde.weights/pde.h, color="#E88B28", s=45, label="左矩形求积采样点", zorder=3)
    ax.set(xlabel="拓扑坐标 ξ", ylabel="核函数值", title="实际参与控制的边界核")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    l2 = np.sqrt(np.sum(error**2*np.array([.5, 1, .5])*pde.h, axis=1))
    ax.semilogy(t, np.maximum(l2, 1e-12), color="#1760AA", label="编队误差离散 L² 范数")
    ax.semilogy(t, np.maximum(np.linalg.norm(a["pde_boundary_residual"], axis=1), 1e-12),
                color="#9651A4", label="领导者边界残差")
    ax.set(xlabel="时间 / s", ylabel="误差 / m（对数轴）", title="完整运输过程的误差")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(t, np.linalg.norm(a["pde_interior_equation_defect"], axis=1), color="#E88B28")
    ax.set(xlabel="时间 / s", ylabel="速度残差 / (m/s)", title="物理跟随者与理想 PDE 的差异")
    ax = axes[1, 2]
    for i in range(3):
        ax.plot(t, np.linalg.norm(a["pde_velocity_command"][:, i], axis=1), color=COLORS[i], label=roles[i])
    ax.axhline(config["pde"]["velocity_limit"], color="#AB2D2D", ls="--", lw=1, label="指令限幅")
    ax.set(xlabel="时间 / s", ylabel="速度指令 / (m/s)", title="PDE 输出的三机速度指令")
    ax.legend(fontsize=8, ncols=2)
    fig.suptitle("申请书 PDE 编队与反步边界控制的三节点实现\n"
                 "空间域表示通信拓扑；误差来自物理仿真状态，图中没有额外虚拟无人机", fontsize=14)
    fig.savefig(Path(output)/"pde_evolution.png", dpi=160)
    fig.savefig(Path(output)/"pde_evolution.pdf")
    plt.close(fig)


def box_vertices(dimensions):
    half = np.asarray(dimensions)/2
    return np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*half


FACES = [[0,1,3,2], [4,5,7,6], [0,1,5,4], [2,3,7,6], [0,2,6,4], [1,3,7,5]]


def setup_scene(a, config, figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlim=(-4.2,4.4), ylim=(-2.3,2.3), zlim=(0,3.5),
           xlabel="x / m", ylabel="y / m", zlabel="z / m")
    ax.set_box_aspect((8.6,4.6,3.5))
    ax.view_init(elev=24, azim=-65)
    ax.plot(a["reference_position"][:,0], a["reference_position"][:,1],
            a["reference_position"][:,2], "--", color="#AAAAAA", label="负载参考", linewidth=1)
    ax.plot([-4,4], [-2,-2], [0,0], color="0.8", linewidth=.8)
    ax.plot([-4,-4], [-2,2], [0,0], color="0.8", linewidth=.8)
    fig.subplots_adjust(left=0, right=.98, bottom=.03, top=.91)
    return fig, ax


def draw_snapshot(ax, position, rotation, plant, alpha=1.0):
    points = box_vertices(plant.c["payload_dimensions"]) @ rotation[0].T + position[0]
    poly = Poly3DCollection([points[f] for f in FACES], facecolors="#5686B1", edgecolors="#23425D", alpha=.65*alpha)
    ax.add_collection3d(poly)
    attachments = plant.anchors @ rotation[0].T + position[0]
    for i in range(3):
        ax.plot(*np.vstack((attachments[i],position[i+1])).T, color=COLORS[i], linewidth=1.5, alpha=alpha)
        rotors = plant.rotor_points @ rotation[i+1].T + position[i+1]
        for ends in ((0,2),(1,3)):
            ax.plot(*rotors[list(ends)].T, color=COLORS[i], linewidth=2, alpha=alpha)
        ax.scatter(*rotors.T, color=COLORS[i], s=22, alpha=alpha)


def animate_run(a, metrics, config, output, speedup=8, fps=15):
    configure_style()
    plant = SlingPlant(config)
    pos, rotation = unpack_log(a)
    fig, ax = setup_scene(a, config)
    trace, = ax.plot([],[],[], color="#1760AA", linewidth=1.8, label="负载响应")
    ropes, arms, dots = [], [], []
    for color in COLORS:
        ropes.append(ax.plot([],[],[], color=color, linewidth=1.8)[0])
        arms.append([ax.plot([],[],[], color=color, linewidth=2.5)[0] for _ in range(2)])
        dots.append(ax.plot([],[],[], linestyle="", marker="o", color=color, markersize=5)[0])
    poly = Poly3DCollection([], facecolors="#5686B1", edgecolors="#23425D", alpha=.8)
    ax.add_collection3d(poly)
    title = fig.suptitle("", fontsize=12)
    caption = fig.text(.05,.04,"", fontsize=9)
    method = "合力预测与逆映射PDE控制" if "force_mapping" in metrics else ("PDE 反步边界控制" if "pde" in metrics else "基础PD")
    role_labels = [ax.text(0, 0, 0, s, color=COLORS[i], fontsize=8)
                   for i, s in enumerate(["1 锚点", "2 跟随者", "3 领导者"])] if "pde" in metrics else []
    vertex = box_vertices(config["payload_dimensions"])
    samples = np.searchsorted(a["time"], np.arange(0,a["time"][-1],speedup/fps))
    samples = np.unique(np.r_[samples,len(a["time"])-1])

    def update(frame):
        idx = samples[frame]
        point = vertex @ rotation[idx,0].T + pos[idx,0]
        poly.set_verts([point[f] for f in FACES])
        attach = plant.anchors @ rotation[idx,0].T + pos[idx,0]
        trace.set_data_3d(*pos[:idx+1,0].T)
        for i in range(3):
            ropes[i].set_data_3d(*np.vstack((attach[i],pos[idx,i+1])).T)
            rotor = plant.rotor_points @ rotation[idx,i+1].T + pos[idx,i+1]
            for j, ends in enumerate(((0,2),(1,3))):
                arms[i][j].set_data_3d(*rotor[list(ends)].T)
            dots[i].set_data_3d(*rotor.T)
            if role_labels:
                role_labels[i].set_position((pos[idx, i+1, 0], pos[idx, i+1, 1]))
                role_labels[i].set_3d_properties(pos[idx, i+1, 2]+.2, zdir=None)
        err = np.linalg.norm(pos[idx,0]-a["reference_position"][idx])
        title.set_text(f"{method}三机吊载  |  t = {a['time'][idx]:.1f} s  |  {speedup}倍速")
        caption.set_text(f"数值仿真 · {method} · 未经实机标定\n"
            f"位置误差 {err:.3f} m  |  张力 " + " / ".join(f"{v:.2f}" for v in a["tension"][idx]) + " N")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(samples), interval=1000/fps, blit=False)
    ani.save(Path(output)/"transport_simulation.gif", writer=animation.PillowWriter(fps=fps), dpi=90)
    update(int(len(samples)*.3))
    fig.savefig(Path(output)/"transport_preview.png", dpi=140)
    plt.close(fig)


def render_saved(folder, make_animation=False):
    import json
    path = Path(folder)
    with np.load(path/"state_log.npz") as f:
        data = {k:f[k] for k in f.files}
    config = json.loads((path/"config_snapshot.json").read_text(encoding="utf-8"))
    metrics = json.loads((path/"metrics.json").read_text(encoding="utf-8"))
    plot_run(data, metrics, config, path)
    if make_animation:
        animate_run(data, metrics, config, path)
