"""Analytic target-IID KDE reference, using archived task geometry only.

No random samples, controller, runner, solver or simulation are executed.
Run from any directory: python -B <this script>.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT/'output/reviews/mixed_sweep_orbit_paired_20261102/runs'
GAIN = ROOT/'output/reviews/p_gain_calibration_20261108'
SHAPE = ROOT/'output/reviews/mixed_sweep_orbit_sparse_20261101/shape_models.py'
SEEDS = (33001, 33002, 33003)
ROLES = ('Fixed-wing', 'FPV', 'Quadrotor')
ROLE_ZH = ('固定翼', '穿越机', '四旋翼')
WINDOWS = {'sweep_regrouping': (8.0, 86.0), 'patrol': (86.0, 178.0)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def write_json(name, value):
    (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf8')


def folder(method, seed):
    return GAIN/f'runs/evaluation/B_s{seed}' if method == 'P' else OLD/f'{method}_s{seed}'


def source_paths():
    paths = [Path(__file__), SHAPE, GAIN/'protocol.json',
             HERE.parent/'archive_manifest.json',
             HERE.parent/'archive_before/execution_error_derivation.tex']
    for method in ('P', 'RAS08', 'MMD19'):
        for seed in SEEDS:
            paths += [folder(method, seed)/name for name in
                      ('configuration.json', 'slow_metrics.npz', 'task_history.npz')]
    return paths


def ratio_formula(Sigma, bandwidth, count):
    """Dimensionless determinant ratios, evaluated through eigenvalues."""
    eigenvalues = np.linalg.eigvalsh(Sigma)
    assert np.all(eigenvalues > 0) and bandwidth > 0 and count > 0
    z = bandwidth**2/eigenvalues
    self_ratio = np.sqrt(np.prod(eigenvalues))/bandwidth**3
    smooth_ratio = np.prod((1+z)**(-0.5))
    cross_ratio = np.prod((1+0.5*z)**(-0.5))
    bias = smooth_ratio-2*cross_ratio+1
    variance = (self_ratio-smooth_ratio)/count
    return np.array([bias, variance, bias+variance,
                     self_ratio/count, (1-1/count)*smooth_ratio,
                     -2*cross_ratio, 1.0])


def gaussian_overlap(cov1, cov2):
    """Integral of two equal-mean normalized Gaussians in R^3."""
    sign, logdet = np.linalg.slogdet(cov1+cov2)
    assert sign > 0
    return np.exp(-1.5*np.log(2*np.pi)-0.5*logdet)


def independent_overlap_formula(Sigma, bandwidth, count):
    """Separate covariance-overlap calculation, without determinant ratios."""
    Q = bandwidth**2*np.eye(3)
    target = gaussian_overlap(Sigma, Sigma)
    self_term = gaussian_overlap(Q, Q)
    smoothed = gaussian_overlap(Sigma+Q, Sigma+Q)
    cross = gaussian_overlap(Sigma+Q, Sigma)
    bias = (smoothed-2*cross+target)/target
    variance = (self_term-smoothed)/(count*target)
    direct_total = (self_term/count+(1-1/count)*smoothed-2*cross+target)/target
    return np.array([bias, variance, direct_total])


def average(times, values, window):
    a, b = window
    mask = (times >= a) & (times <= b)
    assert times[mask][0] == a and times[mask][-1] == b
    return np.trapezoid(values[mask], times[mask], axis=0)/(b-a)


def write_csv(name, rows):
    with (HERE/name).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analytic_limit_checks(Sigma, beta, count):
    base = ratio_formula(Sigma, beta, count)
    large_n = ratio_formula(Sigma, beta, 10**12)
    one = ratio_formula(Sigma, beta, 1)
    small_beta = beta*1e-3
    small = ratio_formula(Sigma, small_beta, count)
    large = ratio_formula(Sigma, beta*1e6, count)
    scaled_small = count*small_beta**3*small[2]/np.sqrt(np.linalg.det(Sigma))
    theta = 0.371
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated = ratio_formula(R@Sigma@R.T, beta, count)
    checks = {
        'N_to_infinity_expected_bias': float(base[0]),
        'N_1e12_total_minus_bias': float(large_n[2]-base[0]),
        'N_1_pair_cross_term': float(one[4]),
        'beta_to_zero_bias_check_beta': float(small_beta),
        'beta_to_zero_bias_observed': float(small[0]),
        'beta_to_zero_scaled_total_expected_1': float(scaled_small),
        'beta_to_infinity_total_expected_1': float(large[2]),
        'beta_to_infinity_variance_expected_0': float(large[1]),
        'rotation_invariance_max_absolute_difference': float(np.max(np.abs(rotated-base))),
    }
    assert abs(large_n[2]-base[0]) < 1e-9
    assert one[4] == 0
    assert abs(small[0]) < 1e-9 and abs(scaled_small-1) < 1e-8
    assert abs(large[2]-1) < 1e-12 and abs(large[1]) < 1e-20
    assert np.max(np.abs(rotated-base)) < 1e-12
    return checks


def manuscript_fragment(window_rows):
    lines = [
        r'\paragraph{同规模目标采样的密度量级参照}',
        r'为解释固定带宽密度指标的量级，在每个时刻令$X_{ik}$独立同分布于目标$\Pi_i=\mathcal N(q_i,\Sigma_i)$，并以与控制实验相同的$N_i$和$\beta_i$构造$\widehat p_i=N_i^{-1}\sum_kG_{\beta_i}(\cdot-X_{ik})$。在$\R^3$上对指标取采样期望，有',
        r'\begin{equation}',
        r'\begin{aligned}',
        r'\mathbb E L_{\rho,i}&=\mathcal B_{\rho,i}+\mathcal V_{\rho,i},\\',
        r'\mathcal B_{\rho,i}&=\sqrt{\frac{\det\Sigma_i}{\det(\Sigma_i+\beta_i^2I)}}-2\sqrt{\frac{\det(2\Sigma_i)}{\det(2\Sigma_i+\beta_i^2I)}}+1,\\',
        r'\mathcal V_{\rho,i}&=\frac1{N_i}\left[\frac{\sqrt{\det\Sigma_i}}{\beta_i^3}-\sqrt{\frac{\det\Sigma_i}{\det(\Sigma_i+\beta_i^2I)}}\right].',
        r'\end{aligned}',
        r'\label{eq:iid-density-reference}',
        r'\end{equation}',
        r'其中$\mathcal B_{\rho,i}$为核平滑的归一化偏差平方，$\mathcal V_{\rho,i}$为有限IID样本的方差贡献。各量由高斯重叠积分解析求得，并按0.5~s时刻和梯形积分计算窗口均值，综合值按角色等权平均。',
        r'\begin{table}[htbp]\centering\small',
        r'\caption{同人数、同物理核带宽及同目标协方差下的IID采样期望参照。各项均为无量纲窗口均值，综合行按三角色等权平均。}\label{tab:iid-density-reference}',
        r'\begin{tabular}{llrrr}\toprule',
        r'窗口 (s) & 角色 & $\mathcal B_{\rho}$ & $\mathcal V_{\rho}$ & $\mathbb E L_{\rho}$\\\midrule',
    ]
    for ri, row in enumerate(window_rows):
        if ri == 4:
            lines.append(r'\midrule')
        label = '8--86' if row['window'] == 'sweep_regrouping' else '86--178'
        role = ROLE_ZH[ROLES.index(row['role'])] if row['role'] in ROLES else '等角色综合'
        lines.append(f"{label} & {role} & {row['smoothing_bias_squared']:.4f} & {row['finite_N_variance']:.4f} & {row['expected_normalized_L2']:.4f}"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}',
              r'固定翼归一化密度指标中的固定自核贡献$\sqrt{\det\Sigma_i}/(N_i\beta_i^3)$为8.3350；本任务保持目标体积不变，因此该项不随伸缩与转向改变。总指标还包含成员间核重叠、与目标的交叉项及目标平方范数。IID参照用于解释量级；受控成员通常具有相关性，因此不能将该期望作为受控点集的性能下界或控制误差。']
    return '\n'.join(lines)+'\n'


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    paths = source_paths()
    before = {relative(p): sha(p) for p in paths}
    config = read(folder('P', SEEDS[0])/'configuration.json')
    counts = np.asarray(config['counts'], dtype=int)
    beta = np.asarray(config['bandwidth_m'], dtype=float)
    assert counts.tolist() == [10, 16, 20]
    assert beta.tolist() == [1.7, 1.5, 1.35]
    protocol = read(GAIN/'protocol.json')
    assert protocol['analysis_windows_s'] == {k: list(v) for k, v in WINDOWS.items()}
    spec = importlib.util.spec_from_file_location('density_reference_shape_geometry', SHAPE)
    shape = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shape)  # inspected pure geometry module, never a runner
    with np.load(folder('P', SEEDS[0])/'slow_metrics.npz', allow_pickle=False) as data:
        times = data['time'].copy()
    np.testing.assert_array_equal(times, np.arange(357)*0.5)
    S, Sdot = shape.shape_state(times)
    Sigma = S@np.swapaxes(S, -1, -2)
    np.testing.assert_allclose(np.diagonal(Sigma[0], axis1=1, axis2=2),
                               np.asarray(config['sigma_m'])**2, atol=1e-12, rtol=0)
    run_checks, observed_p = [], []
    for method in ('P', 'RAS08', 'MMD19'):
        for seed in SEEDS:
            run = folder(method, seed)
            other = read(run/'configuration.json')
            for key in ('counts', 'bandwidth_m', 'sigma_m', 'metric_period_s', 'duration_s'):
                assert other[key] == config[key], (relative(run), key)
            with np.load(run/'slow_metrics.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['time'], times)
                if method == 'P':
                    observed_p.append(data['relative_density_l2_by_role'].copy())
            with np.load(run/'task_history.npz', allow_pickle=False) as data:
                expected_S = shape.shape_state(data['time'])[0]
                mismatch = float(np.max(np.abs(data['shape_matrix']-expected_S)))
                assert mismatch < 1e-12
            run_checks.append({'method': method, 'seed': seed,
                               'configuration_and_metric_clock_match': True,
                               'shape_history_max_absolute_difference': mismatch})
    values = np.empty((len(times), 3, 7))
    overlap_max = 0.0
    for ti in range(len(times)):
        for role in range(3):
            values[ti, role] = ratio_formula(Sigma[ti, role], beta[role], counts[role])
            check = independent_overlap_formula(Sigma[ti, role], beta[role], counts[role])
            overlap_max = max(overlap_max, float(np.max(np.abs(check-values[ti, role, :3]))))
    assert overlap_max < 1e-12
    assert values[:, :, :2].min() >= -1e-12
    np.testing.assert_allclose(values[:, :, 2], values[:, :, 3:].sum(axis=2), atol=1e-12, rtol=0)
    self_range = np.ptp(values[:, :, 3], axis=0)
    assert self_range.max() < 1e-12
    component_names = ['smoothing_bias_squared', 'finite_N_variance', 'expected_normalized_L2',
                       'fixed_diagonal_self_kernel', 'expected_offdiagonal_overlap',
                       'minus_twice_expected_target_overlap', 'target_norm_normalized']
    per_time = []
    for ti, t in enumerate(times):
        for ri, role in enumerate(ROLES):
            row = {'time_s': float(t), 'role': role, 'N': int(counts[ri]), 'bandwidth_m': float(beta[ri])}
            row.update(zip(component_names, map(float, values[ti, ri])))
            per_time.append(row)
        row = {'time_s': float(t), 'role': 'Equal-role aggregate', 'N': '', 'bandwidth_m': ''}
        row.update(zip(component_names, map(float, values[ti].mean(axis=0))))
        per_time.append(row)
    window_rows, contextual_fw = [], []
    for name, bounds in WINDOWS.items():
        means = average(times, values, bounds)
        for ri, role in enumerate(ROLES):
            row = {'window': name, 'start_s': bounds[0], 'end_s': bounds[1], 'role': role}
            row.update(zip(component_names, map(float, means[ri])))
            window_rows.append(row)
        row = {'window': name, 'start_s': bounds[0], 'end_s': bounds[1], 'role': 'Equal-role aggregate'}
        row.update(zip(component_names, map(float, means.mean(axis=0))))
        window_rows.append(row)
        fw = [float(average(times, x[:, 0], bounds)) for x in observed_p]
        contextual_fw.append({'window': name, 'proposed_existing_FW_mean': float(np.mean(fw)),
                              'proposed_existing_FW_per_seed': fw,
                              'iid_FW_expectation': float(means[0, 2]),
                              'fixed_diagonal_self_kernel_FW': float(means[0, 3]),
                              'interpretation': 'Scale context only; not a new controller comparison or error decomposition of controlled data.'})
    limits = {ROLES[i]: analytic_limit_checks(Sigma[0, i], beta[i], counts[i]) for i in range(3)}
    validation = {'status': 'PASS', 'independent_overlap_formula_cases': len(times)*3,
                  'independent_overlap_max_absolute_difference': overlap_max,
                  'fixed_self_term_time_range_by_role': self_range.tolist(),
                  'run_configuration_and_geometry_checks': run_checks,
                  'analytic_limits': limits,
                  'analytic_limit_statements': ['N -> infinity: expected normalized L2 -> smoothing bias squared',
                     'beta -> 0 at fixed N: bias -> 0, N beta^3 E[L]/sqrt(det(Sigma)) -> 1',
                     'beta -> infinity: bias -> 1, variance -> 0',
                     'N = 1: off-diagonal pair term vanishes',
                     'Isotropic kernel: translation and rotation leave this expectation unchanged'],
                  'no_random_sampling_no_simulation': True}
    np.savez_compressed(HERE/'reference_data.npz', times_s=times, roles=np.array(ROLES),
                        counts=counts, bandwidth_m=beta, shape_matrix=S, covariance=Sigma,
                        component_names=np.array(component_names), components=values,
                        equal_role_components=values.mean(axis=1))
    write_csv('per_time_role.csv', per_time)
    write_csv('window_role.csv', window_rows)
    write_json('validation.json', validation)
    write_json('results.json', {'status': 'PASS', 'definition': 'E[||KDE-Pi||_2^2]/||Pi||_2^2 for target-IID samples',
        'sampling_assumption': 'At each time, N independent samples from the current target Gaussian; no trajectory model.',
        'integration': 'Existing 0.5 s samples, endpoint-inclusive trapezoid per fixed window; equal-role aggregation.',
        'counts': counts.tolist(), 'bandwidth_m': beta.tolist(), 'windows': window_rows,
        'fixed_wing_scale_context': contextual_fw,
        'scope': 'Analytic scale reference; not a lower bound, optimum, controlled-data error, or algorithm baseline.'})
    (HERE/'appendix_fragment.tex').write_text(manuscript_fragment(window_rows), encoding='utf8')
    sentence = '为说明有限成员与核平滑对密度指标量级的影响，附录给出相同人数、带宽和目标协方差下的目标IID采样期望参照。\n'
    (HERE/'body_sentence_zh.txt').write_text(sentence, encoding='utf8')
    report = ['# 固定Gaussian KDE的目标IID采样量级参照', '',
              '本计算仅使用既有任务参数、shape_state及0.5 s指标时钟，没有生成随机样本，也没有运行控制器或新增运动仿真。', '',
              '在每个时刻，假设N个成员位置独立服从当前目标Gaussian。期望误差可严格分解为核平滑偏差平方和有限样本方差；这不是实际受控成员误差的分解。', '',
              '|窗口(s)|角色|平滑偏差平方|有限N方差|期望归一化L2|', '|---|---|---:|---:|---:|']
    for row in window_rows:
        report.append(f"|{row['start_s']:g}–{row['end_s']:g}|{row['role']}|{row['smoothing_bias_squared']:.6f}|{row['finite_N_variance']:.6f}|{row['expected_normalized_L2']:.6f}|")
    report += ['', '固定翼量级说明：']
    for row in contextual_fw:
        report.append(f"- {row['window']}：既有P固定翼Lrho={row['proposed_existing_FW_mean']:.6f}；IID期望参照={row['iid_FW_expectation']:.6f}；固定对角自核项={row['fixed_diagonal_self_kernel_FW']:.6f}。")
    report += ['', '固定翼目标三轴乘积13×9×3.5=409.5 m³，N=10、beta=1.7 m，故固定对角自核项为409.5/(10×1.7³)。任务的水平面积保持变形及旋转不改变该项。它解释为何这一归一化指标包含较大的离散核贡献，但不能说6.7全部来自自核项；总量还含成员间重叠、与目标的交叉项及目标平方范数。', '',
               '实际受控点集可能比目标IID样本更规则且具有相关性，其Lrho低于IID期望并不矛盾，也不据此给方法重新排名。不能从实测Lrho减去IID方差后称为控制误差。目标核平滑偏差也不等于全部实际形状误差。', '',
               '独立检验采用另一套Gaussian重叠积分表达，覆盖357×3个时刻/角色，并检查N趋于无穷、beta趋于零和无穷、N=1以及旋转不变性。详见validation.json。', '',
               '复跑：`python -B output/reviews/two_round_revision_20261108/density_reference/compute_density_reference.py`。插入材料见appendix_fragment.tex与body_sentence_zh.txt；原主稿没有修改。']
    (HERE/'report_zh.md').write_text('\n'.join(report)+'\n', encoding='utf8')
    after = {relative(p): sha(p) for p in paths}
    assert before == after, 'A read-only input changed during analysis'
    outputs = [p for p in HERE.iterdir() if p.is_file() and p.name not in ('source_manifest.json', 'compute_density_reference.py')]
    write_json('source_manifest.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': before, 'inputs_unchanged': True,
        'outputs_sha256': {p.name: sha(p) for p in outputs},
        'imported_workspace_module': relative(SHAPE),
        'no_runner_import': True, 'no_new_simulation_or_Monte_Carlo': True,
        'software': {'python': sys.version, 'numpy': np.__version__}})
    print(json.dumps({'status': 'PASS', 'source_files': len(before),
                      'independent_overlap_max_difference': overlap_max,
                      'windows': window_rows, 'fixed_wing_context': contextual_fw}, ensure_ascii=False))


if __name__ == '__main__':
    main()
