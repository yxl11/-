"""Read-only metrics; same formulas as performance_angles_review_20261108.

No control/simulation import. Main windows remain 8--86 and 86--178 s;
0--8 s is the explicitly registered supplement, never a selection input.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

H = .02
COUNTS = np.array([10, 16, 20])
WINDOWS = {'sweep_regrouping': (8, 86), 'patrol': (86, 178), 'initial_deployment': (0, 8)}
METRICS = ('center_m', 'formation_m', 'density', 'shape_m', 'region_pct',
           'acceleration_rms_mps2', 'mapping_rms_mps', 'safety_rms_mps',
           'local_intervention_s', 'backup_intervention_s', 'safety_intervention_s',
           'segment_distance_lower_m', 'center_peak_m')
HIGHER_BETTER = {'region_pct', 'segment_distance_lower_m', 'complete', 'terminal_in_region'}
NUMERICAL_TOL = 1e-9  # original platform-map floating-point check, not engineering slack


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def integrate(t, values, start, end, rms=False):
    mask = (t >= start-1e-10) & (t <= end+1e-10)
    tt, xx = t[mask], values[mask]
    assert abs(tt[0]-start) < 1e-10 and abs(tt[-1]-end) < 1e-10
    mean = np.trapezoid(xx*xx if rms else xx, tt, axis=0)/(end-start)
    return np.sqrt(mean) if rms else mean


def vector_rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=-1))))


def summary_checks(summary):
    """Existing model bounds; retained certificate is the relevant terminal invariant."""
    tol = NUMERICAL_TOL
    return {
        'completed_178s': summary.get('status') == 'completed' and summary.get('completed_seconds') == 178
                          and summary.get('completed_steps') == 8900,
        'worker_no_failure': summary.get('failure') is None,
        'input_hashes_unchanged': summary.get('inputs_unchanged') is True,
        'state_consistency': summary.get('actual_state_consistency') == 'bitwise equal at every committed step',
        'bootstrap_certified': summary.get('bootstrap', {}).get('status') == 'certified',
        'retained_terminal_certified': summary.get('retained_terminal_certificate', {}).get('status') == 'certified',
        'continuous_distance_at_least_2m': summary.get('minimum_continuous_distance_lower_m', -np.inf) >= 2,
        'FW_speed_bounds': summary.get('FW_speed_min_mps', -np.inf) >= 8-tol
                           and summary.get('FW_speed_max_mps', np.inf) <= 15+tol,
        'FW_inclination_bound': summary.get('FW_inclination_max_deg', np.inf) <= 10+tol,
        'FW_rate_bounds': bool(np.all(np.asarray(summary.get('FW_rate_max_mps2_deg_s', [np.inf]*3))
                                     <= np.array([2, 10, 5])+tol)),
        'rotor_acceleration_bounds': bool(np.all(np.asarray(summary.get('rotor_acceleration_max_mps2', [np.inf]*2))
                                                <= np.array([1.5, 1.0])+tol))}


def measure(folder, method, seed, expected_initial):
    folder = Path(folder)
    history, fast, slow = [load_npz(folder/name) for name in ('history.npz', 'fast_metrics.npz', 'slow_metrics.npz')]
    summary, schema = read_json(folder/'summary.json'), read_json(folder/'schema.json')
    checks = summary_checks(summary)
    t, pos, vel, roles = [history[k] for k in ('times', 'positions', 'velocities', 'roles')]
    np.testing.assert_array_equal(t, np.arange(8901)*H)
    np.testing.assert_array_equal(t, fast['times'])
    np.testing.assert_array_equal(slow['time'], np.arange(357)*.5)
    np.testing.assert_array_equal(roles, np.repeat(np.arange(3), COUNTS))
    assert pos.shape == vel.shape == (8901, 46, 3)
    assert all(np.isfinite(x).all() for x in (pos, vel, history['diagnostics']))
    checks['finite_states'] = True
    np.testing.assert_array_equal(pos[0], expected_initial['positions'])
    np.testing.assert_array_equal(vel[0], expected_initial['velocities'])
    np.testing.assert_array_equal(roles, expected_initial['roles'])
    checks['registered_initial_state_equal'] = True
    group = [np.flatnonzero(roles == i) for i in range(3)]
    centers = np.stack([pos[:, g].mean(axis=1) for g in group], axis=1)
    errors = centers-fast['reference']
    ec = np.linalg.norm(errors, axis=-1)
    eb = np.linalg.norm(errors[:, :-1]-errors[:, 1:], axis=-1)
    np.testing.assert_allclose(ec, fast['role_center_error'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(eb, fast['edge_error'], rtol=1e-12, atol=1e-12)
    arrays = {'center_m': (t, ec, True), 'formation_m': (t, eb, True),
              'density': (slow['time'], slow['relative_density_l2_by_role'], False),
              'shape_m': (slow['time'], slow['shape_by_role'], True),
              'region_pct': (t, 100*fast['region_by_role'], False)}
    cols = {name: history['diagnostics'][:, i] for i, name in enumerate(schema['diagnostic_columns'])}
    raw, mapped, accepted = [history[k] for k in ('held_reference', 'envelope_mapped', 'accepted')]
    assert raw.shape == mapped.shape == accepted.shape == (8900, 46, 3)
    assert all(np.isfinite(x).all() for x in (raw, mapped, accepted))
    np.testing.assert_allclose(np.min(cols['continuous_distance_lower_m']),
                               summary['minimum_continuous_distance_lower_m'], rtol=0, atol=1e-12)
    checks['continuous_distance_at_least_2m'] = bool(np.min(cols['continuous_distance_lower_m']) >= 2)
    speed = np.linalg.norm(vel, axis=2)
    checks['all_actual_speed_upper_bounds'] = bool(np.all(speed <= np.repeat([15, 12, 10], COUNTS)+NUMERICAL_TOL))
    rows, role_rows = [], []
    for window, (start, end) in WINDOWS.items():
        row = {'method': method, 'seed': seed, 'window': window, 'start_s': start, 'end_s': end}
        for metric, (tm, values, rms) in arrays.items():
            assert np.isfinite(values).all()
            parts = integrate(tm, values, start, end, rms)
            row[metric] = float(np.sqrt(np.mean(parts*parts)) if rms else np.mean(parts))
            labels = ('edge12', 'edge23') if metric == 'formation_m' else ('FW', 'FPV', 'Quadrotor')
            for label, value in zip(labels, parts):
                role_rows.append({'method': method, 'seed': seed, 'window': window,
                                  'scope': label, 'metric': metric, 'value': float(value)})
        mask = (t[:-1] >= start) & (t[:-1] < end)
        assert int(mask.sum()) == round((end-start)/H)
        local, backup = cols['local_repair_flag'][mask] > 0, cols['retained_backup_flag'][mask] > 0
        assert not np.any(local & backup)
        np.testing.assert_array_equal(accepted[mask][~(local | backup)], mapped[mask][~(local | backup)])
        row.update(acceleration_rms_mps2=vector_rms((np.diff(vel, axis=0)/H)[mask]),
                   mapping_rms_mps=vector_rms((mapped-raw)[mask]),
                   safety_rms_mps=vector_rms((accepted-mapped)[mask]),
                   local_intervention_s=float(H*np.count_nonzero(local)),
                   backup_intervention_s=float(H*np.count_nonzero(backup)),
                   safety_intervention_s=float(H*np.count_nonzero(local | backup)),
                   segment_distance_lower_m=float(np.min(cols['continuous_distance_lower_m'][mask])))
        state_mask = (t >= start) & (t <= end)
        scalar_ec = np.sqrt(np.mean(ec[state_mask]**2, axis=1))
        row['center_peak_m'] = float(scalar_ec.max())
        row['center_peak_first_time_s'] = float(t[state_mask][np.argmax(scalar_ec)])
        rows.append(row)
    orbit = (t >= 102) & (t <= 178)
    origins = []
    for i in range(3):
        q = fast['reference'][orbit, i, :2]
        coeff = np.linalg.lstsq(np.c_[2*q, np.ones(len(q))], np.sum(q*q, axis=1), rcond=None)[0]
        origin, radius = coeff[:2], np.sqrt(coeff[2]+np.dot(coeff[:2], coeff[:2]))
        assert np.max(np.abs(np.linalg.norm(q-origin, axis=1)-radius)) < 1e-6
        origins.append(origin)
    assert np.max(np.linalg.norm(np.array(origins)-origins[0], axis=1)) < 1e-6
    angles = np.unwrap(np.arctan2(pos[orbit, :, 1]-origins[0][1], pos[orbit, :, 0]-origins[0][0]), axis=0)
    completed = np.max(angles-angles[0], axis=0) >= 2*np.pi
    terminal = fast['region_by_role'][-1]*COUNTS
    np.testing.assert_allclose(terminal, np.rint(terminal), atol=1e-10, rtol=0)
    completion = {'method': method, 'seed': seed, 'complete': int(completed.sum()),
                  'terminal_in_region': int(np.rint(terminal).sum())}
    for i in range(3):
        completion[f'complete_role{i}'] = int(completed[group[i]].sum())
        completion[f'terminal_in_region_role{i}'] = int(round(terminal[i]))
    reference_sha = hashlib.sha256(np.ascontiguousarray(fast['reference']).tobytes()).hexdigest()
    return rows, role_rows, completion, checks, reference_sha
