"""Prescribed 64-aircraft corridor service mission and geometry-only initialization."""
from pathlib import Path
import copy
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_CONFIG = ROOT / 'output/simulations/three_group_moving_comparison_20261023/protocol/frozen_mission.json'
ROLE_COUNTS = (10, 24, 30)
SCALES = np.asarray([[13., 9., 3.5], [5.5, 4.5, 3.], [4.8, 4.5, 2.8]])
ALTITUDES = np.asarray([60., 53., 50.])
# Coordinates are offsets from (9 t, 0), in a fixed east/sideways world frame.
WAYPOINT_TIMES = np.asarray([10., 30., 55., 80.])
WAYPOINT_XY = np.asarray([
    [[27., -4.], [0., 0.], [-20., 0.]],
    [[39., 18.], [4., 8.], [-20., 0.]],
    [[39., 18.], [4., 28.], [-16., 35.]],
    [[29., 24.], [0., 24.], [-20., 24.]]
])
HORIZONTAL_SEPARATION = np.asarray([8., 2.7, 2.7])
ALL_PAIR_INITIAL_SEPARATION = 2.7
CENTER_BIASES = np.asarray([[-4., 2., 0.], [3., -2., 0.], [-3., 2., 0.]])
INITIAL_SCALE_FACTOR = 0.85
DEFAULT_SEED = 31031


def configuration():
    """Copy historic P gains/bandwidths and insert the prescribed role mission."""
    original = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))['config']
    cfg = copy.deepcopy(original)
    cfg.update(duration_s=80., counts=list(ROLE_COUNTS), sigma_m=SCALES.tolist(),
               vmax_mps=[15., 12., 10.],
               names=['fixed_wing_scout', 'fpv_relay', 'quadrotor_support'],
               initial_shape_scale=INITIAL_SCALE_FACTOR,
               phase_windows_s=[[0., 10.], [10., 30.], [30., 55.], [55., 80.]],
               metric_windows_s={'whole':[0.,80.], 'initial':[0.,10.], 'advance':[10.,30.],
                                 'side_service':[30.,55.], 'rejoin':[55.,80.]},
               snapshot_times_s=[0., 10., 30., 55., 80.])
    return cfg


def reference_kinematics(t):
    """Return role positions, velocities and accelerations, shape (..., 3, 3)."""
    time = np.asarray(t, dtype=float)
    if not np.isfinite(time).all():
        raise ValueError('Nonfinite reference time')
    frames = time.shape + (3, 3)
    q = np.zeros(frames)
    v = np.zeros(frames)
    a = np.zeros(frames)
    q[..., 0] = 9. * time[..., None]
    q[..., :2] += WAYPOINT_XY[0]
    q[..., 2] = ALTITUDES
    v[..., 0] = 9.
    for start, end, shift in zip(WAYPOINT_TIMES[:-1], WAYPOINT_TIMES[1:], np.diff(WAYPOINT_XY, axis=0)):
        u = np.clip((time - start) / (end - start), 0., 1.)
        blend = u**3 * (10. - 15.*u + 6.*u**2)
        rate = 30. * u**2 * (1.-u)**2 / (end-start)
        accel = 60. * u * (1.-u) * (1.-2.*u) / (end-start)**2
        q[..., :2] += blend[..., None, None]*shift
        v[..., :2] += rate[..., None, None]*shift
        a[..., :2] += accel[..., None, None]*shift
    return q, v, a


def reference(t, cfg=None):
    """Reference input consumed by the original task feedback."""
    q, v, _ = reference_kinematics(t)
    return q, v


def initial_state(seed=DEFAULT_SEED):
    """Draw a reproducible spacing-qualified geometry before looking at any feedback."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Expected a nonnegative integer seed')
    role_seeds = np.random.SeedSequence(seed).spawn(3)
    q0, _ = reference(0.)
    role_arrays = []
    for role, (count, child) in enumerate(zip(ROLE_COUNTS, role_seeds)):
        rng = np.random.default_rng(child)
        for _ in range(64):
            offsets = []
            for _ in range(20000):
                z = rng.standard_normal(3)
                if np.dot(z,z) > 9.:
                    continue
                point = z * (INITIAL_SCALE_FACTOR*SCALES[role])
                if offsets and np.any(np.linalg.norm(np.asarray(offsets)[:,:2]-point[:2], axis=1)
                                      < HORIZONTAL_SEPARATION[role]):
                    continue
                offsets.append(point)
                if len(offsets) == count:
                    break
            if len(offsets) != count:
                continue
            offsets = np.asarray(offsets)
            r = q0[role] + CENTER_BIASES[role] + (offsets-offsets.mean(axis=0))
            if role_arrays:
                previous = np.concatenate(role_arrays)
                if np.any(np.linalg.norm(r[:,None,:]-previous[None,:,:],axis=2)
                          < ALL_PAIR_INITIAL_SEPARATION):
                    continue
            role_arrays.append(r)
            break
        else:
            raise RuntimeError(f'Geometric sampling quota exhausted for role {role}, seed {seed}')
    positions = np.concatenate(role_arrays)
    velocities = np.tile([9., 0., 0.], (len(positions), 1))
    return positions, velocities
