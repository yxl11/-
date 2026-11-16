"""Braking-backup reference governor for the registered surrogate velocity loop.

No real-aircraft, robustness, obstacle, communication-delay, or task-completion
guarantee is implied.  All predictions use the same discrete saturated velocity
update as execution.  Continuous spacing refers to its quadratic interpolation.
"""
from __future__ import annotations

import numpy as np


def step(r, v, command, dt, tau, amax):
    """Execute one model step; caller supplies a speed-admissible command."""
    difference = command - v
    alpha = np.minimum(dt / tau, amax * dt /
                       np.maximum(np.linalg.norm(difference, axis=1), 1e-300))
    vnext = v + alpha[:, None] * difference
    return r + .5 * dt * (v + vnext), vnext


def remaining_braking_path(speed, tau, amax, dt):
    """Exact path length remaining under zero-reference braking, to infinite time."""
    speed = np.asarray(speed, dtype=float)
    m = np.ceil(np.maximum(0., speed - amax * tau) / (amax * dt))
    residual = np.maximum(0., speed - m * amax * dt)
    return dt * (m * speed - .5 * amax * dt * m ** 2) + (tau - .5 * dt) * residual


def segment_clearance_lower_bound(r, rn, v, vn, dt, pairs=None):
    """Vectorized lower bound on every pair's continuous quadratic-step distance.

    Use exact roots only for pairs whose lower bound indicates a possible
    violation.  At h=.02 s, acceleration bounds <=.8 m/s² give <=.00008 m
    conservativeness on any pair.  ``pairs`` is a tuple of equal-length indices.
    """
    if pairs is None:
        pairs = np.triu_indices(len(r), 1)
    i, j = pairs
    p = r[i] - r[j]
    displacement = (rn[i] - rn[j]) - p
    theta = np.clip(-np.sum(p * displacement, axis=-1) /
                    np.maximum(np.sum(displacement ** 2, axis=-1), 1e-300), 0., 1.)
    chord = np.linalg.norm(p + theta[:, None] * displacement, axis=-1)
    acceleration = ((vn - v)[i] - (vn - v)[j]) / dt
    return chord - np.linalg.norm(acceleration, axis=-1) * dt ** 2 / 8.


def segment_min_distances_exact(r, rn, v, vn, dt, pairs=None):
    """Independent cubic-root check; use on a small set of potentially close pairs."""
    if pairs is None:
        pairs = np.triu_indices(len(r), 1)
    a = (vn - v) / dt
    result = []
    for i, j in zip(*pairs):
        p, w, acc = r[i] - r[j], v[i] - v[j], a[i] - a[j]
        coefficients = [np.dot(acc, acc), 3. * np.dot(w, acc),
                        2. * (np.dot(w, w) + np.dot(p, acc)), 2. * np.dot(p, w)]
        roots = np.roots(np.trim_zeros(coefficients, trim="f")) if any(coefficients) else []
        times = [0., dt] + [float(z.real) for z in roots
                            if abs(z.imag) < 1e-9 and 0. < z.real < dt]
        result.append(min(np.linalg.norm(p + t * w + .5 * t ** 2 * acc) for t in times))
    return np.asarray(result)


class BackupGovernor:
    def __init__(self, dt=.02, distance=1., backup_horizon=10.,
                 numerical_margin=1e-7, levels=10):
        self.dt = float(dt)
        self.distance = float(distance)
        self.margin = float(numerical_margin)
        self.steps = int(round(backup_horizon / dt))
        self.levels = int(levels)
        self.backup_indices = np.arange(self.steps + 1, dtype=float)[:, None]
        self.parameters = None
        if not (self.dt > 0. and distance > 0. and self.steps > 0 and self.levels > 0):
            raise ValueError("positive step, distance, backup horizon and levels required")

    def _configure(self, tau, vmax, amax):
        tau, vmax, amax = [np.asarray(x, dtype=float) for x in (tau, vmax, amax)]
        if not (tau.ndim == 1 and tau.shape == vmax.shape == amax.shape and
                np.all(np.isfinite(tau)) and np.all(np.isfinite(vmax)) and
                np.all(np.isfinite(amax)) and self.dt <= tau.min() and
                np.all(vmax > 0.) and np.all(amax > 0.)):
            raise ValueError("finite positive platform bounds and dt <= min(tau) required")
        if self.parameters is not None:
            if not all(np.array_equal(old, new) for old, new in zip(self.parameters, (tau, vmax, amax))):
                raise ValueError("use a new governor when the platform parameters change")
            return
        self.parameters = (tau.copy(), vmax.copy(), amax.copy())
        self.tau, self.vmax, self.amax = self.parameters
        self.pairs = np.triu_indices(len(tau), 1)
        i, j = self.pairs
        maximum_backup = remaining_braking_path(vmax, tau, amax, self.dt)
        self.screen_radius = (self.distance + self.margin +
                              self.dt * (vmax[i] + vmax[j]) +
                              maximum_backup[i] + maximum_backup[j])

    def _candidate_safe(self, r, v, command, pairs, far_margin):
        rn, vn = step(r, v, command, self.dt, self.tau, self.amax)
        if not len(pairs[0]):
            return True, far_margin, 0
        first = segment_clearance_lower_bound(r, rn, v, vn, self.dt, pairs)
        margin = min(float(first.min() - self.distance), far_margin)
        if np.any(first < self.distance + self.margin):
            return False, margin, 0
        i, j = pairs
        speed_all = np.linalg.norm(vn, axis=1)
        tail0 = remaining_braking_path(speed_all, self.tau, self.amax, self.dt)
        tail_lower = np.linalg.norm(rn[i] - rn[j], axis=1) - tail0[i] - tail0[j]
        unresolved = tail_lower < self.distance + self.margin
        if np.any(~unresolved):
            margin = min(margin, float(tail_lower[~unresolved].min() - self.distance))
        if not np.any(unresolved):
            return True, margin, 0
        i, j = i[unresolved], j[unresolved]
        agents, inverse = np.unique(np.concatenate((i, j)), return_inverse=True)
        ip, jp = inverse[:len(i)], inverse[len(i):]
        speed, tau, amax = speed_all[agents], self.tau[agents], self.amax[agents]
        m = np.ceil(np.maximum(0., speed - amax * tau) / (amax * self.dt))
        k = self.backup_indices
        speeds = ((speed - amax * self.dt * np.minimum(k, m)) *
                  (1. - self.dt / tau) ** np.maximum(0., k - m))
        tails = remaining_braking_path(speeds, tau, amax, self.dt)
        directions = vn[agents] / np.maximum(speed, 1e-300)[:, None]
        rb = (rn[agents][None, :, :] +
              (tail0[agents][None, :] - tails)[:, :, None] * directions[None, :, :])
        vb = speeds[:, :, None] * directions[None, :, :]
        relative = rb[:, ip, :] - rb[:, jp, :]
        displacement = relative[1:] - relative[:-1]
        theta = np.clip(-np.sum(relative[:-1] * displacement, axis=-1) /
                        np.maximum(np.sum(displacement ** 2, axis=-1), 1e-300), 0., 1.)
        chord = np.linalg.norm(relative[:-1] + theta[:, :, None] * displacement, axis=-1)
        accel = (vb[1:] - vb[:-1]) / self.dt
        curvature = np.linalg.norm(accel[:, ip, :] - accel[:, jp, :], axis=-1) * self.dt ** 2 / 8.
        lower = chord - curvature
        margin = min(margin, float(lower.min() - self.distance))
        if np.any(lower < self.distance + self.margin):
            return False, margin, self.steps
        terminal = np.linalg.norm(relative[-1], axis=-1) - tails[-1, ip] - tails[-1, jp]
        margin = min(margin, float(terminal.min() - self.distance))
        return bool(np.all(terminal >= self.distance + self.margin)), margin, self.steps

    def filter(self, r, v, c_nom, tau, vmax, amax):
        """Return (speed command, diagnostic dictionary); fail loudly if no backup.

        ``min_margin`` is a conservative prediction certificate margin over the
        admitted step and complete braking backup, not observed closest spacing.
        """
        self._configure(tau, vmax, amax)
        r, v, nominal = [np.asarray(x, dtype=float) for x in (r, v, c_nom)]
        if not (r.shape == v.shape == nominal.shape == (len(self.tau), 3) and
                np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.all(np.isfinite(nominal))):
            raise ValueError("finite n-by-three positions, velocities and commands required")
        if np.any(np.linalg.norm(v, axis=1) > self.vmax + 1e-9):
            raise ValueError("current velocity exceeds registered platform bounds")
        speed = np.linalg.norm(nominal, axis=1)
        nominal = nominal * np.minimum(1., self.vmax / np.maximum(speed, 1e-300))[:, None]
        i, j = self.pairs
        spacing = np.linalg.norm(r[i] - r[j], axis=1)
        active = spacing <= self.screen_radius
        pairs = i[active], j[active]
        far_margin = float((spacing[~active] - self.screen_radius[~active] + self.margin).min()) if np.any(~active) else np.inf
        grid_work = 0
        for attempt, scale in enumerate([2. ** (-k) for k in range(self.levels)] + [0.], start=1):
            passed, margin, backup_steps = self._candidate_safe(r, v, scale * nominal, pairs, far_margin)
            grid_work += backup_steps
            if passed:
                return scale * nominal, {
                    "lambda": float(scale), "intervened": bool(scale < 1.),
                    "min_margin": float(margin), "tested_candidates": attempt,
                    "active_pairs": int(np.sum(active)), "backup_steps": backup_steps,
                    "backup_grid_intervals_total": grid_work,
                    "model_scope": "registered deterministic velocity-loop and quadratic step interpolation"}
        raise RuntimeError("no certified braking backup; preserve failed state and stop the run")
