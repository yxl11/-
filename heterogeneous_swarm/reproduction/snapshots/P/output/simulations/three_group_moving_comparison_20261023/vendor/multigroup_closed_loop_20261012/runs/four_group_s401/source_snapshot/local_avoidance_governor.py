"""Local separation candidates accepted by the original braking-backup check.

This is a mature engineering avoidance component, not a new density theorem.
The task velocity is scaled independently of the separation velocity. Neither
the potential nor its candidate alone guarantees spacing or task completion.
"""
from __future__ import annotations
import numpy as np
from backup_governor import BackupGovernor, remaining_braking_path


class LocalAvoidanceGovernor(BackupGovernor):
    def __init__(self, *args, separation_buffer=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.separation_buffer = float(separation_buffer)
        if not (np.isfinite(self.separation_buffer) and self.separation_buffer > 0.):
            raise ValueError('a finite positive separation buffer is required')

    def filter(self, r, v, c_nom, tau, vmax, amax):
        self._configure(tau, vmax, amax)
        r, v, nominal = [np.asarray(x, dtype=float) for x in (r, v, c_nom)]
        if not (r.shape == v.shape == nominal.shape == (len(self.tau), 3)
                and all(np.all(np.isfinite(x)) for x in (r, v, nominal))):
            raise ValueError('finite n-by-three positions, velocities and commands required')
        speed = np.linalg.norm(v, axis=1)
        if np.any(speed > self.vmax + 1e-9):
            raise ValueError('current velocity exceeds registered platform bounds')
        def cap(command):
            magnitude = np.linalg.norm(command, axis=1)
            return command * np.minimum(1., self.vmax / np.maximum(magnitude, 1e-300))[:, None]
        nominal = cap(nominal)
        i,j = self.pairs
        difference = r[i] - r[j]
        spacing = np.linalg.norm(difference, axis=1)
        paths = remaining_braking_path(speed, self.tau, self.amax, self.dt)
        radius = self.distance + self.separation_buffer + paths[i] + paths[j]
        weight = np.maximum(0., (radius - spacing) / (radius - self.distance))
        weight = np.minimum(1., weight)
        direction = difference / np.maximum(spacing, 1e-300)[:, None]
        separation = np.zeros_like(nominal)
        np.add.at(separation,i,self.vmax[i,None]*weight[:,None]*direction)
        np.add.at(separation,j,-self.vmax[j,None]*weight[:,None]*direction)
        active = spacing <= self.screen_radius
        pairs = i[active],j[active]
        far_margin = (float((spacing[~active]-self.screen_radius[~active]+self.margin).min())
                      if np.any(~active) else np.inf)
        candidates = [(2.**(-k), False) for k in range(self.levels)] + [(0.,False),(0.,True)]
        work = 0
        for attempt,(scale,emergency) in enumerate(candidates,start=1):
            candidate = np.zeros_like(nominal) if emergency else cap(scale*nominal+separation)
            accepted,margin,steps = self._candidate_safe(r,v,candidate,pairs,far_margin)
            work += steps
            if accepted:
                return candidate, {
                    'lambda':float(scale),
                    'intervened':bool(emergency or scale < 1. or np.any(weight > 0.)),
                    'min_margin':float(margin), 'tested_candidates':attempt,
                    'active_pairs':int(np.sum(active)), 'backup_steps':steps,
                    'backup_grid_intervals_total':work,
                    'separation_pairs':int(np.sum(weight>0.)),
                    'maximum_separation_component_mps':float(np.linalg.norm(separation,axis=1).max()),
                    'emergency_zero_reference':bool(emergency),
                    'model_scope':'registered deterministic velocity-loop and quadratic step interpolation',
                    'candidate_scope':'local radial separation plus task reference, with shared braking-backup acceptance'}
        raise RuntimeError('no certified braking backup; preserve failed state and stop the run')
