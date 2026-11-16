"""Heading-aligned common-cruise certificates in the world-coordinate aircraft model."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE.parent / 'mixed_role_corridor_demo_20261031'
spec = importlib.util.spec_from_file_location('frozen_corridor_engine', PREVIOUS / 'engine.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
SOURCE_HASHES = shared.SOURCE_HASHES


def horizontal_backup(tangent):
    """A common 9 m/s horizontal velocity along the prescribed frame tangent."""
    horizontal = np.asarray(tangent, dtype=float)[:2]
    norm = np.linalg.norm(horizontal)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError('Undefined common task-frame tangent')
    return np.array([9. * horizontal[0] / norm, 9. * horizontal[1] / norm, 0.])


def make_engine(counts, mode='tangent'):
    """Each engine and each accepted certificate own their respective backup vectors.

    set_direction changes the proposal context only. It is called before, never
    during, a synchronous certification search in a single closed-loop process.
    """
    if mode not in ('world', 'tangent'):
        raise ValueError('Expected world or tangent')
    engine = shared.make_engine(counts)
    model = engine.model
    ns = engine.certify.__globals__
    original_cert = ns['Certificate']
    original_gov = engine.RetainedBackupInterface
    # The archived axial cruise is represented by an integer vector. Angular
    # directions need floating-point storage across all three search modules.
    common = engine.COMMON.astype(float)
    engine.COMMON = common
    ns['COMMON'] = common
    engine.propose.__globals__['COMMON'] = common
    engine.propose.__globals__['old'].COMMON = common

    class Certificate(original_cert):
        def __init__(self, states_r, states_v, target, evidence):
            super().__init__(states_r, states_v, target, evidence)
            self.backup = engine.COMMON.copy()
            self.backup.setflags(write=False)
            self.evidence = dict(evidence, backup_mps=self.backup[0].tolist())

        def command(self):
            return self.target.copy() if self.cursor == 0 else self.backup.copy()

        def expected_next(self):
            if self.cursor+1 < len(self.R):
                return self.R[self.cursor+1], self.V[self.cursor+1]
            r, v, _ = engine.advance_model(self.expected_r, self.expected_v, self.backup)
            return r, v

        def summary(self):
            record = super().summary()
            payload = bytes.fromhex(record['prefix_sha256']) + self.backup[0].tobytes()
            record['prefix_and_backup_sha256'] = hashlib.sha256(payload).hexdigest()
            return record

    def tail_path(v):
        b = model.B
        speed = np.linalg.norm(b)
        heading = np.arctan2(b[1], b[0])
        q = model.angles(v[model.FW])
        errors = np.stack([np.abs(q[:,0]-speed), np.abs(model.wrap(q[:,1]-heading)),
                           np.abs(q[:,2])],axis=1)
        areas = model.scalar_area(errors,model.T,model.RATE)
        bounds = np.empty(len(v))
        bounds[model.FW] = areas[:,0] + speed*(areas[:,1]+areas[:,2])
        bounds[model.ROT] = model.scalar_area(np.linalg.norm(v[model.ROT]-b,axis=1),
                                              model.TAU[model.ROT],model.ACC[model.ROT])
        return bounds

    class Governor(original_gov):
        def select(self, step, r, v, held_reference):
            c, info = super().select(step,r,v,held_reference)
            info.update(trial_backup_mps=engine.COMMON[0].tolist(),
                        retained_certificate_backup_mps=self.cert.backup[0].tolist())
            return c, info

        def save_current(self,path):
            cert = self.cert
            np.savez_compressed(path,R=cert.R,V=cert.V,target=cert.target,backup=cert.backup,
                expected_r=cert.expected_r,expected_v=cert.expected_v,cursor=cert.cursor,
                generation=cert.generation,created_step=cert.created_step)

    def set_direction(tangent):
        b = horizontal_backup(tangent) if mode == 'tangent' else np.array([9.,0.,0.])
        engine.COMMON[:] = b
        model.B = b.copy()
        if not np.array_equal(engine.COMMON[0], model.B):
            raise RuntimeError('Tail-bound reference differs from actual backup')
        return b

    model.tail_path = tail_path
    ns['Certificate'] = Certificate
    engine.RetainedBackupInterface = Governor
    engine.set_direction = set_direction
    engine.mode = mode
    return engine
