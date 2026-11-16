"""Pairwise escape candidates for the existing all-aircraft predictive safety test."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

BASE = Path(__file__).resolve().parent
PRO = json.loads((BASE / 'protocol.json').read_text(encoding='utf8'))
old_dir = BASE.parent / 'fixedwing_local_repair_20261030'
old_source = old_dir / 'local_interface.py'
old_manifest = json.loads((old_dir / 'artifact_manifest.json').read_text(encoding='utf8'))
assert hashlib.sha256(old_source.read_bytes()).hexdigest() == old_manifest['files_sha256']['local_interface.py']
spec = importlib.util.spec_from_file_location('registered_cruise_repair', old_source)
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
model, COMMON, certify, advance_model = old.model, old.COMMON, old.certify, old.advance_model


def transverse_relative(r, capped, pair):
    i, j = pair
    separation = r[i] - r[j]
    n = separation / np.linalg.norm(separation)
    t = np.array([-n[1], n[0], 0.0])
    if np.linalg.norm(t) < 1e-12:
        t = np.array([1., 0., 0.]) - n * n[0]
    t /= np.linalg.norm(t)
    relative = capped[i] - capped[j]
    if relative @ t < 0:
        t = -t
    floor = model.SAFE / max(model.T)
    return relative + max(0., floor - relative @ n) * n + max(0., floor - relative @ t) * t


def propose(r, v, capped):
    previous, info = old.propose(r, v, capped)
    info['baseline_status'] = info['status']
    info['policy_branch'] = ('nominal' if info['status'] == 'nominal_candidate'
                             else 'local_cruise' if previous is not None else 'retained_backup')
    info['escape_pair'] = []
    if info['status'] == 'nominal_candidate':
        return previous, info
    limiting = info['nominal_trial'].get('limiting_pair')
    if limiting is None:
        return previous, info
    i, j = sorted(limiting)
    base = capped.copy()
    background = info['repair_members'].copy()
    if background:
        base[background] = COMMON[background]
    modified = sorted(set(background) | {i,j})
    if len(modified) > PRO['max_changed_members']:
        return previous, info
    relative = transverse_relative(r, capped, (i,j))
    task_mean = 0.5 * (capped[i] + capped[j])
    for anchor_name, anchor in [('task_mean', task_mean), ('cruise', COMMON[i])]:
        for scale in (1., 0.5):
            candidate = base.copy()
            rows = anchor + np.array([1., -1.])[:,None] * (0.5 * scale * relative)
            rows *= np.minimum(1., model.VMAX[[i,j]] / np.maximum(np.linalg.norm(rows,axis=1),1e-300))[:,None]
            candidate[[i,j]] = rows
            if np.any(np.linalg.norm(candidate[model.FW,:2],axis=1) <= 1e-9):
                continue
            cert, proof = certify(r,v,candidate)
            proof = old._jsonable(proof)
            info['certificate_calls'] += 1
            info['prediction_steps_total'] += int(proof['prediction_steps'])
            assert info['certificate_calls'] <= PRO['max_certificate_calls']
            info['attempts'].append({'branch':'pair_escape','action':'certified' if cert else 'rejected',
                'certificate_called':True,'anchor':anchor_name,'scale':scale,'pair':[i,j],
                'repair_members':modified,'premap_rows':candidate[modified].tolist(),'proof':proof})
            if cert is not None:
                info.update(status='local_repair_candidate',policy_branch='pair_escape',escape_pair=[i,j],
                    escape_anchor=anchor_name,escape_scale=scale,repair_members=modified,
                    premap_rows=candidate[modified].tolist(),trial=proof,
                    stop_reason='pair_escape_certified',
                    changed_reference_members=np.flatnonzero(np.any(capped != candidate,axis=1)).tolist())
                return cert,info
    return previous,info


class RetainedBackupInterface(old.frozen_global_safety.RetainedBackupInterface):
    def select(self, step, r, v, held_reference):
        self.cert.assert_state(r,v)
        trial,info = propose(r,v,held_reference)
        if trial is not None:
            self.generation += 1
            self.cert = trial
            self.cert.generation = self.generation
            self.cert.created_step = step
            self.ledger.append(self.cert.summary())
        else:
            info['status'] = 'retained_backup'
        info.update(certificate_generation=self.cert.generation,
                    certificate_age_steps=self.cert.cursor,
                    inherited_margin_m=self.cert.evidence['certificate_margin_m'])
        return self.cert.command(),info
