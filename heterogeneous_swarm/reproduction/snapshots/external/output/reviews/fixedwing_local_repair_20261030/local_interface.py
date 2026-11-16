"""Sparse command repair with the unchanged whole-fleet trajectory certifier.

The task feedback, aircraft models, certificate conditions and retained backup
are imported from the frozen comparison. Only candidate enumeration changes.
Each call starts with an empty repair set. Accepted inputs are returned from
the resulting certificate, with no post-certification mapping or modification.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
PRO = json.loads((BASE / 'protocol.json').read_text(encoding='utf8'))
SOURCE = BASE.parent / 'mixed_airframe_paper_comparison_20261030' / 'safety_interface.py'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != PRO['certifier_sha256']:
    raise RuntimeError('Frozen whole-fleet certifier source changed')

_spec = importlib.util.spec_from_file_location('frozen_global_safety', SOURCE)
frozen_global_safety = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = frozen_global_safety
_spec.loader.exec_module(frozen_global_safety)
model = frozen_global_safety.model
COMMON = frozen_global_safety.COMMON
Certificate = frozen_global_safety.Certificate
certify = frozen_global_safety.certify
advance_model = frozen_global_safety.advance_model
RESERVE = frozen_global_safety.RESERVE

for _key in ('model_sha256', 'model_protocol_sha256', 'extra_lifetime_position_allowance_m'):
    if PRO[_key] != frozen_global_safety.PRO[_key]:
        raise RuntimeError(f'Local repair changed frozen {_key}')

MAX_ROUNDS = int(PRO['repair_rule']['max_rounds'])
MAX_MEMBERS = int(PRO['repair_rule']['max_members'])
MAX_CALLS = int(PRO['repair_rule']['max_certificate_calls'])
if (MAX_ROUNDS, MAX_MEMBERS, MAX_CALLS) != (8, 16, 25):
    raise RuntimeError('Expected the registered 8-round, 16-member, 25-call policy')


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def propose(r, v, capped):
    """Return a fully certified candidate, or None with complete search records.

    ``capped`` is the saved or current speed-capped *pre-envelope* reference.
    Historical states need no new bootstrap here: when this helper fails, the
    caller retains its existing certificate rather than claiming a fresh one.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    capped = np.asarray(capped, dtype=float)
    if any(x.shape != COMMON.shape or not np.isfinite(x).all() for x in (r, v, capped)):
        raise ValueError('Expected finite (80,3) state and capped reference arrays')
    if np.any(np.linalg.norm(capped, axis=1) > model.VMAX + 1e-9):
        raise ValueError('propose expects the speed-capped reference')
    source = capped.copy()
    attempts = []
    seen = []
    calls = 0
    prediction_steps_total = 0
    mask = set()
    rounds = 0

    def attempt(members, branch, round_number):
        nonlocal calls, prediction_steps_total
        members = sorted(members)
        if len(members) > MAX_MEMBERS:
            raise RuntimeError('Internal repair-member budget violation')
        candidate = source.copy()
        if members:
            candidate[members] = COMMON[members]
        for old_candidate, old_proof, old_attempt in seen:
            if np.array_equal(candidate, old_candidate):
                attempts.append({'round': round_number, 'branch': branch,
                    'repair_members': members, 'action': 'skipped_duplicate',
                    'certificate_called': False, 'proof_from_attempt': old_attempt})
                return None, old_proof, False
        if calls >= MAX_CALLS:
            raise RuntimeError('Internal certificate-call budget violation')
        cert, proof = certify(r, v, candidate)
        calls += 1
        prediction_steps_total += int(proof.get('prediction_steps', 0))
        proof = _jsonable(proof)
        old_attempt = len(attempts)
        attempts.append({'round': round_number, 'branch': branch,
            'repair_members': members, 'action': 'certified' if cert is not None else 'rejected',
            'certificate_called': True, 'proof': proof})
        # Any certified attempt exits immediately, so only failed candidates
        # can subsequently occur as duplicates within this one-state search.
        seen.append((candidate, proof, old_attempt))
        return cert, proof, True

    trial, proof, _ = attempt(mask, 'nominal', 0)
    nominal_proof = proof

    def finish(cert, reason, attempted_mask, final_proof):
        members = sorted(attempted_mask) if cert is not None else []
        status = ('nominal_candidate' if not members else 'local_repair_candidate') if cert is not None else 'no_certified_candidate'
        info = {'status': status, 'nominal_trial': nominal_proof, 'trial': final_proof,
            'attempts': attempts, 'repair_members': members,
            'attempted_repair_members': sorted(attempted_mask),
            'repair_rounds': rounds, 'certificate_calls': calls,
            'prediction_steps_total': prediction_steps_total, 'stop_reason': reason}
        if cert is not None:
            # This reports request-level changes, independently of envelope
            # mapping; the certifier remains the authority for accepted input.
            candidate = source.copy()
            if members:
                candidate[members] = COMMON[members]
            info['changed_reference_members'] = np.flatnonzero(np.any(candidate != source, axis=1)).tolist()
        else:
            info['changed_reference_members'] = []
        return cert, _jsonable(info)

    if trial is not None:
        return finish(trial, 'nominal_certified', mask, proof)

    for round_number in range(1, MAX_ROUNDS + 1):
        pair = proof.get('limiting_pair')
        if pair is None:
            return finish(None, 'no_limiting_pair', mask, proof)
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(k, bool) or not isinstance(k, int) or k < 0 or k >= len(COMMON) for k in pair)
                or pair[0] == pair[1]):
            raise RuntimeError('Certifier returned an invalid limiting pair')
        pending = [k for k in pair if k not in mask]
        if not pending:
            return finish(None, 'no_new_member', mask, proof)
        pending.sort(key=lambda k: (float(np.linalg.norm(source[k] - COMMON[k])), k))
        rounds = round_number
        calls_at_round_start = calls

        if len(pending) == 1:
            next_mask = mask | {pending[0]}
            if len(next_mask) > MAX_MEMBERS:
                return finish(None, 'member_budget', mask, proof)
            trial, proof, distinct = attempt(next_mask, 'single_new_endpoint', rounds)
            if trial is not None:
                return finish(trial, 'local_candidate_certified', next_mask, proof)
            if not distinct:
                return finish(None, 'no_distinct_candidate', next_mask, proof)
            mask = next_mask
            continue

        for endpoint in pending:
            single_mask = mask | {endpoint}
            if len(single_mask) > MAX_MEMBERS:
                continue
            trial, single_proof, distinct = attempt(single_mask, 'single_endpoint', rounds)
            if trial is not None:
                return finish(trial, 'local_candidate_certified', single_mask, single_proof)

        next_mask = mask | set(pending)
        if len(next_mask) > MAX_MEMBERS:
            return finish(None, 'member_budget', mask, proof)
        trial, proof, distinct = attempt(next_mask, 'both_endpoints', rounds)
        if trial is not None:
            return finish(trial, 'local_candidate_certified', next_mask, proof)
        if calls == calls_at_round_start:
            return finish(None, 'no_distinct_candidate', next_mask, proof)
        mask = next_mask

    return finish(None, 'round_budget', mask, proof)


class LocalRepairInterface(frozen_global_safety.RetainedBackupInterface):
    """Keep the existing bootstrap, commit, suffix and state-consistency rules."""

    def select(self, step, r, v, held_reference, force_reject=False):
        self.cert.assert_state(r, v)
        if force_reject:
            trial = None
            proof = {'status': 'forced_unit_test_rejection', 'prediction_steps': 0}
            info = {'status': 'no_certified_candidate', 'nominal_trial': proof, 'trial': proof,
                'attempts': [], 'repair_members': [], 'attempted_repair_members': [],
                'changed_reference_members': [], 'repair_rounds': 0,
                'certificate_calls': 0, 'prediction_steps_total': 0,
                'stop_reason': 'forced_unit_test_rejection'}
        else:
            trial, info = propose(r, v, held_reference)
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
        return self.cert.command(), info


RetainedBackupInterface = LocalRepairInterface
