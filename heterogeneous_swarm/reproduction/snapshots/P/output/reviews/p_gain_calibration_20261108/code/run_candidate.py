"""Run exactly one registered P gain candidate through the derived worker.

The root campaign owns scheduling and selection. This entry point never chooses
gains or initial states and refuses to overwrite an existing run.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import worker

CODE = Path(__file__).resolve().parent
STUDY = CODE.parent
ROOT = STUDY.parents[2]
PAIRED = STUDY.parent / 'mixed_sweep_orbit_paired_20261102'
CALIBRATION = STUDY.parent / 'mixed_sweep_orbit_calibration_20261102'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def verify_sources(protocol):
    if not protocol.get('source_hashes'):
        raise ValueError('The protocol must freeze source_hashes before a run')
    for name, expected in protocol['source_hashes'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Frozen study input changed: '+name)
    static = read(STUDY/'checks/worker_static.json')
    regression = read(STUDY/'checks/regression.json')
    if static['status'] != 'PASS' or regression['status'] != 'PASS':
        raise ValueError('Static and baseline regression checks must pass')
    if static['derived_worker_sha256'] != sha(CODE/'worker.py') or regression['derived_worker_sha256'] != sha(CODE/'worker.py'):
        raise ValueError('Worker differs from the checked derivative')
    if regression['entry_sha256'] != sha(Path(__file__)):
        raise ValueError('Entry point differs from baseline regression record')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['calibration', 'evaluation'])
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    protocol_path = STUDY/'protocol.json'
    protocol = read(protocol_path)
    verify_sources(protocol)
    if args.candidate not in protocol['candidates']:
        raise ValueError('Candidate is not registered')
    if args.seed not in protocol[f'{args.stage}_seeds']:
        raise ValueError('Seed is outside this stage')
    if protocol['duration_s'] != 178 or protocol['counts'] != [10,16,20]:
        raise ValueError('Registered task duration or role counts changed')
    for key, expected in [('task_period_s',.1),('dt_s',.02),('metric_period_s',.5)]:
        if protocol[key] != expected:
            raise ValueError('Frozen clock changed: '+key)
    params = protocol['candidates'][args.candidate]
    if set(params) != {'kc_per_s','kf_per_s'}:
        raise ValueError('Only kc_per_s and kf_per_s may vary')
    initial_source = CALIBRATION if args.stage == 'calibration' else PAIRED
    manifest_path = initial_source/'initial_manifest.json'
    initials = read(manifest_path)
    if initials['status'] != 'PASS':
        raise ValueError('Initial precheck is not PASS')
    initial_record = initials['files'][str(args.seed)]
    initial_path = initial_source/initial_record['path']
    if sha(initial_path) != initial_record['sha256']:
        raise ValueError('Registered initial state changed')
    registered_initial = protocol['initials'][args.stage][str(args.seed)]
    if registered_initial != {'path':relative(initial_path), 'sha256':sha(initial_path)}:
        raise ValueError('Study initial registration differs from the frozen source manifest')
    additional = [relative(STUDY/'checks/worker_static.json'), relative(STUDY/'checks/regression.json')]
    selection_hash = None
    if args.stage == 'evaluation':
        selection_path = STUDY/'selection.json'
        selection = read(selection_path)
        if selection['selected_candidate'] != args.candidate:
            raise ValueError('Evaluation must use the frozen selected_candidate')
        if selection.get('protocol_sha256',sha(protocol_path)) != sha(protocol_path):
            raise ValueError('Selection belongs to a different study protocol')
        selection_hash = sha(selection_path)
        additional.append(relative(selection_path))
    pro = worker.read(worker.BASE/'protocol.json')
    pro.update(seed=args.seed, duration_s=178, counts=[10,16,20],
               scope=protocol.get('scope','Registered P gain calibration and frozen evaluation')+'; '+args.stage,
               budget_s=protocol['cpu_budget_s'], wall_watchdog_s=protocol['wall_watchdog_s'])
    pro['method_parameters']['P'] = dict(params)
    pro['gain_study'] = {'stage':args.stage,'candidate_id':args.candidate,
                         'protocol_path':relative(protocol_path),'initial_manifest_path':relative(manifest_path),
                         'additional_input_paths':additional,'selection_sha256':selection_hash,
                         'source_evaluation_protocol_role':'Execution provenance only; this study protocol defines gains and initial-state split'}
    out = STUDY/'runs'/args.stage/f'{args.candidate}_s{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    worker.save(out/'experiment_identity.json', {
        'registered_utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,
        'candidate_id':args.candidate,'method':'P','parameters':params,'seed':args.seed,
        'protocol_sha256':sha(protocol_path),'selection_sha256':selection_hash,
        'entry_sha256':sha(Path(__file__)),'derived_worker_sha256':sha(CODE/'worker.py'),
        'source_worker_sha256':sha(PAIRED/'worker.py'),'initial_sha256':sha(initial_path),
        'original_initial_path':relative(initial_path),'initial_manifest_sha256':sha(manifest_path),
        'frozen_density_parameters':{'D_m2ps':2.5,'kappa_m2ps':10.0},'local_repair_enabled':True})
    try:
        ok = worker.run(out,pro,'P',initial_path,initial_record)
        verify_sources(protocol)
    except Exception as exc:
        failure = {'reason':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()}
        worker.save(out/'entry_failure.json',failure)
        if not (out/'summary.json').exists():
            worker.save(out/'summary.json',{'status':'stopped','method':'P','seed':args.seed,'failure':failure})
        raise
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
