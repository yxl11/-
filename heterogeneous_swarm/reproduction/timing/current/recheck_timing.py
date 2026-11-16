"""Repeat the frozen 22-state timing protocol with recorded hardware; no trajectories."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT/'output/reviews/p_gain_adoption_20261108/evidence/timing'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_new(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def main():
    assert not (HERE/'protocol_v2.json').exists(), 'Immutable single timing run'
    hardware = subprocess.check_output([
        'powershell', '-NoProfile', '-Command',
        '$cpu=Get-CimInstance Win32_Processor; $machine=Get-CimInstance Win32_ComputerSystem; '
        '[pscustomobject]@{cpu=@($cpu | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors); '
        'memory_bytes=$machine.TotalPhysicalMemory; model=$machine.Model} | ConvertTo-Json -Depth 4'
    ], text=True, encoding='utf-8')
    environment = {'observed_utc':datetime.now(timezone.utc).isoformat(),
                   'hardware':json.loads(hardware), 'python':sys.version,
                   'platform':platform.platform(), 'numpy':np.__version__, 'scipy':scipy.__version__,
                   'thread_environment':{n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    write_new(HERE/'environment.json', environment)
    for name in ('fixed_states.npz', 'state_manifest.json'):
        shutil.copy2(OLD/name, HERE/name)
    pro = read(OLD/'protocol_v2.json')
    pro.update(created_utc=datetime.now(timezone.utc).isoformat(),
               identity='Single repeated fixed-state measurement with contemporaneous CPU metadata; no new closed-loop data',
               predecessor_protocol_sha256=digest(OLD/'protocol_v2.json'),
               environment_sha256=digest(HERE/'environment.json'))
    pro['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()] = digest(Path(__file__))
    write_new(HERE/'protocol_v2.json', pro)
    spec = importlib.util.spec_from_file_location('archived_timing_driver', OLD/'benchmark_fixed_states_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.run()
    current, previous = read(HERE/'timing_results.json'), read(OLD/'timing_results.json')
    # Raw task commands and certifier decisions must remain identical on all frozen inputs.
    keys = ('state_index','seed','origin','time_s','method','repeat','warmup','status','prediction_steps','raw_sha256')
    identity_equal = all(all(a[k] == b[k] for k in keys)
                         for a,b in zip(current['records'], previous['records']))
    assert len(current['records']) == len(previous['records']) == 264 and identity_equal
    write_new(HERE/'checks.json', {'status':'PASS', 'no_trajectory_generated':True,
                                  'same_22_state_pack':digest(HERE/'fixed_states.npz') == digest(OLD/'fixed_states.npz'),
                                  'all_264_command_and_certificate_identities_match':identity_equal,
                                  'environment_sha256':digest(HERE/'environment.json'),
                                  'results_sha256':digest(HERE/'timing_results.json')})
    print(json.dumps({'checks':'PASS','environment':environment,'summary':current['summary']},ensure_ascii=False))

if __name__ == '__main__':
    main()
