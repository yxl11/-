"""Resource-aware scheduler for the thirty registered held-out trajectories."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
SOURCE = BASE.parent / 'mixed_sweep_orbit_external_20261101'
CALIBRATION = BASE.parent / 'mixed_sweep_orbit_calibration_20261102'
MAX_WORKERS = 9
GIB = 1024 ** 3
MIB = 1024 ** 2

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)

def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
          ('physical_total','physical_free','commit_total','commit_free','virtual_total','virtual_free','extended')]

def memory():
    mem=MemoryStatus()
    mem.length=ctypes.sizeof(mem)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
        raise ctypes.WinError()
    return {k:int(getattr(mem,k)) for k in ('physical_total','physical_free','commit_total','commit_free')}

def register():
    directory=BASE/'runs'
    if directory.exists() and any(directory.iterdir()):raise FileExistsError('Existing evidence in runs')
    if (BASE/'campaign_registration.json').exists():raise FileExistsError('Campaign already registered')
    p=read(BASE/'evaluation_protocol.json')
    initial=read(BASE/'initial_manifest.json')
    assert initial['status']=='PASS' and initial['protocol_sha256']==sha(BASE/'evaluation_protocol.json')
    assert read(BASE/'runner_regression.json')['status']=='PASS'
    assert read(BASE/'runner_regression.json')['worker_sha256']==sha(BASE/'worker.py')
    paths=[BASE/n for n in ('evaluation_protocol.json','initial_manifest.json','campaign.py','runner_regression.py','runner_regression.json','worker.py')]
    selection=CALIBRATION/'selected_parameters.json'
    assert sha(selection)==p['selected_parameters_sha256']
    assert read(CALIBRATION/'delivery_check.json')['status']=='PASS'
    params=read(selection)['methods']
    assert p['method_parameters']=={'P':{}, **{m:params[m]['selected_parameters'] for m in ('RAS08','MMD19')}}
    paths.append(selection)
    assert read(SOURCE/'adapter_precheck.json')['status']=='PASS'
    assert read(SOURCE/'precheck_summary.json')['closed_loop_gate']
    for rel,checksum in read(SOURCE/'artifact_manifest.json')['files_sha256'].items():
        assert sha(SOURCE/rel)==checksum, rel
    for rel,digest in read(SOURCE/'precheck_summary.json')['sources_sha256'].items():
        assert sha(ROOT/rel)==digest,rel
        paths.append(ROOT/rel)
    for seed in p['evaluation_seeds']:
        x=initial['files'][str(seed)];target=BASE/x['path']
        assert sha(target)==x['sha256']
        paths.append(target)
    assert p['compute_budget']['closed_loops']==30 and p['evaluation_seeds']==list(range(33001,33011))
    assert set(p['evaluation_seeds']).isdisjoint([31031,32001,32002,32003])
    assert p['methods']==['P','RAS08','MMD19']
    jobs=[{'id':f'{m}_s{seed}','method':m,'seed':seed,'params':p['method_parameters'][m]}
          for seed in p['evaluation_seeds'] for m in p['methods']]
    if shutil.disk_usage(BASE).free<6*GIB:raise OSError('Full archived traces require at least 6 GiB free')
    frozen={x.relative_to(ROOT).as_posix():sha(x) for x in paths}
    out={'registered_utc':utc(),'before_any_evaluation_results':True,'jobs':jobs,
        'frozen_files':frozen,'memory_before':memory(),'free_disk_bytes':shutil.disk_usage(BASE).free,
        'max_workers':MAX_WORKERS,'cpu_logical':os.cpu_count(),'cpu_physical_from_local_CIM':16,
        'admission':'Stagger starts, up to 9 workers; require 720 MiB physical and 2 GiB of commit available before adding another concurrent job.',
        'python':sys.executable,'version':sys.version,'blas_threads':1,'logical_sim_duration_s':178,
        'timeout_semantics':p['compute_budget'],'timing_scope':'Parallel-run times are operational diagnostics only.'}
    (BASE/'logs').mkdir(exist_ok=True)
    (BASE/'runs').mkdir(exist_ok=True)
    save(BASE/'campaign_registration.json',out)
    return p,jobs

def main():
    protocol,queue=register()
    active={};completed=[];watchdog=protocol['compute_budget']['wall_watchdog_seconds']
    logs=BASE/'logs';last_start=-1e9;last_update=-1e9
    env=os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while queue or active:
        now=time.monotonic();mem=memory();changed=False
        for name,(proc,log,started,job) in list(active.items()):
            code=proc.poll()
            timed_out=code is None and now-started>watchdog
            if timed_out:
                proc.kill();code=proc.wait()
            if code is None:continue
            log.close()
            run=BASE/'runs'/name
            summary=read(run/'summary.json') if (run/'summary.json').is_file() else None
            item=dict(job,exit_code=code,ended_utc=utc(),wall_seconds=now-started,
                      status=summary['status'] if summary else ('watchdog' if timed_out else 'process_error'))
            completed.append(item);save(logs/(name+'_process.json'),item)
            print(json.dumps(item),flush=True)
            del active[name];changed=True
        if queue and len(active)<MAX_WORKERS and now-last_start>=2.0 and (
            not active or (mem['physical_free']>720*MIB and mem['commit_free']>2*GIB)):
            job=queue.pop(0);name=job['id']
            file=(logs/(name+'_console.txt')).open('x',encoding='utf8')
            cmd=[sys.executable,str(BASE/'worker.py'),'--method',job['method'],'--seed',str(job['seed'])]
            proc=subprocess.Popen(cmd,stdout=file,stderr=subprocess.STDOUT,env=env,cwd=ROOT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            active[name]=(proc,file,now,job);last_start=now;changed=True
            print(json.dumps({'started':name,'pid':proc.pid,'utc':utc()}),flush=True)
        if changed or now-last_update>=12:
            status={'checked_utc':utc(),'finished':completed,'active':{name:proc.pid for name,(proc,_,_,_) in active.items()},
                    'queue':[item['id'] for item in queue],'memory':mem}
            save(BASE/'campaign_progress.json',status)
            last_update=now
        if queue or active:time.sleep(1.)
    assert len(completed)==30 and len({j['id'] for j in completed})==30
    save(BASE/'campaign_process_results.json',{'closed_loops_attempted':30,'records':completed,'completed_utc':utc()})
    return int(any(x['status'] not in ('completed','stopped') for x in completed))

if __name__=='__main__':raise SystemExit(main())
