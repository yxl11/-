"""Freeze external-only execution after source/interface and adapter checks."""
import os,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,importlib.util
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
OLD=BASE.parent/'mixed_sweep_orbit_sparse_20261101'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    setup=read(BASE/'runner_setup.json')
    for name,expected in setup['identical_common_files'].items():
        assert sha(BASE/name)==sha(OLD/name)==expected,name
    inherited=read(OLD/'precheck_summary.json')
    assert inherited['status']=='PASS' and inherited['closed_loop_gate']
    for name,expected in inherited['sources_sha256'].items():assert sha(ROOT/name)==expected,name
    adapter=read(BASE/'adapter_precheck.json')
    assert adapter['status']=='PASS','External adapter precheck must pass'
    p=read(BASE/'protocol.json')
    assert p['counts']==[10,16,20] and p['seed']==31031 and p['duration_s']==178
    from mission import configuration,initial_state,reference
    from external_adapters import factory
    from shape_feedback import ShapeFeedback
    from p_feedback import load_frozen
    cfg=configuration();r,v=initial_state(p['seed'])
    with np.load(OLD/'runs/P_seed31031/initial.npz') as old:
        assert np.array_equal(r,old['positions']) and np.array_equal(v,old['velocities'])
    _,h,density=load_frozen();metrics=ShapeFeedback(h,density,cfg,reference)
    common={}
    for method in p['external_methods']:
        ctrl=factory(method,cfg,r,p['method_parameters'][method])
        c=ctrl.command(0.,r,v)
        assert c.shape==r.shape and np.isfinite(c).all()
        json.dumps(getattr(ctrl,'identity',{}),allow_nan=False)
        f,s=metrics.fast(r,0.),metrics.slow(r,0.)
        native=ctrl.native_errors(0.,r) if method=='RAS08' else ctrl.native_losses(0.,r)
        assert all(np.isfinite(np.asarray(val)).all() for val in native.values())
        common[method]={'initial_command_finite':True,'nominal_shape':list(c.shape),
             'native_keys':list(native),'task_fields':{k:list(np.asarray(val).shape) for k,val in ctrl.last_fields.items()}}
    names=list(setup['identical_common_files'])+['protocol.json','runner.py','runner_setup.json',
        'external_adapters.py','prepare_external.py','targets.npz','adapter_precheck.json','target_registration.json','target_checks.json','freeze_execution.py']
    files={str((BASE/name).relative_to(ROOT)).replace('\\','/'):sha(BASE/name) for name in names}
    for name in ('external_formation.py','external_mmd.py'):
        path=ROOT/'output/simulations/three_group_moving_comparison_20261023/src'/name
        files[path.relative_to(ROOT).as_posix()]=sha(path)
    result={'status':'PASS','closed_loop_gate':True,'frozen_utc':datetime.now(timezone.utc).isoformat(),
       'inherited_shared_prechecks':{'source':str(OLD/'precheck_summary.json'),
           'scope':'Identical mission, shape, initial states, platform, directed backup and inheritance implementations. Shared checks reused by exact file hash; fresh bootstrap certification performed within each run.'},
       'new_adapter_precheck':str(BASE/'adapter_precheck.json'),'new_initial_controller_interface_checks':common,
       'sources_sha256':files}
    (BASE/'precheck_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'PASS','methods':list(common),'files':len(files)}))
if __name__=='__main__':main()
