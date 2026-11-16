"""Construct and inspect an original frozen controller; no model propagation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,sys,json,hashlib,importlib.util
from pathlib import Path
from package_common import long_path
sys.dont_write_bytecode=True
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--method',required=True,choices=['P','RAS04','MMD19']);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    root=long_path(a.snapshot);paired=root/'output/reviews/mixed_sweep_orbit_paired_20261102';gain=root/'output/reviews/p_gain_calibration_20261108'
    wdir=gain/'code' if a.method=='P' else paired
    sys.path.insert(0,str(wdir))
    import worker
    if a.method=='P':
        import run_candidate
        protocol=read(gain/'protocol.json');run_candidate.verify_sources(protocol)
        selected=read(gain/'selection.json');assert selected['selected_candidate']=='B'
        assert selected['protocol_sha256']==sha(gain/'protocol.json')
        params=protocol['candidates']['B'];method='P';assert a.seed in protocol['evaluation_seeds']
        expected=protocol['initials']['evaluation'][str(a.seed)];assert sha(root/expected['path'])==expected['sha256']
    else:
        registered=read(paired/'campaign_registration.json')
        for name,h in registered['frozen_files'].items():assert sha(root/name)==h,name
        protocol=read(paired/'evaluation_protocol.json');method='RAS08' if a.method=='RAS04' else a.method
        params=protocol['method_parameters'][method];assert a.seed in protocol['evaluation_seeds']
    manifest=read(paired/'initial_manifest.json');record=manifest['files'][str(a.seed)]
    initial=paired/record['path'];assert sha(initial)==record['sha256']
    mission=worker.load_module('snapshot_mission',worker.BASE/'mission.py')
    cfg=worker.make_configuration(mission,'P',params) if a.method=='P' else mission.configuration()
    _,h,density=worker.load_frozen()
    feedback=worker.ShapeFeedback(h,density,cfg,mission.reference)
    with np.load(initial,allow_pickle=False) as z:r=z['positions'].copy();v=z['velocities'].copy();roles=z['roles'].copy()
    controller=worker.make_control(method,feedback,cfg,r,params)
    engine=worker.make_mission_engine(cfg['counts'],178.,'tangent')
    np.testing.assert_array_equal(engine.model.ROLES,roles)
    assert engine.model.H==.02 and list(cfg['counts'])==[10,16,20]
    if a.method=='P':assert cfg['kc_per_s']==.24 and cfg['kf_per_s']==.16 and controller.base.projection_gain==.16
    if a.method=='RAS04':assert hashlib.sha256(controller.assignment_array.astype(np.int64).tobytes()).hexdigest()==record['assignment_sha256']
    # This checks controller construction and a single nominal evaluation only.
    # No advance_model, certificate construction, worker.run or timestep loop.
    raw=controller.command(0.,r,v);assert raw.shape==(46,3) and np.isfinite(raw).all()
    print(json.dumps({'status':'PASS','method':a.method,'seed':a.seed,'parameters':params,
        'source_root':str(root),'initial_sha256':sha(initial),'raw_command_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
        'historical_manuscript_sha256':sha(root/'output/reviews/ppt_execution_error_derivation_20261014/execution_error_derivation.tex'),
        'controller_constructed':True,'initial_assignment_verified':a.method=='RAS04','state_propagation_steps':0,
        'closed_loop_executed':False},ensure_ascii=False))
if __name__=='__main__':main()
