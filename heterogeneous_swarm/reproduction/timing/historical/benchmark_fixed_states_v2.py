"""Frozen common-state timing only; never advances or writes a trajectory."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = ROOT/'output/reviews/mixed_sweep_orbit_paired_20261102'
SRC = ROOT/'output/reviews/mixed_sweep_orbit_external_20261101'
GAIN = ROOT/'output/reviews/p_gain_calibration_20261108'
sys.path.insert(0, str(SRC))
from mission import configuration, reference
from p_feedback import load_frozen, source_files
from shape_feedback import ShapeFeedback
from external_adapters import factory
from mission_engine import make_mission_engine

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rel(p):
    return p.resolve().relative_to(ROOT).as_posix()

def dump_new(p, value):
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def source_manifest():
    frozen = read(GAIN/'protocol.json')['source_hashes']
    paths = {ROOT/p for p in frozen if Path(p).suffix in ('.py', '.json')}
    paths.update(source_files())
    paths.update((Path(__file__), OLD/'time_common_states.py', OLD/'common_state_timing.json',
                  OLD/'evaluation_protocol.json', OLD/'campaign_process_results.json', GAIN/'selection.json'))
    return {rel(p): sha(p) for p in sorted(paths)}

def prepare():
    assert not (HERE/'protocol_v2.json').exists(), 'Do not overwrite frozen protocol'
    plan = read(OLD/'evaluation_protocol.json')
    assert plan['evaluation_seeds'] == list(range(33001,33011))
    assert plan['methods'] == ['P','RAS08','MMD19']
    assert plan['method_parameters']['RAS08'] == {'alpha': .4, 'coupling': .4}
    assert plan['method_parameters']['MMD19'] == {'gamma': 20.}
    selected = read(GAIN/'selection.json')
    assert selected['selected_candidate'] == 'B'
    assert selected['parameters'] == {'kc_per_s': .24, 'kf_per_s': .16}
    rows, positions, velocities, input_hashes = [], [], [], {}
    def add(p, k, seed, origin, t, r, v):
        input_hashes[rel(p)] = sha(p)
        assert r.shape == v.shape == (46,3) and np.isfinite(r).all() and np.isfinite(v).all()
        rows.append({'index':len(rows), 'seed':seed, 'origin':origin, 'time_s':t,
                     'source_path':rel(p), 'source_row':k,
                     'position_float64_sha256':hashlib.sha256(np.ascontiguousarray(r,dtype=np.float64).tobytes()).hexdigest(),
                     'velocity_float64_sha256':hashlib.sha256(np.ascontiguousarray(v,dtype=np.float64).tobytes()).hexdigest()})
        positions.append(np.asarray(r,dtype=np.float64).copy()); velocities.append(np.asarray(v,dtype=np.float64).copy())
    for seed in plan['evaluation_seeds']:
        p=OLD/'initials'/f'seed{seed}.npz'
        with np.load(p) as z: add(p,None,seed,'common_initial',0.,z['positions'],z['velocities'])
    seed=plan['figures']['representative_seed']; assert seed==33001
    for method in plan['methods']:
        folder=OLD/'runs'/f'{method}_s{seed}'
        assert read(folder/'summary.json')['status']=='completed'
        input_hashes[rel(folder/'summary.json')]=sha(folder/'summary.json')
        p=folder/'history.npz'
        with np.load(p) as z:
            for t in (24.,86.,138.,178.):
                k=round(t/.02); assert abs(float(z['times'][k])-t)<1e-10
                add(p,k,seed,'archived_P0' if method=='P' else method,t,z['positions'][k],z['velocities'][k])
    assert len(rows)==22
    pack=HERE/'fixed_states.npz'; assert not pack.exists()
    np.savez_compressed(pack,positions=np.stack(positions),velocities=np.stack(velocities))
    manifest={'scope':'10 archived initial states plus 12 archived P0/RAS04/MMD states; identical pool for all current controllers',
              'count':22,'states':rows,'input_sha256':input_hashes,'state_pack_sha256':sha(pack)}
    dump_new(HERE/'state_manifest.json',manifest)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(), 'kind':'fixed-state command and fresh nominal-candidate certificate timing; no closed-loop simulation',
              'methods':['P_B','RAS04','MMD19'], 'parameters':{'P_B':selected['parameters'],'RAS04':{'alpha':.4,'coupling':.4},'MMD19':{'gamma':20.}},
              'states':22,'warmup_per_state_method':1,'measured_repeats_per_state_method':3,
              'method_order':'rotate P_B,RAS04,MMD19 by state index; each method warmup then three measured repeats',
              'thread_environment':{n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
              'timed_scope':'command only; separate certify after common speed cap, with fresh task tangent backup at that state; excludes local-repair search and retained-certificate execution',
              'clock_ms':{'nominal_ms':100.,'fresh_cert_ms':20.},'summary':'median,min,max,p95,over_clock,n for measured calls; all warmup records also retained',
              'state_manifest_sha256':sha(HERE/'state_manifest.json'),'source_sha256':source_manifest(),
              'offline_preprocessing_identity':'same frozen RAS target set and state-specific initial assignment; new assignment times retained separately; original target generation 0.581 s is archived one-off preparation, not retimed'}
    dump_new(HERE/'protocol_v2.json',protocol)
    print(json.dumps({'prepared':True,'states':len(rows),'protocol_sha256':sha(HERE/'protocol_v2.json')}))

def run():
    assert not (HERE/'timing_results.json').exists(), 'Do not overwrite timing results'
    pro=read(HERE/'protocol_v2.json'); manifest=read(HERE/'state_manifest.json')
    assert sha(HERE/'state_manifest.json')==pro['state_manifest_sha256']
    assert sha(HERE/'fixed_states.npz')==manifest['state_pack_sha256']
    checks={**pro['source_sha256'],**manifest['input_sha256']}
    assert all(sha(ROOT/p)==h for p,h in checks.items()), 'Frozen source changed'
    import ctypes, scipy
    pools=[]
    for package in (np,scipy):
        libdir=Path(package.__file__).parent.parent/(package.__name__+'.libs')
        found=False
        for dll in libdir.glob('*blas*.dll'):
            library=ctypes.CDLL(str(dll))
            for symbol in ('openblas_get_num_threads','openblas_get_num_threads64_',
                           'scipy_openblas_get_num_threads','scipy_openblas_get_num_threads64_'):
                try:
                    function=getattr(library,symbol)
                except AttributeError:
                    continue
                function.restype=ctypes.c_int
                count=function()
                pools.append({'package':package.__name__,'library':str(dll),'symbol':symbol,'num_threads':count})
                assert count==1, pools
                found=True
                break
        assert found, 'Native BLAS thread count could not be verified for '+package.__name__
    cfg=configuration(); assert cfg['kc_per_s']==.12 and cfg['kf_per_s']==.08
    cfg.update(pro['parameters']['P_B'])
    _,h,density=load_frozen()
    pcontroller=ShapeFeedback(h,density,cfg,reference).controller
    assert pcontroller.cfg is cfg and pcontroller.base.projection_gain==.16
    assert pcontroller.base.cfg['kc_per_s']==.24 and pcontroller.base.cfg['kf_per_s']==.16
    controllers={}; assignments={}
    with np.load(HERE/'fixed_states.npz') as z:
        positions=z['positions'].copy(); velocities=z['velocities'].copy()
    for row in manifest['states'][:10]:
        seed=row['seed']; r=positions[row['index']]
        ras=factory('RAS08',cfg,r,pro['parameters']['RAS04'])
        controllers[seed]={'P_B':pcontroller,'RAS04':ras,'MMD19':factory('MMD19',cfg,r,pro['parameters']['MMD19'])}
        assignments[str(seed)]=ras.assignment_seconds
    engine=make_mission_engine(cfg['counts'],178.,'tangent'); records=[]
    eventpath=HERE/'timing_records.jsonl'
    with eventpath.open('x',encoding='utf-8') as events:
        for row in manifest['states']:
            i=row['index']; seed=row['seed']; t=row['time_s']; r=positions[i]; v=velocities[i]
            engine.set_direction(reference(t)[1][1])
            methods=pro['methods']; order=methods[i%3:]+methods[:i%3]
            for method in order:
                for rep in range(4):
                    start=time.perf_counter(); raw=controllers[seed][method].command(t,r,v); nominal=time.perf_counter()-start
                    capped=raw*np.minimum(1.,engine.model.VMAX/np.maximum(np.linalg.norm(raw,axis=1),1e-300))[:,None]
                    start=time.perf_counter(); _,info=engine.certify(r,v,capped); certify=time.perf_counter()-start
                    rec={'state_index':i,'seed':seed,'origin':row['origin'],'time_s':t,'method':method,'repeat':rep,'warmup':rep==0,
                         'nominal_ms':nominal*1000,'fresh_cert_ms':certify*1000,'status':info.get('status'),'prediction_steps':info.get('prediction_steps'),
                         'raw_sha256':hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()}
                    records.append(rec); events.write(json.dumps(rec,allow_nan=False)+'\n'); events.flush()
            print(json.dumps({'state_done':i+1,'total':22}),flush=True)
    assert len(records)==264 and all(sha(ROOT/p)==h for p,h in checks.items())
    summary={}
    for method in pro['methods']:
        summary[method]={}
        for key,limit in pro['clock_ms'].items():
            x=np.array([v[key] for v in records if v['method']==method and not v['warmup']])
            assert len(x)==66
            summary[method][key]={'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max()),'p95':float(np.percentile(x,95)),
                                  'over_clock':int((x>limit).sum()),'clock_ms':limit,'n':len(x)}
    result={'status':'PASS','finished_utc':datetime.now(timezone.utc).isoformat(),'scope':pro['timed_scope'],'states':22,
            'parameters':pro['parameters'],'effective_P_gains':{'kc_per_s':pcontroller.base.cfg['kc_per_s'],'kf_per_s':pcontroller.base.projection_gain},
            'summary':summary,'RAS_assignment_seconds_by_seed':assignments,'platform':platform.platform(),'python':sys.version,'threadpools':pools,
            'protocol_sha256':sha(HERE/'protocol_v2.json'),'state_manifest_sha256':sha(HERE/'state_manifest.json'),'state_pack_sha256':sha(HERE/'fixed_states.npz'),
            'records_sha256':sha(eventpath),'source_and_input_hashes_unchanged':True,'measured_records':198,'warmup_records':66,'records':records}
    dump_new(HERE/'timing_results.json',result)
    print(json.dumps({'status':'PASS','summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args=parser.parse_args()
    prepare() if args.prepare else run()
