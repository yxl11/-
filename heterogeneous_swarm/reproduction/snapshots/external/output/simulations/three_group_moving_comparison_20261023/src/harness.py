"""Common 100-ms task, 20-ms original safety/plant execution and full recording.

No formal evaluation is enabled without root's frozen_execution.json gate.
Command modules implement command(t, positions, velocities), returning an N-by-3
velocity reference; optional last_fields contains numerical task diagnostics.
"""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[n]='1'
import sys
sys.dont_write_bytecode=True
import argparse, hashlib, importlib, json, shutil, time, traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from mission import CFG, config, reference, qref, roles, groups

BASE=Path(__file__).resolve().parents[1]
VENDOR=BASE/'vendor'
FROZEN=VENDOR/'multigroup_closed_loop_20261012/runs/four_group_s401/source_snapshot'
sys.path.insert(0,str(FROZEN)); sys.path.insert(0,str(VENDOR/'projected_mobility_paired_20261014/src'))
from backup_governor import step,segment_clearance_lower_bound,segment_min_distances_exact
from local_avoidance_governor import LocalAvoidanceGovernor
import multigroup
multigroup.references=qref
import point_controller
point_controller.references=qref
from projected_controller import ProjectedController

DIAGNOSTICS=['segment_distance_lower_bound_m','closest_pair_k','closest_pair_l','common_task_scale',
 'separation_pair_count','emergency_zero_reference','accepted_prediction_margin_m','candidate_count',
 'safety_correction_rms_mps','safety_correction_max_mps','safety_changed_aircraft_fraction',
 'speed_capped_aircraft_fraction','backup_grid_intervals']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def utc(): return datetime.now(timezone.utc).isoformat()

class PController(ProjectedController):
    """Unchanged frozen P methods, with a three-role constructor and new q(t)."""
    def __init__(self,cfg=None):
        cfg=config() if cfg is None else cfg
        # Original parent constructor is four-role specific. Generalize arrays
        # here; preserve every online field and Gaussian overlap method verbatim.
        point_controller.FrozenController.__init__(self,cfg,'full')
        self.method='projected'; self.omega=np.asarray(cfg['weights'])
        self.kappa=float(cfg['kappa_m2ps']); self.ell_task=float(cfg['role_kernel_length'])
        self.D=float(cfg['D_m2ps']); self.gradient_order=5
        self.Mc_diag=cfg['kc_per_s']*self.sigma**2/self.D
        self.task_cov_diag=(self.beta[:,None]/self.sigma)**2
        self.projection_gain=cfg['kf_per_s']; self.K_diag=self.D*self.omega[:,None]/self.sigma**2
        self.K_sum_diag=self.K_diag.sum(axis=0)
        self.implementation_metadata={'identity':'P','adaptation':'3 roles, q(t), zero injected excitations',
          'online_methods':'unchanged vendored ProjectedController.fields and PointController.fields',
          'parameters_frozen':True}
        self.last_fields={}
    def command(self,t,r,v):
        self.last_fields=self.fields(r,t)
        return self.last_fields['ctest']

def governor(cfg=None):
    cfg=CFG if cfg is None else cfg
    g=LocalAvoidanceGovernor(dt=cfg['dt_s'],distance=cfg['safe_distance_m'],
        backup_horizon=cfg['backup_horizon_s'],numerical_margin=cfg['numerical_margin_m'],
        levels=cfg['candidate_levels'],separation_buffer=cfg['separation_buffer_m'])
    g._configure(np.asarray(cfg['tau_s'])[roles],np.asarray(cfg['vmax_mps'])[roles],np.asarray(cfg['amax_mps2'])[roles])
    return g

def verify_protocol():
    path=BASE/'protocol/frozen_mission.json'; d=json.loads(path.read_text(encoding='utf-8'))
    if sha(path)!=(BASE/'protocol/frozen_mission.sha256').read_text().strip(): raise RuntimeError('Mission protocol hash changed')
    for item in d['vendor_files']:
        if sha(BASE/item['copied_path'])!=item['sha256']: raise RuntimeError('Vendor changed: '+item['copied_path'])
    for name,h in d['mission_source_sha256'].items():
        if sha(BASE/name)!=h: raise RuntimeError('Mission implementation changed: '+name)
    return d

def load_controller(spec,initial,params=None):
    if spec=='P': return PController()
    module,name=spec.split(':')
    factory=getattr(importlib.import_module(module),name)
    if module=='external_formation' and name=='FormationController':
        from prepare import assign_targets
        slots,mapping,assignment_s=assign_targets(initial)
        model=factory(roles,slots,reference,**(params or {}))
        model.target_assignment=mapping
        model.assignment_seconds=assignment_s
        model.implementation_metadata={'identity':'RenSorensen2008_Eq6_velocity_adaptation',
            'graph':model.graph_metadata,'alpha':model.alpha,'coupling':model.coupling,
            'reference':'all-leader exact-reference specialization','assignment':'once per initial rolewise Hungarian',
            'theorem_transfer':'no inherited constrained plant guarantee'}
        return model
    return factory(config(),initial_positions=initial,**(params or {}))

def run(method,seed,split='smoke',duration=None,controller=None,spec=None,params=None):
    protocol=verify_protocol()
    if split not in ('smoke','calibration','evaluation'): raise ValueError('Unknown split')
    seed_set=protocol['initials']['evaluation_seeds' if split=='evaluation' else 'calibration_seeds']
    if seed not in seed_set: raise ValueError('Seed outside registered split')
    if split=='evaluation':
        gate=json.loads((BASE/'protocol/frozen_execution.json').read_text(encoding='utf-8'))
        if not gate.get('two_valid_external_methods',False): raise RuntimeError('Two valid external methods required')
        if method not in gate['methods']: raise RuntimeError('Method not registered')
        if duration is not None and duration!=CFG['duration_s']: raise RuntimeError('Formal duration is frozen')
        for rel,h in gate['source_sha256'].items():
            if sha(BASE/rel)!=h: raise RuntimeError('Frozen execution source changed: '+rel)
        selected=gate.get('method_specs',{}).get(method)
        if selected is None: raise RuntimeError('Frozen controller identity and selected parameters required')
        if selected['spec']!=(spec or method) or selected['params']!=(params or {}):
            raise RuntimeError('Controller specification differs from frozen selection')
    if split=='calibration' and not (BASE/'protocol/calibration_registration.json').exists():
        raise RuntimeError('Register calibration rules before calibration')
    dur=CFG['duration_s'] if duration is None else float(duration)
    if split=='smoke' and dur>2.: raise RuntimeError('Smoke is limited to 2 s')
    initial=BASE/f'initials/s{seed}.npz'
    if sha(initial)!=protocol['initials']['files'][str(seed)]['sha256']: raise RuntimeError('Initial changed')
    with np.load(initial) as z: r=z['positions'].copy(); v=z['velocities'].copy()
    out=BASE/f'runs/{split}/{method}_s{seed}'
    if out.exists(): raise RuntimeError('Refuse existing result '+str(out))
    out.mkdir(parents=True)
    write(out/'config.json',config())
    shutil.copy2(BASE/'protocol/frozen_mission.json',out/'mission_protocol.json')
    if split=='evaluation':shutil.copy2(BASE/'protocol/frozen_execution.json',out/'execution_protocol.json')
    cfg=config(); dt=cfg['dt_s']; total=round(dur/dt); stride=round(cfg['task_period_s']/dt)
    ms=round(cfg['metric_period_s']/dt); N=len(roles)
    t0=time.perf_counter()
    model=controller if controller is not None else load_controller(spec or method,r,params)
    preprocessing=time.perf_counter()-t0
    gov=governor(); okay,margin,_=gov._candidate_safe(r,v,np.zeros_like(r),gov.pairs,np.inf)
    if not okay: raise RuntimeError('Initial backup infeasible')
    tau=np.asarray(cfg['tau_s'])[roles]; vmax=np.asarray(cfg['vmax_mps'])[roles]; amax=np.asarray(cfg['amax_mps2'])[roles]
    R=np.empty((total+1,N,3)); V=np.empty_like(R); R[0]=r; V[0]=v
    C=np.empty((total,N,3)); Raw=np.empty_like(C); Cap=np.empty_like(C); Acc=np.empty_like(C)
    diagnostics=np.empty((total,len(DIAGNOSTICS))); timing=np.empty((total,2))
    task=[]; metrics=[]; met=multigroup.Metrics(cfg)
    def sample(t):
        item=met.calculate(r,t); item.update(time=t,q=reference(t)[0],centers=np.array([r[g].mean(axis=0) for g in groups])); metrics.append(item)
    sample(0.); completed=0; failure=None; start=time.perf_counter()
    source_hashes={str(p.relative_to(BASE)).replace('\\','/'):sha(p) for p in (BASE/'src').glob('*.py')}
    snapshot=out/'source_snapshot'; snapshot.mkdir()
    for rel in source_hashes:shutil.copy2(BASE/rel,snapshot/Path(rel).name)
    try:
        for k in range(total):
            t=k*dt; nominal_s=0.
            if k%stride==0:
                t0=time.perf_counter(); raw=np.asarray(model.command(t,r,v),dtype=float); nominal_s=time.perf_counter()-t0
                if raw.shape!=(N,3) or not np.isfinite(raw).all(): raise RuntimeError('Invalid nominal command')
                item={'time':t,'raw_reference':raw.copy(),'evaluation_positions':r.copy(),'evaluation_velocities':v.copy()}
                extra={**getattr(model,'last_fields',{}),**getattr(model,'last_diagnostics',{})}
                for key,value in extra.items():
                    if key not in item and np.asarray(value).dtype.kind in 'bifuc': item[key]=np.asarray(value).copy()
                task.append(item)
            t0=time.perf_counter(); command,info=gov.filter(r,v,raw,tau,vmax,amax); safety_s=time.perf_counter()-t0
            rn,vn=step(r,v,command,dt,tau,amax)
            lower=segment_clearance_lower_bound(r,rn,v,vn,dt,gov.pairs); idx=int(np.argmin(lower))
            capped=raw*np.minimum(1.,vmax/np.maximum(np.linalg.norm(raw,axis=1),1e-300))[:,None]
            correction=np.linalg.norm(command-capped,axis=1); cap_delta=np.linalg.norm(capped-raw,axis=1)
            diagnostics[k]=[lower[idx],gov.pairs[0][idx],gov.pairs[1][idx],info['lambda'],info['separation_pairs'],
                info['emergency_zero_reference'],info['min_margin'],info['tested_candidates'],np.sqrt(np.mean(correction**2)),
                correction.max(),np.mean(correction>1e-9),np.mean(cap_delta>1e-9),info['backup_grid_intervals_total']]
            timing[k]=nominal_s,safety_s; Raw[k]=raw; Cap[k]=capped; C[k]=command; Acc[k]=(vn-v)/dt
            completed=k+1; R[completed]=rn; V[completed]=vn; r,v=rn,vn
            if lower[idx]<cfg['safe_distance_m']-1e-9: raise RuntimeError('Segment distance violation')
            if np.max(np.linalg.norm(vn,axis=1)/vmax)>1+1e-9: raise RuntimeError('Speed bound violation')
            if np.max(np.linalg.norm(Acc[k],axis=1)/amax)>1+1e-9: raise RuntimeError('Acceleration bound violation')
            if completed%ms==0: sample(completed*dt)
            if completed%1500==0: print(json.dumps({'method':method,'seed':seed,'t':completed*dt,'wall_s':round(time.perf_counter()-start,2),'min_distance':float(diagnostics[:completed,0].min())}),flush=True)
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'time_s':completed*dt,'traceback':traceback.format_exc()}
        write(out/'failure.json',failure)
    wall=time.perf_counter()-start
    if hasattr(model,'target_assignment'):
        np.savez_compressed(out/'formation_assignment.npz',assignment=model.target_assignment,slot_offsets=model.slot_offsets)
    np.savez_compressed(out/'inner_history.npz',times=np.arange(completed+1)*dt,positions=R[:completed+1],velocities=V[:completed+1],
      raw_held_reference=Raw[:completed],speed_capped_reference=Cap[:completed],accepted_reference=C[:completed],commands=C[:completed],
      acceleration=Acc[:completed],diagnostics=diagnostics[:completed],timing_seconds=timing[:completed],roles=roles)
    keys=set.intersection(*[set(f) for f in task]) if task else set()
    np.savez_compressed(out/'task_history.npz',**{key:np.asarray([f[key] for f in task]) for key in sorted(keys)})
    np.savez_compressed(out/'metrics.npz',**{key:np.asarray([f[key] for f in metrics]) for key in metrics[0]},directions=met.dirs)
    terminal_ok,terminal_margin,_=gov._candidate_safe(r,v,np.zeros_like(r),gov.pairs,np.inf)
    result={'method':method,'seed':seed,'split':split,'status':'completed' if failure is None else 'failed',
      'completed_seconds':completed*dt,'requested_seconds':dur,'wall_seconds':wall,'preprocessing_seconds':preprocessing,
      'assignment_seconds':getattr(model,'assignment_seconds',None),
      'initial_backup_margin_m':float(margin),'terminal_backup_ok':bool(terminal_ok),'terminal_backup_margin_m':float(terminal_margin),
      'minimum_continuous_spacing_lower_bound_m':float(diagnostics[:completed,0].min()) if completed else None,
      'protocol_sha256':sha(BASE/'protocol/frozen_mission.json'),'source_hashes':source_hashes,
      'initial_sha256':sha(initial),'parameters':params or {},'controller_spec':spec or method,'finished_utc':utc(),
      'model_identity':getattr(model,'implementation_metadata',{}),'failure':failure}
    write(out/'summary.json',result)
    write(out/'archive_schema.json',{'states':'row k at k*0.02 s; command row k on [k*dt,(k+1)*dt]',
      'diagnostics_columns':DIAGNOSTICS,'timing_columns':['nominal_update_s_zero_when_held','safety_filter_s'],
      'references':'raw_held_reference is held100ms task reference; no excitation; speed cap then original safety then execution',
      'metrics':'every .5s; shape/density are frozen64direction Gaussian midpoint quantile proxies; not exact Wasserstein'})
    write(out/'manifest.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(result,ensure_ascii=False),flush=True); return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--method',required=True); parser.add_argument('--spec'); parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--split',choices=['smoke','calibration','evaluation'],default='smoke'); parser.add_argument('--duration',type=float)
    parser.add_argument('--params',default='{}'); args=parser.parse_args()
    result=run(args.method,args.seed,args.split,args.duration,spec=args.spec,params=json.loads(args.params))
    sys.exit(0 if result['status']=='completed' else 2)
