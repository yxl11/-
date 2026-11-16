"""Register geometry, copy immutable sources, prepare independent initial states.

Run once before controller calibration. No controller trajectory is simulated.
"""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[n]='1'
import sys
sys.dont_write_bytecode=True
import ast, hashlib, json, shutil, time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from mission import CFG, reference, roles, groups

BASE=Path(__file__).resolve().parents[1]; ROOT=BASE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()

def continuous_slot_minimum(offsets,ids):
    q0=reference(0)[0]; q1=reference(CFG['duration_s'])[0]
    i,j=np.triu_indices(len(offsets),1)
    start=offsets[i]-offsets[j]+q0[ids[i]]-q0[ids[j]]
    end=offsets[i]-offsets[j]+q1[ids[i]]-q1[ids[j]]
    delta=end-start
    s=np.clip(-np.sum(start*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-300),0,1)
    d=np.linalg.norm(start+s[:,None]*delta,axis=1)
    return float(d.min()) if len(d) else float('inf')

def make_targets(spec):
    rng=np.random.default_rng(spec['seed']); offsets=[]; ids=[]; attempts=[]
    sig=np.asarray(CFG['sigma_m'])
    for role,n in enumerate(CFG['counts']):
        for pair in range(n//2):
            accepted=False
            for k in range(spec['proposals_per_pair']):
                z=rng.normal(size=3)
                if np.any(abs(z)>spec['normal_truncation']):continue
                proposed=[*offsets,sig[role]*z,-sig[role]*z]; proposed_ids=[*ids,role,role]
                if continuous_slot_minimum(np.asarray(proposed),np.asarray(proposed_ids))<spec['minimum_m']:continue
                offsets=proposed; ids=proposed_ids; attempts.append(k+1); accepted=True; break
            if not accepted:raise RuntimeError(f'Target geometry failed registered budget role={role}, pair={pair}; no retuning allowed')
    offsets=np.asarray(offsets); ids=np.asarray(ids)
    return offsets,ids,attempts

def assign_targets(initial_positions,slots=None):
    """One fixed minimum-squared-distance bijection per role; no reassignment."""
    t0=time.perf_counter()
    if slots is None:
        with np.load(BASE/'protocol/targets.npz') as z:slots=z['offsets'].copy()
    mapping=np.empty(len(roles),dtype=int); q0=reference(0)[0]
    for role,g in enumerate(groups):
        costs=np.sum((initial_positions[g,None,:]-(q0[role]+slots[g])[None,:,:])**2,axis=-1)
        row,col=linear_sum_assignment(costs); mapping[g[row]]=g[col]
    return slots[mapping],mapping,time.perf_counter()-t0

def make_initial(seed,spec):
    rng=np.random.default_rng(seed); q0=reference(0)[0]; sigma=np.asarray(CFG['sigma_m'])
    # Draw center offsets once, before particle rejection. Retries never alter them.
    offsets=rng.uniform(spec['centroid_low_m'],spec['centroid_high_m'],size=(3,3))
    for retry in range(spec['whole_state_attempts']):
        points=[]; attempts=[]; valid=True
        for role,n in enumerate(CFG['counts']):
            accepted=[]; count=0
            for k in range(spec['particle_proposals_per_role']):
                count+=1; z=rng.normal(size=3)
                if np.any(abs(z)>spec['normal_truncation']):continue
                x=q0[role]+offsets[role]+spec['shape_scale']*sigma[role]*z
                prev=points+accepted
                if prev and np.min(np.linalg.norm(np.asarray(prev)-x,axis=1))<spec['minimum_m']:continue
                accepted.append(x)
                if len(accepted)==n:break
            if len(accepted)!=n:valid=False;break
            points+=accepted;attempts.append(count)
        if valid:return np.asarray(points),np.zeros((len(roles),3)),offsets,attempts,retry
    raise RuntimeError(f'Initial seed {seed} failed registered geometry budget; stop without changing geometry')

def main():
    pd=BASE/'protocol'; pd.mkdir(parents=True,exist_ok=True)
    registration=pd/'generation_registration.json'
    if registration.exists():raise RuntimeError('Preparation already registered; inspect results before any retry')
    (BASE/'initials').mkdir(exist_ok=True)
    original=ROOT/'output/simulations'
    rels=[f'multigroup_closed_loop_20261012/runs/four_group_s401/source_snapshot/{n}' for n in
        ('backup_governor.py','local_avoidance_governor.py','multigroup.py','run_simulation.py')]
    rels += ['projected_mobility_paired_20261014/src/point_controller.py',
        'projected_mobility_paired_20261014/src/projected_controller.py',
        'unified_density_cooperation_20261014/src/task_density_controller.py']
    vendor=[]
    for rel in rels:
        source=original/rel; target=BASE/'vendor'/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(source):raise RuntimeError('Existing vendor mismatch')
        shutil.copy2(source,target)
        vendor.append({'source_path':str(source),'copied_path':str(target.relative_to(BASE)).replace('\\','/'),'sha256':sha(source)})
    spec={'registered_utc':utc(),'before_any_controller_trajectory':True,
        'target_rule':{'seed':771923,'normal_truncation':2.6,'minimum_m':2.7,'proposals_per_pair':20000,
          'symmetry':'append +sigma*z and -sigma*z together, exact role mean zero',
          'distance_test':'exact minimum of pair displacement affine in lambda in [0,1]',
          'assignment':'fixed per-initial per-role Hungarian minimum squared distance; no in-flight reassignment'},
        'initial_rule':{'calibration_seeds':[901,902,903],'evaluation_seeds':list(range(1001,1011)),
          'centroid_low_m':[-6,-4,-1],'centroid_high_m':[6,4,1],'shape_scale':.68,'normal_truncation':2.6,
          'minimum_m':2.7,'particle_proposals_per_role':20000,'whole_state_attempts':100,'v0_mps':[0,0,0]},
        'config':CFG,'vendor_files':vendor,'source_sha256':{f'src/{n}':sha(BASE/'src'/n) for n in ('mission.py','prepare.py','harness.py')},
        'failure_policy':'Stop if registered generation budget fails; do not change geometry after controller outcomes'}
    write(registration,spec)
    # Registration precedes random geometry generation and any controller output.
    from harness import governor,PController,verify_protocol
    start=time.perf_counter(); offsets,ids,attempts=make_targets(spec['target_rule']); elapsed=time.perf_counter()-start
    np.savez_compressed(pd/'targets.npz',offsets=offsets,roles=ids)
    target_check={'generation_seconds':elapsed,'minimum_continuous_target_spacing_m':continuous_slot_minimum(offsets,ids),
      'role_mean_offsets_m':[offsets[g].mean(axis=0).tolist() for g in groups],'pair_proposals':attempts,
      'finite_density_representation':'hard-core symmetric finite target set; quantify separately, not an exact Gaussian sample',
      'sha256':sha(pd/'targets.npz')}
    write(pd/'target_geometry_checks.json',target_check)
    gov=governor(); initial_files={}
    for seed in spec['initial_rule']['calibration_seeds']+spec['initial_rule']['evaluation_seeds']:
        r,v,center_offsets,counts,retry=make_initial(seed,spec['initial_rule'])
        i,j=gov.pairs; mind=float(np.linalg.norm(r[i]-r[j],axis=1).min())
        okay,margin,_=gov._candidate_safe(r,v,np.zeros_like(r),gov.pairs,np.inf)
        if not okay:raise RuntimeError(f'Seed {seed} zero backup infeasible')
        slots,mapping,assignment_s=assign_targets(r,offsets)
        path=BASE/f'initials/s{seed}.npz'
        np.savez_compressed(path,positions=r,velocities=v,roles=roles,target_slots=slots,target_assignment=mapping,requested_centroid_offsets=center_offsets)
        initial_files[str(seed)]={'sha256':sha(path),'minimum_initial_spacing_m':mind,'zero_backup_margin_m':float(margin),
          'actual_initial_center_error_vectors_m':[(r[g].mean(axis=0)-reference(0)[0][i]).tolist() for i,g in enumerate(groups)],
          'particle_proposals':counts,'whole_state_retry':retry,'assignment_seconds':assignment_s}
    design=ROOT/'output/reviews/heterogeneous_experiment_survey_20261018/three_group_engineering_design/build_design.py'
    tree=ast.parse(design.read_text(encoding='utf-8-sig')); nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('smooth','integral','reference')]
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(design),'exec'),ns)
    tt=np.linspace(0,240,12001)
    err=max(float(np.max(abs(a-b))) for a,b in zip(reference(tt),ns['reference'](tt)))
    model=PController(); z=np.load(BASE/'initials/s901.npz'); f=model.fields(z['positions'],140.)
    mean_g=model.means(f['approx_gradient']); chi=model.sigma**2/model.D*mean_g
    common=np.sum(model.omega[:,None]*mean_g,axis=0)/model.K_sum_diag
    independent=(reference(140.)[1]-CFG['kc_per_s']*chi-CFG['kf_per_s']*(chi-common))[roles]-model.center(f['approx_gradient'])
    formula_error=float(np.max(abs(independent-f['cnom'])))
    # Exact graph derivative finite difference along a fixed direction.
    rng=np.random.default_rng(8871); direction=rng.normal(size=z['positions'].shape); direction/=np.linalg.norm(direction)
    h=1e-5; r=z['positions'];q=reference(140.)[0]
    fd=(model._graph_terms(r+h*direction,q)[0]-model._graph_terms(r-h*direction,q)[0])/(2*h)
    scaled=(model.omega[roles]/model.counts[roles])[:,None]*f['graph_gradient']
    graph_error=abs(fd-float(np.sum(scaled*direction)))
    prechecks={'reference_matches_original_design_max_abs':err,'P_direct_formula_max_abs_mps':formula_error,
      'P_sample_mean_error_mps':f['zero_mean_error'],'zero_excitation_max_mps':float(abs(f['excitation']).max()),
      'graph_directional_derivative_abs_error':graph_error,'continuous_target_spacing_m':target_check['minimum_continuous_target_spacing_m'],
      'passed':err<1e-12 and formula_error<1e-12 and graph_error<1e-7 and f['zero_mean_error']<1e-12,
      'scope':'source/formula/reference/initial geometry checks; not a completed controller trajectory or external identity validation'}
    write(pd/'infrastructure_prechecks.json',prechecks)
    if not prechecks['passed']:raise RuntimeError('Infrastructure precheck failed')
    protocol={'registered_utc':utc(),'registered_before_controller_calibration':True,'status':'mission_and_initials_frozen_external_identity_pending',
      'config':CFG,'generation_registration_sha256':sha(registration),'vendor_files':vendor,
      'mission_source_sha256':{f'src/{n}':sha(BASE/'src'/n) for n in ('mission.py','prepare.py')},
      'initials':{'calibration_seeds':spec['initial_rule']['calibration_seeds'],'evaluation_seeds':spec['initial_rule']['evaluation_seeds'],'files':initial_files},
      'target_sha256':sha(pd/'targets.npz'),'no_internal_baselines_or_ablations':True,
      'formal_run_gate':'at least two valid external methods and frozen_execution.json; no outcome-dependent P or mission changes',
      'metrics':{'sample_dt_s':.5,'states_dt_s':.02,'directions':64,'direction_seed':9127,
        'aggregate':'per-initial trapezoidal mean squared error then square root; report all pairs and arithmetic mean of per-initial results',
        'descriptive_thresholds_only':{'center_m':1.,'relative_m':2.},'unknown_transient_engineering_tolerance':True},
      'P_adaptation':'three-role constructor replaces four-role guard and omega length; exact inherited online formulas preserved, graph1-2-3, new analytic reference, zero injected excitation',
      'no_closed_loop_result_yet':True}
    write(pd/'frozen_mission.json',protocol); (pd/'frozen_mission.sha256').write_text(sha(pd/'frozen_mission.json')+'\n',encoding='utf-8')
    print(json.dumps({'initial_count':len(initial_files),'target_check':target_check,'prechecks':prechecks,'mission_sha256':sha(pd/'frozen_mission.json')},indent=2))

if __name__=='__main__':main()
