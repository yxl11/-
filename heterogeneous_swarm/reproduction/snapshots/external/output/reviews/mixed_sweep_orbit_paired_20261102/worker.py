"""One registered held-out trajectory using the common mixed-airframe interface."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import sys
sys.dont_write_bytecode=True
import json
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import hashlib
from collections import Counter

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'mixed_sweep_orbit_external_20261101'
CALIBRATION = HERE.parent / 'mixed_sweep_orbit_calibration_20261102'
sys.path.insert(0, str(ARCHIVE))
import traceback

import numpy as np
from engine import SOURCE_HASHES,PREVIOUS
from mission_engine import make_mission_engine
from shape_feedback import ShapeFeedback
from shape_models import shape_state
from external_adapters import factory
from p_feedback import BASE,ROOT,CAM,load_module,load_frozen,old_identity,source_files,sha

DIAGNOSTICS=['continuous_distance_lower_m','sample_min_m','retained_backup_flag','candidate_prediction_steps',
             'certificate_generation','certificate_age_before_action_steps','inherited_margin_m',
             'world_cap_agent_fraction','FW_envelope_map_fraction','safety_rewrite_agent_fraction',
             'safety_rewrite_rms_mps','local_repair_flag','local_repair_member_count','certificate_calls']

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def save(path,value):
    partial=path.with_suffix(path.suffix+'.tmp')
    partial.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    os.replace(partial,path)

def stack_records(records):
    return {key:np.stack([x[key] for x in records]) for key in records[0]} if records else {}

def interval_count(duration,step):
    value=round(duration/step)
    if abs(value*step-duration)>1e-10:
        raise ValueError('Expected an integer multiple of the execution period.')
    return value

def check_local_record(engine,capped,mapped,command,info):
    """Verify exactly the command unmodified after certification (including masked rows)."""
    branch=info['policy_branch']
    expected={'nominal':'nominal_candidate','pair_escape':'local_repair_candidate',
              'local_cruise':'local_repair_candidate','retained_backup':'retained_backup'}
    if branch not in expected or expected[branch]!=info['status']:
        raise RuntimeError('Status/branch identity mismatch')
    mask=np.zeros(len(capped),bool)
    members=info['repair_members']
    if members!=sorted(set(members)) or any(i<0 or i>=len(mask) for i in members) or len(members)>16:
        raise RuntimeError('Invalid local member set')
    mask[members]=True
    if bool(members)!=(branch in ('pair_escape','local_cruise')) or not 1<=info['certificate_calls']<=29:
        raise RuntimeError('Invalid search bounds')
    if branch=='retained_backup':
        if not np.array_equal(command,np.broadcast_to(info['retained_certificate_backup_mps'],command.shape)):
            raise RuntimeError('Retained cruise input differs from recorded common reference')
    elif branch=='nominal':
        if not np.array_equal(command,mapped):
            raise RuntimeError('Nominal input differs from mapped external command')
    else:
        trial=capped.copy()
        trial[mask]=(np.asarray(info['premap_rows']) if branch=='pair_escape' else engine.COMMON[mask])
        if not np.array_equal(engine.model.map_command(trial)[0],command):
            raise RuntimeError('Local candidate reconstruction differs from accepted input')
    return mask

class JsonArrayLog:
    """Append records immediately with bounded memory; publish the complete array atomically."""
    def __init__(self,path,initial=()):
        self.path=path
        self.temporary=path.with_suffix('.json.partial')
        self.stream=self.temporary.open('x',encoding='utf8',buffering=65536)
        self.stream.write('[')
        self.count=0
        self.branches=Counter()
        for item in initial:self.append(item)

    def append(self,item):
        if self.count:self.stream.write(',\n')
        self.stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False,separators=(',',':')))
        self.count+=1
        if 'policy_branch' in item:self.branches[item['policy_branch']]+=1

    def publish(self):
        self.stream.write(']\n')
        self.stream.close()
        os.replace(self.temporary,self.path)

def make_control(method, feedback, cfg, initial_positions, params):
    if method == 'P':
        if params:
            raise ValueError('P uses the frozen configuration')
        return feedback.controller
    return factory(method,cfg,initial_positions,params)

def record_fields(method, obs):
    if method == 'P':
        names = ('nu','shape','projection_mean_command','projected_common_displacement',
                 'effective_displacement','zero_mean_error','affine_mean','affine_velocity')
        return {name:np.asarray(obs[name]).copy() for name in names}
    return {'controller_'+key:np.asarray(value).copy() for key,value in obs.items()}

def run(out,pro,method,initial_path,initial_record):
    mission=load_module('new_corridor_mission',BASE/'mission.py')
    cfg=mission.configuration()
    seed=pro['seed'];counts=np.asarray(pro['counts'],int);number=int(counts.sum())
    if list(cfg['counts'])!=list(counts) or not np.array_equal(cfg['sigma_m'],pro['sigma_m']):
        raise ValueError('Mission configuration differs from registration')
    if not np.array_equal(cfg['bandwidth_m'],pro['bandwidth_m']):
        raise ValueError('Physical KDE bandwidths differ from registration')
    duration=float(pro['duration_s']);cutoff=float(pro.get('budget_s',2400))
    backup_check=read(BASE/'backup_geometry_check.json')
    mode=backup_check['selected']
    if backup_check['status']!='PASS' or mode not in ('world','tangent'):raise RuntimeError('Backup interface precheck failed')
    engine=make_mission_engine(counts.tolist(),duration,mode);model=engine.model
    if model.H!=float(pro['dt_s']) or model.H!=.02 or pro['task_period_s']!=.1 or pro['metric_period_s']!=.5:
        raise ValueError('Timing changed')
    files=source_files()+[BASE/x for x in ('protocol.json','precheck_summary.json','mission.py','engine.py','p_feedback.py','runner.py','initial_state.npz','shape_feedback.py','shape_models.py','mission_engine.py','backup_geometry_check.json','external_adapters.py','prepare_external.py','targets.npz','adapter_precheck.json','target_registration.json','target_checks.json')]
    files+=[BASE.parent/name for name in SOURCE_HASHES]
    files.extend(PREVIOUS/n for n in ('p_feedback.py','engine.py'))
    paper=BASE.parent/'ppt_execution_error_derivation_20261014/execution_error_derivation.tex'
    files.append(paper)
    files.extend(CAM/'src'/n for n in ('external_formation.py','external_mmd.py'))
    files.append(BASE.parent/'mixed_airframe_paper_comparison_20261030/calibration_selection.json')
    files += [HERE/'evaluation_protocol.json', HERE/'initial_manifest.json', initial_path,
              CALIBRATION/'selected_parameters.json', HERE/'runner_regression.json',
              HERE/'campaign_registration.json', HERE/'worker.py', HERE/'campaign.py']
    hashes={path.relative_to(ROOT).as_posix():sha(path) for path in files}
    save(out/'input_manifest.json',hashes)
    save(out/'registration.json',{'registered_utc':datetime.now(timezone.utc).isoformat(),
        'before_closedloop_results':True,'method':method,'parameters':pro['method_parameters'][method],
        'backup_mode':mode,'lifetime_reserve_m':engine.lifetime_reserve_m,
        'maximum_step_numerical_m':engine.max_numerical_per_step_m,'seed':seed,'protocol_sha256':sha(BASE/'protocol.json'),
        'cpu_budget_s':cutoff,'wall_watchdog_s':pro['wall_watchdog_s'],
        'evaluation_protocol_sha256':sha(HERE/'evaluation_protocol.json'),'roles':counts.tolist(),'task_period_s':.1,'execution_period_s':model.H,
        'single_run_only':True,'inputs':hashes})
    old_mission,h,density=load_frozen()
    feedback=ShapeFeedback(h,density,cfg,mission.reference)
    if not np.array_equal(feedback.roles,model.ROLES):
        raise RuntimeError('Execution and metric role IDs differ')
    with np.load(initial_path,allow_pickle=False) as frozen:
        initial_r,initial_v=frozen['positions'],frozen['velocities']
    r=np.asarray(initial_r,dtype=float).copy();v=np.asarray(initial_v,dtype=float).copy()
    if r.shape!=(number,3) or v.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError('Unexpected state shape or values')
    np.savez_compressed(out/'initial.npz',positions=r,velocities=v,roles=model.ROLES)
    with np.load(initial_path) as z:
        if not (np.array_equal(r,z['positions']) and np.array_equal(v,z['velocities']) and np.array_equal(model.ROLES,z['roles'])):
            raise ValueError('Bootstrap states differ from prechecked initial file')
    save(out/'configuration.json',cfg)
    controller=make_control(method,feedback,cfg,r,pro['method_parameters'][method])
    identity=dict(getattr(controller,'identity',{}))
    if method=='P':
        identity['identity']='Frozen P with prescribed affine transport and instantaneous role metrics'
    if method=='RAS08':
        if hashlib.sha256(controller.assignment_array.astype(np.int64).tobytes()).hexdigest()!=initial_record['assignment_sha256']:
            raise RuntimeError('Assignment differs from independently prepared seed-specific mapping')
        np.savez_compressed(out/'target_assignment.npz',assigned_z=controller.assigned_z,assignment_array=controller.assignment_array,roles=model.ROLES)
    identity.update(method=method,parameters=pro['method_parameters'][method],external_adapter_sha256=sha(BASE/'external_adapters.py'),scope=pro['scope'])
    def slow_metric(r,t):
        result=feedback.slow(r,t)
        if method=='RAS08':
            native=controller.native_errors(t,r)
            result['native_assigned_position_RMS_by_role_m']=np.asarray(native['assigned_position_rms_by_role_m'])
            result['native_assigned_position_RMS_m']=native['assigned_position_rms_m']
        elif method=='MMD19':
            native=controller.native_losses(t,r)
            result['native_mmd_squared_by_role']=np.asarray(native['mmd_squared_per_role'])
            result['native_mmd_squared']=native['mmd_squared_equal_role_mean']
        return result
    identity['shape_feedback_sha256']=sha(BASE/'shape_feedback.py')
    identity['shape_models_sha256']=sha(BASE/'shape_models.py')
    save(out/'controller_identity.json',identity)
    engine.set_direction(mission.reference(0.,cfg)[1][1])
    gov=engine.RetainedBackupInterface(r,v)
    total=interval_count(duration,model.H); stride=interval_count(.1,model.H); metric=interval_count(.5,model.H)
    R=np.empty((total+1,number,3));V=np.empty_like(R);R[0]=r;V[0]=v
    arrays={key:np.empty((total,number,3)) for key in ('held_reference','speed_capped','envelope_mapped','accepted')}
    mask_history=np.empty((total,number),bool); diagnostics=np.empty((total,len(DIAGNOSTICS))); timings=np.empty((total,3))
    fields=[];events=JsonArrayLog(out/'candidate_events.json')
    gov.ledger=JsonArrayLog(out/'certificate_ledger.json',gov.ledger)
    cpu_start=time.process_time()
    fast=[feedback.fast(r,0.)]; slow=[slow_metric(r,0.)]
    fw_rates=np.zeros(3); rotor_acc=np.zeros(2); fw_inclination=0.
    completed=0; failure=None; began=time.perf_counter(); last_log=began
    try:
        for k in range(total):
            t=k*model.H;nominal_time=0.;pending=None
            if k%stride==0:
                start=time.perf_counter()
                held=np.asarray(controller.command(t,r,v),dtype=float).copy()
                nominal_time=time.perf_counter()-start
                if held.shape!=r.shape or not np.isfinite(held).all():raise RuntimeError('Invalid external task reference')
                obs=controller.last_fields
                S,Sdot=shape_state(t)
                fields.append({'time':t,'positions':r.copy(),'velocities':v.copy(),'raw_reference':held.copy(),
                    'shape_matrix':S,'shape_matrix_derivative':Sdot,
                    **record_fields(method,obs)})
            capped=held*np.minimum(1.,model.VMAX/np.maximum(np.linalg.norm(held,axis=1),1e-300))[:,None]
            mapped,mapping=model.map_command(capped)
            start=time.perf_counter()
            direction=mission.reference(t,cfg)[1][1]
            engine.set_direction(direction)
            command,info=gov.select(k,r,v,capped)
            search_time=time.perf_counter()-start
            mask=check_local_record(engine,capped,mapped,command,info)
            start=time.perf_counter()
            rn,vn,seg=engine.advance_model(r,v,command)
            pending=(rn,vn,command)
            _,_,actual=model.check_segment(r,seg,np.full(number,k*engine.max_numerical_per_step_m))
            if actual['lower_m'] < model.SAFE:
                raise RuntimeError('Predicted execution segment did not clear fixed separation condition')
            speed=np.linalg.norm(vn,axis=1);fq=model.angles(vn[model.FW])
            rates=np.max(np.abs(seg['dq']),axis=0)
            acc=np.linalg.norm(seg['ar'],axis=1)
            if (np.any(speed>model.VMAX+1e-9) or speed[model.FW].min()<model.LO-1e-9
                or np.any(rates>model.RATE+1e-9) or np.abs(fq[:,2]).max()>model.GMAX+1e-9
                or np.any(acc>model.ACC[model.ROT]+1e-9)):
                raise RuntimeError('Airframe speed, rate, inclination or acceleration bound violated')
            # commit checks the model state generated independently against the retained certificate.
            gov.commit(rn,vn)
            exec_time=time.perf_counter()-start
            fw_rates=np.maximum(fw_rates,rates);fw_inclination=max(fw_inclination,float(np.abs(fq[:,2]).max()))
            rotor_acc=np.maximum(rotor_acc,[acc[model.ROLES[model.ROT]==role].max() for role in (1,2)])
            fallback=info['status']=='retained_backup'
            delta=np.linalg.norm(command-mapped,axis=1);mask_history[k]=mask
            diagnostics[k]=[actual['lower_m'],actual['sample_min_m'],fallback,info['trial']['prediction_steps'],
                info['certificate_generation'],info['certificate_age_steps'],info['inherited_margin_m'],
                np.mean(np.linalg.norm(capped-held,axis=1)>1e-9),mapping['changed_FW_count']/len(model.FW),
                np.mean(delta>1e-9),np.sqrt(np.mean(delta**2)),bool(mask.any()),int(mask.sum()),info['certificate_calls']]
            for name,value in zip(arrays,(held,capped,mapped,command)):
                arrays[name][k]=value
            timings[k]=[nominal_time,search_time,exec_time];events.append({'time':t,'step':k,**info})
            completed=k+1;r,v=rn,vn;R[completed]=r;V[completed]=v
            fast.append(feedback.fast(r,completed*model.H))
            if completed%metric==0:slow.append(slow_metric(r,completed*model.H))
            wall=time.perf_counter()
            if completed%250==0 or wall-last_log>=15:
                progress={'completed_seconds':completed*model.H,'wall_s':wall-began,'elapsed_simulation_time':completed*model.H,
                    'minimum_segment_lower_m':float(diagnostics[:completed,0].min()),
                    'fallback_steps':int(diagnostics[:completed,2].sum())}
                save(out/'progress.json',progress);print(json.dumps(progress),flush=True);last_log=wall
            if time.process_time()-cpu_start>cutoff:raise TimeoutError('Registered CPU budget reached')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'completed_seconds':completed*model.H,'traceback':traceback.format_exc()}
        save(out/'failure.json',failure)
        if pending is not None and completed==k:
            np.savez_compressed(out/'uncommitted_failed_step.npz',r=r,v=v,observed_r=pending[0],observed_v=pending[1],accepted=pending[2],step=k)
    # Persist committed intervals before any independent terminal certificate call.
    times=np.arange(completed+1)*model.H
    np.savez_compressed(out/'history.npz',times=times,positions=R[:completed+1],velocities=V[:completed+1],
        roles=model.ROLES,**{key:values[:completed] for key,values in arrays.items()},
        diagnostics=diagnostics[:completed],timing_seconds=timings[:completed],local_repair_mask=mask_history[:completed])
    np.savez_compressed(out/'fast_metrics.npz',times=times[:len(fast)],**stack_records(fast))
    np.savez_compressed(out/'slow_metrics.npz',**stack_records(slow))
    np.savez_compressed(out/'task_history.npz',**stack_records(fields))
    events.publish();gov.ledger.publish()
    gov.save_current(out/'terminal_certificate.npz')
    save(out/'schema.json',{'state':'t=k*h; command sample k applies to [t,t+h); no terminal command',
        'metric_periods_s':{'fast':model.H,'slow':.5,'nominal':.1},'diagnostic_columns':DIAGNOSTICS,
        'timing_columns':['nominal_time_s_zero_when_held','candidate_search_s','independent_step_and_audit_s'],
        'nearest':'Slow arrays: Euclidean nearest other member within its own role, aligned with global member row; 3D and horizontal separately',
        'shape':'64 fixed directions and centered projected quantile errors; density = normalized squared physical KDE L2',
        'region':'Fraction inside the instantaneous 3D task ellipsoid radius 2 per role.',
        'backup':'Trial backup follows the normalized horizontal relay-reference velocity; each accepted certificate stores its own frozen backup direction, including past its explicit prefix.'})
    consistent=True
    try:gov.cert.assert_state(r,v)
    except Exception as exc:
        consistent=False;failure=failure or {'type':type(exc).__name__,'message':str(exc)}
    terminal_info=None
    if consistent:
        engine.set_direction(mission.reference(float(times[-1]),cfg)[1][1])
        _,terminal_info=engine.certify(r,v,engine.COMMON)
        terminal_info['terminal_trial_backup_mps']=engine.COMMON[0].tolist()
    changed=[key for key,hsh in hashes.items() if sha(ROOT/key)!=hsh]
    OK=(completed==total and failure is None and consistent and not changed)
    summary={'status':'completed' if OK else 'stopped','method':method,'seed':seed,'counts':counts.tolist(),
        'completed_seconds':float(times[-1]),'requested_seconds':duration,'completed_steps':completed,
        'wall_seconds':time.perf_counter()-began,'failure':failure,'inputs_unchanged':not bool(changed),'changed_inputs':changed,
        'controller_identity_check':identity,'actual_state_consistency':'bitwise equal at every committed step' if consistent else 'mismatch',
        'bootstrap':gov.bootstrap,'retained_terminal_certificate':gov.cert.summary(),'terminal_fresh_backup_certificate':terminal_info,
        'minimum_continuous_distance_lower_m':float(diagnostics[:completed,0].min()) if completed else None,
        'fallback_steps':int(diagnostics[:completed,2].sum()),'local_repair_steps':int(diagnostics[:completed,11].sum()),
        'total_certificate_calls':int(diagnostics[:completed,13].sum()),
        'cpu_seconds':time.process_time()-cpu_start,
        'policy_branch_steps':{name:events.branches[name]
                               for name in ('nominal','pair_escape','local_cruise','retained_backup')},
        'FW_speed_min_mps':float(np.linalg.norm(V[:completed+1,model.FW],axis=2).min()),
        'FW_speed_max_mps':float(np.linalg.norm(V[:completed+1,model.FW],axis=2).max()),
        'FW_inclination_max_deg':fw_inclination/model.RAD,'FW_rate_max_mps2_deg_s':[float(fw_rates[0]),*(fw_rates[1:]/model.RAD).tolist()],
        'rotor_acceleration_max_mps2':rotor_acc.tolist(),'real_time_delays_simulated':False,
        'backup_mode':mode,'lifetime_reserve_m':engine.lifetime_reserve_m,
        'controller_identity_scope':identity['identity'],
        'scope':pro['scope']}
    save(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
    return OK

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',required=True,choices=['P','RAS08','MMD19'])
    parser.add_argument('--seed',required=True,type=int)
    args=parser.parse_args()
    spec=read(HERE/'evaluation_protocol.json')
    registered=read(HERE/'campaign_registration.json')
    for filename,expected in registered['frozen_files'].items():
        if sha(ROOT/filename)!=expected:raise ValueError('Frozen input changed: '+filename)
    if args.seed not in spec['evaluation_seeds']:raise ValueError('Seed outside held-out registration')
    initials=read(HERE/'initial_manifest.json')
    assert initials['status']=='PASS'
    record=initials['files'][str(args.seed)]
    initial_path=HERE/record['path']
    assert sha(initial_path)==record['sha256']
    params=spec['method_parameters'][args.method]
    pro=read(BASE/'protocol.json')
    pro.update(seed=args.seed,scope='Registered ten-seed paired evaluation',budget_s=spec['compute_budget']['per_job_cpu_seconds'],
               wall_watchdog_s=spec['compute_budget']['wall_watchdog_seconds'])
    pro['method_parameters'][args.method]=params
    out=HERE/'runs'/f"{args.method}_s{args.seed}"
    out.mkdir(parents=True,exist_ok=False)
    try:
        ok=run(out,pro,args.method,initial_path,record)
    except Exception as exc:
        failure={'reason':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc()}
        save(out/'stopped_exception.json',failure)
        if not (out/'summary.json').exists():
            save(out/'summary.json',{'status':'stopped','method':args.method,'seed':args.seed,'failure':failure})
        raise
    return 0 if ok else 2

if __name__=='__main__':raise SystemExit(main())
