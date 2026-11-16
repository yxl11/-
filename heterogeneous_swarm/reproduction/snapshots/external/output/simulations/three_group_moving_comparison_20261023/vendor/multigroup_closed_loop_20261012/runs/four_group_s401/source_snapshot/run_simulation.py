"""Archive every executed inner state and command; never generate animation."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse, csv, hashlib, json, sys, time, traceback, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from multigroup import Controller, Metrics, default_config, initial_state, references
from backup_governor import step, segment_clearance_lower_bound, segment_min_distances_exact
from local_avoidance_governor import LocalAvoidanceGovernor

ROOT=Path(__file__).resolve().parents[1]

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py') if p.name in ['multigroup.py','run_simulation.py','backup_governor.py','local_avoidance_governor.py']}

def run(cfg,method,seed,out,initial=None,progress=True):
    out=Path(out)
    if (out/'summary.json').exists(): raise RuntimeError('run already exists: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    source=hashes(); write_json(out/'config.json',cfg)
    (out/'source_snapshot').mkdir(exist_ok=True)
    for name in source:
        shutil.copy2(Path(__file__).parent/name,out/'source_snapshot'/name)
    model=Controller(cfg,method); metrics=Metrics(cfg)
    dt=cfg['dt_s']; total=int(round(cfg['duration_s']/dt)); N=len(model.roles)
    task_stride=int(round(cfg['task_period_s']/dt)); metric_stride=int(round(cfg['metric_period_s']/dt))
    assert task_stride>=1 and metric_stride>=1 and abs(task_stride*dt-cfg['task_period_s'])<1e-10
    if initial is None: r,v=initial_state(cfg,seed)
    else:
        with np.load(initial) as z: r=z['positions'].copy(); v=z['velocities'].copy()
    np.savez_compressed(out/'initial.npz',positions=r,velocities=v,roles=model.roles)
    gov=LocalAvoidanceGovernor(dt=dt,distance=cfg['safe_distance_m'],backup_horizon=cfg['backup_horizon_s'],
                              numerical_margin=cfg['numerical_margin_m'],levels=cfg['candidate_levels'],
                              separation_buffer=cfg['separation_buffer_m'])
    gov._configure(model.tau,model.vmax,model.amax)
    okay,initial_margin,_=gov._candidate_safe(r,v,np.zeros_like(r),gov.pairs,np.inf)
    if not okay: raise RuntimeError('initial zero-reference backup is infeasible')
    states_r=np.empty((total+1,N,3)); states_v=np.empty_like(states_r)
    commands=np.empty((total,N,3)); stat=np.empty((total,13)); timing=np.empty((total,2))
    states_r[0]=r; states_v[0]=v
    task=[]; samples=[]; rows=[]; completed=0; failure=None; start=time.perf_counter()
    min_distance=np.inf; max_speed_ratio=0.; max_acc_ratio=0.; metric_times=[]
    def record_metrics(t):
        nonlocal rows
        item=metrics.calculate(r,t); samples.append(item); metric_times.append(float(t))
        for i in range(model.M):
            rows.append({'time_s':float(t),'group':i+1,'center_error_m':float(item['center_error'][i]),
                         'density_proxy_m':float(item['density_proxy'][i]),'shape_proxy_m':float(item['shape_proxy'][i])})
    record_metrics(0.)
    try:
        for k in range(total):
            t=k*dt; task_time=0.
            if k%task_stride==0:
                t0=time.perf_counter(); fields=model.fields(r,t); task_time=time.perf_counter()-t0
                fields['time']=t; fields['evaluation_positions']=r.copy(); task.append(fields)
                if fields['zero_mean_error']>1e-10 and method!='no_center':
                    raise RuntimeError('nominal sample-mean identity failed')
                if fields['shape_excitation_mean_error']>1e-12: raise RuntimeError('excitation centering failed')
            ts=time.perf_counter()
            c,info=gov.filter(r,v,fields['ctest'],model.tau,model.vmax,model.amax)
            safety_time=time.perf_counter()-ts
            rn,vn=step(r,v,c,dt,model.tau,model.amax)
            lower=segment_clearance_lower_bound(r,rn,v,vn,dt,gov.pairs)
            close=int(np.argmin(lower)); minlower=float(lower[close])
            capped=fields['ctest']*np.minimum(1.,model.vmax/np.maximum(np.linalg.norm(fields['ctest'],axis=1),1e-300))[:,None]
            correction=np.linalg.norm(c-capped,axis=1)
            cap_delta=np.linalg.norm(capped-fields['ctest'],axis=1)
            speed_ratio=float(np.max(np.linalg.norm(vn,axis=1)/model.vmax))
            acc_ratio=float(np.max(np.linalg.norm((vn-v)/dt,axis=1)/model.amax))
            stat[k]=[minlower,gov.pairs[0][close],gov.pairs[1][close],info['lambda'],
                     info['separation_pairs'],info['emergency_zero_reference'],info['min_margin'],
                     info['tested_candidates'],np.sqrt(np.mean(correction**2)),np.max(correction),
                     np.mean(correction>1e-9),np.mean(cap_delta>1e-9),info['backup_grid_intervals_total']]
            timing[k]=[task_time,safety_time]; commands[k]=c
            states_r[k+1]=rn; states_v[k+1]=vn; completed=k+1
            min_distance=min(min_distance,minlower); max_speed_ratio=max(max_speed_ratio,speed_ratio); max_acc_ratio=max(max_acc_ratio,acc_ratio)
            r,v=rn,vn
            if minlower<cfg['safe_distance_m']-1e-9: raise RuntimeError('continuous-segment spacing violation')
            if speed_ratio>1+1e-9 or acc_ratio>1+1e-9: raise RuntimeError('capability violation')
            if completed%metric_stride==0: record_metrics(completed*dt)
            if progress and completed%int(round(30./dt))==0:
                print(json.dumps({'method':method,'seed':seed,'t':round(completed*dt,3),'wall_s':round(time.perf_counter()-start,1),'min_spacing_lb':round(min_distance,6)},ensure_ascii=False),flush=True)
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'time_s':completed*dt}
        write_json(out/'failure.json',failure)
    wall=time.perf_counter()-start
    states_r=states_r[:completed+1]; states_v=states_v[:completed+1]
    commands=commands[:completed]; stat=stat[:completed]; timing=timing[:completed]
    # Commands row k is applied on [k*dt,(k+1)*dt]; states include both endpoints.
    np.savez_compressed(out/'inner_history.npz',times=np.arange(completed+1)*dt,positions=states_r,
                        velocities=states_v,commands=commands,diagnostics=stat,timing_seconds=timing,roles=model.roles)
    task_keys=[key for key in task[0] if key not in ('zero_mean_error','shape_excitation_mean_error')]
    task_archive={key:np.asarray([f[key] for f in task]) for key in task_keys}
    task_archive['zero_mean_error']=np.array([f['zero_mean_error'] for f in task])
    task_archive['shape_excitation_mean_error']=np.array([f['shape_excitation_mean_error'] for f in task])
    np.savez_compressed(out/'task_history.npz',**task_archive)
    sample_archive={key:np.asarray([f[key] for f in samples]) for key in samples[0]}
    sample_archive['times']=np.array(metric_times)
    np.savez_compressed(out/'metrics.npz',**sample_archive,directions=metrics.dirs)
    with (out/'group_metrics.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    exact=None
    if completed:
        # The candidate for the smallest conservative lower bound is independently
        # checked by polynomial roots. Global lower bound remains the conservative one.
        closest=int(np.argmin(stat[:,0])); i,j=stat[closest,1:3].astype(int)
        exact=float(segment_min_distances_exact(states_r[closest],states_r[closest+1],states_v[closest],states_v[closest+1],dt,(np.array([i]),np.array([j])))[0])
    thresholds=cfg['thresholds']; edges=np.array(cfg['B'])>0
    max_relative=np.array([np.max(x['relative_error'][edges]) for x in samples])
    qualified=(sample_archive['center_error'].max(axis=1)<=thresholds['center_m'])&(max_relative<=thresholds['relative_m'])&(sample_archive['shape_proxy'].max(axis=1)<=thresholds['shape_proxy_m'])
    consecutive=0.; first_hold=None
    for idx,good in enumerate(qualified):
        consecutive=consecutive+cfg['metric_period_s'] if good and idx>0 and qualified[idx-1] else 0.
        if consecutive>=thresholds['hold_s'] and first_hold is None: first_hold=float(metric_times[idx])
    final_indices=sample_archive['times']>=cfg['duration_s']-thresholds['hold_s']-1e-9
    final_hold=bool(completed==total and np.any(final_indices) and np.all(qualified[final_indices]))
    overlap=0.; overlap_formation=0.; overlap_density=0.; overlap_safety=0.
    if completed:
        task_idx=np.arange(completed)//task_stride
        form=np.sqrt(np.mean(np.sum(task_archive['formation']**2,axis=-1),axis=-1))[task_idx]
        density=np.sqrt(np.mean(np.sum(task_archive['cross']**2,axis=-1),axis=-1))[task_idx]
        active=(np.arange(completed)*dt>=180.)&(np.arange(completed)*dt<240.)
        threshold=thresholds['activation_mps']; aa=form>threshold; bb=density>threshold; cc=stat[:,8]>threshold
        overlap=float(np.sum(active&aa&bb&cc)*dt)
        overlap_formation=float(np.sum(active&aa)*dt); overlap_density=float(np.sum(active&bb)*dt); overlap_safety=float(np.sum(active&cc)*dt)
    terminal_ok,terminal_margin,_=gov._candidate_safe(r,v,np.zeros_like(r),gov.pairs,np.inf)
    summary={'status':'completed' if failure is None else 'failed','method':method,'seed':seed,
             'completed_seconds':completed*dt,'requested_seconds':cfg['duration_s'],'wall_seconds':wall,
             'N':N,'M':model.M,'initial_backup_margin_m':float(initial_margin),'terminal_backup_ok':bool(terminal_ok),
             'terminal_backup_margin_m':float(terminal_margin),'minimum_continuous_spacing_lower_bound_m':float(min_distance),
             'exact_distance_of_lowest_bound_segment_pair_m':exact,'max_speed_fraction':max_speed_ratio,'max_acceleration_fraction':max_acc_ratio,
             'max_nominal_mean_error_mps':float(task_archive['zero_mean_error'].max()),
             'max_shape_excitation_mean_error_mps':float(task_archive['shape_excitation_mean_error'].max()),
             'max_injected_relay_shape_speed_mps':float(np.linalg.norm(task_archive['excitation'][:,model.groups[1]],axis=-1).max()),
             'final_center_errors_m':sample_archive['center_error'][-1].tolist(),
             'final_shape_proxy_m':sample_archive['shape_proxy'][-1].tolist(),
             'final_density_proxy_m':sample_archive['density_proxy'][-1].tolist(),
             'final_max_relative_error_m':float(max_relative[-1]),
             'first_30s_qualified_at_s':first_hold,'final_30s_qualified':final_hold,
             'combined_overlap_seconds':overlap,'combined_formation_active_seconds':overlap_formation,
             'combined_density_active_seconds':overlap_density,'combined_safety_active_seconds':overlap_safety,
             'safety_changed_aircraft_fraction_mean':float(stat[:,10].mean()) if completed else None,
             'speed_cap_aircraft_fraction_mean':float(stat[:,11].mean()) if completed else None,
             'shared_scaling_below_one_fraction':float(np.mean(stat[:,3]<1.)) if completed else None,
             'zero_reference_steps':int(stat[:,5].sum()) if completed else 0,
             'task_runtime_p99_ms':float(1000*np.quantile(timing[timing[:,0]>0,0],.99)) if completed else None,
             'safety_runtime_p99_ms':float(1000*np.quantile(timing[:,1],.99)) if completed else None,
             'combined_runtime_over_20ms_fraction':float(np.mean(timing.sum(axis=1)>.02)) if completed else None,
             'source_hashes':source,'completed_utc':datetime.now(timezone.utc).isoformat(),
             'metric_semantics':'64-direction Gaussian midpoint-quantile RMS proxy, not exact Wasserstein. Shape metric aligns empirical centroid. Hold is at 0.5 s metric samples; safety is over all interpolated inner segments.',
             'timing_scope':'Nominal calculation plus safety filtering only, not end-to-end flight deployment latency; background workloads may affect measurements.',
             'failure':failure}
    write_json(out/'summary.json',summary)
    write_json(out/'archive_schema.json',{
        'inner_history':'State row k at k*dt. Command/diagnostic/timing row k applies on [k*dt,(k+1)*dt]. float64 throughout.',
        'diagnostics_columns':['observed_segment_distance_lower_bound_m','closest_pair_k','closest_pair_l','common_task_scale','separation_pair_count','emergency_zero_reference','accepted_prediction_margin_m','candidate_count','safety_correction_rms_mps','safety_correction_max_mps','safety_changed_aircraft_fraction','speed_capped_aircraft_fraction','backup_grid_intervals'],
        'timing_columns':['nominal_update_s_zero_when_held','safety_filter_s'],
        'task_history':'Fields evaluated at saved time using evaluation_positions and held until next task update. target/spreading/cross/edge are sample-centered. edge[k,j] is source j to receiver group i(k).',
        'excitation':'ctest=cnom+excitation. cnom group average equals nu before excitation, speed cap and safety.',
        'metrics':'At saved post-execution times; density_proxy includes displacement; shape_proxy removes empirical centroid.'})
    files={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',files)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--method',default='full',choices=['full','no_A','no_B','no_AB','no_shape','no_center']); p.add_argument('--seed',type=int,default=9001); p.add_argument('--out',required=True); p.add_argument('--duration',type=float); p.add_argument('--initial'); p.add_argument('--quiet',action='store_true')
    a=p.parse_args(); cfg=json.loads(Path(a.config).read_text(encoding='utf-8')) if a.config else default_config()
    if a.duration is not None: cfg['duration_s']=a.duration
    result=run(cfg,a.method,a.seed,a.out,a.initial,not a.quiet)
    sys.exit(0 if result['status']=='completed' else 2)
