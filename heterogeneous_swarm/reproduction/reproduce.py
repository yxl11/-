"""Recompute paper tables, paired changes and both figure datasets from saved runs.

Only existing records and an analytic target-shape function are read. No
controller, certifier, platform propagator or closed-loop runner is imported.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,importlib.util,sys,hashlib
sys.dont_write_bytecode=True
import numpy as np
from package_common import long_path
from verify_package import verify
HERE=long_path(Path(__file__).parent)
METHODS=('P','RAS04','MMD19');SEEDS=(33001,33002,33003)
WINDOWS=('initial_deployment','sweep_regrouping','patrol')
ROLE_NAMES=('FW','FPV','Quadrotor')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def csv_write(p,rows):
    with Path(p).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='recomputed');args=parser.parse_args()
    out=long_path(args.out) if Path(args.out).is_absolute() else HERE/args.out
    if out.exists():raise FileExistsError('Result destination must be new; no overwrite: '+str(out))
    if any(out==HERE/p or out.is_relative_to(HERE/p) for p in ('snapshots','data','reference','timing','frozen_analysis')):raise ValueError('Protected destination')
    integrity=verify();out.mkdir(parents=True)
    met=load('frozen_readonly_metrics',HERE/'frozen_analysis/metrics.py')
    expected=read(HERE/'reference/statistics/results.json')
    per_seed=[];role_rows=[];counts=[];checks=[];trace=[];spatial={}
    snapshot=HERE/'snapshots/P'
    shape=load('analytic_task_shape',snapshot/'output/reviews/mixed_sweep_orbit_external_20261101/shape_models.py')
    shapes=np.stack([shape.shape_state(t)[0] for t in (8,16,24,178)])
    with np.load(HERE/'reference/figures/figure2_plot_data.npz',allow_pickle=False) as oldfig:
        # These arrays are display limits, not statistics or trajectory data.
        spatial.update(reference_axes_limits=oldfig['reference_axes_limits'].copy(),reference_inset_limits=oldfig['reference_inset_limits'].copy())
    for method in METHODS:
        method_trace=[]
        for seed in SEEDS:
            folder=HERE/f'data/{method}_s{seed}'
            initial=met.load_npz(snapshot/f'output/reviews/mixed_sweep_orbit_paired_20261102/initials/seed{seed}.npz')
            rows,roles,completion,run_checks,refsha=met.measure(folder,method,seed,initial)
            assert all(run_checks.values()),(method,seed,run_checks)
            per_seed.extend(rows);role_rows.extend(roles);counts.append(completion)
            h=met.load_npz(folder/'history.npz');f=met.load_npz(folder/'fast_metrics.npz')
            t=h['times'];r=h['positions'];ids=h['roles'];groups=[np.flatnonzero(ids==i) for i in range(3)]
            centers=np.stack([r[:,g].mean(axis=1) for g in groups],axis=1);error=centers-f['reference']
            values=np.column_stack((np.sqrt(np.mean(np.sum(error**2,axis=2),axis=1)),np.sqrt(np.mean(np.sum(np.diff(error,axis=1)**2,axis=2),axis=1))))
            method_trace.append(values)
            for row in rows:
                old=next(x for x in expected['per_seed'] if x['method']==method and x['seed']==seed and x['window']==row['window'])
                for k in met.METRICS:
                    delta=abs(row[k]-old[k]);assert delta<1e-9,(method,seed,row['window'],k,delta)
                    checks.append({'method':method,'seed':seed,'window':row['window'],'metric':k,'absolute_error':delta})
            if seed==33001:
                tag='RAS08' if method=='RAS04' else method
                entry={'time':t,'q':f['reference'],'m':centers,'samples':r[np.searchsorted(t,[8,16,24])],
                    'final':r[-1],'roles':ids,'completed_circle':np.asarray(completion['complete']),'fraction':f['region_by_role'][-1]}
                for key,value in entry.items():spatial[f'{tag}_{key}']=value
    # Form figure3 from all nine raw position/reference records.
        trace.append(np.stack(method_trace))
    traces=np.stack(trace);medians=np.median(traces,axis=1)
    fig3={'time_s':t,'per_seed_errors_m':traces,'pointwise_median_errors_m':medians,
        'methods':np.array(['P','RAS08','MMD19']),'seeds':np.array(SEEDS),'metrics':np.array(['Ec_m','EB_m'])}
    spatial.update(methods=np.array(['P','RAS08','MMD19']),seeds=np.array([33001]),snapshot_times_s=np.array([8,16,24]),shape_matrices=shapes,shape_matrix_times_s=np.array([8,16,24,178]))
    figure_checks=[]
    for name,actual in [('figure2_plot_data.npz',spatial),('figure3_plot_data.npz',fig3)]:
        with np.load(HERE/'reference/figures'/name,allow_pickle=False) as reference:
            assert set(actual)==set(reference.files),(name,set(actual)^set(reference.files))
            for key,value in actual.items():
                if value.dtype.kind in 'US':np.testing.assert_array_equal(value,reference[key]);delta=0.
                else:
                    delta=float(np.max(np.abs(value-reference[key])))
                    np.testing.assert_allclose(value,reference[key],rtol=0,atol=3e-10,err_msg=name+'/'+key)
                figure_checks.append({'file':name,'array':key,'max_absolute_error':delta})
        np.savez_compressed(out/name,**actual)
    aggregate={};role_aggregate={};comparisons=[]
    for w in WINDOWS:
        aggregate[w]={};role_aggregate[w]={}
        for m in METHODS:
            rows=[x for x in per_seed if x['method']==m and x['window']==w]
            aggregate[w][m]={k:float((min if k=='segment_distance_lower_m' else max if k=='center_peak_m' else np.mean)([x[k] for x in rows])) for k in met.METRICS}
            for k,v in aggregate[w][m].items():assert abs(v-expected['aggregate'][w][m][k])<1e-9
            role_aggregate[w][m]={}
            for x in role_rows:
                if x['method']==m and x['window']==w:role_aggregate[w][m].setdefault(x['scope'],{}).setdefault(x['metric'],[]).append(x['value'])
            for scope,mm in role_aggregate[w][m].items():
                for k,vals in mm.items():
                    mm[k]=float(np.mean(vals));assert abs(mm[k]-expected['role_aggregate'][w][m][scope][k])<1e-9
        for other in METHODS[1:]:
            for k in met.METRICS:
                p=aggregate[w]['P'][k];v=aggregate[w][other][k]
                diffs=[]
                for seed in SEEDS:
                    get=lambda m:next(x[k] for x in per_seed if x['method']==m and x['seed']==seed and x['window']==w)
                    diffs.append(get('P')-get(other))
                comparisons.append({'window':w,'comparator':other,'metric':k,'P':p,'comparator_value':v,
                    'P_minus_comparator':p-v,'reduction_pct':100*(v-p)/v if abs(v)>1e-12 else None,
                    'P_lower_seed_count':sum(x<0 for x in diffs),'P_equal_seed_count':sum(x==0 for x in diffs),'P_higher_seed_count':sum(x>0 for x in diffs),
                    **{f'difference_s{s}':x for s,x in zip(SEEDS,diffs)}})
    result={'status':'PASS','aggregate':aggregate,'per_seed':per_seed,'role_aggregate':role_aggregate,'completion_per_seed':counts,'comparisons':comparisons}
    write(out/'results.json',result);csv_write(out/'per_seed.csv',per_seed);csv_write(out/'role_per_seed.csv',role_rows)
    csv_write(out/'aggregates.csv',[{'window':w,'method':m,**x} for w,mm in aggregate.items() for m,x in mm.items()])
    csv_write(out/'comparisons.csv',comparisons);csv_write(out/'completion_terminal.csv',counts)
    csv_write(out/'numerical_checks.csv',checks)
    report={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),'integrity':integrity,
        'per_seed_metric_checks':len(checks),'maximum_statistical_error':max(x['absolute_error'] for x in checks),
        'figure_array_checks':figure_checks,'new_closed_loop_trajectories':0,
        'computation_scope':'Original read-only integral definitions applied to audited fast/slow/history records. Ec/EB and both figure datasets reconstructed from raw positions/reference; target shapes analytic. Slow density/shape records are integrated, not a new KDE estimator.',
        'source_root':'.','output_files_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'report.json',report)
    print(json.dumps({'status':'PASS','checks':len(checks),'maximum_error':report['maximum_statistical_error'],'figure_arrays':len(figure_checks),'output':str(out),'new_trajectories':0},ensure_ascii=False))
if __name__=='__main__':main()
