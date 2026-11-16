"""Verify relocation, results and all nine full-rerun preflights without trajectories."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,shutil,hashlib
import numpy as np
from package_common import long_path
from verify_package import verify
HERE=long_path(Path(__file__).parent)
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run(args,cwd):
    x=subprocess.run([sys.executable,'-X','utf8','-B',*map(str,args)],cwd=cwd,capture_output=True,text=True,encoding='utf8')
    if x.returncode:raise RuntimeError(x.stderr+'\n'+x.stdout)
    try:return json.loads(x.stdout)
    except json.JSONDecodeError:return json.loads(x.stdout.strip().splitlines()[-1])
def main():
    verify();reports=HERE/'reports';reports.mkdir(exist_ok=True)
    moved=HERE/'migration_probe'
    if not moved.exists():
        shutil.copytree(HERE,moved,ignore=shutil.ignore_patterns('migration_probe','recomputed','reports','work','__pycache__'))
    existing=moved/'recomputed_after_move/report.json'
    if existing.exists():
        saved=json.loads(existing.read_text(encoding='utf8'));assert saved['status']=='PASS'
        results={'status':'PASS','checks':saved['per_seed_metric_checks'],'maximum_error':saved['maximum_statistical_error'],
            'figure_arrays':len(saved['figure_array_checks']),'output':str(existing.parent),'new_trajectories':0,'reused_completed_relocation_recomputation':True}
    else:results=run([moved/'reproduce.py','--out','recomputed_after_move'],HERE)
    preflights=[]
    for method in ('P','RAS04','MMD19'):
        snapshot=moved/'snapshots'/('P' if method=='P' else 'external')
        for seed in (33001,33002,33003):
            answer=run([moved/'preflight_context.py','--snapshot',snapshot,'--method',method,'--seed',seed],moved)
            with np.load(moved/f'data/{method}_s{seed}/history.npz',allow_pickle=False) as h:
                expected=hashlib.sha256(h['held_reference'][0].tobytes()).hexdigest()
            assert answer['raw_command_sha256']==expected,(method,seed)
            answer['initial_nominal_command_matches_original_bitwise']=True
            preflights.append(answer)
            print('PREFLIGHT',method,seed,flush=True)
    # Verify the frozen copies again after all read-only calculations.
    moved_integrity=run([moved/'verify_package.py'],HERE);original_integrity=verify()
    report={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'relocation':'Copied package to migration_probe and launched its scripts from another working directory; all inputs resolved within copied package.',
        'recomputed_results':results,'preflights':preflights,'new_closed_loop_trajectories':0,
        'total_model_propagation_steps':sum(x['state_propagation_steps'] for x in preflights),
        'full_rerun_execution':'Not run. Original complete 178-second entry points are supplied; source/hash/initial/assignment/controller construction and initial nominal command preflights passed.',
        'copied_package_integrity':moved_integrity,'original_package_integrity':original_integrity,
        'probe_cleanup':'The copied migration_probe directory may be removed after retaining this report; it is redundant delivery-test material.'}
    write(reports/'delivery_validation.json',report)
    shutil.copyfile(moved/'recomputed_after_move/report.json',reports/'relocated_reproduction_report.json')
    print(json.dumps({'status':'PASS','preflights':len(preflights),'new_trajectories':0,'model_steps':0,'report':str(reports/'delivery_validation.json')},ensure_ascii=False))
if __name__=='__main__':main()
