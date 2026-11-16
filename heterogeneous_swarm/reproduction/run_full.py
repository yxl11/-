"""Original 178 s rerun entry. Default is preflight only; --execute is explicit.

Historical files are copied to a fresh working directory before any new run.
The frozen package and supplied trajectories are never output destinations.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,subprocess,sys,shutil
from verify_package import verify
from package_common import long_path
HERE=long_path(Path(__file__).parent)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['P','RAS04','MMD19']);p.add_argument('--seed',required=True,type=int,choices=[33001,33002,33003]);p.add_argument('--execute',action='store_true');p.add_argument('--work-dir');a=p.parse_args()
    verify();context='P' if a.method=='P' else 'external';snapshot=HERE/'snapshots'/context
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(HERE/'preflight_context.py'),'--snapshot',str(snapshot),'--method',a.method,'--seed',str(a.seed)],cwd=snapshot,capture_output=True,text=True,encoding='utf8')
    if result.returncode:
        print(result.stderr,file=sys.stderr);result.check_returncode()
    print(result.stdout.strip())
    if not a.execute:return
    target=(long_path(a.work_dir) if a.work_dir else HERE/'work'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{a.method}_s{a.seed}')
    protected=[HERE/x for x in ('snapshots','data','reference','timing','frozen_analysis')]
    if any(target==x or target.is_relative_to(x) for x in protected):raise ValueError('Run destination is inside protected package evidence')
    if target.exists():raise FileExistsError('Rerun directory must be new; no overwriting: '+str(target))
    shutil.copytree(snapshot,target)
    if a.method=='P':
        entry=target/'output/reviews/p_gain_calibration_20261108/code/run_candidate.py'
        args=['--stage','evaluation','--candidate','B','--seed',str(a.seed)]
    else:
        entry=target/'output/reviews/mixed_sweep_orbit_paired_20261102/worker.py'
        args=['--method','RAS08' if a.method=='RAS04' else a.method,'--seed',str(a.seed)]
    subprocess.run([sys.executable,'-X','utf8','-B',str(entry),*args],cwd=target,check=True)
if __name__=='__main__':main()
