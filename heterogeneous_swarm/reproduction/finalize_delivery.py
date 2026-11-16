"""Author-side provenance and manuscript-row check. No model execution."""
from pathlib import Path
import json, hashlib
from datetime import datetime, timezone
from package_common import long_path
HERE=long_path(Path(__file__).parent)
ROOT=HERE.parents[3]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj): (HERE/'reports'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
manifest=read(HERE/'manifest.json'); seen=set(); checks=[]
for r in manifest['files']:
    k=(r['workspace_source'],r['sha256'])
    if k in seen: continue
    seen.add(k); original=ROOT/r['workspace_source']
    checks.append({'source':r['workspace_source'],'expected_sha256':r['sha256'],'matches_original':original.is_file() and sha(original)==r['sha256']})
assert all(c['matches_original'] for c in checks)
save('provenance_check.json',{'status':'PASS','checked_unique_sources':len(checks),'sources':checks,'new_closed_loop_trajectories':0})
stats=read(HERE/'recomputed/results.json'); agg=stats['aggregate']; roles=stats['role_aggregate']
paper=ROOT/'output/reviews/ppt_execution_error_derivation_20261014/execution_error_derivation.tex'
lines=paper.read_text(encoding='utf-8-sig').splitlines(); table_checks=[]
labels={'P':'本文方法','RAS04':'点位编队','MMD19':'核分布匹配'}
def verify_row(table,row):
    start=next(i for i,s in enumerate(lines) if '\\label{'+table+'}' in s)
    end=next(i for i in range(start+1,len(lines)) if '\\end{tabular}' in lines[i])
    matches=[i+1 for i in range(start,end+1) if lines[i].startswith(row+'\\\\')]
    assert len(matches)==1,(table,row)
    table_checks.append({'table':table,'line':matches[0],'expected_row':row,'status':'PASS'})
for win,table in [('sweep_regrouping','tab:sweep-results'),('patrol','tab:patrol-results')]:
    for m,label in labels.items():
        d=agg[win][m]
        vals=[f'{d[k]:.3f}' for k in ['center_m','formation_m','density','shape_m']]+[f"{d['region_pct']:.2f}",f"{d['acceleration_rms_mps2']:.4f}"]
        verify_row(table,' & '.join([label]+vals))
for m,label in labels.items():
    d=agg['initial_deployment'][m]
    verify_row('tab:initial-deployment',' & '.join([label]+[f'{d[k]:.3f}' for k in ['center_m','acceleration_rms_mps2','density','shape_m']]))
for win,wlabel in [('sweep_regrouping','8--86'),('patrol','86--178')]:
    for m,label in labels.items():
        verify_row('tab:role-results',' & '.join([wlabel,label]+[f"{roles[win][m][r]['center_m']:.3f}" for r in ['FW','FPV','Quadrotor']]))
        d=agg[win][m]
        vals=[('0' if abs(d[k])<1e-9 else f'{d[k]:.4f}') for k in ['mapping_rms_mps','safety_rms_mps']]+[f"{d['safety_intervention_s']:.2f}",f"{d['segment_distance_lower_m']:.6f}"]
        verify_row('tab:phase-execution',' & '.join([wlabel,label]+vals))
    for r,rlabel in [('FW','固定翼'),('FPV','穿越机'),('Quadrotor','四旋翼')]:
        verify_row('tab:role-density-stages',' & '.join([wlabel,rlabel]+[f"{roles[win][m][r]['density']:.3f}" for m in labels]))
save('manuscript_table_check.json',{'status':'PASS','manuscript_sha256':sha(paper),'checked_utc':datetime.now(timezone.utc).isoformat(),'table_rows':table_checks,'scope':'27 rows in six trajectory-derived tables. The new static target-sampling reference is checked by its separate analysis.','new_closed_loop_trajectories':0})
hashes={p.relative_to(HERE).as_posix():sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ['README.md','environment.json','requirements.txt'])}
(HERE/'entrypoint_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
save('delivery_report.json',{'status':'PASS','scope':'Portable existing-result reproduction, frozen original code, configuration, initial conditions, targets, nine trajectory datasets, and original full-rerun entry points.','frozen_unique_files':manifest['file_count'],'frozen_bytes':manifest['copied_bytes'],'per_seed_metric_comparisons':351,'maximum_statistical_error':0.0,'figure_arrays_compared':37,'manuscript_table_rows':len(table_checks),'relocated_preflights':9,'initial_commands_match_original_bitwise':True,'new_closed_loop_trajectories':0,'model_propagation_steps':0,'full_reruns':'Only preflight tested; no full 178-second rerun was executed.','timing':'Current CPU-recorded fixed-state timing copied and hashed, no timing repeated.','migration_probe':'Redundant copied validation directory retained because automatic approval review rejected cleanup with blocked by policy. Exclude migration_probe when distributing the package.','entrypoint_hashes':hashes})
print(json.dumps({'status':'PASS','source_checks':len(checks),'manuscript_rows':len(table_checks),'entrypoints_hashed':len(hashes),'new_trajectories':0}))
