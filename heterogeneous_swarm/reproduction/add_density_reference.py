"""Author-side byte-copy supplement; never imports or runs the analytic script."""
from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone
from package_common import long_path
HERE=long_path(Path(__file__).parent); ROOT=HERE.parents[3]
SRC=HERE.parent/'density_reference'; DST=HERE/'reference/density_reference'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
manifest=read(HERE/'manifest.json');before={r['path']:r['sha256'] for r in manifest['files']}
assert not DST.exists(),'Supplement must not overwrite an earlier copy.'
DST.mkdir(parents=True);rows=[]
for p in sorted(SRC.rglob('*')):
    if not p.is_file():continue
    assert not p.is_symlink()
    rel=p.relative_to(SRC); target=DST/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target);h=sha(p);assert h==sha(target)
    row={'path':target.relative_to(HERE).as_posix(),'sha256':h,'bytes':target.stat().st_size,
         'workspace_source':p.relative_to(ROOT).as_posix(),
         'purpose':'Independent analytic fixed-N Gaussian/KDE sampling reference; copied existing output only, not one of the nine trajectories and not automatically recomputed.'}
    rows.append(row)
manifest['files'].extend(rows)
unique={r['path']:r for r in manifest['files']}
manifest['file_count']=len(unique);manifest['copied_bytes']=sum(r['bytes'] for r in unique.values())
manifest['independent_analytic_supplement']={'path':'reference/density_reference','files':len(rows),'bytes':sum(r['bytes'] for r in rows),'automatic_recomputation':False}
assert all(unique[k]['sha256']==v for k,v in before.items())
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert read(DST/'validation.json')['status']=='PASS'
report={'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),'copied_files':rows,
        'original_frozen_hash_entries_unchanged':len(before),'total_frozen_files':manifest['file_count'],
        'analytic_script_executed':False,'new_closed_loop_trajectories':0,
        'validation_status':'Original archived validation reports PASS; its calculation was not repeated.'}
(HERE/'reports/density_reference_copy_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='copied_files'},ensure_ascii=False))
