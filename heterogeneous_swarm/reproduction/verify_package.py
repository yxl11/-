"""Hash verification only. Never imports a controller or executes a trajectory."""
from pathlib import Path
import hashlib,json,argparse
from package_common import long_path
HERE=long_path(Path(__file__).parent)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
    manifest=read(HERE/'manifest.json');seen=set();failures=[]
    for row in manifest['files']:
        key=row['path']
        if key in seen:continue
        seen.add(key);p=HERE/key
        if not p.is_file() or p.is_symlink() or sha(p)!=row['sha256']:failures.append(key)
    entry_manifest=HERE/'entrypoint_hashes.json'
    if entry_manifest.exists():
        for key,h in read(entry_manifest).items():
            if sha(HERE/key)!=h:failures.append(key)
    assert not failures,failures
    current=read(HERE/'timing/current/timing_results.json')
    checks=read(HERE/'timing/current/checks.json')
    assert current['status']==checks['status']=='PASS'
    return {'status':'PASS','checked_copied_files':len(seen),'new_trajectories':0,
        'current_timing_results':'PASS; hash verification only, not repeated timing'}
if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
