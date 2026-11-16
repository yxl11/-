"""Builder supplement: include transitive import dependencies omitted by run manifests."""
from pathlib import Path
import hashlib,json,shutil
from package_common import long_path
HERE=long_path(Path(__file__).parent);ROOT=HERE.parents[3]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
manifest=read(HERE/'manifest.json')
cam='output/simulations/three_group_moving_comparison_20261023'
p=read(ROOT/cam/'protocol/frozen_mission.json')
required={f'{cam}/{x["copied_path"]}':x['sha256'] for x in p['vendor_files']}
required.update({f'{cam}/{n}':h for n,h in p['mission_source_sha256'].items()})
for name in ('artifact_manifest.json','mission.py'):
    key='output/reviews/mixed_role_corridor_demo_20261031/'+name;required[key]=sha(ROOT/key)
added=[];existing={x['path']:x for x in manifest['files']}
for context in ('P','external'):
    for name,h in required.items():
        src=ROOT/name;assert sha(src)==h
        key=f'snapshots/{context}/{name}';dest=HERE/key
        if key in existing:assert existing[key]['sha256']==h;continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(dest)==h
        else:shutil.copyfile(src,dest)
        assert sha(dest)==h
        row={'path':key,'sha256':h,'bytes':dest.stat().st_size,'workspace_source':name,
            'purpose':'Transitive original import dependency, verified by frozen mission vendor/source manifest or archived corridor launch manifest'}
        manifest['files'].append(row);added.append(row)
manifest['file_count']=len({x['path'] for x in manifest['files']})
manifest['copied_bytes']=sum(x['bytes'] for x in {x['path']:x for x in manifest['files']}.values())
manifest['transitive_import_supplement']={'files':len(added),'basis':'Original mission import reads corridor manifest; harness.verify_protocol checks its full vendor and mission-source manifests.'}
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({'added':len(added),'file_count':manifest['file_count'],'MiB':manifest['copied_bytes']/2**20}))
