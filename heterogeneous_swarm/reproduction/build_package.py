"""Build an isolated reproduction package by copying hash-pinned existing material.

This builder reads the research workspace once. Distributed entry points use only
their package directory; no original source/data file is edited or hard-linked.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,shutil,sys,importlib.metadata,platform
from package_common import long_path

HERE=long_path(Path(__file__).parent)
ROOT=HERE.parents[3]
GAIN=ROOT/'output/reviews/p_gain_calibration_20261108'
PAIRED=ROOT/'output/reviews/mixed_sweep_orbit_paired_20261102'
ADOPT=ROOT/'output/reviews/p_gain_adoption_20261108'
PAPER='output/reviews/ppt_execution_error_derivation_20261014/execution_error_derivation.tex'
SEEDS=(33001,33002,33003)
METHODS=('P','RAS04','MMD19')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def run_path(m,s):return GAIN/f'runs/evaluation/B_s{s}' if m=='P' else PAIRED/f'runs/{"RAS08" if m=="RAS04" else m}_s{s}'

def main():
    if (HERE/'manifest.json').exists():raise RuntimeError('Package manifest already exists; never rebuild in place')
    records=[]
    def copy(source,destination,expected=None,purpose=''):
        source=Path(source);destination=HERE/destination
        actual=sha(source)
        if expected is not None:assert actual==expected,(source,actual,expected)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert sha(destination)==actual
        else:shutil.copyfile(source,destination)
        assert sha(destination)==actual
        assert not destination.is_symlink()
        records.append({'path':destination.relative_to(HERE).as_posix(),'sha256':actual,'bytes':destination.stat().st_size,
                        'workspace_source':source.relative_to(ROOT).as_posix(),'purpose':purpose})
    all_inputs={}
    for m in METHODS:
        for s in SEEDS:all_inputs.update(read(run_path(m,s)/'input_manifest.json'))
    all_inputs.update(read(GAIN/'protocol.json')['source_hashes'])
    all_inputs.update(read(PAIRED/'campaign_registration.json')['frozen_files'])
    documents={'P':ADOPT/'archive_before/execution_error_derivation.tex',
        'external':ROOT/'output/reviews/ppt_execution_error_derivation_20261014/engineering_three_seed_revision_20261102/execution_error_derivation.before.tex.archive'}
    expected_docs={context:read(run_path('P' if context=='P' else 'RAS04',33001)/'input_manifest.json')[PAPER] for context in documents}
    for context in ('P','external'):
        for name,expected in all_inputs.items():
            if name==PAPER:
                source=documents[context];expected=expected_docs[context]
            else:source=ROOT/name
            copy(source,f'snapshots/{context}/{name}',expected,'Frozen full-rerun dependency; original bytes')
        for name in ('protocol.json','selection.json'):
            source=GAIN/name
            copy(source,f'snapshots/{context}/'+source.relative_to(ROOT).as_posix(),purpose='Frozen selection and registration')
    included=('history.npz','fast_metrics.npz','slow_metrics.npz','initial.npz','configuration.json',
        'controller_identity.json','input_manifest.json','registration.json','summary.json','schema.json','terminal_certificate.npz')
    run_records=[]
    for m in METHODS:
        for s in SEEDS:
            folder=run_path(m,s)
            extras=[n for n in ('experiment_identity.json','target_assignment.npz') if (folder/n).exists()]
            for name in (*included,*extras):copy(folder/name,f'data/{m}_s{s}/{name}',purpose='Nine completed trajectories; immutable input to result reproduction')
            run_records.append({'method':m,'seed':s,'path':f'data/{m}_s{s}','original_folder':folder.relative_to(ROOT).as_posix(),
                                'original_method':'RAS08' if m=='RAS04' else m,'controller_parameters':read(folder/'controller_identity.json')['parameters']})
    for name in ('results.json','checks.json','source_hashes.json','snapshot_convention_check.json'):
        copy(ADOPT/'evidence'/name,f'reference/statistics/{name}',purpose='Previously independently checked expected statistics')
    for name in ('figure2_plot_data.npz','figure3_plot_data.npz','plot_data_sources.json'):
        copy(ADOPT/'figures'/name,f'reference/figures/{name}',purpose='Archived expected figure data and provenance')
    copy(GAIN/'analysis/metrics.py','frozen_analysis/metrics.py',purpose='Original read-only metric integrator, unchanged')
    for source,dest in [(GAIN/'audit/evaluation_audit.json','reference/audit/evaluation_audit.json'),
        (GAIN/'selection.json','reference/selection.json')]:
        copy(source,dest,purpose='Existing completed evaluation audit or frozen selection')
    copy(HERE.parent/'archive_before/execution_error_derivation.tex','reference/manuscript_before_revision.tex',purpose='Manuscript before this two-round revision; numerical tables unchanged')
    for context,source_folder in [('historical',ADOPT/'evidence/timing'),('current',HERE.parent/'timing')]:
        for p in sorted(source_folder.iterdir()):
            if p.is_file() and p.suffix in ('.py','.json','.jsonl','.npz','.md'):
                copy(p,f'timing/{context}/{p.name}',purpose='Existing fixed-state timing only; do not rerun for reproduction')
    env={'python':sys.version,'platform':platform.platform(),
        'versions':{name:importlib.metadata.version(name) for name in ('numpy','scipy','matplotlib')},
        'scope':'Environment used to build and verify the portable package; timing environment is separately recorded in timing/current/environment.json.'}
    write(HERE/'environment.json',env)
    (HERE/'requirements.txt').write_text('\n'.join(f'{n}=={v}' for n,v in env['versions'].items())+'\n',encoding='utf8')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Portable existing-result reproduction and separately gated original full-rerun entry; no new trajectory generated during packaging',
        'methods':METHODS,'seeds':SEEDS,'runs':run_records,'historical_manuscripts':expected_docs,
        'snapshot_layout':'snapshots/P and snapshots/external are complete independent research-root layouts; separate original manuscript hashes are restored without modifying original protocols.',
        'omitted_large_records':'candidate_events.json and certificate_ledger.json are not needed for the supplied table/figure reproduction; this package does not independently re-audit every historical certificate. Full archived evaluation audit and terminal certificates are included.',
        'files':records,'file_count':len({x['path'] for x in records}),'copied_bytes':sum(x['bytes'] for x in {x['path']:x for x in records}.values()),
        'copy_mode':'Independent byte copies, no hard links/symlinks; no run outputs are inside frozen execution snapshots.'}
    write(HERE/'manifest.json',manifest)
    print(json.dumps({'files':manifest['file_count'],'MiB':manifest['copied_bytes']/2**20,'status':'BUILT'},ensure_ascii=False))

if __name__=='__main__':main()
