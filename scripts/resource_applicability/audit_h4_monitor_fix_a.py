"""Freeze actual SOURCE blobs and read-only preparation profiles, never launch."""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[2]
BASE='814212cb1c25f27e3300294e72fef452d953fbc9'
OLD_SOURCE='6d365257770e99022b91d6a38dbee49ee0077503'
OLD_BUNDLE='artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06'
PREP='artifacts/resource_applicability/track_a_h4_new_server_preparation/2026-10-07'
EXTRA=[
    'src/trottertracks/resource_applicability/h4_geometry/streaming.py',
    'src/trottertracks/resource_applicability/h4_geometry/observer.py',
    'scripts/resource_applicability/run_h4_monitor_fix_a_tests.py',
    'scripts/resource_applicability/audit_h4_monitor_fix_a.py',
    'tests/tracks/resource_applicability/test_h4_monitor_fix_a.py',
    'tests/tracks/resource_applicability/h4_run05_serializer_reference.py']
REASONS={
    'circuits.py':'Replace ndarray.tolist/exact matrix tree and circuit-wide instruction allocation with call-local lazy traversal. Preserve readonly canonical bytes; validation occurs at streaming consumption. No scientific builder/metrics changes.',
    'identity.py':'Accept lazy sequences; bound encode/hash chunks to 64KiB. Canonical scalar representation/order unchanged.',
    'execution.py':'Require separately approved observer role, add reserved independent trace/wall accounting and driver phase boundaries. Old environment/source/authorization gates remain closed; no new production adoption.'}


def sha(data):return hashlib.sha256(data).hexdigest()
def require(value,reason):
    if not value:raise RuntimeError(reason)
def git(*args):return subprocess.check_output(['git','-c','maintenance.auto=false','-c','gc.auto=0','-C',str(ROOT),*args])
def blob(commit,path):return git('show',commit+':'+path)
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,sort_keys=True);f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--evidence',required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    require(re.fullmatch('[0-9a-f]{40}',args.source_commit),'actual source SHA')
    evidence,bundle=Path(args.evidence),ROOT/args.bundle
    require(evidence.is_absolute() and evidence.is_relative_to('/home/AbeHiromu') and
            bundle.is_relative_to(ROOT/'artifacts/resource_applicability') and '..' not in bundle.parts,
            'home-local dedicated artifact scope')
    require(not any(p.is_symlink() for p in [evidence,*evidence.parents,bundle,*bundle.parents]),'artifact symlink')
    bundle.mkdir(parents=True,exist_ok=False)
    old=json.loads(blob(BASE,OLD_BUNDLE+'/source_freeze_v1.json'))
    previous={**old['new_source_hashes'],**old['namespace_parent_hashes']}
    require(len(previous)==19,'old source19')
    paths=sorted(set(previous)|set(EXTRA))
    changed,closure=[],{}
    for path in paths:
        current=(ROOT/path).read_bytes();source=blob(args.source_commit,path)
        require(current==source,'SOURCE blob mismatch '+path)
        tree=ast.parse(source,filename=path)
        entry=dict(sha256=sha(source),blob=git('rev-parse',args.source_commit+':'+path).decode().strip(),
           imports=sorted({n.module or '.' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|
               {a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}))
        if path in previous:
            before=blob(OLD_SOURCE,path)
            require(sha(before)==previous[path],'old source byte hash '+path)
            entry.update(old_sha256=sha(before),old_blob=git('rev-parse',OLD_SOURCE+':'+path).decode().strip(),
                         changed=before!=source)
            if entry['changed']:
                require(Path(path).name in REASONS,'unrelated old19 changed')
                entry['change_reason']=REASONS[Path(path).name];changed.append(path)
        else:entry.update(added=True,change_reason='Dedicated lazy traversal/observer or artificial validation/audit source.')
        closure[path]=entry
    require(len(changed)==3,'expected three source19 differences')
    write(bundle/'source_freeze_v1.json',dict(schema_version='h4-monitor-fix-a-source-closure-v1',
        base_commit=BASE,old_source_commit=OLD_SOURCE,source_commit=args.source_commit,
        source_checkout_root=str(ROOT),closure=closure,closure_count=len(closure),old_source19_count=19,
        changed_source19_paths=changed,unchanged_source19_count=16,
        seed_rebinding_required=True,old_random_partial_results_reused=False,
        scientific_conditions_compiler_options_unchanged=True,production_authorized=False))
    exact=json.loads(blob(BASE,PREP+'/exact_source_environment_audit_v1.json'))
    profile=json.loads((evidence/'preparation_environment_profile_v1.json').read_text())
    observed=sorted([dict(name=d.metadata['Name'],version=d.version,
        RECORD_sha256=sha(d.read_text('RECORD').encode()) if d.read_text('RECORD') is not None else None)
        for d in metadata.distributions()],key=lambda d:(d['name'].lower(),d['version']))
    require(observed==profile['all_distributions'],'preparation environment changed during tests')
    require(profile['python']==sys.executable,'preparation interpreter')
    diffs=[]
    for name,expected in old['dependency_observations'].items():
        dist=metadata.distribution(name)
        actual=dict(version=dist.version,installed_record_sha256=sha(dist.read_text('RECORD').encode()))
        diffs.append(dict(name=name,old=expected,new=actual,
            version_differs=expected['version']!=actual['version'],
            RECORD_differs=expected['installed_record_sha256']!=actual['installed_record_sha256']))
    require(len(diffs)==45 and sum(d['version_differs'] for d in diffs)==18 and
            sum(d['RECORD_differs'] for d in diffs)==45,'accepted18/45 environment differences')
    for item in profile['installed_source11']:
        require(sha(Path(item['new_absolute_path']).read_bytes())==item['observed_sha256']==item['expected_sha256'],
                'installed source changed')
    write(bundle/'environment_profile_v1.json',profile)
    write(bundle/'environment_differences_v1.json',dict(dependencies=diffs,version_difference_count=18,
        RECORD_difference_count=45,installed_source_matches=11,environment_equivalence_established=False,
        production_environment_adopted=False,install_upgrade_settings_changes=0))
    require(re.fullmatch('attempt-[0-9]{2}',args.attempt),'final synthetic attempt name')
    result=json.loads((evidence/args.attempt/'test_result_v1.json').read_text())
    require(result['status']=='PASS' and result['failures']==result['errors']==result['skipped']==0,'final artificial suite')
    compiler=json.loads((evidence/args.attempt/'preparation_compiler_profile_v1.json').read_text())
    require(compiler['explicit_options']==exact['compiler_options_reference'] and
            compiler['explicit_options']['num_processes']==1,'compiler options preserved')
    write(bundle/'compiler_profile_v1.json',compiler)
    write(bundle/'test_results_v1.json',result)
    records=[]
    for path in sorted(evidence.glob('attempt-*/*')):
        if path.is_file():records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path.read_bytes())))
    write(bundle/'local_evidence_manifest_v1.json',dict(files=records,
        raw_logs_external_to_git=True,failed_attempts_preserved=True,
        plan_sha256=sha((evidence/'ARTIFICIAL_TEST_PLAN_v1.json').read_bytes())))
    write(bundle/'ARTIFICIAL_TEST_PLAN_v1.json',json.loads((evidence/'ARTIFICIAL_TEST_PLAN_v1.json').read_text()))
    print(json.dumps(dict(source_commit=args.source_commit,closure_count=len(closure),
        changed_old19=len(changed),tests=result['tests'],environment_versions_differ=18,records_differ=45)))


if __name__=='__main__':main()
