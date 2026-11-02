#!/usr/bin/env python3
"""Zero-science source audit and draft binding. No execution path is provided."""
import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates, identity, resources

BASE = '5245a29ca26cad7421640410934907647459b822'
OLD_SOURCE = '6a121725ce751affd2d3d131a84944728e6b2343'
OLD_REVIEW = '88461f3930b9fef511739f91edae88231c33a3f5'
OLD_AUDIT = 'artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06/source_freeze_v1.json'
OLD_AUTH = 'artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06'
SOURCE_BUNDLE = 'artifacts/resource_applicability/track_a_h4_geometry_resource_observer_fix/2026-10-06'
AUTH_BUNDLE = 'artifacts/resource_applicability/track_a_h4_geometry_input_generation_authorization/2026-10-06-v2'
STATUS = 'H4_INPUT_GENERATION_RESOURCE_FIX_FROZEN_AWAITING_REVIEW'
BLOCKED = 'H4_INPUT_GENERATION_RESOURCE_OBSERVER_BLOCKED'
CHANGED = {'src/trottertracks/resource_applicability/h4_geometry/resources.py',
           'src/trottertracks/resource_applicability/h4_geometry/gates.py'}
PREPARATION_SOURCE = ('scripts/resource_applicability/prepare_h4_resource_observer_fix.py',
    'scripts/resource_applicability/run_h4_resource_observer_fix_tests.py',
    'tests/tracks/resource_applicability/test_h4_resource_observer_fix.py')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def blob(commit,path):
    return gates.git_blob(ROOT,commit,path)


def write_new(relative,value):
    identity.require(relative.startswith((SOURCE_BUNDLE+'/',AUTH_BUNDLE+'/'))
                     and '..' not in Path(relative).parts, 'new resource/draft evidence scope')
    path=ROOT/relative;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out:
        json.dump(value,out,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);out.write('\n')


def source_audit(source_commit=None):
    require,sha=identity.require,identity.sha
    require(Path(gates.__file__).absolute()==ROOT/'src/trottertracks/resource_applicability/h4_geometry/gates.py', 'loaded new checkout')
    require(gates.SOURCE_AUDIT==SOURCE_BUNDLE+'/source_freeze_v1.json', 'new source audit reference')
    for older,newer in ((gates.BASE,OLD_SOURCE),(OLD_SOURCE,OLD_REVIEW),(OLD_REVIEW,BASE)):
        require(subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',older,newer]).returncode==0, 'ancestry')
    old_raw=blob(BASE,OLD_AUDIT)
    require(sha(old_raw)=='5c9a1997339fa0f1f5479c62b11b6e2ef2ee024ce5a584aa958cfa80c4addd5f','old source audit')
    old=json.loads(old_raw);closure={};unchanged={};changes={};imports={}
    for path,expected in old['new_source_hashes'].items():
        require(sha(blob(OLD_SOURCE,path))==expected,'old SOURCE blob')
        data=(ROOT/path).read_bytes();closure[path]=sha(data)
        tree=ast.parse(data,filename=path)
        imports[path]=sorted({n.module or '.' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|
                            {a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names})
        if path in CHANGED:
            changes[path]={'old_sha256':expected,'new_sha256':sha(data)}
        else:
            require(sha(data)==expected,'unrelated source changed');unchanged[path]=expected
        if source_commit:require(sha(blob(source_commit,path))==sha(data),'new SOURCE blob')
    require(len(closure)==17 and set(changes)==CHANGED and len(unchanged)==15,'minimal source inventory')
    parent=old['namespace_parent_hashes']
    for path,expected in parent.items():
        require(sha((ROOT/path).read_bytes())==expected and sha(blob(OLD_SOURCE,path))==expected,'namespace parent changed')
        if source_commit:require(sha(blob(source_commit,path))==expected,'new SOURCE parent blob')
    old_gate=blob(OLD_SOURCE,'src/trottertracks/resource_applicability/h4_geometry/gates.py')
    new_gate=(ROOT/'src/trottertracks/resource_applicability/h4_geometry/gates.py').read_bytes()
    expected_gate=old_gate.replace(b"SOURCE_AUDIT = '"+OLD_AUDIT.encode()+b"'", b"SOURCE_AUDIT = '"+gates.SOURCE_AUDIT.encode()+b"'")
    require(new_gate==expected_gate,'gate change exceeds audit path')
    def nodes(data):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    old_nodes=nodes(blob(OLD_SOURCE,'src/trottertracks/resource_applicability/h4_geometry/resources.py'))
    new_nodes=nodes((ROOT/'src/trottertracks/resource_applicability/h4_geometry/resources.py').read_bytes())
    invariant_nodes={name for name in old_nodes if name not in ('cgroup_directories','observe_memory')}
    require(all(old_nodes[name]==new_nodes[name] for name in invariant_nodes),'non-observer resource condition changed')
    contract=gates.verify_contract(ROOT)
    observed=gates.environment_matches(old['dependency_observations'])
    options=gates.compiler_matches(contract['compiler_environment_reference']['compiler'])
    for path,expected in old['installed_source_hashes'].items():
        require(sha(Path(path).read_bytes())==expected,'installed critical source mismatch')
    directories={}
    for directory,count in (
        ('artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05',25),
        ('artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06',26),
        (gates.CONTRACT,30),('artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06',28),
        (str(Path(OLD_AUDIT).parent),11),(OLD_AUTH,18)):
        paths=git('ls-tree','-r','--name-only','-z',BASE,'--',directory).decode().split('\0')[:-1]
        require(len(paths)==count,'old bundle inventory')
        hashes={}
        for path in paths:
            expected=sha(blob(BASE,path));require(sha((ROOT/path).read_bytes())==expected,'old bundle changed');hashes[path]=expected
        directories[directory]={'files':count,'sha256':hashes,'byte_identical':True}
    static=json.loads(blob(BASE,'artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05/static_audit_v0.json'))
    for entry in static['source_hashes']+static['allowed_json_identity']:
        require(entry['path'].endswith(('.py','.json')) and sha((ROOT/entry['path']).read_bytes())==entry['sha256'],'old source/evidence changed')
    require(len(static['source_hashes'])==247 and len(static['allowed_json_identity'])==6,'old evidence inventory')
    independent={p:sha((ROOT/p).read_bytes()) for p in PREPARATION_SOURCE}
    result={'schema_version':'h4-resource-fix-source-audit-v1',
        'status':'SOURCE_BLOBS_VERIFIED' if source_commit else 'PRE_FREEZE_SOURCE_IDENTITY_VERIFIED',
        'source_commit':source_commit,'branch_base_commit':BASE,'base_commit':gates.BASE,
        'old_source_commit':OLD_SOURCE,'old_parallel_review':OLD_REVIEW,'source_checkout_root':str(ROOT),
        'artifact_anchor':gates.ARTIFACT_ANCHOR,'future_output_root':gates.OUTPUT,
        'source_audit_reference':gates.SOURCE_AUDIT,'new_source_hashes':closure,'namespace_parent_hashes':parent,
        'AST_import_closure':imports,'changed_previous_sources':changes,'unchanged_previous_source_hashes':unchanged,
        'invariant_resource_AST_nodes':sorted(invariant_nodes),'gate_change':'SOURCE_AUDIT path only; schema/permission/fingerprint guards unchanged',
        'preparation_only_source_hashes':independent,'contract_manifest_entries_verified':37,
        'contract_plan_sha256':gates.PLAN_SHA,'contract_plan_fingerprint':gates.PLAN_FP,'contract_manifest_sha256':gates.MANIFEST_SHA,
        'dependency_count':len(observed),'dependency_observations':observed,'dependency_mismatches':[],
        'installed_source_hashes':old['installed_source_hashes'],'compiler_options':options,
        'compiler_defaults_and_plugins_match':True,'compiler_fingerprint':old['compiler_fingerprint'],
        'environment_fingerprint':old['environment_fingerprint'],'old_bundles':directories,
        'old_sources_verified':247,'saved_JSON_verified':6,'source_series_transpile_cumulative':28,'additional_transpile':0,
        'observed_utc':datetime.now(timezone.utc).isoformat(),'execution_ready':False,'mandatory_stop':True}
    return contract,result


def live_observation():
    metadata={'schema_version':'h4-resource-fix-live-observation-v1','observed_utc':datetime.now(timezone.utc).isoformat(),
        'kernel_release':os.uname().release,'cgroup_namespace':os.readlink('/proc/self/ns/cgroup'),
        'own_cgroup_membership':Path('/proc/self/cgroup').read_text(),
        'cgroup_mountinfo_lines':[l for l in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup' in l],
        'metadata_only':True,'actual_launch_admission_performed':False,'requested_workers':6,'actual_workers':None,
        'allowed_cpus':[],'saved_review_approved':False,'affinity_cgroup_priority_changed':False,
        'fresh_launch_observation_still_required':True,'host_available_is_reservation':False,'execution_ready':False,
        'production_input_output_registry_access_creation':False,'additional_transpile':0,'mandatory_stop':True}
    try:
        observation=resources.observe_memory();observation['process_cpus']=sorted(observation['process_cpus'])
        metadata.update(status='READ_ONLY_OBSERVER_SUCCEEDED',observation=observation,error=None,
            operational_status=STATUS)
    except (identity.Stop,OSError,ValueError,KeyError) as exc:
        metadata.update(status='READ_ONLY_OBSERVER_BLOCKED',observation=None,error=str(exc),operational_status=BLOCKED)
    return metadata


def updated_documents(source_commit,audit_bytes,contract,audit):
    identity.require(audit['source_commit']==source_commit and audit['status']=='SOURCE_BLOBS_VERIFIED','frozen new source first')
    old_plan=json.loads(blob(BASE,OLD_AUTH+'/input_generation_plan_v1.json'))
    old_auth=json.loads(blob(BASE,OLD_AUTH+'/authorization_draft_v1.json'))
    old_review=json.loads(blob(BASE,OLD_AUTH+'/stage_review_v1.json'))
    plan={**old_plan,'source_commit':source_commit,'source_hashes':{**audit['new_source_hashes'],**audit['namespace_parent_hashes']},
          'source_root':str(ROOT),'source_audit_sha256':identity.sha(audit_bytes)}
    identity.require(plan['templates']==contract['templates'] and plan['requested_workers']==6
                     and plan['inputs'] is None and plan['generation_freeze_digest'] is None,'generation scope')
    plan_fp=identity.fingerprint('h4-execution-plan-v1',plan)
    auth={**old_auth,'plan_fingerprint':plan_fp,'allowed_cpus':[]}
    review={**old_review,'plan_fingerprint':plan_fp,'authorization_digest':identity.fingerprint('h4-authorization-v1',auth),'approved':False}
    gates.structural_gate(plan,auth,review)
    return plan,auth,review


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('source-audit','observe','draft'),required=True)
    parser.add_argument('--source-commit')
    parser.add_argument('--output')
    args=parser.parse_args(argv)
    if args.mode=='observe':
        result=live_observation();write_new(args.output,result)
        print(json.dumps(result,sort_keys=True));return 0 if result['observation'] is not None else 1
    if args.mode=='source-audit':
        _contract,result=source_audit(args.source_commit);write_new(args.output,result)
        print(json.dumps({'source_commit':args.source_commit,'source_paths':17,'parents':2,'unchanged_paths':15,'dependency_count':45,'additional_transpile':0}));return 0
    identity.require(args.source_commit is not None,'new SOURCE_COMMIT required before draft')
    data=(ROOT/gates.SOURCE_AUDIT).read_bytes();audit=json.loads(data)
    contract,verified=source_audit(args.source_commit)
    identity.require(verified['new_source_hashes']==audit['new_source_hashes'],'frozen draft source closure')
    plan,auth,review=updated_documents(args.source_commit,data,contract,audit)
    for name,value in [('input_generation_plan_v2.json',plan),('authorization_draft_v2.json',auth),('stage_review_v2.json',review)]:
        write_new(AUTH_BUNDLE+'/'+name,value)
    print(json.dumps({'status':STATUS,'source_commit':args.source_commit,'source_root':str(ROOT),
        'source_audit_sha256':identity.sha(data),'plan_fingerprint':auth['plan_fingerprint'],
        'authorization_digest':review['authorization_digest'],'review_digest':identity.fingerprint('h4-review-v1',review),
        'allowed_cpus':[],'approved':False,'requested_workers':6,'execution_ready':False},sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
