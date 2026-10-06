#!/usr/bin/env python3
"""Read-only source/allowed-JSON audit; emits lightweight review artifacts only."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates
from trottertracks.resource_applicability.h4_geometry.identity import sha,require

BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_parallel_source/2026-10-06'
OLD_BUNDLE='artifacts/resource_applicability/track_a_h4_geometry_source/2026-10-06'
OLD_SOURCE='d6c7afd02dc0603216982a3cd4ea71b3d047dd8d'
OLD_REVIEW='4d5eba454dda06bc2735730cf7c3f132e456db29'
OLD_AUDIT_SHA='8c7d68c2f50753e22152079206e6d9a8e6e174a87f4b940b4cfed218c92e5eef'
CHANGED_SOURCE={
    'src/trottertracks/resource_applicability/h4_geometry/'+name+'.py'
    for name in ('execution','gates','ledger','workers')}
CHANGED_SOURCE.update({'scripts/resource_applicability/audit_h4_geometry_source.py',
    'scripts/resource_applicability/run_h4_geometry_source_tests.py',
    'tests/tracks/resource_applicability/test_h4_geometry_source.py'})
PREP='artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05'
V1='artifacts/resource_applicability/track_a_h4_geometry_contract_preparation/2026-10-06'


def source_paths():
    return sorted([str(p.relative_to(ROOT)) for p in (ROOT/'src/trottertracks/resource_applicability/h4_geometry').glob('*.py')]+
        ['scripts/resource_applicability/'+p for p in ('run_h4_geometry_input_generation.py','run_h4_geometry_signal_compile.py',
         'run_h4_geometry_source_tests.py','audit_h4_geometry_source.py')]+
        ['tests/tracks/resource_applicability/test_h4_geometry_source.py'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    require(args.output.startswith(BUNDLE+'/') and '..' not in Path(args.output).parts,'review audit output scope')
    plan=gates.verify_contract(ROOT)
    env=json.loads(gates.git_blob(ROOT,gates.BASE,V1+'/environment_binding_audit_v1.json'))
    observed=gates.environment_matches(env['dependency_observations'])
    options=gates.compiler_matches(plan['compiler_environment_reference']['compiler'])
    old=json.loads(gates.git_blob(ROOT,gates.BASE,PREP+'/static_audit_v0.json'))
    preserved=[]
    for entry in old['source_hashes']:
        require(entry['path'].endswith('.py') and sha((ROOT/entry['path']).read_bytes())==entry['sha256'],'old source changed')
        preserved.append(entry['path'])
    require(len(preserved)==247,'old247 inventory')
    saved=old['allowed_json_identity']
    for entry in saved:
        require(entry['path'].endswith('.json') and sha((ROOT/entry['path']).read_bytes())==entry['sha256'],'saved6 JSON changed')
    require(len(saved)==6,'saved6 inventory')
    bundles={}
    for rel,expected in ((PREP,25),(V1,26),(gates.CONTRACT,30),(OLD_BUNDLE,28)):
        commit=OLD_REVIEW if rel==OLD_BUNDLE else gates.BASE
        paths=subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',commit,'--',rel],text=True).splitlines()
        require(len(paths)==expected,'old bundle inventory')
        for path in paths:
            require(sha((ROOT/path).read_bytes())==sha(gates.git_blob(ROOT,commit,path)),'old bundle changed')
        bundles[rel]={'files':len(paths),'byte_identical':True}
    old_audit_bytes=(ROOT/OLD_BUNDLE/'source_freeze_v1.json').read_bytes()
    require(sha(old_audit_bytes)==OLD_AUDIT_SHA,'old source audit changed')
    old_audit=json.loads(old_audit_bytes)
    unchanged={};changed={}
    for path,expected in old_audit['new_source_hashes'].items():
        require(sha(gates.git_blob(ROOT,OLD_SOURCE,path))==expected,'old source blob identity')
        actual=sha((ROOT/path).read_bytes())
        if path in CHANGED_SOURCE:
            changed[path]={'old_sha256':expected,'new_sha256':actual}
        else:
            require(actual==expected,'unrelated source changed')
            unchanged[path]=actual
    require(set(changed)==CHANGED_SOURCE,'minimal changed source inventory')
    installed=json.loads(gates.git_blob(ROOT,gates.BASE,gates.CONTRACT+'/static_source_audit_v2.json'))
    installed_hashes={}
    for entry in installed['source_findings']:
        if 'absolute_path' in entry:
            require(sha(Path(entry['absolute_path']).read_bytes())==entry['sha256'],'installed source changed')
            installed_hashes[entry['absolute_path']]=entry['sha256']
    # Additional static sources explaining the in-memory DIIS and AO2MO ports.
    for rel in ('pyscf/lib/diis.py','pyscf/ao2mo/__init__.py','qiskit/circuit/controlledgate.py'):
        path=Path('/home/AbeHiromu/venvs/trotter-common/lib/python3.12/site-packages')/rel
        installed_hashes[str(path)]=sha(path.read_bytes())
    closure={};imports={}
    for path in source_paths():
        data=(ROOT/path).read_bytes();closure[path]=sha(data)
        tree=ast.parse(data,filename=path)
        imports[path]=sorted({node.module or '.' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}|
                            {alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names})
        if args.source_commit:
            require(sha(gates.git_blob(ROOT,args.source_commit,path))==closure[path],'SOURCE_COMMIT blob mismatch')
    parent_hashes={p:sha((ROOT/p).read_bytes()) for p in ('src/trottertracks/__init__.py','src/trottertracks/resource_applicability/__init__.py')}
    for path,value in parent_hashes.items():
        require(sha(gates.git_blob(ROOT,gates.BASE,path))==value,'namespace parent changed')
    migrations=[]
    for path,difference in [
        ('src/trotterlib/rte.py','paired weights/order/product/finite polynomial preserved; D3 per-occurrence streams replace legacy sequential PCG64'),
        ('src/trotterlib/df_rte_tail.py','exact I/Z/ZZ coefficients and exact-zero pruning ported; full registered Gaussian basis used'),
        ('src/trotterlib/df_trotter/ops.py','dense Gaussian JW fallback and diagonal controlled phase ported; no qiskit-nature alternative; full basis explicit new wrapper identity'),
        ('src/trotterlib/df_trotter/decompose.py','descending-absolute intra-block eigenvalue order; not DF fragment reranking'),
        ('src/trotterlib/pr2_s0_s1_validation.py','generated fragment order, bit reversal, phase and shot rules ported; new environment/seed/controller'),
        ('src/trotterlib/pr2_matched_accuracy_m1_execution.py','same PF application order and finite corrected polynomial; new input and source bindings'),
        ('src/trotterlib/pr2_matched_accuracy_m1_b1_execution.py','full two-axis wrapper cost scope; new durable ledger and stage controls; no legacy SQLite/runtime reuse')]:
        migrations.append({'original_path':path,'original_commit':gates.BASE,'sha256':sha(gates.git_blob(ROOT,gates.BASE,path)),'semantic_difference':difference})
    tests=sorted((ROOT/BUNDLE).glob('test-attempt-*.json'))
    final=json.loads(tests[-1].read_text())
    require(final['status']=='PASS' and final['errors']==final['failures']==final['skipped']==0,'synthetic suite gate')
    require(final['synthetic_transpile_cumulative']<=64,'synthetic64 gate')
    reservations=(ROOT/BUNDLE/'synthetic_transpile_reservations.jsonl').read_text().splitlines()
    require(final['synthetic_transpile_prior']==25 and final['synthetic_transpile_new_total']==len(reservations)
            and final['synthetic_transpile_cumulative']==25+len(reservations),'old25 plus new accounting')
    require([json.loads(line)['source_series_cumulative'] for line in reservations]==list(range(26,26+len(reservations))),'new reservation continuity')
    artifact={'schema_version':'h4-native-source-audit-v1','status':'SOURCE_BLOBS_VERIFIED' if args.source_commit else 'PRE_FREEZE_SYNTHETIC_GATES_PASSED',
        'base_commit':gates.BASE,'branch_base_commit':OLD_REVIEW,'old_source_commit':OLD_SOURCE,
        'old_review_commit':OLD_REVIEW,'old_source_audit_sha256':OLD_AUDIT_SHA,
        'source_checkout_root':str(ROOT),'artifact_anchor':gates.ARTIFACT_ANCHOR,
        'future_output_root':gates.OUTPUT,'source_audit_reference':gates.SOURCE_AUDIT,
        'source_commit':args.source_commit,'observed_utc':datetime.now(timezone.utc).isoformat(),
        'contract_manifest_entries_verified':37,'contract_manifest_sha256':gates.MANIFEST_SHA,
        'plan_sha256':gates.PLAN_SHA,'plan_fingerprint':gates.PLAN_FP,
        'new_source_hashes':closure,'namespace_parent_hashes':parent_hashes,'AST_import_closure':imports,
        'unchanged_previous_source_hashes':unchanged,'changed_previous_sources':changed,
        'added_source_paths':sorted(set(closure)-set(old_audit['new_source_hashes'])),
        'dependency_observations':observed,'dependency_count':len(observed),'dependency_mismatches':[],
        'installed_source_hashes':installed_hashes,'compiler_options':options,'compiler_defaults_and_plugins_match':True,
        'compiler_fingerprint':plan['compiler_environment_reference']['compiler_fingerprint'],
        'environment_fingerprint':plan['compiler_environment_reference']['environment_fingerprint'],
        'migrations':migrations,'preservation':{'old_sources':len(preserved),'saved_JSON':len(saved),'bundles':bundles,
        'remaining_base_paths_changed':subprocess.check_output(['git','-C',str(ROOT),'diff','--name-only',gates.BASE,'--'],text=True).splitlines(),
        'excluded_scientific_paths_inspected':False},'final_synthetic_tests':final,
        'scientific_counts':dict(molecular_access=0,molecular_generation=0,SCF_DF_state_generation=0,real_signal=0,
        real_sampling=0,real_circuit_build=0,actual_science_transpile=0,GPU=0,shared_environment_changes=0,
        other_job_changes=0,execution_authorizations_issued=0,production_runner_launches=0,real_worker_launches=0),
        'untested':['real SCF/minao convergence and strict gradient','real DF returned-rank/ties/degeneracy',
        'real six input snapshots and physical operators','actual live cgroup/AS/RSS/worker handshake enforcement',
        'power-loss/fsync behavior on production filesystem','full74784 campaign memory/wall/output feasibility'],
        'mandatory_stop':True,'next_stage_authorized':False,'research_decision':None}
    with (ROOT/args.output).open('x') as out:
        json.dump(artifact,out,indent=2,sort_keys=True,ensure_ascii=False);out.write('\n')
    print(json.dumps({'source_paths':len(closure),'old_sources':len(preserved),'saved_JSON':len(saved),
                      'dependency_count':len(observed),'tests':final['tests'],'synthetic_transpiles':final['synthetic_transpile_cumulative']}))


if __name__=='__main__':
    main()
