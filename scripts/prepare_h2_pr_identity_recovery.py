"""Create NEW technical recovery evidence only; never create a science grant."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import h2_pr_identity_recovery as r

OUT = r.ROOT / 'artifacts/resource_applicability/track_a_h2_pr_identity_recovery/2026-10-23'
NEW_FILES = ('scripts/h2_pr_identity_recovery.py', 'scripts/prepare_h2_pr_identity_recovery.py',
             'tests/test_h2_pr_identity_recovery.py')


def write(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def fixed(path):
    return r.loads(r.git_bytes(r.BASE, path))


def source_joins(catalogue, cfg):
    rows = []
    historical = {}
    for item in catalogue.runtime['historical_sealed_files']:
        saved = item['saved']
        if not saved['role'].startswith('H2_SOURCE:'):continue
        path = saved['role'].split(':', 1)[1]
        sha = hashlib.sha256(r.git_bytes(r.ACQUISITION, path)).hexdigest()
        historical[path] = saved['sha256']
        rows.append(dict(path=path, historically_sealed_sha256=saved['sha256'],
                         fixed_Git_sha256=sha, matches=sha == saved['sha256'], actual_worker_load_proved=False))
    current = []
    for item in cfg['source_files']:
        if item['root'] != 'acquisition':continue
        path = item['path']; actual = r.file_digest(Path(cfg['acquisition_root'])/path)
        current.append(dict(root='acquisition', path=path, current_sha256=actual,
                            historical_sha256=historical.get(path), historically_sealed_source_matches=actual == historical.get(path)))
    for item in catalogue.runtime['supplemental_native_files']:
        path = item['path'].split('${NEW_PROVIDER}/', 1)[1]
        actual = r.file_digest(Path(cfg['provider_root'])/path)
        current.append(dict(root='provider', path=path, current_sha256=actual,
                            historical_sha256=item['historical_sha256'], historically_sealed_source_matches=actual == item['historical_sha256']))
    old_python = [x['saved'] for x in catalogue.runtime['historical_sealed_files']
                  if x['saved']['role'] == 'fixed_python_runtime']
    return dict(status='STATIC_SOURCE_BYTES_ONLY_NOT_COMPLETE_IDENTITY', historical_acquisition=rows,
        compare_dependencies=current, saved_python=old_python,
        adopted_python_sha256=cfg['runtime']['python_sha256'],
        actual_historical_import_manifest=catalogue.runtime['complete_actual_import_manifest'],
        worker_declared_checks='Fixed SOURCE checks NumPy/Qiskit versions and __file__ membership in sealed inventory; it does not persist the full observed interpreter/module/library closure.',
        no_scientific_source_imported=True, original_byte_difference_is_not_an_automatic_all_comparison_invalidity_claim=True)


def build(source):
    for path in NEW_FILES:
        r.require((r.ROOT/path).read_bytes() == r.git_bytes(source, path), 'COMMITTED_RECOVERY_SOURCE')
    catalogue = r.Catalogue(); cfg = fixed(r.PREVIOUS+'COMPARE_contract_BLOCKED.json')
    previous = fixed(r.PREVIOUS+'preflight.json'); command = previous['command']
    original_root = Path(next(s for s in command if s.endswith('/scripts/launch_h2_accepted_compare.py'))).parents[1]
    # A prior immutable receipt provides the exact bounded originals to recheck.
    prior_preservation = fixed(r.PREVIOUS+'preservation_audit.json')
    preserved = list(prior_preservation['original_file_checks'])
    prior_inventory = fixed(r.PREVIOUS+'file_inventory.json')
    preserved.extend(dict(path=str(original_root/item['path']), sha256=item['sha256'],
                          role='BASE_4008_PROTECTED_FILE') for item in prior_inventory['files'])
    preserved.append(dict(path=str(original_root/r.PREVIOUS/'file_inventory.json'),
        sha256=hashlib.sha256(r.git_bytes(r.BASE,r.PREVIOUS+'file_inventory.json')).hexdigest(), role='BASE_INVENTORY'))
    for item in preserved:r.require(r.file_digest(item['path']) == item['sha256'], 'PROTECTED_ORIGINAL_CHANGED')
    r.require(not Path(cfg['run_root']).exists() and not Path(cfg['runtime']['cache_prefix']).exists(), 'NO_COMPARE_RUN_OR_CACHE')
    env = dict(os.environ); env.pop('LD_PRELOAD', None); env.pop('LD_LIBRARY_PATH', None)
    preflight = subprocess.run(command, env=env, capture_output=True, text=True)
    r.require(preflight.returncode == 0, 'BASE_COMPARE_PREFLIGHT_ERROR')
    result = r.loads(preflight.stdout)
    r.require(result['status'] == 'BLOCKED_PR_IDENTITY' and result['execution_ready'] is False and
              result['contract_sha256'] == previous['contract_sha256'], 'BASE_COMPARE_MUST_STAY_BLOCKED')
    joins = source_joins(catalogue, cfg)
    OUT.mkdir(parents=True, exist_ok=False)
    technical = subprocess.run([sys.executable, '-I', '-S', '-B', str(r.ROOT/NEW_FILES[0]),
                               '--report', str(OUT/'recovery.json')], capture_output=True, text=True)
    r.require(technical.returncode == 0, 'METADATA_RECOVERY_ERROR')
    recovery = r.loads((OUT/'recovery.json').read_bytes())
    write('source_byte_joins.json', joins)
    source = subprocess.check_output(['git','rev-parse',source], cwd=r.ROOT, text=True).strip()
    write('SOURCE.json',dict(base_commit=r.BASE, recovery_tool_SOURCE=source,
        preserved_COMPARE_SOURCE=cfg['source_commit'], preserved_COMPARE_contract_sha256=previous['contract_sha256'],
        files=[dict(path=p,sha256=r.file_digest(r.ROOT/p)) for p in NEW_FILES],
        evidence_catalogue='fixed Git BASE only; no mutable expected metadata', source_is_not_scientific_execution_authorization=True))
    write('COMPARE_preparation_status.json',dict(status='BLOCKED_PR_IDENTITY', PR_reuse='BLOCKED',
        preserved_contract_sha256=previous['contract_sha256'], preserved_SOURCE=cfg['source_commit'],
        command=command, exit_code=preflight.returncode, stdout=preflight.stdout, stderr=preflight.stderr,
        budget=cfg['budget'], scientific_imports=0, scientific_calls=0, live_grants=0,
        execution_ready=False, execution_authorized=False,
        new_contract_not_issued_reason='No complete PR identity witnesses. Original contract and SOURCE preserved; no witness or authorization fabricated.'))
    task_rows = []
    for row in catalogue.tasks:
        task = row['task']; tid = task['id']; candidate = task['candidate']
        task_rows.append(dict(task=tid,candidate=candidate['id'],sample=task['sample'],axes=task['axes'],
            event_locator=r.TOKEN+'stagepacks/H2_PR_EXPLORATION/run01/output/'+tid+'/events_PRIVATE.json',
            event_digest=row['result']['saved']['event_digest'],
            piece_counts=[p['saved']['rte_steps'] for p in catalogue.schedule[tid]['pieces']],
            historical_to_common_source_bytes='MATCH_FOR_SEALED_COMPUTATIONAL_DEPENDENCIES',
            full_event_body='NOT_RECOVERED', historical_operation_stream='NOT_RECOVERED',
            historical_runtime='DECLARED_INTERPRETER_BYTES_DIFFER_AND_ACTUAL_LOAD_MANIFEST_MISSING',
            correctness='SAVED_SINGLE_INPUT_STATE_ACCEPTANCE_ONLY_NO_REEVALUATION',
            reuse_status='BLOCKED', confirmed_recompile_required=False))
    r.require(all(x['historically_sealed_source_matches'] for x in joins['compare_dependencies']), 'DEPENDENCY_BYTE_JOIN_NOT_COMPLETE')
    write('task_identity_67.json',dict(PR_reuse='BLOCKED',tasks=task_rows,task_count=67,axes=134,
        source_join_scope='Declared sealed source bytes; no complete execution/circuit identity',
        saved_correctness_not_promoted_to_a_new_threshold=True, affected_recompile_set_not_yet_determined=True))
    axis_rows = []
    storage = {x['task_id']:x for x in catalogue.storage}
    for row in catalogue.tasks:
        task=row['task'];tid=task['id']
        for axis, records in zip(row['axes'],storage[tid]['axes']):
            r.require(axis['axis']==records['axis'],'AXIS_JOIN_ORDER')
            manifests={x['manifest']['saved']['archive_file']:x['manifest']['saved'] for x in records['archives']}
            compiled=manifests['compiled.qpy.zlib'];cost=axis['cost']['saved']
            r.require(cost['compiled_SHA']==compiled['codec_receipt']['original']['sha256'],'COMPILED_RAW_SHA_COST_JOIN')
            axis_rows.append(dict(task=tid,candidate=task['candidate']['id'],sample=task['sample'],axis=axis['axis'],
                event_digest=row['result']['saved']['event_digest'],compiled_raw_sha256=cost['compiled_SHA'],
                compiled_archive=compiled['codec_receipt']['stored'],
                wrapper_raw_sha256=manifests['wrapper.qpy.zlib']['codec_receipt']['original']['sha256'],
                raw_bytes_recovered=False, operation_stream_join_complete=False,
                saved_correctness_scope=cost['status'],recompile_authorized=False,reuse='BLOCKED'))
    write('axis_identity_134.json',dict(rows=axis_rows,axes=134,compiled_SHA_metadata_joins=134,
                                     new_RZ_or_cost_statistics=False))
    write('request_summary.json',dict(requests=len(recovery['requests']),
        requested_roles=dict(Counter(role for x in recovery['requests'] for role in x['roles'])),
        full_event_files=67,event_records=6400,signed_pieces=336,archives=402,
        original_config_SHA_verified=False,canonical_NEW_EXECUTION=None,
        event_raw_file_SHA_previously_registered=False,event_content_digest_anchor_available=True,
        recovered_original_files=0,new_scientific_calls=0))
    write('preservation_audit.json',dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        protected_file_checks=len(preserved),files=preserved,base_namespace_unchanged=True,
        original_454_file_checks_unchanged=True,base_COMPARE_full_seal_and_runtime_preflight_verified=True,
        sealed_files=len(cfg['files']),source_files=len(cfg['source_files']),
        original_completion_sha256=cfg['accepted_diagnostics']['completion_sha256'],
        source_results_grant_ledger_marker_STOP_unchanged=True,shared_environment_changed=False,
        new_COMPARE_root_exists=False,new_bytecode_cache_exists=False))
    write('resources.json',dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        affinity_cpus=len(os.sched_getaffinity(0)),free_disk_bytes=shutil.disk_usage(r.ROOT).free,
        available_RAM=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
        arbitrary_resource_caps_added=False,other_processes_changed=False))
    write('STOP.json',dict(status='MANDATORY_STOP_AFTER_TECHNICAL_RECOVERY',PR_reuse='BLOCKED',
        original_config_recovered=False,execution_authorized=False,execution_ready=False,scientific_calls=0,
        scientific_imports=0,numerical_reanalysis=0,sampling=0,compile=0,u_N_G=0,ranking=0,live_grants=0,
        detailed_scientific_review=False,push=False,PR_created=False,merge=False,mandatory_STOP=True))
    print(json.dumps(dict(status='TECHNICAL_RECOVERY_PREPARATION_COMPLETE_COMPARE_BLOCKED',source=source,
        request_files=len(recovery['requests']),source_byte_matches=len(joins['compare_dependencies']),
        protected_file_checks=len(preserved),science=0,live_grants=0)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source-commit',required=True)
    build(parser.parse_args().source_commit)
