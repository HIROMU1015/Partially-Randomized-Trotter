"""Final metadata/byte audit only. No scientific imports or new cost evaluation.

Read only explicitly catalogued originals; write a new report exclusively within
this technical artifact directory. Never issue or alter a grant or science run.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import h2_pr_identity_recovery as r

BASE = 'e7a63b2b06a124f9305483aa39bcf1ba8cafdd0d'
SOURCE = '9d413d7fb9cbe302ff064a4078ff2a747a8f3db2'
NEW_NAMESPACE = str(OUT.relative_to(ROOT)) + '/'
DOCS = {'PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'artifacts/validation_manifest.json',
        'docs/README.md', 'scripts/README.md', 'src/trotterlib/README.md',
        'docs/research/研究概要・現状.md',
        'docs/research/track_a_h2_pr_original_identity_20261023.md',
        'docs/research/研究ノート/2026-10-23_h2_pr_original_identity.md'}


def read(path):
    return r.loads(Path(path).read_bytes())


def verify(config):
    catalogue = r.Catalogue()
    old = catalogue.verify_config(config)
    execution = config.parents[2]
    compare = r.loads(r.git_bytes(r.BASE, r.PREVIOUS + 'COMPARE_contract_BLOCKED.json'))
    source = read(OUT / 'SOURCE.json')
    binding = read(OUT / 'SOURCE_binding.json')
    r.require(binding['audit_source_commit'] == SOURCE, 'AUDIT_SOURCE_BINDING')
    r.require(r.file_digest(OUT / 'SOURCE.json') == binding['original_SOURCE_json_sha256'], 'SOURCE_REPORT_SHA')
    for row in source['files']:
        actual = r.file_digest(ROOT / row['path'])
        r.require(actual == row['sha256'], 'CURRENT_AUDIT_FILE_BYTES')
        r.require(actual == __import__('hashlib').sha256(r.git_bytes(SOURCE, row['path'])).hexdigest(), 'COMMITTED_SOURCE_BYTES')
    r.require(source['unchanged_COMPARE_SOURCE'] == compare['source_commit'], 'COMPARE_SOURCE_UNCHANGED')
    r.require(source['unchanged_contract_sha256'] == r.digest(compare), 'COMPARE_CONTRACT_UNCHANGED')
    # Match the original audit's exact explicit watch set and ordering.
    watched = {}
    for request in catalogue.requests():
        watched[str(r.resolve(execution, request['locator']))] = request['locator']
    for item in catalogue.origins.values():
        if any(role in ('old_consumed_marker_hash_only', 'old_grant_hash_only_never_published',
                        'existing_target_ledger_hash_only') for role in item['roles']):
            watched[str(r.resolve(execution, item['locator']))] = item['locator']
    for item in old['sealed_files']:
        watched[item['path']] = 'ORIGINAL_SEALED:' + item['role']
    for key in ('log', 'exit', 'entered'):
        watched[old['supervisor'][key]] = 'ORIGINAL_SUPERVISOR:' + key
    origin = Path(compare['accepted_diagnostics']['run_root'])
    watched[str(origin / 'results/preparation_1.json')] = 'ACCEPTED_DIAGNOSTICS:preparation_1'
    watched[str(origin / 'completion.json')] = 'ACCEPTED_DIAGNOSTICS:completion'
    for item in compare['files']:
        watched[item['path']] = 'COMPARE_SEAL:' + item.get('role', 'input')
    previous = read(OUT / 'preservation_audit.json')
    paths = sorted(watched)
    r.require(len(paths) == previous['checked_files'] == 16804, 'EXACT_WATCH_SET')
    def observe(path):
        p = Path(path)
        return dict(bytes=p.stat().st_size, sha256=r.file_digest(p)) if p.is_file() else None
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        observations = list(pool.map(observe, paths))
    for path, current, before in zip(paths, observations, previous['files']):
        r.require(watched[path] == before['locator'], 'WATCH_LOCATOR_ORDER')
        r.require(current == before['before'] == before['after'], 'PROTECTED_ORIGINAL_CHANGED')
    recovery = read(OUT / 'recovery_verified.json')
    r.require(Counter(x['status'] for x in recovery['recovered']) ==
              Counter(ORIGINAL_BYTES_MATCH=2085, ARCHIVE_AND_RAW_BYTES_MATCH=402, SAVED_EVENT_CONTENT_MATCH=67),
              'RECOVERY_STATUS_COUNTS')
    axes = read(OUT / 'axis_identity_134.json')
    rows = axes['rows']
    r.require(len(rows) == 134 and len({(x['task'], x['axis']) for x in rows}) == 134, 'EXACT_AXES')
    r.require(Counter(x['candidate'] for x in rows) ==
              Counter(H2O32_c7ec590a32a859bc=66, H2O32_8a2ecdd0294ffb74=68), 'EXACT_CANDIDATE_AXES')
    tasks = read(OUT / 'task_identity_67.json')['rows']
    r.require(len(tasks) == 67 and len({x['task'] for x in tasks}) == 67, 'EXACT_TASKS')
    r.require(all(x['strict_reuse'] == 'BLOCKED' for x in rows + tasks), 'NO_PARTIAL_IDENTITY_PROMOTION')
    r.require(axes['strict_reuse_accepted_axes'] == 0 and axes['saved_artifact_binding_pass'] == 134, 'ACCEPTED_VS_BOUND')
    preflight = read(OUT / 'COMPARE_preflight.json')
    r.require(read(OUT / 'COMPARE_contract_DRAFT_BLOCKED.json')['science_authorized'] is False, 'DRAFT_NOT_AUTHORITY')
    r.require(not preflight['execution_ready'] and not preflight['execution_authorized'], 'COMPARE_BLOCKED')
    r.require(r.loads(preflight['stdout'])['status'] == 'BLOCKED_PR_IDENTITY', 'OLD_PREFLIGHT_STATUS')
    for filename in ('alternative_equivalence_contract_DRAFT.json', 'minimal_additional_validation_DRAFT.json'):
        r.require(read(OUT / filename)['mandatory_STOP'] is True, 'PROPOSAL_STOP')
    tests = read(OUT / 'test_results.json')
    r.require(tests['tests'] == sum(x['passed'] for x in tests['suites']) == 200, 'TECHNICAL_TEST_COUNT')
    for row in tests['suites']:
        r.require(row['passed'] == row['tests'] and row['exit_code'] == 0 and
                  r.file_digest(ROOT / row['path']) == row['sha256'], 'TEST_RESULT_SOURCE')
    manifest = read(ROOT / 'artifacts/validation_manifest.json')
    original_manifest = r.loads(r.git_bytes(BASE, 'artifacts/validation_manifest.json'))
    r.require(all(manifest[k] == v for k, v in original_manifest.items()), 'PRIOR_MANIFEST_FIELDS')
    r.require(set(manifest) - set(original_manifest) == {'track_a_h2_pr_original_identity_20261023'}, 'ONLY_NEW_MANIFEST_KEY')
    # All tracked old scientific namespaces remain byte-identical to the base.
    changed = subprocess.check_output(['git', 'diff', '--name-only', '-z', BASE], cwd=ROOT).decode().split('\0')
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    changed = sorted(set(filter(None, changed + untracked)))
    source_files = {x['path'] for x in source['files']}
    r.require(all(p in DOCS or p in source_files or p.startswith(NEW_NAMESPACE) for p in changed), 'ONLY_AUTHORIZED_CHANGE_PATHS')
    new_docs = [ROOT / p for p in DOCS if p.endswith('.md')]
    new_docs += [OUT / 'REPORT.md']
    required_target = 'track_a_h2_pr_original_identity_20261023.md'
    r.require(all(required_target in (ROOT/p).read_text() for p in DOCS
                  if p.endswith('.md') and p not in
                  ('docs/research/' + required_target, 'docs/research/研究ノート/2026-10-23_h2_pr_original_identity.md')), 'ALL_INDEX_ENTRIES')
    # Validate new local links, not every historical entry in large inherited indexes.
    for p in (OUT / 'REPORT.md', ROOT/'docs/research'/required_target,
              ROOT/'docs/research/研究ノート/2026-10-23_h2_pr_original_identity.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            resolved = (p.parent / target).resolve()
            deferred = {OUT/'final_verification.json', OUT/'file_inventory.json'}
            r.require(resolved.is_file() or resolved in deferred, 'NEW_REPORT_LINK_TARGET:' + target)
    # Public projections contain digests and joins; reject original payload fields.
    forbidden = {'application_sequence', 'rotation_component_id', 'product_component_ids',
                 'basis_change_operations', 'science_master_seed', 'seed_hex', 'seed_bytes'}
    def inspect(value):
        if isinstance(value, dict):
            r.require(not forbidden.intersection(value), 'PRIVATE_EVENT_OR_SEED_PAYLOAD')
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    for path in OUT.glob('*.json'):
        inspect(read(path))
    stop = read(OUT / 'STOP.json')
    r.require(stop['mandatory_STOP'] is True and stop['PR_reuse'] == 'BLOCKED' and
              not stop['execution_ready'] and not stop['execution_authorized'] and
              all(stop[k] == 0 for k in ('new_sampling', 'new_simulation', 'new_circuit_generation', 'compile',
                  'new_u_N_G', 'new_cost_or_ranking', 'scientific_imports', 'live_grants')) and
              not stop['push'] and not stop['detailed_scientific_review'] and not stop['final_evidence_freeze'], 'FINAL_STOP')
    r.require(not Path(compare['run_root']).exists(), 'NO_NEW_COMPARE_RUN_ROOT')
    cache = compare['runtime']['cache_prefix']
    r.require(not Path(cache).exists(), 'NO_COMPARE_BYTECODE_CACHE')
    r.require(not any(name.split('.')[0] in ('numpy','qiskit','scipy','trotterlib','trottertracks')
                      for name in sys.modules), 'NO_SCIENTIFIC_IMPORTS')
    return dict(status='PASS', performed_at_UTC=datetime.now(timezone.utc).isoformat(),
        base_commit=BASE, audit_SOURCE=SOURCE, verifier_sha256=r.file_digest(__file__),
        execution_ready=False, strict_reuse_accepted_axes=0, blocked_axes=134,
        original_saved_artifact_bindings=134, original_evidence_files_recovered=2554,
        protected_paths_rechecked=16804, protected_path_changes=0,
        prior_manifest_keys_unchanged=True, previous_scientific_NAMESPACES_unchanged=True,
        changed_paths=changed, scientific_calls=0, live_grants_issued=0, raw_copies_written=0,
        private_event_and_seed_payloads_in_new_JSON=False, detailed_scientific_review=False,
        original_historical_lib_drift_is_not_a_change_during_this_task=True,
        compare_run_root_exists=False, compare_pycache_exists=False, push=False,
        tests=200, mandatory_STOP=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    output = args.report.resolve()
    r.require(output.parent == OUT and output.suffix == '.json', 'REPORT_IN_NEW_NAMESPACE_ONLY')
    r.require(not output.exists(), 'NO_REPORT_OVERWRITE')
    report = verify(args.config.resolve(strict=True))
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'protected_paths_rechecked', 'protected_path_changes',
                                            'tests', 'scientific_calls', 'strict_reuse_accepted_axes', 'mandatory_STOP')}))
