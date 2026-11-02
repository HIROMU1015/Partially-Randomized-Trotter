#!/usr/bin/env python3
"""Saved-byte and structure audit only; never dispatch B/C or scientific arithmetic."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = '2bd212288e8530e519c4a8135257df37f7608662'
BASE = 'ebe20403bcb5c4439162614f64d991a769d7ed68'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def saved_checks():
    before = read(HERE/'preservation_before.json')
    prior = Path(before['prior_investigation_root'])
    old = load_module('old_preservation_only',
        prior/'artifacts/non_df_h4_evidence_investigation/2026-10-23/structure_checks.py')
    after = old.preservation()
    after['prior_investigation_root'] = str(prior)
    after['prior_investigation_files'] = {
        name: sha(prior/name) for name in before['prior_investigation_files']}
    require(after == before, 'OLD_SOURCE_RESULTS_CLAIMS_OR_PRIOR_REPORT_CHANGED')
    control = load_module('saved_B_control_only', ROOT/'src/trotterlib/h4_precision_control.py')
    B = control.verify(HERE/'B_run01')  # hashes, registry, saved native result only
    rows = read(HERE/'B_run01/diagnostics.json')['rows']
    mapped = read(HERE/'C_precision_method_map.json')['rows']
    key = lambda row: (row['case'], row['epsilon_sig'], row['axis'])
    expected = {(c, e, a) for c in ('pauli_jw_PF', 'pauli_jw_PR', 'df_native_PF', 'df_native_PR')
                for e in (.05, .01, .005, .001, .0005) for a in ('cosine', 'sine')}
    require(len(rows) == len(mapped) == 40, 'ROW_COUNT')
    require(set(map(key, rows)) == set(map(key, mapped)) == expected, 'REGISTRY')
    indexed = {key(row): row for row in rows}
    require(sum(row['group'] == 'R' for row in rows) == 22, 'R_COUNT')
    require(sum(row['group'] == 'H' for row in rows) == 18, 'H_COUNT')
    for row in mapped:
        b = indexed[key(row)]
        require(row['necessary_u_ceiling_exact'] == b['u_max_conservative']['exact'], 'BUDGET_COPY')
        require(row['design_total_error_target_exact'] == b['C_design_target_only']['exact'], 'TARGET_COPY')
        require(row['target_is_certificate'] is False, 'TARGET_NOT_CERTIFICATE')
        require(row['current_N'] is None and row['current_G'] is None
                and row['current_status'] == 'UNDETERMINED', 'OLD_STATUS')
    csv_counts = {}
    for name in ('B_run01/diagnostics.csv', 'C_precision_method_map.csv'):
        with (HERE/name).open(newline='') as stream:
            csv_counts[name] = len(list(csv.DictReader(stream)))
        require(csv_counts[name] == 40, 'CSV_ROW_COUNT')
    contract = read(HERE/'C_execution_contract_DRAFT.json')
    require(all(contract[name] is False for name in
        ('execution_authorized', 'grant_valid', 'dispatch_enabled', 'consumed')), 'C_ACTIVE')
    require(contract['grant_id'] is None and contract['launch_command'] is None, 'C_GRANT_OR_LAUNCH')
    require(contract['C_production_implementation_SOURCE'] is None
            and contract['C_production_runner_implemented'] is False, 'FALSE_IMPLEMENTATION_CLAIM')
    require(contract['C_design_spec_sha256'] == sha(ROOT/'docs/non_df_h4_numerical_enclosure_design_20261023.md'), 'C_DESIGN_HASH')
    require(contract['C_artificial_test_sha256'] == sha(ROOT/'tests/check_h4_precision_c_backend.py'), 'C_ARTIFICIAL_CODE_HASH')
    for name, digest in contract['input_files_sha256'].items():
        require(sha(ROOT/name) == digest == sha(old.OLD/name), 'C_OLD_INPUT_HASH:'+name)
    require(sha(HERE/'B_run01/diagnostics.json') == contract['B_outputs_sha256'], 'B_TO_C_BINDING')
    tests = read(HERE/'technical_tests_final.json')
    require(tests['status'] == 'PASS' and tests['tests'] == 33
            and tests['failures'] == tests['errors'] == tests['skips'] == 0, 'TECHNICAL_TEST_STATUS')
    backend = read(HERE/'C_artificial_checks_final.json')
    require(backend['status'] == 'PASS_C_ARTIFICIAL_BACKEND_ONLY' and backend['check_count'] == 13
            and len(backend['checks']) == 13, 'C_TEST_STATUS')
    require(backend['real_H4_inputs_loaded'] is False and backend['real_H4_C_calls'] == 0
            and backend['real_H4_certificates_generated'] == 0, 'C_REAL_SCOPE')
    for name, digest in backend['runtime_files'].items():
        require(sha(name) == digest, 'PRIVATE_BACKEND_RUNTIME_CHANGED:'+name)
    current = json.loads((ROOT/'artifacts/validation_manifest.json').read_text())
    original = json.loads(subprocess.check_output(
        ['git', 'show', SOURCE+':artifacts/validation_manifest.json'], cwd=ROOT))
    entry = current.pop('non_df_h4_precision_bc_20261023')
    require(current == original, 'UNRELATED_VALIDATION_STATUS_CHANGED')
    require(entry['real_H4_C_calls'] == entry['new_N_G_accounting'] == 0, 'MANIFEST_SCOPE')
    docs = [ROOT/'docs/non_df_h4_precision_bc_20261023.md',
            ROOT/'docs/non_df_h4_numerical_enclosure_design_20261023.md',
            ROOT/'docs/research/研究ノート/2026-10-23_H4数値精度BC.md',
            HERE/'README.md', HERE/'CONTINUATION.md']
    links = []
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https://', 'http://', '#')):
                continue
            linked = (path.parent/target.split('#')[0]).resolve()
            # These three audit outputs are created only after all checks pass.
            require(linked.exists() or linked in (HERE/'completion_audit.json', HERE/'STOP.json',
                    HERE/'preservation_after.json'),
                    'MISSING_REPORT_LINK:'+str(path)+':'+target)
            links.append(str(linked.relative_to(ROOT)))
    return after, dict(
        status='PASS_BC_SAVED_STRUCTURE_AND_PRESERVATION',
        scope='Primary-agent byte/structure checks only; no new scalar diagnosis, H4 enclosure or accounting',
        checked_UTC=datetime.now(timezone.utc).isoformat(),
        branch='codex/non-df-h4-precision-bc-20261023',
        base_commit=BASE, B_SOURCE=SOURCE,
        B_saved_verification=B, B_actual_axes=40, independent_arithmetic_checks_saved=40,
        exact_boundary_checks_saved=40, R_axes=22, H_axes=18, csv_rows=csv_counts,
        old_scientific_SOURCE=after['scientific_SOURCE'],
        old_SOURCE_files=after['source_file_count'], old_evidence_and_claim_files=after['evidence_file_count'],
        old_investigation_files=len(after['prior_investigation_files']), preservation_matches_before=True,
        B_input_file_count=10, C_input_file_count=len(contract['input_files_sha256']),
        backend_runtime_files=len(backend['runtime_files']), unit_and_regression_tests_saved=33,
        C_artificial_backend_checks_saved=13, method_mapping_axes=40,
        C_real_H4_calls=0, new_N_G_accounting=0, old_precision_rows_N_G_null=20,
        old_axes_UNDETERMINED=40, C_grant_issued=False, C_production_runner_implemented=False,
        original_native_completion_verified=False, original_supervisor_cause='UNKNOWN',
        unrelated_manifest_entries_unchanged=True, checked_local_links=links,
        push_performed=False, final_scientific_evidence_freeze=False, mandatory_STOP=True)


def write_exclusive(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='Exclusively record this new BC audit/STOP only')
    args = parser.parse_args()
    after, audit = saved_checks()
    if args.record:
        write_exclusive(HERE/'preservation_after.json', after)
        write_exclusive(HERE/'completion_audit.json', audit)
        write_exclusive(HERE/'STOP.json', dict(
            status='B_COMPLETE_C_DESIGN_COMPLETE_MANDATORY_STOP',
            UTC=datetime.now(timezone.utc).isoformat(), scope='NEW_BC_WORK_ONLY_NOT_ORIGINAL_H4',
            B_actual_saved_scalar_axes=40, B_native_exit_code=0, B_direct_child_waited=True,
            C_real_H4_calls=0, C_certificates_generated=0, new_N_G_accounting=0,
            C_grant_issued=False, execution_authorized=False, grant_valid=False, dispatch_enabled=False,
            automatic_retry=False, original_native_completion_verified=False,
            original_supervisor_cause='UNKNOWN', next_execution_requires_explicit_approval=True,
            push_performed=False))
    print(json.dumps({name: audit[name] for name in ('status', 'B_actual_axes',
        'old_SOURCE_files', 'old_evidence_and_claim_files', 'old_investigation_files',
        'C_real_H4_calls', 'new_N_G_accounting', 'mandatory_STOP')}, sort_keys=True))


if __name__ == '__main__':
    main()
