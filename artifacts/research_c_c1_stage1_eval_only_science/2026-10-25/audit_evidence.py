"""Audit JSON declarations and file bytes only; never decode numerical arrays."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE/'run01'
BASE = 'bec41ac7dc7e034bd664817acfdcfed858ddadd0'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (HERE/name).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


result = read(OUT/'result.json')
contract = read(OUT/'contract.json')
records = result['records']
gates = [row for row in records if row['gate']]
assert result['status'] == 'STAGE1_REPRESENTATION_PASS'
assert len(records) == 39 and len(gates) == 37
assert all(row['passed'] is True for row in gates)
rows = {phase: [json.loads(line) for line in (OUT/f'{phase}_calls.jsonl').read_text().splitlines()]
        for phase in ['load', 'evaluation']}
counts = {}
for phase, phase_rows in rows.items():
    counter = Counter()
    for row in phase_rows:
        counter[row['kind']] += row['count']
    counts[phase] = dict(counter)
assert counts == contract['call_limits']
assert counts['evaluation'] == result['calls']
assert sum(sum(v.values()) for v in counts.values()) == result['scientific_counter_units'] == 141
spectral = [row['label'] for row in rows['evaluation'] if row['kind'] == 'spectral_norm']
assert spectral == ['full_base']+[f'full_frame_{i}' for i in range(8)]+['sector_adapter', 'sector_DF_CDF', 'sector_DF_XDF']
assert [row['id'] for row in records][19:28] == ['full_base']+[f'full_frame_{i}' for i in range(8)]
operators = [json.loads(line) for line in (OUT/'operator_inventory.jsonl').read_text().splitlines()]
assert len(operators) == len(list((OUT/'operators').glob('*.npz'))) == 24
assert all(sha(OUT/'operators'/row['file']) == row['sha256'] for row in operators)
assert not any((OUT/name).exists() for name in ['fit.claim', 'fit_export.npz', 'failure.json'])
stop = read(OUT/'STOP.json')
assert stop['mandatory_stop'] and stop['retry'] == stop['resume'] == stop['new_fit_calls'] == 0
assert read(OUT/'approval.json') == read(HERE/'approval_run01.json')
source = read(OUT/'evaluation_SOURCE.json')
assert all(sha(ROOT/name) == want for name, want in source['files'].items())
old = read(Path('/tmp/prt-c1-eval-science-old-artifacts-20261025.json'))
assert all(sha(ROOT/name) == want for name, want in old.items())
original = Path('/home/abe/Project/Partially Randomized Trotter')
baseline = read(Path('/tmp/prt-c1-eval-science-preservation-20261025.json'))
assert all(sha(original/name) == want for name, want in baseline.items() if name != 'tracked_diff_sha256')
assert hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=original)).hexdigest() == baseline['tracked_diff_sha256']
audit = dict(status='PASS_JSON_AND_BYTE_AUDIT_ONLY', gate_pass=37, gate_fail=0,
             gate_not_run=0, non_gate_physical_error_reports=2, load_call_units=2,
             evaluation_call_units=139, total_call_units=141, maximum_call_units=141,
             ledger_lines={phase: len(value) for phase, value in rows.items()},
             counting='Sum row.count, including batched xdf_matvec36/full_sum2/sector_restrict2; not JSONL line count',
             spectral_order=spectral, operator_files_byte_verified=24,
             source_files_byte_verified=len(source['files']), call_counts=counts,
             new_fit_calls=0, optimizer_calls=0, retry=0, resume=0,
             additional_scientific_calls=0, numerical_matrix_decodes_during_audit=0)
summary = dict(schema='C1_STAGE1_SAVED_FIT_EVAL_ONLY_EXECUTION_SUMMARY_V1',
               scope=contract['scope'], date_JST='2026-10-25', status=result['status'],
               source_commit=BASE, provenance=result['provenance'],
               contract_sha256=sha(ROOT/'docs/research/research_c_c1_stage1_eval_only_contract_unapproved_20261025.json'),
               physical_conditions=contract['input'], scientific_conditions=contract['scientific_conditions'],
               actual_CDF_rank=8, saved_XDF_rank=10, gate_PASS=37, gate_FAIL=0, gate_NOT_RUN=0,
               non_gate_reports=2, full_base_and_frames_PASS=True,
               previous_frame0_mismatch_resolved_in_this_fixed_evaluation=True,
               old_failure_record_unchanged=True, frame0=records[20],
               sector_adapter=next(row for row in records if row['id'] == 'sector_adapter'),
               CDF_physical_sector_error_Hartree=records[-2]['error'],
               XDF_physical_sector_error_Hartree=records[-1]['error'],
               physical_DF_absolute_gate=None, allowance_version=result['allowance_version'],
               rigorous_certificate=False, new_fit_calls=0, optimizer_calls=0,
               saved_fit_load_calls=1, scientific_counter_units=141, maximum_scientific_counter_units=141,
               scientific_retry=0, scientific_resume=0, operators_saved=24,
               artificial_regression_tests=19, mandatory_stop=True,
               Stage2_authorized=False, Stage3_authorized=False,
               Stage2_eligibility='Separate user/research-C decision on physical DF error budget and circuit-validation scope',
               local_execution_not_external_reproduction=True)
save('evidence_audit.json', audit)
save('execution_summary.json', summary)
save('call_summary.json', dict(counts=counts, load=2, evaluation=139, total=141, maximum=141,
                               new_fit=0, new_optimizer=0, retry=0, resume=0,
                               historical_fit_units_not_new_calls=6))
save('preservation.json', dict(status='PASS', frozen_source_files_verified=len(source['files']),
                              old_research_c_artifact_files_verified=len(old),
                              old_artifacts_unchanged=True, root_workspace_files_and_tracked_diff_unchanged=True,
                              root_baseline=baseline, source_commit=BASE, shared_environment_modifications=0,
                              old_grant_result_ledger_marker_STOP_reused_or_overwritten=False))
print(json.dumps(audit, ensure_ascii=False))
