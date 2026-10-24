"""Metadata/SHA audit only; never replay, load QPY, compile or rerun statistics."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
NS = BASE / 'run01'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = types.ModuleType('cohort_audit_private')
    package.__path__ = [str(ROOT / 'src/trotterlib')]
    sys.modules[package.__name__] = package
    from cohort_audit_private import h4_a_cohort as D
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    seal, contract, grant = read(BASE / 'seal.json'), read(BASE / 'active_contract.json'), read(NS / 'grant.json')
    D.verify_seal(seal)
    D.scope(contract)
    check('SOURCE_code_inputs_runtime_old_preservation', True)
    check('grant_digest', grant['grant_id'] == D.C.digest({k: v for k, v in grant.items() if k != 'grant_id'}))
    check('grant_binding', grant['SOURCE'] == seal['SOURCE'] == contract['science_SOURCE']
          and grant['contract_sha256'] == sha(BASE / 'active_contract.json')
          and grant['seal_sha256'] == contract['seal_sha256'] == sha(BASE / 'seal.json')
          and grant['approval_sha256'] == contract['approval_sha256'] == seal['approval_sha256'])
    check('grant_consumed_nonrefundable', read(NS / 'CONSUMED.json')['grant_id'] == grant['grant_id']
          and read(NS / 'CONSUMED.json')['nonrefundable'] is True)
    ledger, stop = read(NS / 'call_ledger.json'), read(NS / 'STOP.json')
    check('exact_call_limits', ledger['used'] == ledger['limits'] == D.LIMITS == grant['call_limits'])
    counts = Counter(row['call'] for row in ledger['events'])
    check('calls_reserved_before_execution', all(row['reserved_before_call'] for row in ledger['events'])
          and all(counts[key] == count for key, count in D.LIMITS.items()))
    check('mandatory_STOP_no_next_stage', stop['status'] == 'A_COHORT_COS_COMPLETE_MANDATORY_STOP'
          and stop['mandatory_STOP'] and stop['automatic_next_stage'] is False
          and stop['ledger'] == ledger and stop['preservation_error'] is None)
    check('old_unknown_and_certificate_not_promoted', stop['original_H4_native_verified'] is False
          and stop['formal_certificate_accepted'] is False)
    check('engine_native_exit_reaped', stop['native_engine_exit']['exit_code'] == 0
          and stop['native_engine_exit']['reaped'] and stop['all_owned_reaped'])
    reused = D.reuse_results()
    check('saved_index0_index1_reuse', read(BASE / 'reuse_compatibility.json')['records'] == reused
          and all(row['new_scientific_calls'] == 0 for row in reused))
    targets = D.records()
    rows = read(NS / 'cohort_costs.json')['rows']
    stats = read(NS / 'statistics.json')
    check('all_256_unique_complete_cosine_rows', len(rows) == 256 and [r['index'] for r in rows] == list(range(256))
          and [r['trajectory_id'] for r in rows] == [r['trajectory_id'] for r in targets]
          and len({r['trajectory_id'] for r in rows}) == 256 and all(r['axis'] == 'cosine' for r in rows))
    check('statistics_complete_finite_cohort_only', stats['status'] == 'COMPLETE_EXACT_SAVED_COHORT'
          and stats['n'] == 256 and stats['new_successes'] == 254 and stats['reused'] == 2
          and stats['cross_representation_pairing'] is False and stats['new_N_G'] is False
          and stats['population_mean_upper'] is None and stats['compiled_error_certificate'] is False)
    # Exact saved-sum identities verify accounting; no CI/quantile recomputation.
    check('saved_exact_mean_sum_identities', Fraction(sum(r['RZ'] for r in rows), 256) == Fraction(stats['central_cosine_RZ']['mean_exact'])
          and Fraction(sum(r['generic_RZ'] for r in rows), 256) == Fraction(stats['generic_cosine_RZ']['mean_exact']))
    identities = [read(NS / 'native/worker_identity.json')]
    schema = read(BASE / 'output_schema.json')
    errors = []
    for index in range(256):
        row, target = rows[index], targets[index]
        join = read(ROOT / target['price_join'])
        order_counts = Counter(e['taylor_order'] for e in join['events'])
        okay = (row['generic_RZ'] == join['compiled_RZ']['cosine']
                and row['event_record_hash'] == join['event_record_hash']
                and row['trajectory_record_hash'] == join['trajectory_record_hash']
                and all(row[f'order{k}_events'] == order_counts[k] for k in (0, 2, 4)))
        if index < 2:
            m = reused[index]['metrics']
            okay = okay and row['reused'] is True and row['SOURCE'] == reused[index]['SOURCE']
        else:
            folder = NS / 'trajectories' / f'{index:04d}'
            task = NS / 'native_tasks' / ('compile_' + f'{index:04d}')
            m, inp = read(folder / 'metrics.json'), read(folder / 'input.json')
            binding, mapping = read(folder / 'source_binding.json'), read(folder / 'controlled_mapping.json')
            receipt, reservation, launch = read(task / 'native_exit.json'), read(task / 'reservation.json'), read(task / 'launch.json')
            okay = (okay and all((folder / path).is_file() for path in schema['per_new_trajectory'])
                    and all(key in m for key in schema['metrics']) and m['target'] == target
                    and row['reused'] is False and row['SOURCE'] == seal['SOURCE'] == m['SOURCE'] == inp['SOURCE']
                    and row['trajectory_id'] == inp['trajectory_id'] == target['trajectory_id']
                    and binding['numeric_tokens_match'] is True and binding['new_certificate'] is False
                    and binding['source_instruction_count'] == mapping['source_instruction_count'] == len(mapping['instruction_mapping'])
                    and binding['replay_sidecar']['aggregate_phase'] == mapping['aggregate_source_global_phase']
                    and mapping['phase_owner'] == 'saved_body_aggregate_only_no_event_phase_readdition'
                    and inp['source_body_sha256'] == sha(folder / 'source_body.qpy')
                    and m['input_qpy_sha256'] == inp['input_qpy_sha256'] == sha(folder / 'input.qpy')
                    and m['output_qpy_sha256'] == sha(folder / 'output.qpy')
                    and m['input_fingerprint'] == inp['input_fingerprint'] and len(m['output_fingerprint']) == 64
                    and m['pretranspile_histogram'] == inp['pretranspile_histogram']
                    and m['RZ'] == m['posttranspile_histogram']['rz'] and m['CX'] == m['posttranspile_histogram']['cx']
                    and m['axis'] == inp['axis'] == 'cosine' and m['compiler'] == contract['fixed']['compiler']
                    and receipt['exit_code'] == 0 and receipt['native_wait'] and receipt['reaped']
                    and reservation['grant_id'] == grant['grant_id'] and reservation['case'] == f'{index:04d}')
            identities.append(launch['identity'])
        okay = okay and all(row[key] == m[key] for key in ('RZ', 'CX', 'depth', 'rz_depth', 'cx_depth'))
        if not okay:
            errors.append(index)
    check('per_trajectory_source_phase_QPY_profile_ID_metrics_native', not errors)
    check('new_task_scope_and_native_count', {p.name for p in (NS / 'trajectories').iterdir()} == set(D.CASES)
          and len(list((NS / 'native_tasks').glob('*/native_exit.json'))) == 254
          and stop['native_task_count'] == 254 and stop['all_tasks_exit0_reaped'])
    inventory = read(NS / 'output_inventory.json')['files']
    check('run_inventory_bytes', all(sha(NS / path) == digest for path, digest in inventory.items()))
    check('run_inventory_complete', set(inventory) == {str(p.relative_to(NS)) for p in NS.rglob('*')
          if p.is_file() and p.name != 'output_inventory.json'})
    old_manifest = json.loads(subprocess.check_output(['git', 'show',
        '5eaf32d2f01e95f514634bbbdefd43e971da5d16:artifacts/validation_manifest.json'], cwd=ROOT))
    current_manifest = read(ROOT / 'artifacts/validation_manifest.json')
    check('all_prior_manifest_entries_unchanged', all(current_manifest[key] == value for key, value in old_manifest.items()))
    remaining = [row for row in identities if D.C._proc_identity(row['pid']) == row]
    check('no_recorded_owned_process_remaining', not remaining)
    check('no_unobserved_order4_cost_imputation', all(group['central_RZ'] is None for group in stats['order_groups'].values() if group['n'] == 0))
    return dict(status='PASS', scope='PRIMARY_AUTHOR_METADATA_SHA_AND_SAVED_SUM_IDENTITIES_NOT_NEW_STATISTICS_OR_CERTIFICATE',
                SOURCE=seal['SOURCE'], checks=checks, failed_indices=errors, remaining_owned_processes=remaining,
                counts=dict(code_files=len(seal['code_files']), input_files=len(seal['input_files']),
                            runtime_files=len(read(ROOT / seal['runtime_manifest'])['files']),
                            preserved_files=len(read(BASE / 'preservation_before.json')),
                            run_inventory_files=len(inventory), native_tasks=254, new_rows=254, reused_rows=2,
                            prior_manifest_entries=len(old_manifest)),
                raw_token_digest_equality_required=False,
                fingerprint_scope='SAVED_METADATA_BINDING_AND_QPY_SHA_NO_QPY_RELOAD',
                new_scientific_calls=0, new_statistics_batches=0)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))
