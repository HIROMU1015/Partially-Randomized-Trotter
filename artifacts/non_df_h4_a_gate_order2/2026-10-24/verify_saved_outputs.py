"""Read-only metadata/SHA audit. No QPY load, replay, compile or science.

Reuses the frozen A_GATE/D2 seal verifier (stdlib-only import path), then checks
the saved paired outputs and native receipts. This is the primary author's
technical audit, not an independent review or a numerical certificate.
"""
import collections
import hashlib
import json
import pathlib
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = pathlib.Path(__file__).resolve().parent
NS = BASE / 'run01'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    package = types.ModuleType('a_gate_audit_private')
    package.__path__ = [str(ROOT / 'src/trotterlib')]
    sys.modules[package.__name__] = package
    from a_gate_audit_private import h4_a_gate as gate

    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise AssertionError(name)

    seal = read(BASE / 'seal.json')
    gate.verify_seal(seal)  # Original D2 code/input/runtime/approval/SOURCE checks.
    check('frozen_seal_and_runtime', True)
    contract = read(BASE / 'active_contract.json')
    gate.scope(contract)
    grant = read(NS / 'grant.json')
    check('grant_digest', grant['grant_id'] == gate.C.digest({k: v for k, v in grant.items() if k != 'grant_id'}))
    check('grant_binding', grant['SOURCE'] == seal['SOURCE'] == contract['science_SOURCE']
          and grant['contract_sha256'] == sha(BASE / 'active_contract.json')
          and grant['seal_sha256'] == contract['seal_sha256'] == sha(BASE / 'seal.json')
          and grant['approval_sha256'] == contract['approval_sha256'] == seal['approval_sha256'])
    check('consumed_once_nonrefundable', read(NS / 'CONSUMED.json')['grant_id'] == grant['grant_id']
          and read(NS / 'CONSUMED.json')['nonrefundable'] is True)
    ledger = read(NS / 'call_ledger.json')
    stop = read(NS / 'STOP.json')
    check('exact_call_limits', ledger['used'] == ledger['limits'] == gate.LIMITS == grant['call_limits'])
    counts = collections.Counter(event['call'] for event in ledger['events'])
    check('all_calls_reserved_before_execution', all(event['reserved_before_call'] for event in ledger['events'])
          and all(counts[key] == count for key, count in gate.LIMITS.items()))
    check('STOP_complete_without_continuation', stop['status'] == 'A_GATE_COMPLETE_MANDATORY_STOP'
          and stop['mandatory_STOP'] is True and stop['automatic_next_stage'] is False
          and stop['ledger'] == ledger and stop['preservation_error'] is None)
    check('old_native_and_certificate_not_promoted', stop['original_H4_native_verified'] is False
          and stop['formal_certificate_accepted'] is False)
    stage_a = read(NS / 'stage_A.json')
    stage_b = read(NS / 'stage_B.json')
    source = read(NS / 'body_source_check.json')
    mapping = read(NS / 'controlled_mapping.json')
    check('saved_order2_target', stage_a['status'] == 'PASS_FIXED_ORDER2_SOURCE'
          and stage_a['trajectory_id'] == gate.ID and stage_a['event_record_hash'] == gate.EVENT
          and stage_a['trajectory_record_hash'] == gate.TRAJECTORY
          and 2 in stage_a['event_orders'] and 4 not in stage_a['event_orders'])
    check('body_numeric_tokens_match', source['numeric_tokens_match'] is True
          and source['historical_program_identity'] == 'NOT_ESTABLISHED')
    check('central_source_instruction_and_phase_mapping', stage_b['status'] == 'PASS_BODY_ONLY_AND_CENTRAL_ONCE'
          and source['source_instruction_count'] == mapping['source_instruction_count'] == len(mapping['instruction_mapping'])
          and source['sidecar']['aggregate_phase'] == mapping['aggregate_source_global_phase'] == stage_b['aggregate_phase']
          and mapping['ancilla'] == 8 and mapping['phase_owner'] == 'saved_body_aggregate_only_no_event_phase_readdition'
          and stage_b['new_certificate'] is False)
    check('shared_QPY_bytes', sha(NS / 'source_body.qpy') == stage_b['source_body_sha256']
          and sha(NS / 'controlled_body.qpy') == stage_b['controlled_body_sha256'])
    schema = read(BASE / 'output_schema.json')
    paired = {}
    for axis in gate.AXES:
        folder = NS / 'variants' / axis
        inputs, metrics = read(folder / 'input.json'), read(folder / 'metrics.json')
        check(axis + '_output_schema', all((folder / name).is_file() for name in schema['per_axis'])
              and all(key in metrics for key in schema['metrics']))
        check(axis + '_bytes_and_saved_fingerprint_binding', sha(folder / 'input.qpy') == inputs['input_qpy_sha256']
              and sha(folder / 'output.qpy') == metrics['output_qpy_sha256']
              and inputs['input_fingerprint'] == metrics['input_fingerprint']
              and len(metrics['output_fingerprint']) == 64
              and inputs['controlled_body_sha256'] == stage_b['controlled_body_sha256'])
        check(axis + '_fixed_compiler_and_target', metrics['SOURCE'] == inputs['SOURCE'] == seal['SOURCE']
              and metrics['axis'] == inputs['axis'] == axis
              and metrics['trajectory_id'] == inputs['trajectory_id'] == gate.ID
              and metrics['compiler'] == contract['fixed']['compiler'])
        check(axis + '_saved_histogram_consistency', metrics['RZ'] == metrics['posttranspile_histogram']['rz']
              and metrics['CX'] == metrics['posttranspile_histogram']['cx']
              and metrics['pretranspile_histogram'] == inputs['pretranspile_histogram'])
        task = NS / 'native_tasks' / ('compile_' + axis)
        receipt = read(task / 'native_exit.json')
        check(axis + '_native_exit_reaped', receipt['exit_code'] == 0 and receipt['reaped'] and receipt['native_wait'])
        check(axis + '_single_reserved_compile', read(task / 'reservation.json')['grant_id'] == grant['grant_id']
              and read(folder / 'compile_claim.json')['axis'] == axis)
        paired[axis] = metrics
    check('both_wrappers_saved', read(NS / 'stage_C_inputs.json')['axes'] == list(gate.AXES))
    check('engine_native_exit_reaped', stop['native_engine_exit']['exit_code'] == 0
          and stop['native_engine_exit']['reaped'] and stop['all_owned_reaped']
          and len(stop['task_native_receipts']) == 2)
    identities = [read(NS / 'native/worker_identity.json')]
    identities += [read(NS / 'native_tasks' / ('compile_' + axis) / 'launch.json')['identity'] for axis in gate.AXES]
    observations = []
    for identity in identities:
        path = pathlib.Path('/proc') / str(identity['pid']) / 'stat'
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            same = int(fields[19]) == identity['start_ticks']
        except FileNotFoundError:
            same = False
        observations.append(dict(**identity, same_process_present=same))
    check('no_recorded_owned_process_remaining', not any(row['same_process_present'] for row in observations))
    inventory = read(NS / 'output_inventory.json')['files']
    check('run_inventory_bytes', all(sha(NS / path) == digest for path, digest in inventory.items()))
    check('run_inventory_complete', set(inventory) == {str(path.relative_to(NS)) for path in NS.rglob('*')
          if path.is_file() and path.name != 'output_inventory.json'})
    comparison = read(NS / 'comparisons.json')
    check('paired_saved_comparison_binding', comparison['metrics'] == paired and comparison['new_N_G'] is False)
    old_manifest = json.loads(subprocess.check_output(['git', 'show',
        '966c2cb323e3fe34032dc3be9f6b962936415e77:artifacts/validation_manifest.json'], cwd=ROOT))
    current_manifest = read(ROOT / 'artifacts/validation_manifest.json')
    check('prior_manifest_entries_unchanged', all(current_manifest[key] == value for key, value in old_manifest.items()))
    return dict(status='PASS', scope='PRIMARY_AUTHOR_METADATA_AND_SHA_ONLY_NOT_NUMERICAL_CERTIFICATE',
                SOURCE=seal['SOURCE'], checks=checks, process_observations=observations,
                counts=dict(code_files=len(seal['code_files']), input_files=len(seal['input_files']),
                            runtime_files=len(read(ROOT / seal['runtime_manifest'])['files']),
                            preserved_files=len(read(BASE / 'preservation_before.json')),
                            run_inventory_files=len(inventory), prior_manifest_entries=len(old_manifest)),
                new_scientific_calls=0, raw_token_digest_equality_required=False,
                fingerprint_check_scope='SAVED_METADATA_BINDING_AND_QPY_SHA_NO_QPY_RELOAD')


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2, sort_keys=True))
