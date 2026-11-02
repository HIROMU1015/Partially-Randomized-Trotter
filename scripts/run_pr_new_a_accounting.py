"""Separate one-shot gate for new A total-cost accounting (not A/B/D replay)."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_new_a_accounting as a
c = a.c
ROOT = Path(__file__).resolve().parents[1]
FILES = ('scripts/pr_new_a_accounting.py', 'scripts/run_pr_new_a_accounting.py',
         'scripts/prepare_pr_new_a_accounting.py', 'tests/test_pr_new_a_accounting.py',
         'scripts/pr_cost_contract.py', 'scripts/pr_cost_saved.py',
         'src/trotterlib/pr_cost_integration.py', 'src/trotterlib/h8_six_axis_reaccount.py',
         'references/pr_cost_integration_evidence_v1.json')


def technical_digest(spec):
    value = dict(spec)
    value.pop('technical_digest', None)
    return c.digest(value)


def verify(spec):
    c.require(spec['schema'] == a.SCHEMA and spec['technical_sealed'] is True, 'TECHNICAL_SEAL')
    c.require(spec['technical_digest'] == technical_digest(spec), 'TECHNICAL_DIGEST')
    c.require(spec['approved'] is False and spec['runtime_authorization'] is False,
              'TECHNICAL_SEAL_NOT_APPROVAL')
    c.require(spec['epsilons'] == list(c.core.EPS) and spec['factors'] == [1., 10.]
              and spec['bias_modes'] == list(c.core.MODES) and spec['P'] == 0
              and spec['retry'] is False and spec['next_stage_authorized'] is False,
              'FIXED_SCIENTIFIC_EXECUTION_CONTRACT')
    c.require(set(spec['source_hashes']) == set(FILES), 'COMPLETE_SOURCE_CLOSURE')
    c.verify_source(ROOT, spec['SOURCE'], spec['source_hashes'])
    c.verify_byte_rows(spec['inputs'])
    c.require(c.sha(spec['python']) == spec['python_sha256'], 'PYTHON_BINARY_CHANGED')
    return dict(status='PASS_BYTES_ONLY', scientific_calls=0)


def approve(grant, system):
    """Executed before loading a technical spec or any scientific input."""
    c.require(system in a.OLD_LABELS and grant.get('system') == system
              and grant.get('scope') == system + '_NEW_A_TOTAL_COST', 'EXACT_NEW_SCOPE')
    c.require(all(grant.get(k) is True for k in ('approved', 'approved_by_user',
              'runtime_authorization', 'one_shot')) and grant.get('retry') is False
              and grant.get('next_stage_authorized') is False, 'NEW_EXPLICIT_APPROVAL_REQUIRED')
    c.require(grant.get('SOURCE') and grant.get('approval_receipt_sha256'), 'APPROVAL_SOURCE_RECEIPT')


def load_data(spec):
    """Only after approval for execution; preparation calls it for structural audit."""
    by = {r['role']: r['path'] for r in spec['inputs']}
    origin = c.read(by['origin_A_spec'])
    payload = c.read(by['A_payload'])
    c.core.same_binding(origin['binding'], spec['binding'])
    d_spec = c.read(by['D_spec'])
    c.core.same_binding(d_spec['binding'], spec['binding'])
    d_payload = c.read(by['D_payload'])
    c.require(d_payload['PR_ids'] == spec['old_PR_ids'] and
              d_payload['deterministic_sets'] == spec['deterministic_sets'], 'FIXED_COMPARATOR_IDS')
    result = c.read(by['A_result'])
    d_result = c.read(by['D_result'])
    for kind, value in [('A', result), ('D', d_result)]:
        terminal = c.read(by[kind + '_terminal'])
        c.require(value['status'] == terminal['status'] == 'COMPLETE_MANDATORY_STOP'
                  and terminal['result_sha256'] == c.sha(by[kind + '_result']), 'COMPLETED_RESULT_SHA')
        proof = spec['completion_acceptance'][kind]
        c.require(terminal['SOURCE'] == proof['SOURCE'] and proof['native_exit_code'] == 0
                  and proof['reaped'] is True and proof['result_sha256'] == terminal['result_sha256'],
                  'ACCEPTED_COMPLETION_SOURCE')
    c.require(c.digest(payload['signal_record_hashes']) == spec['signal_hashes_digest'], 'SAVED_SIGNAL_HASHES')
    return dict(kind='SAVED_REAL_VALUES', system=spec['system'], binding=spec['binding'],
        cells=spec['cells'], signals=c.read(by['A_signal'])['signals'],
        signal_record_hashes=payload['signal_record_hashes'], tasks=payload['new_seed_registry']['tasks'],
        pairs=result['paired_costs'], old_rows=d_result['rows'], old_PR_ids=d_payload['PR_ids'],
        deterministic_sets=d_payload['deterministic_sets'])


def execute(system, spec_path, grant_path):
    grant = c.read(grant_path)
    approve(grant, system)
    spec = c.read(spec_path)
    c.require(spec['system'] == system and spec['scope'] == grant['scope']
              and grant['SOURCE'] == spec['SOURCE']
              and grant['technical_digest'] == spec['technical_digest'], 'EXACT_SPEC_GRANT_BINDING')
    receipt = c.read(grant['approval_receipt'])
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256']
              and receipt['approved_by_user'] is True
              and receipt['SOURCE'] == spec['SOURCE']
              and receipt['selected_contracts'].get(spec['scope']) == spec['technical_digest'], 'USER_SELECTED_DIGEST_RECEIPT')
    c.require(receipt.get('scope') == a.SCHEMA and receipt.get('missing_endpoints_stay_unresolved') is True,
              'NEW_ACCOUNTING_NOT_OLD_A_APPROVAL')
    verify(spec)
    # Standard-library scalar code. Reuse bound interpreter, no provider import.
    c.require(Path(sys.executable).resolve() == Path(spec['python']).resolve(), 'BOUND_INTERPRETER_REQUIRED')
    c.require(sys.flags.safe_path and sys.dont_write_bytecode, 'PYTHON_P_B_REQUIRED')
    output, lock = c.home_path(spec['output']), c.home_path(spec['lock'])
    c.require(not output.exists() and not lock.exists() and output != lock
              and not output.is_relative_to(lock) and not lock.is_relative_to(output), 'UNUSED_OUTPUT_AND_LOCK')
    fresh = c.fresh(output.parent)
    c.require(fresh['disk_free_bytes'] >= spec['output_estimate_bytes']
              and fresh['free_inodes'] >= spec['inode_estimate']
              and fresh['memory_available_bytes'] >= spec['memory_estimate_bytes'], 'FRESH_ADMISSION')
    c.require(grant.get('quota_policy') in ('AVAILABLE_VERIFIED', 'NO_QUOTA_CONFIRMED',
              'USER_ACCEPTS_UNEXPOSED_QUOTA_ENOSPC_FAIL_CLOSED'), 'QUOTA_POLICY_REQUIRED')
    if grant['quota_policy'] == 'AVAILABLE_VERIFIED':
        c.require(grant['quota_remaining_bytes'] >= spec['output_estimate_bytes']
                  and grant['quota_remaining_inodes'] >= spec['inode_estimate'], 'QUOTA_ADMISSION')
    lock.parent.mkdir(parents=True, exist_ok=True)
    c.claim_attempt(lock, spec['SOURCE'], spec['scope'])
    try:
        output.mkdir(parents=True, exist_ok=False)
        c.save(output / 'startup_gate.json', fresh)
        budget = c.Budget(output, dict(saved_scalar_pass=1))
        budget.take('saved_scalar_pass')
        data = load_data(spec)
        c.require(a.audit_structure(data) == spec['structure_audit'], 'SEALED_STRUCTURE')
        result = a.account(data, a.Permit(token=a._REAL))
        c.save(output / 'result.json', result)
        verify(spec)
        c.save(output / 'terminal.json', dict(status='COMPLETE_MANDATORY_STOP',
            SOURCE=spec['SOURCE'], result_sha256=c.sha(output / 'result.json'), next_stage_authorized=False))
    except BaseException as error:
        if output.is_dir():
            c.first_stop(output, str(error) or type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=tuple(a.OLD_LABELS), required=True)
    parser.add_argument('--spec', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-bytes', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--grant', type=Path)
    args = parser.parse_args()
    try:
        if args.verify_bytes:
            spec = c.read(args.spec)
            c.require(spec['system'] == args.system, 'SYSTEM_SPEC')
            print(verify(spec))
        else:
            c.require(args.grant is not None, 'GRANT_REQUIRED')
            execute(args.system, args.spec, args.grant)
    except (Exception, KeyboardInterrupt) as error:
        print('STOP: ' + (str(error) or type(error).__name__), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
