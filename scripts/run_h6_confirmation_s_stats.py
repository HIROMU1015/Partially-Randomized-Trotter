"""Exactly one explicitly approved H6 saved-scalar design pass, then STOP."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import pr_cost_contract as c

POLICY = 'references/h6_confirmation_s_stats_policy_v1.json'
FILES = ('scripts/run_h6_confirmation_s_stats.py', 'scripts/h6_confirmation_sample_design.py',
         'scripts/h8_confirmation_sample_design.py', 'scripts/pr_cost_contract.py',
         'src/trotterlib/pr_cost_integration.py', 'tests/test_h6_confirmation_sample_design.py', POLICY)
SCOPE = 'H6_CONFIRMATION_S_STATS'


def verify(spec, grant):
    c.require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True
              and grant.get('one_shot') is True and grant.get('stage') == SCOPE
              and grant.get('science_sampling_compile_authorized') is False
              and grant.get('retry') is False and grant.get('next_stage_authorized') is False,
              'EXPLICIT_NUMERICAL_ONLY_APPROVAL_REQUIRED')
    c.require(spec['scope'] == SCOPE and grant['SOURCE'] == spec['SOURCE']
              and grant['spec_digest'] == c.digest(spec) and set(spec['source_hashes']) == set(FILES), 'SCOPED_SOURCE_SPEC')
    c.verify_source(ROOT, spec['SOURCE'], spec['source_hashes'])
    c.verify_byte_rows(spec['inputs'])
    c.require(c.sha(spec['python']) == spec['python_sha256'], 'PYTHON_BINARY')
    for name, expected in spec['runtime_hashes'].items():
        c.require(c.sha(name) == expected, 'NUMERICAL_RUNTIME_BYTES')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT')
    approval = c.read(grant['approval_receipt'])
    policy = c.read(ROOT/POLICY)
    c.require(approval['scope'] == SCOPE and approval['approved_by_user'] is True
              and approval['SOURCE'] == spec['SOURCE'] and approval['plan_sha256'] == policy['plan_sha256']
              and approval['policy_sha256'] == c.sha(ROOT/POLICY) and approval['saved_scalar_passes'] == 1
              and approval['science_sampling_compile_authorized'] is False, 'EXACT_NUMERICAL_APPROVAL_SCOPE')
    c.require(spec['plan'] == str(ROOT/policy['plan_path']) and c.sha(spec['plan']) == policy['plan_sha256'],
              'FIXED_PHASE0_PLAN_BYTES')
    covered = {(r['path'],r['sha256']) for r in spec['inputs']}
    c.require((spec['plan'],policy['plan_sha256']) in covered
              and (grant['approval_receipt'],grant['approval_receipt_sha256']) in covered, 'COVERED_APPROVAL_INPUT')
    c.require(spec['calls'] == dict(saved_scalar_pass=1) and policy['allowed_passes'] == 1, 'EXACT_ONE_PASS_BUDGET')
    output, lock = c.home_path(spec['output']), c.home_path(spec['lock'])
    c.require(not output.is_relative_to(lock) and not lock.is_relative_to(output), 'SEPARATE_OUTPUT_LOCK')


def run(spec, grant):
    verify(spec, grant)
    output = c.home_path(spec['output'])
    c.require(not output.exists() and not Path(spec['lock']).exists(), 'UNUSED_OUTPUT_ONE_SHOT')
    c.claim_attempt(spec['lock'], spec['SOURCE'], SCOPE)
    output.mkdir(exist_ok=False)
    try:
        budget = c.Budget(output, spec['calls'])
        budget.take('saved_scalar_pass')
        from h6_confirmation_sample_design import analyze
        result = analyze(c.read(spec['plan']), c.read(ROOT/POLICY))
        result.update(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec), actual_calls=budget.used)
        c.save(output/'result.json', result)
        verify(spec, grant)
        c.save(output/'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
            result_sha256=c.sha(output/'result.json'), actual_calls=budget.used,
            scientific_execution_approved=False, next_stage_authorized=False))
        return 0
    except BaseException as error:
        c.first_stop(output, type(error).__name__+':'+str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(c.read(args.spec), c.read(args.grant)))
