"""Read-only checks of the H8 proposal, not a scientific execution entry point.

Standard library only. Does not read molecular arrays or saved scientific cost
values, import providers, create grants, fit models, or launch workers.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(rows):
    result = set()
    for row in rows:
        require(set(row) == {'id', 'method', 'prefix', 'q', 'formula'}, 'CANDIDATE_SCHEMA')
        require(type(row['prefix']) is int and type(row['q']) is int, 'INTEGER_GRID')
        key = (row['method'], row['prefix'], row['q'], row['formula'])
        require(key not in result, 'DUPLICATE_CANDIDATE')
        require(row['id'] == '{}_p{}_q{}_{}'.format(*key), 'CANDIDATE_ID')
        result.add(key)
    return result


def validate(plan):
    require(plan['status'] == 'PROPOSAL_ONLY_NOT_EXECUTABLE', 'PLAN_STATUS')
    auth = plan['authorization']
    require(all(auth[k] is False for k in ('approved', 'runtime_authorization',
                'grant_issued', 'sealed', 'next_stage_authorized')), 'EXECUTION_NOT_AUTHORIZED')
    require(auth['SOURCE'] is None and auth['allowed_cpus'] == [] and
            auth['absolute_launch_command'] is None, 'UNBOUND_RUNTIME')
    require(auth['mandatory_stop'] is True and auth['retry'] is False and
            auth['resume'] is False, 'STOP_ONE_SHOT_POLICY')
    reg = plan['registrations']
    old = keys(reg['existing_deterministic'])
    new = keys(reg['additional_deterministic'])
    qs = (1, 2, 4, 8)
    expected_old = {(m, p, q, '2nd') for m in ('B0', 'B0I')
                    for p in (7, 14, 21) for q in qs}
    expected_old |= {('B1', 28, q, f) for f in ('2nd', '4th') for q in qs}
    requested = {(m, p, q, f) for m in ('B0', 'B0I')
                 for f, ps in (('2nd', range(8, 21)), ('4th', range(15, 22)))
                 for p in ps for q in qs}
    require(old == expected_old and new == requested - old, 'FIXED_GRID_CHANGED')
    require(old.isdisjoint(new), 'DUPLICATE_SCIENCE')
    require(keys(reg['reused_requested']) == requested & old, 'REUSE_GAP')
    caps = plan['phase_A']['call_caps']
    require(caps['B0_signals'] == len(new)//2 == 76 and
            caps['B0I_analytic_overlays'] == 76, 'SIGNAL_DEDUP')
    require(caps['truncated_reference_pairs_max'] ==
            len({p for m, p, q, f in new}) == 13, 'PREFIX_DEDUP')
    require(all(caps[k] == 0 for k in ('compile', 'native_full_wrapper_build',
                'SCF', 'DF_decomposition', 'state_solver', 'PR_sampling',
                'PR_law_or_normalization', 'cost_fit', 'cost_reaccount')), 'PHASE_A_SCOPE')
    require(plan['phase_A']['reference_policy'] == 'REUSE_HASH_BOUND_SAVED_FULL_H_REFERENCE',
            'FULL_REFERENCE_POLICY')
    b = plan['phase_B2']
    require(b['max_compile_axes'] == 4 and b['compile_concurrency'] == 1 and
            b['stop_relative_error'] == 'abs(predicted-measured)/measured >= 1/50' and
            b['check_before_next_axis'] is True and b['automatic_refit'] is False,
            'PER_AXIS_IMMEDIATE_STOP')
    require(b['axis_selection'] == 'B0_MIN_ELIGIBLE_S4_PREFIX_Q__B0I_MAX_ELIGIBLE_S4_PREFIX_Q__BOTH_AXES',
            'PREREGISTERED_SELECTION')
    for phase in ('phase_A', 'phase_B1', 'phase_B2', 'phase_C'):
        value = plan[phase]
        require(value['separate_explicit_approval_required'] is True and
                value['mandatory_stop'] is True, 'PHASE_BOUNDARY')
    require(plan['phase_B1']['coefficients'] is None and
            plan['phase_B1']['H6_coefficients_transferred'] is False, 'NO_H6_MODEL_TRANSFER')
    require(plan['phase_C']['PR_scope'] == 'H8_CONFIRMATION_UNION6_n32', 'CONFIRMATION_SCOPE')
    require(plan['V2']['status'] == 'CLOSED_NO_GO_BY_USER_DECISION' and
            plan['V2']['conditions_relaxed'] is False and
            plan['V2']['alternative_execution_authorized'] is False, 'V2_CLOSURE')
    return dict(status='PASS_METADATA_ONLY', existing=len(old), additional=len(new),
                additional_S2=sum(f == '2nd' for m, p, q, f in new),
                additional_S4=sum(f == '4th' for m, p, q, f in new),
                combined_deterministic=len(old | new), scientific_execution=False)


def verify_public_files(directory):
    directory = Path(directory)
    receipt = json.loads((directory/'publication_files_v1.json').read_text())
    for row in receipt['files']:
        path = Path(row['path'])
        require(not path.is_absolute() and '..' not in path.parts, 'RELATIVE_PUBLIC_PATH')
        raw = (ROOT/path).read_bytes()
        require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'],
                'PUBLIC_FILE_CHANGED:' + row['path'])
    return len(receipt['files'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    args = parser.parse_args()
    result = validate(json.loads((args.bundle/'execution_plan_UNAPPROVED_v1.json').read_text()))
    result['public_file_hashes_checked'] = verify_public_files(args.bundle)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
