"""Metadata-only cost-object joins and fixture-only call contract rehearsal.

No scientific imports, QPY circuit decoding, compiler, simulator or grant API.
Historical receipts and a proposed common-runtime cost object are distinct.
"""
from copy import deepcopy
import h2_pr_identity_recovery as r
import h2_pr_equivalence_contract as alternative

BASE = '65085ff7f818fd92af3424b1980f6bd9dbf1c742'
PF_RESULT = 'f36402f90bcfcbcc4d556c7a952837dc5d7dd291'
PF_SOURCE = '1b35f9ac2efab4c15bc00ee2d3c81643aca422f9'
PF_PREFIX = 'artifacts/resource_applicability/track_a_pf_common_H2_results/2026-10-22/'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PY312 = 'e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f'
GUARD = 'H2_QISKIT130_SYMBOLIC_ZERO_SCALAR_GUARD_V1'
FULL_MAP = {'B1_p3_q4_2nd': 'H2_FULL_3_4_2nd', 'B1_p3_q2_4th': 'H2_FULL_3_2_4th'}
PR_CANDIDATES = ('H2O32_c7ec590a32a859bc', 'H2O32_8a2ecdd0294ffb74')
PROBE_TASKS = ('H2E1__H2O32_c7ec590a32a859bc__0', 'H2E1__H2O32_8a2ecdd0294ffb74__4')
SETTINGS = dict(basis_gates=['rz', 'sx', 'x', 'cx'], optimization_level=1,
                seed_transpiler=17, backend=None, coupling_map=None,
                layout_method=None, routing_method=None, qiskit_version='1.3.0', num_processes=1)
ZERO_CALLS = dict.fromkeys(('compile', 'transpile', 'simulate', 'sampling', 'setup',
    'prepare', 'lowering', 'wrapper', 'allowance', 'accounting', 'ranking', 'COMPARE'), 0)


def guard_receipt(value):
    r.require(value['policy'] == GUARD and value['unchanged_gate_stream'] is True and
              value['state_fit'] is False and value['raw_validation_threshold_changed'] is False,
              'GUARD_POLICY_OR_SCOPE')
    r.require(bool(value['passes']), 'MISSING_TRANSLATOR_PASS_RECEIPT')
    for p in value['passes']:
        r.require(p['policy'] == GUARD and p['unchanged_gate_stream'] is True and
                  p['input_or_output_state_fit'] is False and p['postcompile_phase_correction'] is False,
                  'TRANSLATOR_GUARD_PASS')


def join_saved_cost(row, expected):
    """Reject cross-candidate/cohort/runtime/source/axis substitutions, even equal RZ."""
    fields = ('candidate', 'task', 'axis', 'cohort', 'runtime_id', 'compiler_hash',
              'compiler_source_sha256', 'settings', 'input_sha256', 'wrapper_raw_sha256',
              'compiled_raw_sha256', 'receipt_sha256', 'RZ', 'unit', 'correctness')
    r.require(set(row) == set(expected) == set(fields), 'COST_OBJECT_FIELDS')
    r.require(alternative.strict.policy.strict_equal(row, expected),
              'COST_OBJECT_BINDING_OR_SAVED_VALUE')
    r.require(type(row['RZ']) is int and row['RZ'] > 0 and
              row['unit'] == 'compiled_RZ_per_full_Hadamard_wrapper_axis', 'RZ_DEFINITION')
    r.require(alternative.strict.policy.strict_equal(row['settings'], SETTINGS) and
              row['compiler_hash'] == COMPILER, 'COMPILER_SETTINGS')
    alternative.validate_correctness(row['correctness'])
    for name in ('input_sha256', 'wrapper_raw_sha256', 'compiled_raw_sha256',
                 'receipt_sha256', 'compiler_source_sha256'):
        r.require(type(row[name]) is str and len(row[name]) == 64 and
                  all(c in '0123456789abcdef' for c in row[name]), 'COST_OBJECT_SHA')
    return deepcopy(row)


def join_projection_seals(pf_closure, pr_seals):
    """Match public role/SHA/size to original seal entries, never infer actual load."""
    groups = {}
    for role, count in (('fixed_python_binary', 1), ('loaded_runtime_identity', 8),
                        ('compiler_array_runtime_binary', 21), ('pinned_provider', 236)):
        rows = [x for x in pf_closure if x['role'] == role]
        r.require(len(rows) == count, 'EXPECTED_PF_PROJECTION_ROLE_SCOPE')
        pairs = {(x['sha256'], x['bytes']) for x in pr_seals}
        groups[role] = dict(entries=len(rows), matched=sum((x['sha256'], x['bytes']) in pairs for x in rows),
            missing=[dict(sha256=x['sha256'], bytes=x['bytes']) for x in rows
                     if (x['sha256'], x['bytes']) not in pairs])
    py = next(x for x in pf_closure if x['role'] == 'fixed_python_binary')
    r.require(py['sha256'] == PY312, 'PF_HISTORICAL_PYTHON_IDENTITY')
    return dict(role_groups=groups, full_historical_load_proven=False,
        full_runtime_closure_proven=False, scientific_output_equivalence_proven=False,
        scope='Published PF projection versus original PR seal; equal version names are insufficient')


def preflight():
    return dict(status='BLOCKED_POLICY_AND_SAVED_PF_NATIVE_RUNTIME_BINDING',
        same_Python312_saved_PR_and_PF_costs_exist=True,
        new_compile_required_by_current_evidence=False, minimum_new_scientific_calls_now=0,
        alternative_adopted=False, execution_ready=False, science_authorized=False,
        grants_issued=0, COMPARE='BLOCKED', actual_calls=deepcopy(ZERO_CALLS),
        strict_runtime_output_condition='UNRESOLVED_FOR_ORIGINAL_PY312_VS_ADOPTED_PY311_CONTRACT',
        remaining=['Retrieve exact common-PF original plan/QPY/native preparation receipts',
                   'Bind relevant runtime paths and dependency closure, not SHA-set membership alone',
                   'Relate archived PF preparation to current common-native preparation without substitution',
                   'Resolve A independent absolute diagonal witness separately',
                   'User adopts runtime/cost-object policy; separate COMPARE SOURCE and approval'],
        mandatory_STOP=True)


def probe_draft(rows, original_plan):
    """Preserve the previous four-axis recipe; no real execution entry exists."""
    selected = [x for x in rows if x['task'] in PROBE_TASKS]
    r.require(len(selected) == 4 and {(x['task'], x['axis']) for x in selected} ==
              {(t, a) for t in PROBE_TASKS for a in ('cosine', 'sine')}, 'EXACT_PROBE_AXES')
    calls = deepcopy(original_plan['prospective_exact_calls'])
    r.require(calls['compile'] == calls['transpile'] == 4 and calls['simulate'] == 8 and
              calls['setup'] == calls['prepare'] == calls['sampling'] == calls['lowering'] == 0,
              'ORIGINAL_PROBE_BUDGET')
    return dict(schema='H2_SAVED_WRAPPER_RUNTIME_PROBE_DRAFT_V1', status='PRESERVED_OPTION_ONLY',
        tasks=list(PROBE_TASKS), axes=deepcopy(selected), prospective_exact_calls=calls,
        original_recipe_unchanged=True, science_authorized=False, runtime_sealed=False,
        grant=None, run_root=None, cache=None, production_backend_connected=False,
        execution_entry_implemented=False, policy_adopted=False, retry=False,
        historical_correctness_scope='SINGLE_SAVED_INPUT_STATE_CT01',
        acceptance='Exact saved/new RZ equality; before/after saved-state strict <1e-7; no phase fit',
        scope='Runtime output sensitivity on these four axes only; not a new full cohort',
        internal_calls='4 cost calls each invoke 1 guarded transpile. Each guard pass invokes '
                       'BasisTranslator.run twice; number of passes is data-dependent and not sealed. '
                       'No extra transpile or all-state simulation is hidden in this recipe.',
        readiness_blockers=['New actual-load/runtime/input/source seal and unused authority root',
                            'Production adapter and internal translator-pass ledger not implemented',
                            'Separate explicit scientific execution approval'],
        proof_of_134_axes=False, reconstructs_historical_load=False, mandatory_STOP=True)


class FixtureLedger:
    """Artificial rehearsal only. No callbacks, files, scientific backend or authority."""
    def __init__(self, caps, *, fixture):
        r.require(fixture == 'ARTIFICIAL_ONLY' and type(fixture) is str, 'REAL_EXECUTION_UNAUTHORIZED')
        r.require(type(caps) is dict and all(type(n) is int and n >= 0 for n in caps.values()), 'EXACT_CAPS')
        self.caps = deepcopy(caps)
        self.used = dict.fromkeys(caps, 0)
        self.stopped = False

    def take(self, name):
        r.require(not self.stopped, 'ATTEMPT_STOPPED')
        if name not in self.caps or self.used[name] >= self.caps[name]:
            self.stopped = True
            raise r.EvidenceError('CALL_NOT_APPROVED_OR_BUDGET_EXCEEDED')
        self.used[name] += 1

    def finish(self):
        r.require(not self.stopped and self.used == self.caps, 'INCOMPLETE_EXACT_CALLS')
        return deepcopy(self.used)


def deny_scientific_validation(*args, **kwargs):
    raise r.EvidenceError('EXISTING_VALIDATOR_REQUIRES_UNAUTHORIZED_SCIENTIFIC_EXECUTION')


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        deny_scientific_validation()
    print(json.dumps(preflight(), indent=2))
