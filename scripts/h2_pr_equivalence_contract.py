"""Unadopted alternative PR conformance policy, metadata-only and non-executable.

Scientific circuit conformance (A) and comparability of historical compiled
costs (B) are separate. A verified historical input-state check is never an
operator proof. Even a complete artificial packet cannot authorize execution.
The old strict runner, its witnesses, SOURCE and contracts remain unchanged.
"""
from copy import deepcopy
from functools import lru_cache
import h2_pr_identity_recovery as recovery
import h2_accepted_compare_contract as strict

require = recovery.require
STATES = ('VERIFIED', 'UNRESOLVED', 'MISMATCH', 'NOT_APPLICABLE')
BASE = '588d19544a2afdfb55ec4b40475a58114bd47b0b'
OLD_SOURCE = '46a3b33e3717c4fadf11846e922caf17d10156d9'
OLD_DIGEST = 'a65f3e90b7ed70701f57bcf1519e1d74acdfd1c45c320a5fe91d00d4d44d9ef2'
A = ('original_binding', 'event_digest_order_law', 'gaussian_forward_inverse',
     'basis_reuse_boundaries', 'deterministic_diagonal_shape_and_mirror',
     'event_control_order_wire_angle', 'relative_event_and_identity_phase',
     'scalar_once_and_full_wrapper', 'complete_instruction_coverage',
     'serialized_global_phase_relation', 'absolute_deterministic_diagonal_vs_common_preparation',
     'historical_input_state_correctness')
B = ('original_compiled_raw_receipt', 'compiler_source_settings_and_phase_guard',
     'compiler_relevant_runtime_output_equivalence')
OUTSIDE = ('all_state_operator_equivalence', 'new_runtime_state_correctness')


def condition_spec(name):
    table = {
      'original_binding': ('Verified input/H/state/preparation hashes and exact cohort/trajectory/axis', 'Existing sealed input/result and common receipt', 'Reject any SHA or candidate/cohort mismatch', 'No new setup required for archived provenance'),
      'event_digest_order_law': ('Original complete event digest and ordered law/signed pieces', '67 full files with saved result digest', 'Check all original event components/phases', 'No RNG, no resampling'),
      'gaussian_forward_inverse': ('Every saved Gaussian gate/hex/wire and exact source-defined inverse order', 'Common six registered basis field lists + QPY', 'Consume inverse and forward at exact boundaries', 'No rotation reconstruction'),
      'basis_reuse_boundaries': ('Every switch/identity flush/piece end follows original adjacent-basis policy', 'Full events + saved QPY', 'Reject missing or surplus basis operations', 'No circuit generation'),
      'deterministic_diagonal_shape_and_mirror': ('Fixed H2 nonzero 4 RZ, 6 RZZ/4 RZ and directional wire/sign mirror', 'Serialized diagonal and its paired directional fields', 'Reject omission/order/support/sign changes', 'Does not independently certify diagonal magnitudes from H'),
      'event_control_order_wire_angle': ('All application central support/wire/angle/sign and saved control definitions match', 'Original event fields and saved QPY custom definitions', 'Check base/body/control states and every central operation', 'No matrix or operator evaluation'),
      'relative_event_and_identity_phase': ('Event/product/identity phases accumulated in fixed SOURCE order', 'Saved event phase + saved relative ancilla PhaseGate', 'Exact binary64 parameter relation', 'No phase fit'),
      'scalar_once_and_full_wrapper': ('One final -scalar*T and correct H/U/Sdg/H/measure, system4/ancilla4', 'Original scalar == common diagnostic scalar + QPY', 'Reject duplicate scalar or wrapper/measurement change', 'No new signal evaluation'),
      'complete_instruction_coverage': ('Every saved instruction consumed exactly once by fixed grammar', 'Full saved wrapper QPY', 'Reject unrecognized or unconsumed instructions', 'Static shape does not imply all-state physical equivalence'),
      'serialized_global_phase_relation': ('Saved global phase matches source normalization of paired serialized ancilla phases exactly', 'QPY global phase and paired directional phase fields', 'Unknown normalization stays UNRESOLVED, no tolerance relaxation', 'No new Hamiltonian diagonal/phase computation'),
      'absolute_deterministic_diagonal_vs_common_preparation': ('Absolute diagonal parameters have an independent common-preparation binding', 'No independent diagonal-parameter receipt in existing common prep JSON', 'Search only explicitly saved witnesses; source/preparation hashes alone insufficient', 'If no independent receipt, separately authorized RNG-free preparation/probe or justified restricted policy is needed'),
      'historical_input_state_correctness': ('Preserve original CT01 SINGLE_SAVED_INPUT_STATE strict <1e-7 and no phase fit', '134 original compiled-vector acceptance receipts', 'SHA and copied receipt scope/PASS/strict limit only; no error reevaluation', 'No generalization to other states or runtimes'),
      'original_compiled_raw_receipt': ('Original compiled raw/hash and saved RZ receipt match, exact paired cohort', 'Original archive manifest/raw + accepted result + cost receipt', 'Verify original byte/content binding, no new cost derivation', 'Historical cost object only'),
      'compiler_source_settings_and_phase_guard': ('Pinned target basis/settings/guard and relevant saved SOURCE match', 'Original compiler settings/hash/guard/raw metadata; common SOURCE join', 'Verify source and receipt hashes; matching names alone insufficient', 'Does not by itself evaluate ABI-output sensitivity'),
      'compiler_relevant_runtime_output_equivalence': ('Evidence covers output-sensitive compiler/runtime differences for all 134 archived axes', 'Qiskit bytes equal; Python/NumPy/SymEngine ABI differs; no complete historical actual-load manifest', 'Map source/binary dependencies and unresolved relevance; never claim not loaded without proof', 'Probe can test chosen outputs only; cohort-wide proof or explicitly justified cost-object policy still required'),
      'all_state_operator_equivalence': ('Outside original single-state acceptance scope', 'No operator certificate', 'Never promote historical acceptance', 'Needs separate scientific design if required'),
      'new_runtime_state_correctness': ('Not claimed for historical PR compiled circuits under adopted runtime', 'No new numerical execution authorized', 'Explicitly NOT_APPLICABLE to archived acceptance', 'Separate statevector validation if future adopted policy requires it')}
    acceptance, existing, static, scientific = table[name]
    return dict(condition=name, category='A' if name in A else 'B' if name in B else 'OUTSIDE',
                required=name in A+B, acceptance=acceptance, existing_evidence=existing,
                additional_static_scope=static, scientific_boundary=scientific,
                STOP_if_unverified=name in A+B)


def draft():
    original = recovery.loads(recovery.git_bytes(recovery.BASE, recovery.PREVIOUS+'COMPARE_contract_BLOCKED.json'))
    return dict(schema='H2_PR_ALTERNATIVE_EQUIVALENCE_DRAFT_V1', status='DRAFT_UNADOPTED_NONEXECUTABLE',
        base_commit=BASE, strict_source_unchanged=OLD_SOURCE, strict_contract_digest_unchanged=OLD_DIGEST,
        physical=deepcopy(original['physical']), candidates=deepcopy(original['candidates']),
        compiler=deepcopy(original['compiler']), native_policy=original['native_policy'],
        allowance_policy=original['allowance_policy'], reference_policy=original['reference_policy'],
        correctness_scope='SINGLE_SAVED_INPUT_STATE_CT01_STRICT_LT_1E_MINUS7',
        alternative_policy_adopted=False, scientific_execution_authorized=False,
        grant=None, ledger=None, marker=None, run_root=None, retry=False, mandatory_STOP=True,
        cost_object='ORIGINAL_ARCHIVED_PR_COMPILED_AXIS_WITH_UNCHANGED_PAIRED_RECEIPT',
        source_or_runtime_names_alone_suffice=False,
        historical_load_may_be_reconstructed=False,
        A_does_not_imply_B=True, condition_specs=[condition_spec(x) for x in A+B+OUTSIDE])


def validate_draft(value):
    # The exact draft policy cannot be relaxed or mistaken for a live contract.
    require(strict.policy.strict_equal(value, draft()), 'ALTERNATIVE_DRAFT_POLICY_MUTATION_OR_ADOPTION')


@lru_cache(maxsize=1)
def anchors():
    return tuple(recovery.loads(recovery.git_bytes(recovery.BASE, recovery.RECEIVED+'task_acceptance_67.json'))['tasks'])


@lru_cache(maxsize=1)
def original_origin_hashes():
    return {x['id']:x['original_sha256'] for x in recovery.loads(
        recovery.git_bytes(recovery.BASE, recovery.RECEIVED+'origin_inventory.json'))['originals']}

def original_correctness_sha(anchor, axis):
    entry = next(x for x in anchor['saved_correctness'] if x['saved']['check'] == 'compiled_vector_'+axis)
    return original_origin_hashes()[entry['origin']]


def validate_correctness(receipt):
    require(set(receipt) == {'scope','strictly_below','PASS','phase_alignment','receipt_sha256'}, 'CORRECTNESS_FIELDS')
    require(receipt['scope'] == 'SINGLE_SAVED_INPUT_STATE_CT01', 'NO_CORRECTNESS_SCOPE_PROMOTION')
    require(type(receipt['strictly_below']) is float and receipt['strictly_below'] == 1e-7 and
            receipt['PASS'] is True and receipt['phase_alignment'] is False, 'HISTORICAL_CORRECTNESS_GATE')
    strict.policy.sha(receipt['receipt_sha256'], 'ORIGINAL_CORRECTNESS_SHA')


def preflight(value, rows):
    validate_draft(value)
    lookup = {x['task']['id']:x for x in anchors()}
    expected = {(tid, axis) for tid in lookup for axis in ('cosine','sine')}
    require(type(rows) is list and len(rows) == 134 and
            {(x['task'],x['axis']) for x in rows} == expected, 'EXACT_67_TASK_134_PAIRED_AXES')
    stats = {'A':dict.fromkeys(STATES,0), 'B':dict.fromkeys(STATES,0)}
    blockers = []
    for row in rows:
        require(set(row) == {'task','candidate','sample','axis','cohort','result_sha256','conditions','correctness'}, 'AXIS_SCHEMA')
        anchor = lookup[row['task']];task = anchor['task']
        require(row['candidate'] == task['candidate']['id'] and type(row['sample']) is int and
                row['sample'] == task['sample'] and row['cohort'] == 'EXPLORATION' and
                row['result_sha256'] == anchor['acceptance']['saved']['result_sha256'], 'ORIGINAL_CANDIDATE_COHORT_RECEIPT')
        validate_correctness(row['correctness'])
        require(row['correctness']['receipt_sha256'] == original_correctness_sha(anchor, row['axis']), 'ORIGINAL_CORRECTNESS_RECEIPT_SHA')
        require(set(row['conditions']) == set(A+B+OUTSIDE), 'ALL_CONDITIONS_REQUIRED')
        for item in row['conditions'].values():
            require(set(item) == {'status','evidence_sha256','scope'} and item['status'] in STATES and
                    type(item['scope']) is str and bool(item['scope']), 'CONDITION_EVIDENCE_SCHEMA')
            strict.policy.sha(item['evidence_sha256'], 'CONDITION_EVIDENCE_SHA')
        for category, names in (('A',A),('B',B)):
            observed = []
            for name in names:
                item = row['conditions'][name]
                require(set(item) == {'status','evidence_sha256','scope'} and item['status'] in STATES and
                        type(item['scope']) is str and bool(item['scope']), 'CONDITION_EVIDENCE_SCHEMA')
                strict.policy.sha(item['evidence_sha256'], 'CONDITION_EVIDENCE_SHA')
                require(item['status'] != 'NOT_APPLICABLE', 'REQUIRED_CONDITION_NOT_APPLICABLE')
                observed.append(item['status'])
                if item['status'] != 'VERIFIED':
                    blockers.append(dict(task=row['task'],axis=row['axis'],condition=name,status=item['status']))
            overall = 'MISMATCH' if 'MISMATCH' in observed else 'UNRESOLVED' if 'UNRESOLVED' in observed else 'VERIFIED'
            stats[category][overall] += 1
        for name in OUTSIDE:
            require(row['conditions'][name]['status'] == 'NOT_APPLICABLE', 'OUTSIDE_SCOPE_MUST_NOT_BE_PROMOTED')
    return dict(status='BLOCKED_ALTERNATIVE_UNADOPTED', alternative_adopted=False,
        technical_axis_counts=stats, blockers=blockers, scientific_GO=False,
        execution_ready=False, science_authorized=False, strict_reuse_accepted_axes=0,
        interpretation='Technical A/B fulfillment is not policy adoption or PR cost reuse certification',
        grants_issued=0, scientific_calls=0, mandatory_STOP=True)


def deny_execute(*args, **kwargs):
    raise recovery.EvidenceError('UNADOPTED_ALTERNATIVE_HAS_NO_SCIENTIFIC_EXECUTION_ENTRY')


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--draft'); p.add_argument('--axes'); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.execute: deny_execute()
    require(bool(args.draft) and bool(args.axes), 'METADATA_INPUTS_REQUIRED')
    print(json.dumps(preflight(recovery.loads(Path(args.draft).read_bytes()),
                               recovery.loads(Path(args.axes).read_bytes())['rows'])))
