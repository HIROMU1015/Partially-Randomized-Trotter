"""Static joins for unadopted policy 2a. No scientific execution interface."""
from copy import deepcopy
import h2_runtime_cost_reuse as previous

r = previous.r
B = 'ca29dd9ddc8a09cace84310f0dfdf7124652e07a'
RECOVERY = 'bb9e0bd167daa8e618ec203f8ea3b3cd26440ebf'
A = '43be11d648ca35302bd2c7267e23327fe37e34cb'
B_DIR = 'artifacts/resource_applicability/track_a_h2_runtime_cost_reuse/2026-10-23/'
REC_DIR = 'artifacts/resource_applicability/track_a_h2_common_pf_original_recovery/2026-10-23/'
OUT_DIR = 'artifacts/resource_applicability/track_a_h2_saved_cost_static_join_b2/2026-10-23'
PLAN = '6eef028aff888dd64d8c6bb1833f01112df1026e452f8b89318c866f5e1a1892'
PREP = '98c1beb120c22077b409550a4d61c31638b9a52637246fae4cb1ac55a776f1d9'
FULL = {'H2_FULL_3_4_2nd': ('2nd', 4, 1655), 'H2_FULL_3_2_4th': ('4th', 2, 2463)}
UNIT = 'compiled_RZ_per_full_Hadamard_wrapper_axis'


def same(a, b, reason):
    r.require(previous.alternative.strict.policy.strict_equal(a, b), reason)


def join_full(saved, expected, recovered, packet, archives, plan):
    """Bind saved fields, not recomputed counts or simulated vectors."""
    same(saved, expected, 'FIXED_FULL_COST_OBJECT')
    cid, ax = saved['saved_candidate'], saved['axis']
    r.require(cid in FULL and ax in ('cosine', 'sine'), 'FULL_CANDIDATE_AXIS')
    formula, q, rz = FULL[cid]
    candidate = dict(system='H2', method='FULL', prefix=3, q=q, formula=formula)
    same(saved['candidate_fields'], candidate, 'FULL_CANDIDATE_SCOPE')
    same(packet['candidate'], candidate, 'RESULT_CANDIDATE_SCOPE')
    r.require(candidate in plan['rows'] and plan['private_original_plan_sha256'] == PLAN,
              'ORIGINAL_PLAN_BINDING')
    same(saved['compiler_settings'], previous.SETTINGS, 'COMPILER_SETTINGS')
    r.require(saved['compiler_hash'] == recovered['compiler_hash'] ==
              packet['prices'][ax]['compiler'] == plan['compiler'] == previous.COMPILER,
              'COMPILER_HASH')
    r.require(saved['runtime_id'] == previous.PY312 and saved['cohort'] ==
              'COMMON_PF_DETERMINISTIC_20261022', 'FULL_RUNTIME_COHORT')
    r.require(type(saved['RZ']) is int and saved['RZ'] == rz == recovered['saved_RZ_per_axis'] ==
              packet['prices'][ax]['metrics']['rz_count'] and saved['unit'] == UNIT, 'SAVED_RZ_UNIT')
    r.require(recovered['original_plan_sha256'] == PLAN and
              recovered['SOURCE'] == packet['SOURCE'] == previous.PF_SOURCE, 'RESULT_PLAN_SOURCE')
    r.require(recovered['preparation_sha256'] == saved['preparation_hash'] ==
              packet['preparation_hash'] == PREP, 'FULL_PREPARATION')
    for name, field in (('H', 'hamiltonian'), ('state', 'state')):
        r.require(saved[name] == recovered[name] == packet[field] == plan['binding'][field],
                  'HAMILTONIAN_STATE_BINDING')
    same(saved['scalar'], packet['scalar'], 'SCALAR_BINDING')
    same(saved['guard'], packet['prices'][ax]['phase_guard'], 'GUARD_RESULT_BINDING')
    previous.guard_receipt(saved['guard'])
    r.require(r.digest(saved['guard']) == recovered['phase_guard_receipt_SHA256'] and
              recovered['phase_guard_QPY_metadata_identical'] is True, 'QPY_GUARD_BINDING')
    check = next(x for x in packet['checks'] if x['check'] == 'compiled_vector_'+ax)
    same(check, saved['correctness'], 'CORRECTNESS_RECEIPT_BINDING')
    same(check, recovered['correctness']['receipt'], 'ORIGINAL_CORRECTNESS_BINDING')
    r.require(check['PASS'] is True and check['phase_alignment'] is False and
              check['strictly_below'] == 1e-7 and
              recovered['correctness']['scope'] == 'SINGLE_SAVED_INPUT_STATE' and
              recovered['correctness']['all_state_or_operator_proof'] is False,
              'CORRECTNESS_SCOPE_NOT_OPERATOR')
    r.require(recovered['wrapper_axis_wire_check'] is True, 'WRAPPER_AXIS_WIRE_CHECK')
    for name in ('wrapper', 'compiled'):
        rec = saved[name]
        matches = [a for a in archives if a['candidate'] == cid and
                   a['locator'].rsplit('/', 1)[-1] == rec['path']]
        r.require(len(matches) == 1, 'UNIQUE_ARCHIVE_BINDING')
        a = matches[0]
        same(a['archive']['sha256'], rec['archive_sha256'], 'ARCHIVE_SHA')
        same(a['archive']['bytes'], rec['compressed_bytes'], 'ARCHIVE_BYTES')
        same(a['raw']['sha256'], rec['original_sha256'], 'RAW_SHA')
        same(a['raw']['bytes'], rec['original_bytes'], 'RAW_BYTES')
        s = a['QPY_static']
        r.require(s is not None and s['qpy_version'] == 13 and s['nq'] == 5 and s['nc'] == 1 and
                  s['circuit_object_created'] is False and s['scientific_libraries_imported'] is False,
                  'STATIC_QPY_SCOPE')
    return dict(candidate=cid, axis=ax, saved_cost_identity='VERIFIED_INHERITED_GPU01',
                original_byte_observer='GPUA100-gleap-01',
                original_bytes_rehashed_GPU02=False,
                detailed_gate_parameter_phase_stream_binding='UNRESOLVED',
                all_state_operator_proof=False)


def join_runtime_projection(closure, projection):
    r.require(len(closure) == len(projection) == 365, 'FULL_PLAN_CLOSURE_365')
    logical = set()
    for i, (entry, expected) in enumerate(zip(closure, projection)):
        r.require(type(entry['index']) is int and entry['index'] == i, 'CLOSURE_ORDER')
        same({k:entry[k] for k in ('role','sha256','bytes')}, expected, 'CLOSURE_ROLE_SHA_BYTES')
        p = entry['logical_path']
        r.require(type(p) is str and p.startswith(('${PROVIDER}/', '${RUNTIME}/',
                  '${SCIENCE_SOURCE}/', '${SEALED}/')) and '..' not in p.split('/') and
                  p not in logical, 'CLOSURE_LOGICAL_PATH')
        logical.add(p)
        r.require(entry['historical_actual_load_proven'] is False, 'HISTORICAL_LOAD_PROMOTION')
    return dict(closure_entries=365, fixed_projection_join=True,
                actual_absolute_path_map_recovered_GPU02=False,
                original_plan_path_byte_observer='GPUA100-gleap-01',
                historical_actual_load_complete=False)


def policy_preflight(conditions):
    r.require(type(conditions) is list and len({x['condition'] for x in conditions}) == len(conditions),
              'UNIQUE_CONDITIONS')
    allowed = {'VERIFIED', 'CONDITIONAL', 'UNRESOLVED', 'MISMATCH'}
    r.require(all(x['status'] in allowed for x in conditions), 'CONDITION_STATUS')
    return dict(schema='H2_SAVED_COST_POLICY_2A_PREFLIGHT_V1',
        status='BLOCKED_UNADOPTED_POLICY_AND_REMAINING_EVIDENCE',
        costs_bound=dict(PR=134, Full=4),
        evidence_complete_for_unconditional_common_native_reuse=False,
        policy2a_adopted=False, alternative_adopted=False,
        strict_contract_changed=False, science_authorized=False, execution_ready=False,
        COMPARE='BLOCKED', grant=None, actual_calls=deepcopy(previous.ZERO_CALLS),
        additional_compile_required_now=False, private_transfer_required_now=False,
        conditions=deepcopy(conditions), mandatory_STOP=True)


def forbid_execution(*args, **kwargs):
    raise r.EvidenceError('B2_METADATA_ONLY_NO_SCIENTIFIC_EXECUTION')


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        forbid_execution()
    print(json.dumps(policy_preflight([]), indent=2))
