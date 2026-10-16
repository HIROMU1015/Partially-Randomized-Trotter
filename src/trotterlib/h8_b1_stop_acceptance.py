"""Review-only H8 saved-STOP acceptance checks. No fit, grant or executor.

A successful check establishes consistency of pinned historical evidence, not
acquisition completion, historical thread compliance, or runtime permission.
"""
import hashlib
import json
import math

B0_SOURCE = 'db559a518214fd27fa96e2f1451911df05c4f4a8'
B0_RESULT = 'af10b55420d31e227905f7762971e4b39c6c6a05'
AUDIT_COMMIT = 'b19c337f0b16513d68b59eb3366dc6eff175c313'
CANONICAL_SOURCE = '6fb8a0a71986fda571d82b5658686f5e74a32f8b'
EXISTING_B1_RESULT = '83a3c4a6688e66576f85471bf41a45e29cbfb705'
UNSELECTED_SOURCE = '1e25d5691b2a5c4738ca8b67c398ae06b70c4e0f'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
MODEL = '8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
ROLES = ('stop', 'binding', 'plan', 'train', 'holdout', 'workers', 'controls',
         'audit', 'B1_contract', 'B1_binding', 'B1_terminal', 'B1_phase',
         'B1_cleanup', 'B1_model')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'DUPLICATE_JSON_KEY')
            result[k] = v
        return result

    def nonfinite(value):
        raise ValueError('NONFINITE_JSON:' + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def validate_draft(contract, evidence_sha256):
    require(contract['schema'] == 'H8_B1_SAVED_STOP_ACCEPTANCE_DRAFT_V1', 'DRAFT_SCHEMA')
    for key in ('approved', 'accepted', 'runtime_authorization', 'existing_result_adopted',
                'B2_authorized', 'C_authorized', 'next_stage_authorized'):
        require(contract[key] is False, 'DRAFT_CANNOT_AUTHORIZE:' + key)
    require(contract['allowed_cpus'] == [] and contract['absolute_launch_command'] is None,
            'NO_EXECUTION_CONFIGURATION')
    require(contract['canonical_numerical_SOURCE'] == CANONICAL_SOURCE and
            contract['unselected_numerical_SOURCE'] == UNSELECTED_SOURCE,
            'SINGLE_CANONICAL_IMPLEMENTATION')
    require(contract['existing_B1_result_commit'] == EXISTING_B1_RESULT and
            contract['duplicate_fit_permitted'] is False, 'EXISTING_RUN_NO_DUPLICATE_FIT')
    require(contract['evidence_set_sha256'] == evidence_sha256, 'EVIDENCE_SET_BINDING')
    require(contract['original_acquisition_status'] == 'STOP' and
            contract['original_final_gate_replaced'] is False and
            contract['historical_thread_compliance_certified'] is False,
            'HISTORY_MUST_REMAIN_STOP')


def validate_records(records):
    """Compare saved metadata only; never compute a fit or residual."""
    require(set(records) == set(ROLES), 'EXACT_EVIDENCE_ROLES')
    stop, binding, plan, audit = (records[k] for k in ('stop', 'binding', 'plan', 'audit'))
    require(stop['status'] == 'STOP_AFTER_EIGHT_SAVED_MEASUREMENTS' and
            stop['acquisition_contract_status'] == 'STOP' and
            stop['first_observer_stop'] == 'RuntimeError:PROCESS_THREADS', 'ORIGINAL_STOP')
    require(stop['final_gate_exists'] is False and stop['STOP_changed_to_PASS'] is False and
            stop['final_protected_worktree_audit_completed'] is False, 'NO_RETROACTIVE_FINAL_GATE')
    require(stop['exact_process_causing_thread_violation'] == 'UNRESOLVED_NOT_RECORDED' and
            stop['Git_threading_cause_proven'] is False, 'NO_INFERRED_THREAD_CAUSE')
    require(stop['all_owned_processes_reaped'] is True and not stop['cleanup_errors'] and
            stop['all_eight_workers_exit0_and_reaped'] is True and
            stop['unknown_reserved_compile_calls'] == 0 and stop['compile_entered'] == 8,
            'SAVED_WORK_COMPLETENESS')
    require(stop['completed_calls_lower_bound'] == plan['call_caps'], 'FIXED_ACQUISITION_CALL_CAPS')
    require(audit['status'] == 'SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL' and
            audit['execution_SOURCE'] == B0_SOURCE and audit['saved_result_commit'] == B0_RESULT and
            audit['original_acquisition_status'] == 'STOP', 'PINNED_POSTHOC_AUDIT')
    for k in ('original_final_gate_replaced', 'historical_thread_compliance_certified',
              'posthoc_audit_retroactively_closes_contract', 'Git_cause_of_original_violation_proven',
              'B1_authorized'):
        require(audit[k] is False, 'AUDIT_NOT_AUTHORIZATION:' + k)
    require(audit['SOURCE_input_runtime_preparation_compiler_binding'] is True and
            audit['original_config_grant_raw_unchanged'] is True and
            audit['final_preservation_audit_passed_posthoc'] is True and
            (audit['measured_axes_verified'], audit['TRAIN_axes'], audit['HOLDOUT_axes']) == (8, 6, 2),
            'POSTHOC_BINDING')
    require(binding['SOURCE'] == B0_SOURCE and binding['profile_sha256'] == PROFILE and
            binding['compiler']['compiler_hash'] == COMPILER and len(binding['input_receipts']) == 8,
            'SOURCE_PROFILE_COMPILER_INPUT')
    require(binding['preparation_binding'] == plan['binding']['prefix_preparations'], 'PREPARATION_PLAN')
    cases = {x['id']: x for x in plan['cases']}
    require(len(cases) == 4, 'FOUR_FIXED_CASES')
    expected = {(c, a) for c in cases for a in ('cosine', 'sine')}
    waves = records['workers']['waves']
    require(len(waves) == 2 and all(len(w['workers']) == 4 and w['status'] == 'PASS' and
            w['all_reaped'] is True and not w['cleanup_errors'] and w['automatic_retry'] is False
            for w in waves), 'WORKER_WAVES')
    flat = [w for wave in waves for w in wave['workers']]
    workers = {w['task_id']: w for w in flat}
    require(len(workers) == 8, 'WORKER_IDENTITIES')
    controls = {x['case']['id']: x for x in records['controls']['cases']}
    require(len(controls) == len(records['controls']['cases']) == 4, 'FOUR_CONTROL_RECEIPTS')
    seen = set()
    for name, count in (('train', 6), ('holdout', 2)):
        group = records[name]
        require(group['role'] == name.upper() and len(group['axes']) == count and
                group['fit_performed'] is False, 'FIXED_SPLIT_NO_FIT')
        for row in group['axes']:
            key = row['case_id'], row['axis']
            require(key in expected and key not in seen, 'EXACT_EIGHT_AXES')
            seen.add(key)
            case = cases[row['case_id']]
            require(case['role'] == row['role'] == group['role'] and
                    case['role'] == ('HOLDOUT' if (case['prefix'], case['q']) == (21, 8) else 'TRAIN'),
                    'NO_HOLDOUT_REASSIGNMENT')
            prep = binding['preparation_binding'][str(case['prefix'])]
            ctrl = controls[row['case_id']]
            require(row['SOURCE'] == ctrl['SOURCE'] == B0_SOURCE and
                    row['scope'] == plan['scope'] and row['profile_sha256'] == ctrl['profile_sha256'] == PROFILE and
                    row['compiler_hash'] == COMPILER, 'AXIS_LINEAGE')
            require(row['control'] == plan['control'] and row['scalar_policy'] == ctrl['scalar_policy'] ==
                    plan['scalar_policy'] and row['preparation_binding'] == ctrl['preparation_binding'] == prep and
                    ctrl['case'] == case and ctrl['basis_observed_from_restored_live_operations'] is True and
                    ctrl['observed_ordered_block_description_sha256'] == prep['ordered_block_description_sha256'],
                    'LIVE_PREPARATION_CONTROL')
            require(row['fingerprint_before'] == row['fingerprint_after'] ==
                    ctrl['wrapper_fingerprints'][row['axis']], 'WRAPPER_FINGERPRINT')
            err = row['max_control_error']
            require(type(err) in (int, float) and math.isfinite(err) and 0 <= err < 1e-8 and
                    err == ctrl['max_control_error'], 'CONTROL_TOLERANCE')
            require(all(type(row[k]) is int and row[k] > 0 for k in ('measured_RZ', 'CX', 'depth', 'size')),
                    'SAVED_COUNTS')
            w = workers.get(row['case_id'] + '_' + row['axis'], {})
            require(w.get('exit_code') == row['native_exit_code'] == 0 and
                    w.get('reaped') is row['reaped'] is True and
                    w.get('identity') == row['worker_identity'] and
                    w.get('result_sha256') == row['result_sha256'], 'WORKER_EXIT_REAP_RESULT')
    require(seen == expected, 'EXACT_EIGHT_AXES')
    # This is evidence of an already executed B1, never permission to repeat it.
    hist = records['B1_binding']
    require(hist['SOURCE'] == CANONICAL_SOURCE and hist['profile_sha256'] == PROFILE and
            hist['compiler_hash'] == COMPILER and hist['original_B0_acquisition_status'] == 'STOP' and
            hist['historical_thread_compliance_certified'] is False, 'EXISTING_B1_BINDING')
    require(records['B1_contract']['authorized_B1_executed_once'] is True and
            records['B1_contract']['original_B0_run02_status'] == 'STOP', 'HISTORICAL_AUTHORIZATION_RECORD')
    for name in ('B1_terminal', 'B1_phase'):
        x = records[name]
        require(x['status'] == 'COMPLETE_MANDATORY_STOP' and x['SOURCE'] == CANONICAL_SOURCE and
                x['model_sha256'] == MODEL and all(x[k] is False for k in
                ('B2_authorized', 'C_authorized', 'next_stage_authorized')), 'EXISTING_B1_TERMINAL')
    require(digest(records['B1_model']) == MODEL, 'EXISTING_MODEL_BYTES_ONLY')
    cleanup = records['B1_cleanup']
    require(cleanup['native_exit_code'] == 0 and cleanup['all_reaped'] is True and
            cleanup['first_stop'] is None and not cleanup['cleanup_errors'] and
            cleanup['automatic_retry'] is False and 0 < cleanup['max_monitor_gap_seconds'] <= 5,
            'EXISTING_B1_CLEANUP')
    return dict(status='TECHNICALLY_ELIGIBLE_PENDING_EXPLICIT_ACCEPTANCE',
                saved_B0_axes=8, TRAIN_axes=6, HOLDOUT_axes=2,
                original_acquisition_status='STOP', original_final_gate_replaced=False,
                historical_thread_compliance_certified=False,
                canonical_numerical_SOURCE=CANONICAL_SOURCE, existing_B1_result_commit=EXISTING_B1_RESULT,
                existing_B1_result_adopted=False, duplicate_fit_permitted=False,
                approved=False, runtime_authorization=False, new_compile=0, new_fit=0,
                new_residual=0, new_reaccount=0, new_valid_grant=0, mandatory_STOP=True)
