"""H8 B0 anchor acquisition proposal and receipt checks; no science launcher.

The determinant uses fixed design coordinates only, never measured costs.
Existing grant classes, model coefficients and molecular libraries are unused.
"""
from copy import deepcopy
import hashlib
import json
import math

SCOPE = 'H8_B0_CURRENT_PROFILE_S2_ANCHOR4_ACQUISITION_V1'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
TRAIN = ((14, 1), (21, 1), (14, 8))
HOLDOUT = ((21, 8),)
AXES = ('cosine', 'sine')
FORBIDDEN = ('cost_fit', 'cost_residual', 'cost_reaccount', 'S4_compile',
             'B0I_compile', 'new_signal', 'new_full_H_reference', 'SCF', 'DF',
             'state_solver', 'PR_sampling', 'law', 'tail_scalar')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def cases():
    return [dict(id=f'B0_p{p}_q{q}_2nd', method='B0', prefix=p, q=q,
                 formula='2nd', role='TRAIN' if (p, q) in TRAIN else 'HOLDOUT')
            for p, q in TRAIN + HOLDOUT]


def axes():
    return [dict(case_id=c['id'], axis=a, role=c['role']) for c in cases() for a in AXES]


def design_proof():
    x = [[1, q, p*q] for p, q in TRAIN]
    a, b, c = x
    determinant = (a[0]*(b[1]*c[2]-b[2]*c[1])
                   -a[1]*(b[0]*c[2]-b[2]*c[0])
                   +a[2]*(b[0]*c[1]-b[1]*c[0]))
    require(determinant != 0, 'TRAIN_DESIGN_SINGULAR')
    return dict(columns=['1', 'q', 'prefix*q'], train_matrix=x,
                determinant=determinant, rank_by_nonzero_determinant=3,
                measured_RZ_used=False, fitted_coefficients=None)


def proposal(binding):
    caps = dict(snapshot_load=1, saved_preparation_decode=1, prefix_view=2,
                native_evolution_build=4, measured_wrapper_build=8,
                probe_wrapper_copy=8, control_probe=16, compile=8,
                **{k: 0 for k in FORBIDDEN})
    return dict(schema='H8_B0_ANCHOR_ACQUISITION_PROPOSAL_V1', scope=SCOPE,
        status='UNAPPROVED_PLAN_NOT_EXECUTABLE', cases=cases(), axes=axes(),
        waves=[['B0_p14_q1_2nd', 'B0_p21_q1_2nd'],
               ['B0_p14_q8_2nd', 'B0_p21_q8_2nd']],
        design=design_proof(), binding=deepcopy(binding),
        preparation_policy='RESTORE_SAVED_ORDERED_OPERATIONS_NO_DIAGONALIZATION',
        scalar_policy='ORIGINAL_CONSTANT_ONCE_NO_TAIL_IDENTITY',
        control='symmetric_directional', compiler_hash=COMPILER,
        full_runtime_profile_sha256=PROFILE, provider_SOURCE=PROVIDER,
        call_caps=caps,
        acquisition_STOP=dict(on_any_technical_failure=True, stop_new_admissions=True,
            terminate_and_reap_owned_inflight_only=True, preserve_partial_and_reserved=True,
            retry=False, resume=False, extra_compile=False, model_error_evaluated=False,
            all_pass_still_mandatory_stop=True),
        later_B1=dict(separate_approval=True, automatic_start=False,
            B0_training_case_ids=[c['id'] for c in cases() if c['role']=='TRAIN'],
            B0_holdout_case_ids=[c['id'] for c in cases() if c['role']=='HOLDOUT'],
            fit_S2='a+b*q+d*prefix*q', holdout_in_fit=False,
            independent_axes=True, relative_error_denominator='MEASURED_RZ',
            holdout_PASS='abs(predicted-measured)/measured < 1/50',
            at_or_above_2_percent='STOP_NO_REFIT_NO_REASSIGNMENT_NO_EXTRA_COMPILE',
            B0I_source='REUSE_EXISTING_16_BOUND_MEASURED_AXES_NO_NEW_COMPILE',
            old_B0='ARCHIVE_ONLY_NOT_FIT_OR_VALIDATION_ANCHORS',
            all_four_strata_required=True, coefficients=None, H6_transfer=False,
            heldout_test_is_global_grid_guarantee=False, mandatory_stop=True),
        B2=dict(separate_approval=True, automatic_start=False, max_axes=4,
            compile_concurrency=1, unchanged_selection=['B0_p15_q1_4th','B0I_p21_q8_4th'],
            per_axis_STOP='abs(predicted-measured)/measured >= 1/50', mandatory_stop=True),
        C=dict(separate_approval=True, automatic_start=False, mandatory_stop=True),
        resources=dict(driver_processes=1, max_compile_workers=4, observer_processes=1,
            physical_cores=6, threads_per_role=1, max_cases_in_parent=2,
            driver_AS_bytes=64*2**30, driver_RSS_bytes=32*2**30,
            worker_AS_bytes=64*2**30, worker_RSS_bytes=32*2**30,
            observer_AS_bytes=512*2**20, observer_RSS_bytes=256*2**20,
            aggregate_RSS_bytes=160*2**30+256*2**20, admission_reserve_bytes=16*2**30,
            cumulative_CPU_seconds=24*3600, phase_wall_seconds=12*3600,
            observer_CPU_seconds=1800, stage_storage_bytes=8*2**30,
            output_bytes=256*2**20, logs_bytes=128*2**20,
            fresh_free_disk_bytes=16*2**30, fresh_free_inodes=40000,
            output_files=20000, native_instructions=4_000_000, transpiled_instructions=10_000_000,
            fresh_gate_max_age_seconds=30, observer_interval_seconds=1,
            observer_max_gap_seconds=5, host_full_PSI_stop_percent=1,
            cgroup_full_PSI_must_be_zero=True, OOM_increment_allowed=False,
            unknown_quota_or_capacity='STOP', other_job_or_environment_changes=False),
        runtime_prerequisites=['NEW_ACQUISITION_SOURCE_AND_RUNTIME_CLOSURE',
            'REAL_ADAPTER_RESTORATION_AND_CONTROL_FIXTURES_PASS',
            'EXPLICIT_USER_APPROVAL_AND_NEW_SCOPE_GRANT',
            'UNUSED_EXCLUSIVE_OUTPUT_AND_ONE_SHOT_LOCK',
            'FULL_34273_RUNTIME_BYTES_MATCH', 'PROVIDER_AND_INPUT_BYTES_MATCH',
            'FRESH_SIX_IDLE_PHYSICAL_CORES_AND_EFFECTIVE_RAM_QUOTA_PSI_OOM_GATE',
            'RESTORED_BASIS_DESCRIPTION_AND_SCALAR_MATCH_BEFORE_COMPILE',
            'COST_TASK_RECEIPTS_BIND_SOURCE_PROFILE_BASIS_AND_WRAPPER'],
        authorization=dict(approved=False, runtime_authorization=False, grant_issued=False,
            SOURCE=None, allowed_cpus=[], command=None, next_stage_authorized=False),
        implementation=dict(metadata_guard=True, scientific_adapter_sealed=False,
            scientific_launcher_available=False, valid_grant_issuer_available=False),
        mandatory_stop=True, retry=False, resume=False)


def validate(plan):
    expected = proposal(plan['binding'])
    require(plan == expected, 'FIXED_ACQUISITION_PROPOSAL_CHANGED')
    require(set(plan['binding']['prefix_preparations']) == {'14', '21'}, 'PREFIX_BINDING')
    for row in plan['binding']['prefix_preparations'].values():
        for field in ('saved_preparation_hash', 'ordered_block_description_sha256', 'view_recipe_sha256'):
            h = row[field]
            require(type(h) is str and len(h)==64 and all(x in '0123456789abcdef' for x in h), 'PREPARATION_DIGEST')
    require(plan['binding']['profile_sha256'] == PROFILE, 'FULL_PROFILE_BINDING')
    return dict(status='PASS_UNAPPROVED_METADATA_PLAN', cases=4, axes=8,
                training_axes=6, independent_validation_axes=2,
                new_compile=0, new_fit=0, executable=False)


def validate_terminal(plan, receipt):
    """Artificial/metadata receipt validator; cannot grant or launch anything."""
    validate(plan)
    require(receipt['scope'] == SCOPE and receipt['SOURCE']
            and receipt['SOURCE'] != 'cb8a56c6b2a772827275f9200b02638030f33f72'
            and receipt['SOURCE'] != PROVIDER, 'NEW_ACQUISITION_SOURCE_REQUIRED')
    require(receipt['status'] == 'COMPLETE_MANDATORY_STOP'
            and receipt['next_stage_authorized'] is False, 'ACQUISITION_MANDATORY_STOP')
    expected = {(x['case_id'], x['axis']) for x in axes()}
    require(len(receipt['axes'])==8 and {(r['case_id'],r['axis']) for r in receipt['axes']}==expected,
            'EXACT_EIGHT_AXES_REQUIRED')
    roles = {c['id']: c['role'] for c in cases()}
    by_case = {c['id']: c for c in cases()}
    for row in receipt['axes']:
        require(row['role'] == roles[row['case_id']], 'HOLDOUT_ASSIGNMENT_CHANGED')
        require(row['SOURCE'] == receipt['SOURCE'] and row['scope']==SCOPE,
                'AXIS_SOURCE_SCOPE')
        require(row['compiler_hash']==COMPILER and row['profile_sha256']==PROFILE
                and row['control']=='symmetric_directional'
                and row['scalar_policy']==plan['scalar_policy'], 'AXIS_RUNTIME_POLICY')
        prep = plan['binding']['prefix_preparations'][str(by_case[row['case_id']]['prefix'])]
        require(row['preparation_binding'] == prep, 'COST_PREPARATION_BINDING')
        require(row['native_exit_code']==0 and row['reaped'] is True, 'NATIVE_EXIT_OR_REAP')
        require(row['fingerprint_before'] and row['fingerprint_before']==row['fingerprint_after'],
                'WRAPPER_MUTATION')
        require(type(row['measured_RZ']) is int and row['measured_RZ']>0, 'MEASURED_RZ')
        error = row['max_control_error']
        require(type(error) in (int, float) and math.isfinite(error)
                and 0 <= error < 1e-8, 'CONTROL_VALIDATION')
    require(set(receipt['actual_calls']) == set(plan['call_caps']), 'CALL_ACCOUNTING')
    for key, cap in plan['call_caps'].items():
        used = receipt['actual_calls'][key]
        require(type(used) is int and 0 <= used <= cap, 'CALL_CAP:'+key)
    require(receipt['actual_calls'] == plan['call_caps'], 'COMPLETE_CALL_ACCOUNTING')
    return dict(status='PASS_TERMINAL_METADATA_ONLY', next_phase_started=False)
