"""Synthetic receipts only; no saved scientific costs or numerical fitting."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_h8_b1_stop_acceptance as checker
r = checker.review


def fixture():
    plan = dict(scope='artificial', control='symmetric_directional', scalar_policy='artificial',
                call_caps={'compile': 8}, cases=[], binding={'prefix_preparations': {}})
    for p in (14, 21):
        plan['binding']['prefix_preparations'][str(p)] = {'ordered_block_description_sha256': str(p)}
    train, holdout, workers, controls = [], [], [], []
    for p, q in ((14, 1), (21, 1), (14, 8), (21, 8)):
        role = 'HOLDOUT' if (p, q) == (21, 8) else 'TRAIN'
        case = dict(id=f'artificial_{p}_{q}', prefix=p, q=q, role=role)
        plan['cases'].append(case)
        prep = plan['binding']['prefix_preparations'][str(p)]
        ctrl = dict(case=case, SOURCE=r.B0_SOURCE, profile_sha256=r.PROFILE, scalar_policy='artificial',
                    preparation_binding=prep, basis_observed_from_restored_live_operations=True,
                    observed_ordered_block_description_sha256=str(p), max_control_error=1e-12,
                    wrapper_fingerprints=dict(cosine='fixture_cos', sine='fixture_sin'))
        controls.append(ctrl)
        for axis in ('cosine', 'sine'):
            identity = [len(workers) + 1, 100, 200, 300]
            row = dict(case_id=case['id'], axis=axis, role=role, SOURCE=r.B0_SOURCE,
                       scope='artificial', profile_sha256=r.PROFILE, compiler_hash=r.COMPILER,
                       control='symmetric_directional', scalar_policy='artificial', preparation_binding=prep,
                       fingerprint_before=ctrl['wrapper_fingerprints'][axis],
                       fingerprint_after=ctrl['wrapper_fingerprints'][axis], max_control_error=1e-12,
                       measured_RZ=1, CX=1, depth=1, size=1, native_exit_code=0, reaped=True,
                       worker_identity=identity, result_sha256='fixture_hash_' + str(len(workers)))
            (train if role == 'TRAIN' else holdout).append(row)
            workers.append(dict(task_id=case['id'] + '_' + axis, exit_code=0, reaped=True,
                                identity=identity, result_sha256=row['result_sha256']))
    stop = dict(status='STOP_AFTER_EIGHT_SAVED_MEASUREMENTS', acquisition_contract_status='STOP',
                first_observer_stop='RuntimeError:PROCESS_THREADS', final_gate_exists=False,
                STOP_changed_to_PASS=False, final_protected_worktree_audit_completed=False,
                exact_process_causing_thread_violation='UNRESOLVED_NOT_RECORDED', Git_threading_cause_proven=False,
                all_owned_processes_reaped=True, cleanup_errors=[], all_eight_workers_exit0_and_reaped=True,
                unknown_reserved_compile_calls=0, compile_entered=8, completed_calls_lower_bound={'compile': 8})
    audit = dict(status='SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL',
                 execution_SOURCE=r.B0_SOURCE, saved_result_commit=r.B0_RESULT, original_acquisition_status='STOP',
                 original_final_gate_replaced=False, historical_thread_compliance_certified=False,
                 posthoc_audit_retroactively_closes_contract=False, Git_cause_of_original_violation_proven=False,
                 B1_authorized=False, SOURCE_input_runtime_preparation_compiler_binding=True,
                 original_config_grant_raw_unchanged=True, final_preservation_audit_passed_posthoc=True,
                 measured_axes_verified=8, TRAIN_axes=6, HOLDOUT_axes=2)
    terminal = dict(status='COMPLETE_MANDATORY_STOP', SOURCE=r.CANONICAL_SOURCE,
                    model_sha256=r.digest({'fixture': True}), B2_authorized=False,
                    C_authorized=False, next_stage_authorized=False)
    return dict(stop=stop, audit=audit, plan=plan,
                binding=dict(SOURCE=r.B0_SOURCE, profile_sha256=r.PROFILE,
                             compiler={'compiler_hash': r.COMPILER}, input_receipts=[{} for _ in range(8)],
                             preparation_binding=deepcopy(plan['binding']['prefix_preparations'])),
                train=dict(role='TRAIN', axes=train, fit_performed=False),
                holdout=dict(role='HOLDOUT', axes=holdout, fit_performed=False),
                workers={'waves': [dict(workers=workers[i:i+4], status='PASS', all_reaped=True,
                                       cleanup_errors=[], automatic_retry=False) for i in (0, 4)]},
                controls={'cases': controls}, B1_model={'fixture': True},
                B1_contract=dict(authorized_B1_executed_once=True, original_B0_run02_status='STOP'),
                B1_binding=dict(SOURCE=r.CANONICAL_SOURCE, profile_sha256=r.PROFILE,
                                compiler_hash=r.COMPILER, original_B0_acquisition_status='STOP',
                                historical_thread_compliance_certified=False),
                B1_terminal=terminal, B1_phase=deepcopy(terminal),
                B1_cleanup=dict(native_exit_code=0, all_reaped=True, first_stop=None,
                                cleanup_errors=[], automatic_retry=False, max_monitor_gap_seconds=1))


def draft():
    return dict(schema='H8_B1_SAVED_STOP_ACCEPTANCE_DRAFT_V1', approved=False, accepted=False,
                runtime_authorization=False, existing_result_adopted=False, B2_authorized=False,
                C_authorized=False, next_stage_authorized=False, allowed_cpus=[], absolute_launch_command=None,
                canonical_numerical_SOURCE=r.CANONICAL_SOURCE, unselected_numerical_SOURCE=r.UNSELECTED_SOURCE,
                existing_B1_result_commit=r.EXISTING_B1_RESULT, duplicate_fit_permitted=False,
                evidence_set_sha256='artificial', original_acquisition_status='STOP',
                original_final_gate_replaced=False, historical_thread_compliance_certified=False)


class AcceptanceTests(unittest.TestCase):
    def validate(self, data):
        with patch.object(r, 'MODEL', r.digest({'fixture': True})):
            return r.validate_records(data)

    def reject(self, change, message):
        data = fixture(); change(data)
        with self.assertRaisesRegex(ValueError, message):
            self.validate(data)

    def test_eligible_is_never_permission(self):
        result = self.validate(fixture())
        self.assertFalse(result['approved']); self.assertFalse(result['runtime_authorization'])
        self.assertFalse(result['duplicate_fit_permitted'])
        self.assertEqual(result['original_acquisition_status'], 'STOP')

    def test_original_stop_cannot_be_pass(self):
        self.reject(lambda x: x['stop'].update(acquisition_contract_status='PASS'), 'ORIGINAL_STOP')

    def test_final_gate_cannot_be_fabricated(self):
        self.reject(lambda x: x['stop'].update(final_gate_exists=True), 'FINAL_GATE')

    def test_thread_cause_cannot_be_inferred(self):
        self.reject(lambda x: x['stop'].update(Git_threading_cause_proven=True), 'THREAD_CAUSE')

    def test_unknown_reservation_stops(self):
        self.reject(lambda x: x['stop'].update(unknown_reserved_compile_calls=1), 'COMPLETENESS')

    def test_audit_is_not_authorization(self):
        self.reject(lambda x: x['audit'].update(B1_authorized=True), 'AUDIT_NOT_AUTHORIZATION')

    def test_missing_axis_stops(self):
        self.reject(lambda x: x['train']['axes'].pop(), 'FIXED_SPLIT')

    def test_duplicate_axis_stops(self):
        self.reject(lambda x: x['train']['axes'].__setitem__(1, deepcopy(x['train']['axes'][0])), 'EIGHT_AXES')

    def test_wrong_profile_stops(self):
        self.reject(lambda x: x['train']['axes'][0].update(profile_sha256='wrong'), 'AXIS_LINEAGE')

    def test_preparation_mismatch_stops(self):
        self.reject(lambda x: x['binding']['preparation_binding']['14'].update(extra='changed'), 'PREPARATION_PLAN')

    def test_fingerprint_mismatch_stops(self):
        self.reject(lambda x: x['train']['axes'][0].update(fingerprint_after='changed'), 'FINGERPRINT')

    def test_worker_exit_or_reap_failure_stops(self):
        for update in ({'exit_code': 1}, {'reaped': False}, {'identity': [999]}, {'result_sha256': 'changed'}):
            with self.subTest(update=update):
                self.reject(lambda x: x['workers']['waves'][0]['workers'][0].update(update), 'EXIT_REAP_RESULT')

    def test_first_stop_reason_preserved(self):
        self.reject(lambda x: x['stop'].update(first_observer_stop='another'), 'ORIGINAL_STOP')

    def test_nonfinite_or_large_control_error_stops(self):
        for value in (float('nan'), float('inf'), 1e-8):
            with self.subTest(value=value):
                self.reject(lambda x: x['train']['axes'][0].update(max_control_error=value), 'CONTROL_TOLERANCE')

    def test_existing_b1_must_stay_bound(self):
        self.reject(lambda x: x['B1_phase'].update(SOURCE='wrong'), 'B1_TERMINAL')

    def test_existing_model_hash_only(self):
        self.reject(lambda x: x['B1_model'].update(changed=True), 'MODEL_BYTES')

    def test_existing_b1_cleanup_failure_stops(self):
        self.reject(lambda x: x['B1_cleanup'].update(all_reaped=False), 'B1_CLEANUP')

    def test_draft_rejects_acceptance_execution_and_duplicate_fit(self):
        r.validate_draft(draft(), 'artificial')
        for key in ('approved', 'accepted', 'runtime_authorization', 'existing_result_adopted',
                    'B2_authorized', 'C_authorized', 'next_stage_authorized', 'duplicate_fit_permitted'):
            x = draft(); x[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_draft(x, 'artificial')

    def test_draft_binds_selected_implementation_and_evidence(self):
        for key in ('canonical_numerical_SOURCE', 'evidence_set_sha256', 'existing_B1_result_commit'):
            x = draft(); x[key] = 'wrong'
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.validate_draft(x, 'artificial')

    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                r.strict_json(data)

    def test_evidence_bytes_and_fixed_object_are_checked_before_parsing(self):
        role, (commit, path) = next(iter(checker.EVIDENCE_OBJECTS.items()))
        row = dict(role=role, commit=commit, path=path, bytes=1, sha256='bad')
        inventory = dict(schema='H8_B1_ACCEPTANCE_EVIDENCE_V1', files=[row])
        proposal = draft(); proposal['evidence_set_sha256'] = r.digest(inventory)
        with patch.object(checker.subprocess, 'check_output', return_value=b'{}'):
            with self.assertRaisesRegex(ValueError, 'EVIDENCE_BYTES'):
                checker.verify(ROOT, inventory, proposal)
        row['commit'] = 'a' * 40
        proposal['evidence_set_sha256'] = r.digest(inventory)
        with patch.object(checker.subprocess, 'check_output') as git:
            with self.assertRaisesRegex(ValueError, 'FIXED_EVIDENCE_OBJECT'):
                checker.verify(ROOT, inventory, proposal)
            git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
