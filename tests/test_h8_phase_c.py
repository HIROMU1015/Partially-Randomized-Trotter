"""Artificial costs/shots/proofs only; candidate coordinates are fixed metadata.

Adapted from the 27 snapshot tests to the final B2 four-axis contract, plus checks
for per-axis B2 SOURCEs, noncancellation exclusion, set independence and the
no-cap launcher. No saved H8 value is read.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as F
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import h8_phase_c_runtime as r
import run_h8_phase_c as runner
import prepare_h8_phase_c as prepare
c = r.c
REGISTRY = json.loads((ROOT/r.REGISTRY).read_text())
CONFIRMED = REGISTRY['confirmation_ids']


def account(cid, eligible=True):
    return dict(schema='track_a_u_aware_diagnostic_v1', evidence_ref=cid + '_signal.json', evidence_kind='EMPIRICAL',
        allowance_certification_claimed_by_caller=False, cost_acquisition_eligible=eligible,
        eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE', N_total=30 if eligible else None,
        axes=dict(real=dict(shots=10, headroom=.1, u_over_headroom=1e-6), imag=dict(shots=20, headroom=.1, u_over_headroom=1e-6)))


def model_fixture():
    return dict(models=[dict(method=m, axis=a, coefficients=[str(100+10*j), '10', '2'],
        status='S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS', S4_status='UNVALIDATED_SCHEDULE_COMPOSITION_HYPOTHESIS',
        full_grid_guaranteed=False) for j, (m, a) in enumerate((('B0', 'cosine'), ('B0', 'sine'), ('B0I', 'cosine'), ('B0I', 'sine')))])


def b2_proof(model_sha, measured=(168000, 168001, 1852000, 1852001)):
    axes = []
    for (cid, axis, source, prediction), m in zip(c.B2_AXES, measured):
        axes.append(dict(case_id=cid, axis=axis, SOURCE=source, measured_RZ=m, model_sha256=model_sha,
            profile_sha256=c.PROFILE, compiler_hash=c.COMPILER, native_exit_code=0, reaped=True, fingerprint_before='fx',
            fingerprint_after='fx', max_control_error=1e-12, result_sha256='a'*64,
            model_check=dict(exact_prediction=prediction, measured_RZ=m, PASS=True,
                             exact_relative_error=str(abs(F(prediction)-m)/m))))
    final = dict(status='PASS_MANDATORY_STOP', SOURCE=c.B2_FINAL_SOURCE, profile_sha256=c.PROFILE, next_stage_authorized=False)
    binding = dict(schema='H8_B2_REGISTERED_FOUR_AXES_PASS_BINDING_V1', status=c.B2_STATUS, final_execution_SOURCE=c.B2_FINAL_SOURCE,
        source_per_axis=True, registered_axes=4, saved_axes_accepted=3, new_axes_completed=1, axes=axes, model_sha256=model_sha,
        compiler_sha256=c.COMPILER, profile_sha256=c.PROFILE, relative_error='abs(predicted-measured)/measured',
        STOP_threshold='1/50', comparison='strictly_less_than', final_gate=dict(final),
        original_runs_retroactively_changed=False, old_STOPs_and_unsettled_reservations_preserved=True,
        single_original_attempt_completed=False, full_grid_guaranteed=False, Phase_C_authorized=False, mandatory_STOP=True)
    receipts = dict(raw_driver_terminal=dict(status='COMPLETE_MANDATORY_STOP', SOURCE=c.B2_FINAL_SOURCE, fit_performed=False,
            retry=False, next_stage_authorized=False, axes=deepcopy(axes[3:]), saved_B0_axes=deepcopy(axes[:2]),
            saved_cosine_axis=deepcopy(axes[2]), validation=dict(C_authorized=False, full_grid_guaranteed=False)),
        raw_launcher_terminal=dict(status='COMPLETE_MANDATORY_STOP', native_exit_code=0, all_reaped=True, cleanup_errors=[],
            first_stop=None, process_thread_violation=None),
        native_launcher_wait=dict(status='NATIVE_LAUNCHER_WAIT_RECORDED', native_exit_code=0, wait_completed=True,
            witness_stop_requested=False),
        admission_terminal=dict(status='COMPLETE_MANDATORY_STOP', SOURCE=c.B2_FINAL_SOURCE, reported_exit_code=0),
        final_gate=dict(final), new_compile_calls=1, extra_unregistered_compile_calls=0, Phase_C_authorized=False,
        fit_performed=False, scientific_retry_after_mismatch=False)
    return dict(binding=binding, receipts=receipts)


def fixture():
    model = model_fixture()
    data = dict(fixture_kind='ARTIFICIAL', model=model, B1_result_commit=c.B1_RESULT,
        B1_terminal=dict(status='COMPLETE_MANDATORY_STOP', model_sha256=c.digest(model)),
        B2=b2_proof(c.digest(model)), accounting=[], measured_axes=[])
    det = REGISTRY['old'] + REGISTRY['additional']
    for cid in [x['id'] for x in det] + CONFIRMED:
        for e in c.EPSILONS:
            for f in c.FACTORS:
                row = dict(cell_id=cid, epsilon=e, factor=f, accounting=account(cid), bias_kind=c.USUAL_BIAS)
                if cid in CONFIRMED:
                    row.update(sample_phase='confirmation', sample_count=32, PR_metric=dict(
                        G_exact=dict(numerator='300', denominator='1'), paired_samples=32,
                        G_standard_error=1., population_winner_certified=False))
                data['accounting'].append(row)
    for cell in REGISTRY['old']:
        if cell['method'] == 'B1':
            for axis in c.AXES:
                data['measured_axes'].append(dict(cell_id=cell['id'], axis=axis, RZ=1000,
                    compiler_hash=c.COMPILER, provenance_verified=True, kind='LEGACY_FULL_PF_DECLARED_LIMITS'))
    return data


def raw_fixture(data):
    old = {x['id']: x for x in REGISTRY['old']}; extra = {x['id']: x for x in REGISTRY['additional']}
    raw = dict(B2=data['B2'], model=data['model'], B1_terminal=data['B1_terminal'], B1_anchors={'rows': []},
        Phase_A=dict(status='COMPLETE_MANDATORY_STOP', signals=[]), B0I_accounting=[],
        D0=dict(H8_EXPLORATION92=[], H8_CONFIRMATION_UNION6=[]))
    for cell in extra.values():
        raw['Phase_A']['signals'].append(dict(cell=cell, precision_rows={str(e): {str(f): account(cell['id'])
            for f in c.FACTORS} for e in c.EPSILONS}))
    for row in data['accounting']:
        cid = row['cell_id']; z = {k: v for k, v in deepcopy(row).items() if k != 'bias_kind'}
        if cid in old:
            z.update(status='DERIVED_SAVED_SCALAR')
            if old[cid]['method'] == 'B0I': raw['B0I_accounting'].append(z)
            else:
                if old[cid]['method'] == 'B1': z['metrics'] = {'RZ': {'axis_means': {'cosine': 1000, 'sine': 1000}}}
                raw['D0']['H8_EXPLORATION92'].append(z)
        elif cid in CONFIRMED:
            z.update(method='B2', metrics={'RZ': z['PR_metric']}); raw['D0']['H8_CONFIRMATION_UNION6'].append(z)
    for m in ('B0', 'B0I'):
        for p in (14, 21):
            for q in ((1, 8) if m == 'B0' else (1, 2, 4, 8)):
                for axis in c.AXES:
                    raw['B1_anchors']['rows'].append(dict(method=m, prefix=p, q=q, axis=axis,
                        formula='2nd', saved_RZ=777, control='symmetric_directional', evidence={'fixture': True},
                        runtime_profile_sha256=c.PROFILE, compiler_hash=c.COMPILER,
                        origin_SOURCE='db559a518214fd27fa96e2f1451911df05c4f4a8' if m == 'B0' else
                        '54ed98c5fa5b7190f77255271f4e3bec8db15630'))
    return raw


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture(); self.patch = patch.object(c, 'MODEL_SHA', c.digest(self.data['model'])); self.patch.start()
        self.addCleanup(self.patch.stop)

    def evaluate(self, data=None):
        calls = []; events = []
        out = c.evaluate(self.data if data is None else data, REGISTRY, take=calls.append,
                         emit=lambda kind, row: events.append((kind, row)))
        return out, calls, events

    def row(self, events, cid, e=.05, f=1.):
        return next(v for k, v in events if k == 'accounting_row' and v['cell_id'] == cid and v['epsilon'] == e and v['factor'] == f)

    def test_exact_domain_caps_scopes_and_labels(self):
        result, calls, events = self.evaluate()
        self.assertEqual(result['accounting_rows'], 1900); self.assertEqual(len(result['comparisons']), 10)
        self.assertEqual(calls.count('comparison'), 20); self.assertEqual(calls.count('reaccounting_pass'), 1)
        self.assertLessEqual(calls.count('model_axis_estimate'), 352)
        self.assertFalse(result['all_candidates_measured']); self.assertFalse(result['next_stage_authorized'])
        self.assertEqual((result['bias_kind'], result['noncancellation_diagnostic_used']), (c.USUAL_BIAS, False))
        statuses = {row['cost_status'] for kind, row in events if kind == 'accounting_row'}
        self.assertIn('MODEL_CONDITIONAL_POINT', statuses); self.assertIn('SAVED_PR_FINITE_SAMPLE_MEAN_N32', statuses)

    def test_saved_shot_weighted_exact_rational_cost(self):
        _, _, events = self.evaluate()
        self.assertEqual(F(self.row(events, 'B0_p7_q1_2nd')['G_exact']), 10*(100+10+2*7)+20*(110+10+2*7))

    def test_cosine_and_sine_unit_costs_are_separate_axes(self):
        _, _, events = self.evaluate()
        row = self.row(events, 'B0I_p21_q8_4th')
        self.assertEqual((row['axis_costs']['cosine']['RZ'], row['axis_costs']['sine']['RZ']), ('1852000', '1852001'))
        self.assertEqual(F(row['G_exact']), 10*1852000+20*1852001)

    def test_B2_measurement_precedes_model_and_keeps_per_axis_SOURCE(self):
        _, _, events = self.evaluate()
        row = self.row(events, 'B0_p15_q1_4th')
        self.assertEqual(F(row['G_exact']), 10*168000+20*168001)
        for axis, value in row['axis_costs'].items():
            self.assertEqual(value['status'], 'SAVED_MEASURED_AXIS')
            self.assertEqual(value['provenance']['kind'], 'B2_REGISTERED_S4_MEASUREMENT')
            self.assertEqual(value['provenance']['origin_SOURCE'], '244c45e752fa703e286d40ece9ad641430ea9c6e')
            self.assertEqual(value['provenance']['B2_acceptance'], 'SAVED_AXIS_ACCEPTED_BY_FINAL_CONTRACT')
        sine = self.row(events, 'B0I_p21_q8_4th')['axis_costs']['sine']['provenance']
        self.assertEqual((sine['origin_SOURCE'], sine['B2_acceptance']), (c.B2_FINAL_SOURCE, 'NEW_AXIS_COMPLETED'))

    def test_unmeasured_S4_and_S2_are_labelled_model_estimates(self):
        _, _, events = self.evaluate()
        s4 = self.row(events, 'B0_p16_q2_4th')['axis_costs']['cosine']
        self.assertEqual((s4['status'], s4['model_formula'], s4['full_grid_error_guaranteed']), ('FIXED_MODEL_ESTIMATE', 'S4', False))
        self.assertIn('FOUR_REGISTERED_B2_AXES', s4['model_validation'])
        s2 = self.row(events, 'B0_p8_q1_2nd')['axis_costs']['sine']
        self.assertEqual((s2['model_formula'], s2['prefix_outside_B1_fit_range']), ('S2', True))

    def test_any_B2_failure_prevents_every_C_call(self):
        for i in range(4):
            data = deepcopy(self.data); data['B2']['binding']['axes'][i]['model_check']['PASS'] = False; calls = []
            with self.subTest(axis=i), self.assertRaisesRegex(ValueError, 'B2_AXIS'):
                c.evaluate(data, REGISTRY, take=calls.append)
            self.assertEqual(calls, [])

    def test_B2_STOP_missing_partial_final_or_native_cleanup_rejected(self):
        mutations = {
            'status': lambda b: b['binding'].update(status='STOP'),
            'partial': lambda b: b['binding']['axes'].pop(),
            'counts': lambda b: b['binding'].update(saved_axes_accepted=4, new_axes_completed=0),
            'final_gate': lambda b: b['binding']['final_gate'].update(status='STOP'),
            'single_attempt_claim': lambda b: b['binding'].update(single_original_attempt_completed=True),
            'history_changed': lambda b: b['binding'].update(original_runs_retroactively_changed=True),
            'C_authorized_by_B2': lambda b: b['binding'].update(Phase_C_authorized=True),
            'driver': lambda b: b['receipts']['raw_driver_terminal'].update(status='STOP'),
            'reaped': lambda b: b['receipts']['raw_launcher_terminal'].update(all_reaped=False),
            'native_wait': lambda b: b['receipts']['native_launcher_wait'].update(native_exit_code=1),
            'axis_native': lambda b: b['binding']['axes'][2].update(native_exit_code=-9),
            'extra_compile': lambda b: b['receipts'].update(extra_unregistered_compile_calls=1),
            'receipt_axis_differs': lambda b: b['receipts']['raw_driver_terminal']['axes'][0].update(measured_RZ=1852002),
        }
        for name, change in mutations.items():
            data = deepcopy(self.data); change(data['B2'])
            with self.subTest(change=name), self.assertRaises(ValueError): self.evaluate(data)

    def test_B2_per_axis_SOURCEs_required_not_single_old_SOURCE(self):
        for source in ('244c45e752fa703e286d40ece9ad641430ea9c6e', c.B2_FINAL_SOURCE):
            data = deepcopy(self.data)
            for row in data['B2']['binding']['axes']:
                row['SOURCE'] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'EXACT_B2_FOUR_AXES_AND_SOURCES'):
                self.evaluate(data)

    def test_two_percent_boundary_rejected(self):
        data = deepcopy(self.data)
        data['B2']['binding']['axes'][0]['model_check']['exact_relative_error'] = '1/50'
        with self.assertRaisesRegex(ValueError, 'B2_AXIS'): self.evaluate(data)
        data = deepcopy(self.data); row = data['B2']['binding']['axes'][2]
        row['measured_RZ'] = row['model_check']['measured_RZ'] = 1815000
        row['model_check']['exact_relative_error'] = str(abs(F('1852640')-1815000)/1815000)
        with self.assertRaisesRegex(ValueError, 'B2_AXIS'): self.evaluate(data)

    def test_changed_model_compiler_or_profile_rejected(self):
        for change in (lambda d: d['model'].update(changed=True),
                       lambda d: d['B2']['binding']['axes'][0].update(profile_sha256='wrong'),
                       lambda d: d['B2']['binding'].update(compiler_sha256='wrong'),
                       lambda d: d['model']['models'][0].update(S4_status='VALIDATED_FULL_GRID')):
            data = deepcopy(self.data); change(data)
            with self.subTest(change=change), self.assertRaises(ValueError): self.evaluate(data)

    def test_noncancellation_rows_are_never_accounted(self):
        data = deepcopy(self.data); data['accounting'][0]['bias_kind'] = 'EMPIRICAL_NONCANCELLATION_DIAGNOSTIC'
        with self.assertRaisesRegex(ValueError, 'ONLY_SAVED_USUAL_SIGNED_BIAS_ROWS'): self.evaluate(data)
        for marker in ('usual_saved', 'shot_ratio_to_usual', 'bias_kind'):
            raw = raw_fixture(self.data); raw['Phase_A']['signals'][0]['precision_rows']['0.05']['1.0'][marker] = 'x'
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, 'NONCANCELLATION_ROWS_NOT_ALLOWED'):
                r.inputs.normalize(raw, REGISTRY)
        raw = raw_fixture(self.data); raw['D0']['H8_EXPLORATION92'][0]['accounting']['evidence_ref'] = 'other'
        with self.assertRaisesRegex(ValueError, 'SAVED_USUAL_ACCOUNTING_SCHEMA'): r.inputs.normalize(raw, REGISTRY)

    def test_ineligible_is_not_zero_or_a_winner(self):
        cid = 'B0_p7_q1_2nd'
        for row in self.data['accounting']:
            if row['cell_id'] == cid: row['accounting'] = account(cid, eligible=False)
        result, _, events = self.evaluate()
        row = self.row(events, cid)
        self.assertIsNone(row['G_exact']); self.assertFalse(row['eligible'])
        self.assertNotEqual(result['comparisons'][0]['scopes']['old32']['deterministic']['best']['cell_id'], cid)

    def test_missing_eligibility_prevents_full_domain_minimum(self):
        self.data['accounting'][0]['accounting'] = None
        result, _, _ = self.evaluate(); r0 = result['comparisons'][0]
        self.assertIsNone(r0['scopes']['old32']['deterministic']['best'])
        self.assertEqual(r0['scopes']['old32']['deterministic']['status'], 'UNRESOLVED_FULL_DOMAIN')
        self.assertIsNone(r0['reduction_change_percentage_points_exact'])

    def test_old32_and_184_sets_are_independent(self):
        base, _, _ = self.evaluate()
        data = deepcopy(self.data)
        target = next(x['id'] for x in REGISTRY['additional'] if x['formula'] == '2nd')
        for row in data['accounting']:
            if row['cell_id'] == target: row['accounting'] = None
        changed, _, _ = self.evaluate(data)
        for a, b in zip(base['comparisons'], changed['comparisons']):
            self.assertEqual(a['scopes']['old32'], b['scopes']['old32'])
            self.assertEqual(b['scopes']['strengthened184']['deterministic']['status'], 'UNRESOLVED_FULL_DOMAIN')
            self.assertIsNone(b['scopes']['strengthened184']['PR_reduction_percent_exact'])

    def test_missing_eligible_full_PF_cost_stops(self):
        self.data['measured_axes'].pop()
        with self.assertRaisesRegex(ValueError, 'FULL_PF_COST_MISSING'): self.evaluate()

    def test_no_PR_exploration_pooling(self):
        for change in (dict(sample_phase='exploration'), dict(sample_count=8)):
            data = deepcopy(self.data); next(x for x in data['accounting'] if x['cell_id'] in CONFIRMED).update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'CONFIRMATION_N32'): self.evaluate(data)

    def test_missing_duplicate_or_wrong_factor_rows_rejected(self):
        for change in (lambda d: d['accounting'].pop(), lambda d: d['accounting'].append(d['accounting'][0]),
                       lambda d: d['accounting'][0].update(factor=.1)):
            data = deepcopy(self.data); change(data)
            with self.subTest(change=change), self.assertRaises(ValueError): self.evaluate(data)

    def test_nonfinite_or_unreliable_shots_rejected(self):
        for field, value in (('shots', 2**52), ('shots', True), ('headroom', float('nan')), ('u_over_headroom', .011)):
            data = deepcopy(self.data); data['accounting'][0]['accounting']['axes']['real'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.evaluate(data)

    def test_exact_tie_lexical_and_no_eligible_cases(self):
        rows = [dict(cell_id=x, eligible=True, G_exact='3/2') for x in ('z', 'a')]
        best = c.minimum(rows)
        self.assertEqual((best['best']['cell_id'], best['exact_ties']), ('a', ['a', 'z']))
        self.assertEqual(c.minimum([dict(cell_id='a', eligible=False)])['status'], 'NO_ELIGIBLE_CANDIDATE')

    def test_reduction_formula_and_same_policy_old_vs_new(self):
        result, _, _ = self.evaluate(); row = result['comparisons'][0]
        for scope in row['scopes'].values():
            det = F(scope['deterministic']['best']['G_exact']); pr = F(scope['confirmed_PR']['best']['G_exact'])
            self.assertEqual(F(scope['PR_reduction_percent_exact']), 100*(1-pr/det))
        self.assertLessEqual(F(row['scopes']['strengthened184']['deterministic']['best']['G_exact']),
                             F(row['scopes']['old32']['deterministic']['best']['G_exact']))
        before, after = (F(row['scopes'][s]['PR_reduction_percent_exact']) for s in ('old32', 'strengthened184'))
        self.assertEqual(F(row['reduction_change_percentage_points_exact']), after-before)

    def test_model_axis_cache_shared_across_precision_and_scopes(self):
        _, calls, _ = self.evaluate()
        self.assertEqual(calls.count('model_axis_estimate'), 352-4)
        self.assertEqual(calls.count('saved_row'), 1900)

    def test_adapter_saved_schema_without_new_shot_rule(self):
        normalized = r.inputs.normalize(raw_fixture(self.data), REGISTRY)
        self.assertTrue(all(x['bias_kind'] == c.USUAL_BIAS for x in normalized['accounting']))
        result, _, _ = self.evaluate(normalized)
        self.assertEqual(result['accounting_rows'], 1900)

    def test_adapter_preserves_legacy_profile_unknown_and_uses_B1_anchors(self):
        normalized = r.inputs.normalize(raw_fixture(self.data), REGISTRY)
        self.assertEqual(sum(x['kind'] == 'B1_REGISTERED_ANCHOR' for x in normalized['measured_axes']), 24)
        self.assertTrue(all(x['profile_sha256'] is None for x in normalized['measured_axes']
                            if x['kind'] == 'LEGACY_FULL_PF_DECLARED_LIMITS'))

    def test_adapter_rejects_phase_A_domain_or_anchor_change(self):
        for change in (lambda d: d['Phase_A']['signals'].pop(),
                       lambda d: d['B1_anchors']['rows'][0].update(runtime_profile_sha256='wrong'),
                       lambda d: d['B1_anchors']['rows'][0].update(origin_SOURCE='0'*40)):
            raw = raw_fixture(self.data); change(raw)
            with self.subTest(change=change), self.assertRaises(ValueError): r.inputs.normalize(raw, REGISTRY)

    def test_lineage_joins(self):
        raw = dict(D0_terminal=dict(status='COMPLETE_MANDATORY_STOP', first_stop=None, output_file_receipts=[
                       dict(name='candidate_accounting_v1.json', bytes=5, sha256='d'*64)]),
                   B0I_file_receipts=dict(files=[dict(name='case_accounting_v1.json', sha256='b'*64)]),
                   B1_terminal=dict(SOURCE='6fb8a0a71986fda571d82b5658686f5e74a32f8b'),
                   B1_launcher_native_exit=dict(native_exit_code=0, wait_completed=True))
        receipts = dict(D0=dict(bytes=5, sha256='d'*64), B0I_accounting=dict(sha256='b'*64))
        self.assertTrue(r.inputs.lineage(raw, receipts))
        for name, change in (('D0', lambda x: x['D0_terminal'].update(status='FAIL_CLOSED_STOP')),
                             ('B0I', lambda x: x['B0I_file_receipts']['files'][0].update(sha256='0'*64)),
                             ('B1', lambda x: x['B1_launcher_native_exit'].update(native_exit_code=1))):
            bad = deepcopy(raw); change(bad)
            with self.subTest(name=name), self.assertRaises(ValueError): r.inputs.lineage(bad, receipts)


class RuntimeTests(unittest.TestCase):
    def grant_fixture(self):
        cfg = dict(SOURCE='a'*40, execution_sealed=True, plan_sha256=c.digest(r.plan()),
                   resource_policy=deepcopy(r.RESOURCE_POLICY), B2_receipt_set_sha256='b'*64)
        g = dict(schema='H8_PHASE_C_NEW_GRANT_V2', scope=r.SCOPE, approved_by_user=True, runtime_authorization=True,
            approval_text='ARTIFICIAL', SOURCE=cfg['SOURCE'], config_sha256=c.digest(cfg), plan_sha256=cfg['plan_sha256'],
            model_sha256=c.MODEL_SHA, B1_result_commit=c.B1_RESULT, B2_result_commit=c.B2_RESULT,
            B2_receipt_set_sha256=cfg['B2_receipt_set_sha256'], resource_policy=deepcopy(r.RESOURCE_POLICY),
            call_caps=deepcopy(r.CAPS), one_shot=True, retry=False, resume=False, refit=False,
            noncancellation_shots_allowed=False, next_stage_authorized=False, allowed_cpus=[101, 103])
        return cfg, g

    def test_runtime_draft_rejected_before_inputs_output_or_affinity(self):
        with (patch.object(r, 'read', return_value={}), patch.object(r, 'verify') as verify,
              patch.object(os, 'sched_setaffinity') as affinity):
            with self.assertRaisesRegex(ValueError, 'NEW_SEALED_C'): runner.launch('unused_config', 'unused_grant')
            verify.assert_not_called(); affinity.assert_not_called()

    def test_new_C_grant_is_source_model_B2_policy_and_scope_bound(self):
        cfg, g = self.grant_fixture()
        with patch.object(os, 'sched_getaffinity', return_value={101, 103}):
            self.assertEqual(r.authorize(cfg, g), [101, 103])
            for key, value in (('approved_by_user', False), ('runtime_authorization', False), ('SOURCE', 'bad'),
                ('model_sha256', 'bad'), ('B2_receipt_set_sha256', None), ('B2_result_commit', '0'*40), ('scope', 'B1'),
                ('retry', True), ('resource_policy', {}), ('call_caps', {}), ('noncancellation_shots_allowed', True),
                ('allowed_cpus', []), ('allowed_cpus', [999])):
                bad = deepcopy(g); bad[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError): r.authorize(cfg, bad)
            unsealed = dict(cfg, execution_sealed=False); bad = dict(g, config_sha256=c.digest(unsealed))
            with self.assertRaises(ValueError): r.authorize(unsealed, bad)

    def test_resource_policy_imposes_no_caps(self):
        policy = r.plan()['resource_policy']
        self.assertIsNone(policy['agent_imposed_CPU_RAM_worker_wall_caps'])
        self.assertEqual(policy['PSI'], 'OBSERVATION_ONLY')
        self.assertNotIn('AS_bytes', json.dumps(r.plan()))

    def test_fresh_gate_staleness_and_binding(self):
        cfg, g = self.grant_fixture()
        fresh = dict(SOURCE=cfg['SOURCE'], config_sha256=c.digest(cfg), grant_sha256=c.digest(g),
            allowed_cpus=g['allowed_cpus'], monotonic_seconds=100, source_input_verified=True,
            CPU_selection_verified=True, resources_admission_verified=True, quota_verified=True,
            protected_jobs_unchanged=True, launcher_wait_ready=True)
        r.fresh_binding(cfg, g, fresh, now=101)
        with self.assertRaises(ValueError): r.fresh_binding(cfg, g, fresh, now=131)
        fresh['protected_jobs_unchanged'] = False
        with self.assertRaises(ValueError): r.fresh_binding(cfg, g, fresh, now=101)

    def test_first_launch_stop_is_saved_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = dict(output=str(d/'out'), control=str(d/'control'), lock=str(d/'lock'),
                SOURCE='a'*40, protected_files=[], protected_worktrees=[])
            with (patch.object(r, 'read', return_value=cfg), patch.object(r, 'authorize', return_value=[101, 103]),
                  patch.object(r, 'verify'), patch.object(runner, 'fresh', side_effect=ValueError('FIRST_FIXTURE_STOP'))):
                with self.assertRaisesRegex(ValueError, 'FIRST_FIXTURE_STOP'): runner.launch(d/'cfg', d/'grant')
            stop = json.loads((d/'C_phase_STOP.json').read_text())
            self.assertEqual(stop['first_stop'], 'FIRST_FIXTURE_STOP'); self.assertFalse(stop['automatic_retry'])
            self.assertTrue((d/'C_LAUNCH_ATTEMPT').exists()); self.assertFalse((d/'out').exists())

    def test_strict_JSON_nan_overflow_and_duplicate_rejected(self):
        for text in ('{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError): r.loads(text.encode())

    def test_output_exclusive_and_supervisor_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = dict(output=str(d/'out'), control=str(d/'control'), lock=str(d/'lock'),
                                            protected_files=[], protected_worktrees=[])
            r.path_set(cfg); (d/'control').mkdir(); r.path_set(cfg, supervisor_control_exists=True)
            (d/'lock').touch()
            with self.assertRaisesRegex(ValueError, 'SPENT'): r.path_set(cfg, supervisor_control_exists=True)

    def test_output_cannot_overlap_a_protected_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); cfg = dict(output=str(d/'out'), control=str(d/'control'), lock=str(d/'lock'),
                protected_files=[], protected_worktrees=[dict(path=str(d))])
            with self.assertRaisesRegex(ValueError, 'OVERLAPS'): r.path_set(cfg)

    def test_pinned_inputs_require_exact_roles_bytes_and_SHA(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); files = []; mapping = {}
            for role in ('x', 'y'):
                (d/role).write_bytes(role.encode()); mapping[role] = str(d/role)
                files.append(dict(role=role, commit=None, path=role, bytes=1, sha256=hashlib.sha256(role.encode()).hexdigest()))
            contract = dict(schema='H8_PHASE_C_INPUT_CONTRACT_V2', real_numeric_pipeline_run=False, files=files)
            with patch.object(r, 'read', side_effect=lambda p: contract if str(p).endswith(r.INPUT_CONTRACT) else json.loads(Path(p).read_text())):
                self.assertEqual(len(prepare.pinned_inputs(mapping, git=False)), 2)
                with self.assertRaisesRegex(ValueError, 'EXACT_INPUT_ROLES'):
                    prepare.pinned_inputs(dict(mapping, z=str(d/'x')), git=False)
                (d/'y').write_bytes(b'Y')
                with self.assertRaisesRegex(ValueError, 'PINNED_SAVED_INPUT_BYTES'):
                    prepare.pinned_inputs(mapping, git=False)

    def test_B2_job_detection_matches_scripts_not_shell_text(self):
        self.assertTrue(runner.is_B2_script([b'/venv/bin/python', b'-B', b'/w/scripts/run_h8_phase_b2_recovery.py']))
        self.assertTrue(runner.is_B2_script([b'python', b'scripts/h8_phase_b2_recovery_witness.py', b'--x']))
        self.assertFalse(runner.is_B2_script([b'/bin/bash', b'-c', b'ps | grep -E "h8_phase_b2|setup_h8"']))
        self.assertFalse(runner.is_B2_script([b'python', b'scripts/run_h8_phase_c.py']))

    def test_default_CLIs_only_print_plan(self):
        for name in ('run_h8_phase_c.py', 'prepare_h8_phase_c.py'):
            value = json.loads(subprocess.check_output([sys.executable, '-P', '-B', str(ROOT/'scripts'/name)]))
            self.assertEqual(value['scope'], r.SCOPE); self.assertFalse(value['new_shot_calculation'])
            self.assertFalse(value['noncancellation_shots_used'])


class ReviewScopeAndCleanupTests(unittest.TestCase):
    """Only artificial files and at most one tiny child per case; never H8 data."""

    def test_other_worktree_HEAD_and_status_progress_are_observed_not_STOP(self):
        cfg = dict(protected_files=[], protected_worktrees=[dict(path='/artificial/other',
            HEAD='old', status_sha256='old')])
        with patch.object(subprocess, 'check_output', side_effect=[b'new\n', b' M ordinary-note.md\0']):
            report = r.protected(cfg)
        self.assertTrue(report['worktrees'][0]['changed'])
        self.assertFalse(report['worktrees'][0]['enforced'])

    def test_other_worktree_observation_error_is_not_STOP(self):
        cfg = dict(protected_files=[], protected_worktrees=[dict(path='/artificial/other',
            HEAD='old', status_sha256='old')])
        with patch.object(subprocess, 'check_output', side_effect=FileNotFoundError('moved')):
            report = r.protected(cfg)
        self.assertEqual(report['worktrees'][0]['observation_error'], 'FileNotFoundError')

    def test_only_own_dependency_receipts_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            own, other = Path(directory)/'input', Path(directory)/'unrelated-ledger'
            own.write_bytes(b'original'); other.write_bytes(b'original')
            rows = [dict(path=str(p), bytes=p.stat().st_size, sha256=r.sha(p)) for p in (own, other)]
            cfg = dict(inputs=[dict(path=str(own))], protected_files=rows, protected_worktrees=[])
            other.write_bytes(b'normal unrelated progress')
            report = r.protected(cfg)
            self.assertEqual(report['critical_files_checked'], 1)
            self.assertEqual(report['unrelated_file_receipts_observed'], 1)
            own.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'PROTECTED_BYTES'): r.protected(cfg)

    def test_own_result_and_lock_receipts_stay_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root/'out'; out.mkdir(); path = out/'receipt'
            path.write_bytes(b'original')
            cfg = dict(output=str(out), protected_files=[dict(path=str(path), bytes=8, sha256=r.sha(path))],
                       protected_worktrees=[])
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'PROTECTED_BYTES'): r.protected(cfg)

    def fake_child(self):
        identity = [99999999, 123, os.getpid(), os.getuid()]
        with patch.object(r, 'identity', return_value=identity):
            owned = runner.OwnedChild(SimpleNamespace(pid=identity[0], returncode=None))
        return owned, identity

    def test_pidfd_ESRCH_still_requires_wait_and_closes_FD(self):
        owned, identity = self.fake_child()
        with (patch.object(r, 'identity', return_value=identity), patch.object(os, 'pidfd_open', return_value=123),
              patch.object(signal, 'pidfd_send_signal', side_effect=ProcessLookupError(errno.ESRCH, 'exited')),
              patch.object(os, 'close') as close):
            owned.send(signal.SIGTERM)
            self.assertIsNone(owned.native); close.assert_called_once_with(123)
        usage = SimpleNamespace(ru_utime=0., ru_stime=0., ru_maxrss=1)
        with patch.object(os, 'wait4', return_value=(owned.child.pid, 0, usage)):
            self.assertTrue(owned.wait()['reaped'])

    def test_pidfd_non_ESRCH_and_ownership_mismatch_are_not_success(self):
        owned, identity = self.fake_child()
        with (patch.object(r, 'identity', return_value=identity), patch.object(os, 'pidfd_open', return_value=123),
              patch.object(signal, 'pidfd_send_signal', side_effect=PermissionError(errno.EPERM, 'denied')),
              patch.object(os, 'close') as close):
            with self.assertRaises(PermissionError): owned.send(signal.SIGTERM)
            close.assert_called_once_with(123)
        with patch.object(r, 'identity', return_value=[identity[0], 999, os.getpid(), os.getuid()]), \
             patch.object(os, 'pidfd_open') as opened:
            with self.assertRaisesRegex(ValueError, 'OWN_CHILD_IDENTITY'): owned.send(signal.SIGTERM)
            opened.assert_not_called()

    def test_double_cleanup_does_not_signal_or_reap_twice(self):
        owned, _ = self.fake_child()
        owned.native = dict(native_exit_code=0, reaped=True)
        with patch.object(os, 'wait4') as wait, patch.object(os, 'pidfd_open') as opened:
            self.assertEqual(owned.cleanup(), []); self.assertEqual(owned.cleanup(), [])
            wait.assert_not_called(); opened.assert_not_called()

    def test_cleanup_errors_remain_errors_and_not_reaped(self):
        owned, _ = self.fake_child()
        with patch.object(owned, 'wait', return_value=None), \
             patch.object(owned, 'send', side_effect=PermissionError('denied')):
            self.assertIn('PermissionError:denied', owned.cleanup()[0])
            self.assertIsNone(owned.native)

    def interrupted_launcher(self, *, via_signal, ignore_TERM=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); ready = root/'ready'
            cfg = dict(output=str(root/'out'), control=str(root/'control'), lock=str(root/'lock'),
                       SOURCE='a'*40, python=sys.executable, protected_files=[], protected_worktrees=[])
            created = []
            popen, sleep = subprocess.Popen, time.sleep
            def launch_dummy(command, **kwargs):
                code = ('import signal,time;from pathlib import Path;'
                        + ('signal.signal(signal.SIGTERM,signal.SIG_IGN);' if ignore_TERM else '')
                        + 'Path(' + repr(str(ready)) + ').write_text("ready");time.sleep(30)')
                child = popen([sys.executable, '-P', '-B', '-c', code], stdout=kwargs['stdout'], stderr=kwargs['stderr'])
                created.append(child)
                deadline = time.monotonic()+5
                while not ready.exists() and time.monotonic() < deadline:
                    sleep(.01)
                self.assertTrue(ready.exists())
                return child
            interrupted = False
            def trigger(delay):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    if via_signal:
                        os.kill(os.getpid(), signal.SIGTERM)
                    else:
                        raise KeyboardInterrupt('FIRST_WAIT_INTERRUPTION')
                else:
                    sleep(delay)
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                with (patch.object(r, 'read', return_value=cfg), patch.object(r, 'authorize', return_value=[min(os.sched_getaffinity(0))]),
                      patch.object(r, 'verify', return_value=dict(worktrees=[])), patch.object(runner, 'fresh', return_value={}),
                      patch.object(runner, 'observe', return_value={}), patch.object(subprocess, 'Popen', side_effect=launch_dummy),
                      patch.object(time, 'sleep', side_effect=trigger)):
                    with self.assertRaises((RuntimeError, KeyboardInterrupt)):
                        runner.launch(root/'cfg', root/'grant')
                stop = json.loads((root/'C_phase_STOP.json').read_text())
                native = json.loads((root/'control/native_exit.json').read_text())
                self.assertTrue(stop['all_children_reaped']); self.assertEqual(stop['cleanup_errors'], [])
                self.assertIn('C_LAUNCHER_SIGNAL:' if via_signal else 'FIRST_WAIT_INTERRUPTION', stop['first_stop'])
                self.assertEqual(native['native_exit_code'], -signal.SIGKILL if ignore_TERM else -signal.SIGTERM)
                self.assertFalse((root/'C_final_gate.json').exists())
                self.assertFalse(Path('/proc', str(created[0].pid)).exists())
                for sig in handlers: self.assertEqual(signal.getsignal(sig), handlers[sig])
                # File contexts closed both launcher stdout/stderr handles on exit.
                for name in ('stdout.log', 'stderr.log'):
                    with (root/'control'/name).open('ab') as stream: stream.write(b'closed-parent-handles\n')
            finally:
                for child in created:
                    if child.returncode is None:
                        child.kill(); child.wait()

    def test_launcher_SIGTERM_stops_and_reaps_only_its_own_child(self):
        self.interrupted_launcher(via_signal=True)

    def test_launcher_wait_exception_stops_and_reaps_own_child(self):
        self.interrupted_launcher(via_signal=False)

    def test_own_child_ignoring_TERM_is_KILLed_and_reaped(self):
        self.interrupted_launcher(via_signal=True, ignore_TERM=True)


class EndToEndArtificialTests(unittest.TestCase):
    """Real launcher -> driver child on ARTIFICIAL bytes, in a throwaway git copy of SOURCE."""

    @unittest.skipUnless(Path(r.PYTHON).exists(), 'pinned interpreter required')
    def test_launcher_driver_complete_then_one_shot_spent_and_source_change_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); src = d/'src'
            model = model_fixture(); model_sha = c.digest(model)
            for rel in r.SOURCES:
                (src/rel).parent.mkdir(parents=True, exist_ok=True)
                data = (ROOT/rel).read_bytes()
                if rel.endswith('h8_phase_c_accounting.py'):
                    data = data.replace(c.MODEL_SHA.encode(), model_sha.encode())  # artificial model digest only
                (src/rel).write_bytes(data)
            data = fixture(); data['model'] = model; data['B1_terminal']['model_sha256'] = model_sha
            data['B2'] = b2_proof(model_sha); raw = raw_fixture(data)
            inputs = d/'inputs'; inputs.mkdir()
            enc = lambda v: json.dumps(v, sort_keys=True).encode()
            blobs = {k: enc(raw[k]) for k in ('D0', 'B0I_accounting', 'model', 'B1_anchors')}
            blobs.update(B1_terminal=enc(dict(raw['B1_terminal'], SOURCE='6fb8a0a71986fda571d82b5658686f5e74a32f8b')),
                         B1_launcher_native_exit=enc(dict(native_exit_code=0, wait_completed=True)),
                         Phase_A=gzip.compress(enc(raw['Phase_A']), mtime=0), B2_binding=enc(raw['B2']['binding']),
                         B2_execution_receipts=enc(raw['B2']['receipts']), registry=(ROOT/r.REGISTRY).read_bytes())
            sha = lambda b: hashlib.sha256(b).hexdigest()
            blobs['D0_terminal'] = enc(dict(status='COMPLETE_MANDATORY_STOP', first_stop=None, output_file_receipts=[
                dict(name='candidate_accounting_v1.json', bytes=len(blobs['D0']), sha256=sha(blobs['D0']))]))
            blobs['B0I_file_receipts'] = enc(dict(files=[dict(name='case_accounting_v1.json', sha256=sha(blobs['B0I_accounting']))]))
            files = []
            for role, blob in blobs.items():
                (inputs/role).write_bytes(blob)
                files.append(dict(role=role, path=str(inputs/role), bytes=len(blob), sha256=sha(blob)))
            contract = dict(schema='H8_PHASE_C_INPUT_CONTRACT_V2', real_numeric_pipeline_run=False,
                            files=[dict(role=x['role'], commit=None, path=x['role'], bytes=x['bytes'], sha256=x['sha256']) for x in files])
            (src/r.INPUT_CONTRACT).write_text(json.dumps(contract))
            env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t', GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
            for args in (['init', '-q'], ['add', '-A'], ['commit', '-q', '-m', 'artificial']):
                subprocess.run(['git', *args], cwd=src, check=True, env=env, capture_output=True)
            source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src).decode().strip()
            sys.path.insert(0, str(src/'scripts'))
            try:
                code = ('import json,sys;sys.path.insert(0,sys.argv[1]);import h8_phase_c_runtime as r;c=r.c;'
                        'print(json.dumps(dict(plan=c.digest(r.plan()),hashes={p:r.sha(r.ROOT/p) for p in r.SOURCES})))')
                info = json.loads(subprocess.check_output([r.PYTHON, '-P', '-B', '-c', code, str(src/'scripts')]))
            finally:
                sys.path.pop(0)
            stage = d/'stage'; stage.mkdir()
            by = {x['role']: x for x in files}
            cfg = dict(schema='H8_PHASE_C_CONFIG_V2', SOURCE=source, execution_sealed=True, approved=False, runtime_authorization=False,
                plan_sha256=info['plan'], resource_policy=r.RESOURCE_POLICY, source_hashes=info['hashes'], python=r.PYTHON,
                python_sha256=r.sha(r.PYTHON), profile_sha256=c.PROFILE, compiler_hash=c.COMPILER, inputs=files,
                B2_receipt_set_sha256=c.digest([by[k] for k in r.B2_ROLES]), protected_files=[], protected_worktrees=[],
                output=str(stage/'output01'), control=str(stage/'control01'), lock=str(stage/'C_ONE_SHOT'))
            cpus = sorted(os.sched_getaffinity(0))[-2:]
            grant = dict(schema='H8_PHASE_C_NEW_GRANT_V2', scope=r.SCOPE, approved_by_user=True, runtime_authorization=True,
                approval_text='ARTIFICIAL_UNITTEST_ONLY', SOURCE=source, config_sha256=c.digest(cfg), plan_sha256=info['plan'],
                model_sha256=model_sha, B1_result_commit=c.B1_RESULT, B2_result_commit=c.B2_RESULT,
                B2_receipt_set_sha256=cfg['B2_receipt_set_sha256'], resource_policy=r.RESOURCE_POLICY, call_caps=r.CAPS,
                one_shot=True, retry=False, resume=False, refit=False, noncancellation_shots_allowed=False,
                next_stage_authorized=False, allowed_cpus=cpus)
            (d/'config.json').write_text(json.dumps(cfg)); (d/'grant.json').write_text(json.dumps(grant))
            launch = [r.PYTHON, '-P', '-B', str(src/'scripts/run_h8_phase_c.py'), '--launch-approved',
                      '--config', str(d/'config.json'), '--grant', str(d/'grant.json')]
            done = subprocess.run(launch, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            terminal = json.loads((stage/'output01/terminal.json').read_text())
            self.assertEqual((terminal['status'], terminal['actual_calls']['saved_row']), ('COMPLETE_MANDATORY_STOP', 1900))
            native = json.loads((stage/'control01/native_exit.json').read_text())
            self.assertEqual((native['native_exit_code'], native['reaped'], native['PSI_is_STOP_gate']), (0, True, False))
            self.assertEqual(json.loads((stage/'C_final_gate.json').read_text())['status'], 'PASS_MANDATORY_STOP')
            result = json.loads((stage/'output01/comparisons.json').read_text())
            self.assertEqual((len(result['comparisons']), result['B2_binding']['status']), (10, c.B2_STATUS))
            again = subprocess.run(launch, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0); self.assertIn('ONE_SHOT_ALREADY_SPENT', again.stderr)
            shutil.rmtree(stage); stage.mkdir()
            (src/'scripts/h8_phase_c_inputs.py').write_text((src/'scripts/h8_phase_c_inputs.py').read_text() + '\n# changed\n')
            changed = subprocess.run(launch, capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0); self.assertIn('C_SOURCE_CLOSURE_CHANGED', changed.stderr)
            self.assertFalse((stage/'output01').exists())


if __name__ == '__main__': unittest.main()
