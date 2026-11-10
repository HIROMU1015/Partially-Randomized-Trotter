"""C1 R-price intake: pinned metadata plus artificial C-schema records only.

Artificial publications are formatted by C's own pinned price_records with
made-up evaluation numbers. No real C1 price, fit, compile, N or G is used.
"""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import rpe_r_price_intake as ci
from trottertracks import h4_df10_price_provider as h4
from trottertracks import h6_df19_price_provider as h6
from check_pf_pr_rpe_pipeline import fake_shot
p = ci.p; r = ci.r; b = ci.b
RUN = 'docs/research/r_price_phase1_results_20261111/run01'
PUB = dict(commit='a'*40, run_dir=RUN)
FROZEN_COMMIT = 'f'*40


def mock_N(menu, row, value=None):
    n = fake_shot(menu, row)
    if value is not None:
        n['plan'].update(N_X=value, N_Y=value)
    n.update(law_binding=r.request_context(menu, row)['law_binding'], N_comparison_policy=b.N_POLICY,
             diagnostics=dict(synthetic_only=True, allowance={'policy': b.N_POLICY}))
    return n


def artificial(intake, *, stages=('CALIBRATION', 'VALIDATION', 'rare')):
    """Schema-faithful artificial C1 publication; every number is made up."""
    c, plan = intake.c, intake.plan
    groups = {s: 'R_PRICE_P1:'+s+':'+'0'*64 for s in ('H4', 'H6')}
    models = {}
    for s in ('H4', 'H6'):
        cells = [v for v in plan['cells'] if v['system'] == s]
        models[s] = dict(coefficients=[[1., 2.], [3., 4.], [5., 6.]], rank=3, cell_level_lack_of_fit_DOF=0,
            features=plan['features'], axes=list(c.AXES), calibration_cells=[v['id'] for v in cells if v['role'] == 'CALIBRATION'],
            point_roles={v['id']: v['role'] for v in cells}, unobserved_q_R_qualified=False,
            nominal_coefficient_covariance=[[0.]*6 for _ in range(6)], covariance_order='feature_then_axis',
            covariance_qualification='ARTIFICIAL', shared_calibration_group=groups[s], R_q_systematic=None,
            all_law_expected_price=None, total_CI=None, qualified_for_candidate=False)
    frozen = dict(plan_digest=intake.digest, model=plan['model'], models=models, metrics=plan['metrics'],
                  validation_prices_observed=False, validation_refit=False, acceptance_threshold=None, full_RPE_qualified=False)

    def evaluation(cell, k):
        return dict(cell_id=cell['id'], role=cell['role'], summaries=[dict(artificial=True)],
            observed_cohort_mean=[100.+k, 200.+k], predicted_cohort_mean=[101.+k, 199.+k],
            nominal_mean_price_covariance=[[1., .5], [.5, 1.]], paired_price_covariance=[[8., 4.], [4., 8.]],
            paired_residual_covariance=[[2., 1.], [1., 2.]], nominal_mean_residual_covariance_given_frozen_fit=[[.25, .125], [.125, .25]],
            price_unit='FINITE_COHORT_ONLY', validation='CALIBRATION' if cell['role'] == 'CALIBRATION' else 'FINITE_INDEPENDENT_POINT_EVALUATED_NOT_ADOPTED',
            qualified_for_candidate=False, total_CI=None, all_law_expected_price=None)

    files = dict(plan=deepcopy(plan), frozen=frozen)
    for stage in ('CALIBRATION', 'VALIDATION'):
        if stage not in stages:
            continue
        evs, prices = [], []
        for k, cell in enumerate(plan['cells']):
            if cell['role'] != stage:
                continue
            ev = evaluation(cell, k); evs.append(ev)
            source = dict(execution_source=plan['source'], result_path=RUN+'/'+stage+'.json')
            if stage == 'VALIDATION':
                source['frozen_model_commit'] = FROZEN_COMMIT
            prices += c.price_records(plan, cell, ev, frozen, source)
        doc = dict(evaluations=evs, prices=prices, full_rpe_resources=None)
        if stage == 'VALIDATION':
            doc.update(frozen_model_commit=FROZEN_COMMIT, data={'ARTIFICIAL': []}, validation_refit=False,
                       all_law_expected_price=None, total_CI=None, mandatory_stop=True)
        files[stage] = doc
    if 'rare' in stages:
        pairs = []
        for g in plan['rare_groups']:
            path = dict(cell_id='ARTIFICIAL', candidate_id=g['candidate_id'], cohort=g['cohort'], conditional=True, C=[1, 2], S=[1, 2])
            pairs.append(dict(system=g['system'], pair=g['pair'], unit='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT',
                              paths=[path, deepcopy(path)], compiled_difference=[0, 0], all_law_weighted_contribution=None))
        files['rare'] = dict(pairs=pairs, cohort_pooling=False, all_law_contribution=None, total_CI=None, mandatory_stop=True)
        files['stop'] = dict(status='PHASE1_COMPLETE_MANDATORY_STOP', mandatory_stop=True, new_N_G=0, full_RPE_qualified=False)
    return files


@contextmanager
def no_science(intake=None):
    with ExitStack() as stack:
        targets = [(p, 'evaluate'), (r, 'evaluate_menu'), (p.accounting, 'aggregate_supplied_axes'),
                   (b.nmodel, 'analyze'), (h6, '_affine'), (h6.H6AffinePriceProvider, 'predict_affine'),
                   (h4.H4PriceProvider, 'predict_path')]
        if intake is not None:
            targets += [(intake.c, name) for name in ('calibrate', 'evaluate_cell', 'price_records', 'saved_cell', 'helpers', 'make_plan')]
        for obj, name in targets:
            stack.enter_context(patch.object(obj, name, side_effect=AssertionError('No numerical work: '+name)))
        yield
    assert 'trottertracks.r_price_phase1_science' not in sys.modules


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intake = ci.C1PriceIntake(ROOT)
        cls.files = artificial(cls.intake)
        with no_science(cls.intake):
            cls.accepted = cls.intake.accept(cls.files, publication=PUB, frozen_committed=cls.files['frozen'])
            cls.inputs = b.saved.load_inputs(ROOT, p.SavedArtifacts(ROOT))

    def accept(self, files):
        return self.intake.accept(files, publication=PUB, frozen_committed=files.get('frozen'))

    def menu(self, system, mapping):
        return b.prepare_menu(ROOT, system, mapping, inputs=self.inputs)

    def test_01_fixed_R_saved_prices_come_from_existing_providers(self):
        read = p.SavedArtifacts(ROOT).read
        for system, mapping, m in [('H4', {3: [4, 8]}, 3), ('H6', {4: [16, 32]}, 4)]:
            menu = self.menu(system, mapping)['menu']
            rows = [v for v in menu['candidates'] if v['m'] == m]
            sel = {v['candidate']['id']: (dict(kind='direct_compiled', price_unit='cohort_mean', cohort='EXPLORATION')
                                          if p.arm(v['candidate']) == 'PR' else {}) for v in rows}
            with no_science():
                old = b.FixedSavedPrices(read, system, selections=sel)(menu, rows)
                new = ci.RScopedFixedPrices(read, system, selections=sel, intake=self.intake)(menu, rows)
            self.assertEqual([{k: v for k, v in x.items() if k != 'R_price_scope'} for x in new], old)
            anchor = [x for x in new if x['identity']['candidate']['R_outer'] == ci.FIXED_R[system]]
            self.assertTrue(anchor and all(x['mean']['rz_count'] is not None for x in anchor))
            pf = [x for x in new if p.arm(x['identity']['candidate']) == 'PF']
            self.assertTrue(all(x['R_price_scope']['R_price_status'] == 'PF_DETERMINISTIC_NO_R' and x['mean']['rz_count'] is not None for x in pf))

    def test_02_new_R_has_no_transplanted_fixed_coefficients(self):
        menu = self.menu('H4', {3: [4, 8]})['menu']
        rows = [v for v in menu['candidates'] if v['m'] == 3]
        provider = ci.RScopedFixedPrices(p.SavedArtifacts(ROOT).read, 'H4')
        with no_science():
            records = provider(menu, rows)
        new = [x for x in records if x['identity']['candidate']['R_outer'] == 8]
        self.assertTrue(new and all(x['mean']['rz_count'] is None and x['classification'] == 'UNAVAILABLE' for x in new))
        self.assertTrue(all(x['R_price_scope']['R_price_status'] == 'NEW_R_PRICE_NOT_ACQUIRED' for x in new))
        self.assertTrue(all(x['R_price_scope']['R_evidence_status'] == 'NEW_R_CALIBRATION_REQUIRED' for x in new))

        real = provider.fixed

        def transplant(menu, rows):
            out = real(menu, rows)
            for x in out:
                if x['identity']['candidate']['R_outer'] == 8:
                    x['mean'] = {'rz_count': 7}
            return out
        with patch.object(provider, 'fixed', Mock(side_effect=transplant)):
            with self.assertRaisesRegex(ValueError, 'TRANSPLANTED'):
                provider(menu, rows)

    def test_03_H6_minimum_R_allocation_delta_prefix_K_are_enforced(self):
        base = b.nmodel.saved_task(ROOT, 'H6')
        with self.assertRaises(ValueError):
            r.build_menu(base, {4: [10]}, sampling_law=b.LAW)
        menu = self.menu('H6', {4: [16, 32]})['menu']
        row = next(v for v in menu['candidates'] if v['m'] == 4 and v['candidate']['R_outer'] == 32)
        self.assertEqual(row['structure']['allocation'], [3, 3, 3, 3, 2, 5, 2, 3, 2, 3, 3])
        bad = deepcopy(menu)
        next(v for v in bad['candidates'] if p.same(v, row))['structure']['allocation'] = [3]*10 + [2]
        with self.assertRaisesRegex(ValueError, 'Recipe differs'):
            p.validate_task(bad)
        early = next(v for v in menu['candidates'] if v['m'] == 0 and p.arm(v['candidate']) == 'PR')
        self.assertEqual(ci.price_scope(menu, early, 'X')['R_price_status'], 'OUTSIDE_R_VARIABLE_PRICE_DESIGN')
        for field, value in [('delta', .8), ('prefix', 5), ('K', 3), ('R_outer', 31)]:
            files = deepcopy(self.files)
            record = next(x for x in files['CALIBRATION']['prices'] if x['identity']['candidate']['id'] == row['candidate']['id'])
            record['identity']['candidate'][field] = value
            with self.assertRaisesRegex(ValueError, 'outside the fixed C1 cells'):
                self.accept(files)

    def test_04_eight_cells_bind_candidate_identity_compiler_and_law(self):
        ready = self.intake.readiness()
        self.assertEqual(len(ready['cells']), 8)
        self.assertEqual(sorted(v['role'] for v in ready['cells']),
                         ['INDEPENDENT_VALIDATION']*2 + ['NEW_CALIBRATION']*2 + ['SAVED_CALIBRATION']*4)
        for v in ready['cells']:
            self.assertEqual(v['B_request']['identity']['profile']['compiler'], p.PROFILE['compiler'])
            self.assertEqual(v['B_request']['law_binding']['definition'], b.LAW)
            self.assertEqual(v['C_law_binding']['definition'], ci.C_LAW)
            self.assertEqual(v['event_count'], v['q_PF']*v['R_outer'])
        self.assertFalse(ready['law_correspondence']['ideal_sampler_distance_qualified'])
        original = r.request_context
        def changed(menu, row):
            out = original(menu, row); out['identity']['profile']['compiler'] = 'OTHER'; return out
        with patch.object(r, 'request_context', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                ci.C1PriceIntake(ROOT)

    def test_05_calibration_validation_and_model_are_separate_records(self):
        kinds = {v['c1']['price_kind'] for v in self.accepted['prices']}
        self.assertEqual(kinds, {'SAVED_CALIBRATION_DIRECT', 'NEW_CALIBRATION_DIRECT', 'INDEPENDENT_VALIDATION_DIRECT',
                                 'MODEL_PREDICTION_AT_CALIBRATION_POINT', 'MODEL_PREDICTION_AT_VALIDATION_POINT'})
        self.assertEqual(len(self.accepted['prices']), 32)
        self.assertTrue(all(v['cohort'].startswith('C1:') and v['cohort'] == 'C1:'+v['source_cohort'] for v in self.accepted['prices']))
        packet = self.menu('H4', {3: [4, 8], 4: [4, 8]})
        prices = packet['prices'] + ci.records_for(self.accepted, packet['menu'])
        inventory = p.plan(packet['menu'], packet['shots'], prices)
        self.assertIsNone(inventory['totals'])
        saved = next(v for v in self.accepted['prices'] if v['c1']['cell_id'] == 'H4_q8_R4')
        self.assertEqual(saved['c1']['same_saved_cohort_as_fixed_catalog'], 'EXPLORATION')
        self.assertTrue(any(x['cohort'] == 'DIRECT:EXPLORATION' for x in packet['prices']
                            if x['identity']['candidate']['id'] == saved['identity']['candidate']['id']))

    def test_06_validation_points_never_enter_calibration(self):
        files = deepcopy(self.files)
        files['CALIBRATION']['evaluations'].append(deepcopy(files['VALIDATION']['evaluations'][0]))
        with self.assertRaisesRegex(ValueError, 'outside its role'):
            self.accept(files)
        files = deepcopy(self.files)
        files['CALIBRATION']['prices'].append(deepcopy(files['VALIDATION']['prices'][0]))
        with self.assertRaisesRegex(ValueError, 'other calibration/validation role'):
            self.accept(files)
        files = deepcopy(self.files)
        files['frozen']['models']['H4']['calibration_cells'][-1] = 'H4_q16_R8'
        with self.assertRaisesRegex(ValueError, 'frozen calibration cells'):
            self.accept(files)
        files = deepcopy(self.files)
        files['frozen']['validation_prices_observed'] = True
        with self.assertRaisesRegex(ValueError, 'frozen model'):
            self.accept(files)
        files = deepcopy(self.files)
        direct = next(x for x in files['VALIDATION']['prices'] if x['classification'] != 'MODEL_PREDICTION')
        direct['uncertainty']['calibration_data_group'] = 'R_PRICE_P1:H4:'+'0'*64
        with self.assertRaisesRegex(ValueError, 'calibration-data group'):
            self.accept(files)
        changed = deepcopy(self.files['frozen']); changed['models']['H6']['coefficients'][0][0] = 9.
        with self.assertRaisesRegex(ValueError, 'frozen commit'):
            self.intake.accept(self.files, publication=PUB, frozen_committed=changed)
        with self.assertRaisesRegex(ValueError, 'committed frozen'):
            self.intake.accept(self.files, publication=PUB)
        files = deepcopy(self.files); files.pop('CALIBRATION')
        with self.assertRaisesRegex(ValueError, 'stage order'):
            self.accept(files)
        validation = [v for v in self.accepted['prices'] if v['c1']['stage'] == 'VALIDATION']
        self.assertTrue(all(v['source']['frozen_model_commit'] == FROZEN_COMMIT and v['c1']['frozen_model_commit'] == FROZEN_COMMIT for v in validation))

    def test_07_finite_cohort_forced_rare_and_law_expectation_stay_distinct(self):
        self.assertEqual(len(self.accepted['rare_diagnostics']), 4)
        self.assertTrue(all(v['price_record'] is False and v['pooled_with_cohort'] is False for v in self.accepted['rare_diagnostics']))
        self.assertFalse(any('FORCED' in v['cohort'] for v in self.accepted['prices']))
        self.assertTrue(all(v['expectation'] == 'FINITE_COHORT_ONLY' and not v['c1']['full_law_expected_price_qualified'] for v in self.accepted['prices']))
        files = deepcopy(self.files)
        record = files['CALIBRATION']['prices'][0]
        record['cohort'] = 'DIRECT:R_PRICE_P1_20261111_H4_FORCED_DIAGNOSTIC'
        with self.assertRaisesRegex(ValueError, 'cohort'):
            self.accept(files)
        files = deepcopy(self.files); files['rare']['cohort_pooling'] = True
        with self.assertRaisesRegex(ValueError, 'rare'):
            self.accept(files)
        files = deepcopy(self.files); files['CALIBRATION']['prices'][0]['expectation'] = 'QUALIFIED_EXPECTATION'
        with self.assertRaisesRegex(ValueError, 'expectation'):
            self.accept(files)
        menu = self.menu('H6', {4: [16, 32], 5: [16, 32]})['menu']
        prices = ci.records_for(self.accepted, menu)
        row = next(v for v in menu['candidates'] if v['candidate']['id'] == '4269eafc8c0ef8922c23')
        shots = [mock_N(menu, row)]
        select = {(row['candidate']['id'], a): 'C1:DIRECT:R_PRICE_P1_20261111_H6_R32_CALIBRATION' for a in p.AXES}
        strict = r.coverage(menu, shots, prices, cohort_selection=select)
        scenario = r.coverage(menu, shots, prices, cohort_selection=select, price_policy='finite_cohort_scenario')
        pick = lambda out: next(v for v in out['rows'] if v['candidate_id'] == row['candidate']['id'])
        self.assertIn('UNQUALIFIED_C_X', pick(strict)['reasons'])
        self.assertTrue(pick(scenario)['eligible'])
        model = {(row['candidate']['id'], a): 'C1:MODEL:R_PRICE_P1_20261111_H6_R32_CALIBRATION' for a in p.AXES}
        self.assertIn('UNQUALIFIED_C_X', pick(r.coverage(menu, shots, prices, cohort_selection=model, price_policy='finite_cohort_scenario'))['reasons'])

    def test_08_xy_pairing_and_shared_calibration_dependence_are_kept(self):
        prices = self.accepted['prices']
        for v in prices:
            twin = next(x for x in prices if x is not v and x['c1']['xy_pair'] == v['c1']['xy_pair'])
            self.assertNotEqual(twin['axis'], v['axis'])
            self.assertEqual(v['c1']['independent_classical_samples'], v['n_paths'])
            self.assertFalse(v['c1']['classical_paths_are_quantum_shots'])
            self.assertIsNotNone(v['c1']['uncertainty']['path_price_covariance'])
        group = self.accepted['shared_calibration_groups']['H4']
        dep = self.accepted['dependence_groups'][group]
        self.assertEqual(len(dep['model_predictions']), 8); self.assertEqual(len(dep['calibration_data']), 6)
        self.assertFalse(any('H4_q16_R8' in v and 'DIRECT' in v for v in dep['calibration_data']))
        self.assertEqual(p.joint_uncertainty(['a'], [1], ['PR'], dict(labels=['a'], basis='mean_price_estimator',
            source='ARTIFICIAL', covariance=[[1.]], cross_round_method_complete=True, shared_calibration_included=False))['status'],
            'INCOMPLETE_DEPENDENCE')

    def test_09_law_source_compiler_cohort_mismatches_are_rejected(self):
        edits = [(lambda x: x['law_binding']['definition'].update(sampling='IDEAL_FINITE_RTE_MEAN'), 'law binding'),
                 (lambda x: x['law_context'].update(price_law='IDEAL'), 'measurement'),
                 (lambda x: x['source'].update(execution_source='0'*40), 'SOURCE'),
                 (lambda x: x['identity']['profile'].update(compiler='OTHER'), 'outside the fixed C1 cells'),
                 (lambda x: x.update(cohort='DIRECT:OTHER'), 'cohort'),
                 (lambda x: x.update(n_paths=16), 'path count'),
                 (lambda x: x.update(outcome_convention='P1_minus_P0'), 'measurement'),
                 (lambda x: x['mean'].update(rz_count=1.), 'saved evaluation')]
        for edit, message in edits:
            files = deepcopy(self.files); edit(files['CALIBRATION']['prices'][1])
            with self.assertRaisesRegex(ValueError, message):
                self.accept(files)
        files = deepcopy(self.files); files['plan']['cells'][0]['n'] = 10
        with self.assertRaisesRegex(ValueError, 'PLAN differs'):
            self.accept(files)
        menu = self.menu('H4', {3: [4, 8]})['menu']
        bad = deepcopy(menu); bad['sampling_law'] = deepcopy(ci.C_LAW)
        with self.assertRaisesRegex(ValueError, 'law binding'):
            ci.records_for(self.accepted, bad)

    def test_10_missing_unqualified_and_null_uncertainty_are_retained(self):
        files = artificial(self.intake, stages=('CALIBRATION',))
        out = self.accept(files)
        self.assertEqual(out['completion'], 'CALIBRATION_RECEIVED_VALIDATION_PENDING')
        self.assertEqual({v['cell_id'] for v in out['cells'] if v['intake_status'] == 'NOT_RECEIVED'}, {'H4_q16_R8', 'H6_q2_R32'})
        stop = dict(files, stop=dict(status='PHASE1_FAILED_NO_RETRY', stage='validate', reason='ARTIFICIAL', mandatory_stop=True))
        self.assertEqual(self.accept(stop)['completion'], 'STOP')
        for v in self.accepted['prices']:
            self.assertTrue(all(v['c1']['uncertainty'][k] is None for k in ci.UNQUALIFIED))
            self.assertFalse(v['c1']['qualified_for_candidate'])
        files = deepcopy(self.files); files['CALIBRATION']['prices'][0]['uncertainty']['total_CI'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'unqualified uncertainty'):
            self.accept(files)
        files = deepcopy(self.files)
        files['CALIBRATION']['evaluations'][0]['observed_cohort_mean'][0] = None
        next(x for x in files['CALIBRATION']['prices'] if x['axis'] == 'X' and x['classification'] != 'MODEL_PREDICTION'
             and x['identity']['candidate']['id'] == self.intake.plan['cells'][0]['candidate_id'])['mean']['rz_count'] = None
        null = self.accept(files)
        self.assertTrue(any(v['mean']['rz_count'] is None for v in null['prices']))
        menu = self.menu('H4', {3: [4, 8], 4: [4, 8]})
        cov = ci.coverage(menu['menu'], menu['shots'], menu['prices'], cohort_selection={}, intake=self.intake)
        new = next(v for v in cov['rows'] if v['candidate_id'] == '6e1efff1ba1f7ac95a3b')
        self.assertEqual((new['C_status'], new['signal_phase_radius_N'], new['G']), ('C1_NOT_INGESTED', 'NOT_EVALUATED', 'NOT_EVALUATED'))
        self.assertEqual(new['R_price_status'], 'C1_NEW_CALIBRATION')

    def test_11_A0_normalization_is_never_N_or_C(self):
        menu = self.menu('H6', {4: [16, 32], 12: [16, 8192]})
        target = next(v for v in menu['menu']['candidates'] if v['m'] == 12 and v['candidate']['R_outer'] == 8192)
        a0 = dict(rows=[dict(candidate_id=target['candidate']['id'], normalization_status='NORMALIZATION_SCORED_NOT_N_QUALIFIED',
                             record=dict(normalization=dict(log_B=123456.789, optimistic_shot_inflation_log=246913.578)))])
        with no_science(self.intake):
            out = ci.coverage(menu['menu'], menu['shots'], menu['prices'], cohort_selection={}, intake=self.intake, a0=a0)
        row = next(v for v in out['rows'] if v['candidate_id'] == target['candidate']['id'])
        self.assertEqual(row['normalization']['status'], 'NORMALIZATION_SCORED_NOT_N_QUALIFIED')
        self.assertFalse(row['normalization']['used_as_N'] or row['normalization']['used_as_C_multiplier'])
        self.assertEqual((row['signal_phase_radius_N'], row['C_status'], row['G']), ('NOT_EVALUATED', 'NOT_ACQUIRED', 'NOT_EVALUATED'))
        self.assertNotIn('123456.789', json.dumps(out)); self.assertIsNone(out['costs'])
        self.assertFalse(out['normalization_is_price_multiplier'])
        self.assertTrue(all(v['normalization_is_price_multiplier'] is False for v in self.accepted['prices']))

    def test_12_intake_provider_and_coverage_call_no_numerical_work(self):
        packet = self.menu('H6', {4: [16, 32], 5: [16, 32]})
        menu = packet['menu']
        prices = packet['prices'] + ci.records_for(self.accepted, menu)
        scoped = ci.RScopedFixedPrices(p.SavedArtifacts(ROOT).read, 'H6', intake=self.intake)
        calls = []

        def provider(m, rows):
            self.assertFalse(any(self.intake.cell_for(v) for v in rows), 'C1 rows already have records')
            calls.append([v['candidate']['id'] for v in rows])
            return scoped(m, rows)
        with no_science(self.intake):
            self.intake.accept(self.files, publication=PUB, frozen_committed=self.files['frozen'])
            ci.coverage(menu, packet['shots'], prices, cohort_selection={}, intake=self.intake, accepted=self.accepted)
            c1_rows = [v for v in menu['candidates'] if self.intake.cell_for(v)]
            self.assertEqual(len(c1_rows), 4)
            shots = packet['shots'] + [mock_N(menu, v) for v in c1_rows if v['candidate']['R_outer'] == 32]
            out = b.run_missing(menu, shots, prices, c_provider=provider)
            again = b.run_missing(menu, out['shots'], out['prices'], c_provider=provider)
        self.assertEqual(out['status'], 'BATCH_PREPARED_NO_TOTALS'); self.assertIsNone(out['totals'])
        self.assertEqual(len(calls), 1); self.assertEqual(again['plan']['C_provider_requests'], [])

    def test_13_long_q_and_large_R_get_no_automatic_price_qualification(self):
        packet = self.menu('H4', {12: [4, 8192]})
        menu = packet['menu']
        kept = ci.records_for(self.accepted, menu)
        self.assertEqual({v['c1']['cell_id'] for v in kept}, {'H4_q8_R4', 'H4_q16_R4'})
        self.assertFalse(any(v['identity']['m'] == 12 for v in kept))
        row = next(v for v in menu['candidates'] if v['m'] == 12 and v['candidate']['R_outer'] == 8192)
        with no_science(self.intake):
            records = ci.RScopedFixedPrices(p.SavedArtifacts(ROOT).read, 'H4', intake=self.intake)(menu, [row])
        self.assertTrue(all(v['mean']['rz_count'] is None for v in records))
        self.assertEqual(records[0]['R_price_scope']['R_price_status'], 'NEW_R_PRICE_NOT_ACQUIRED')
        moved = deepcopy(self.accepted); moved['prices'][0]['identity']['m'] = 12
        moved['prices'][0]['identity']['candidate'] = deepcopy(row['candidate'])
        with self.assertRaisesRegex(ValueError, 'does not match'):
            ci.records_for(moved, menu)
        files = deepcopy(self.files)
        next(x for x in files['VALIDATION']['prices'] if x['classification'] == 'MODEL_PREDICTION')['model']['qualified_for_candidate'] = True
        with self.assertRaisesRegex(ValueError, 'qualification must stay false'):
            self.accept(files)
        self.assertFalse(self.accepted['long_q_or_new_R_transfer_qualified'])
        self.assertTrue(all(v['c1']['model_application_domain'] == 'C1_EVALUATED_POINT_ONLY' for v in self.accepted['prices']))

    def test_14_huge_integer_N_and_exact_accounting_survive(self):
        menu = self.menu('H6', {4: [16, 32], 5: [16, 32]})['menu']
        huge = (1 << 1200) + 7
        target = next(v for v in menu['candidates'] if v['candidate']['id'] == '4269eafc8c0ef8922c23')
        shots = [mock_N(menu, v, huge if v is target else None) for v in menu['candidates']]
        c1 = ci.records_for(self.accepted, menu)
        rows = [v for v in menu['candidates'] if not any(p.key(x['identity']) == p.key(v) for x in c1)]
        mock = []
        for row in rows:
            for axis in p.AXES:
                pr = p.arm(row['candidate']) == 'PR'
                mock.append(dict(**p.measurement(row, axis), **r.request_context(menu, row), axis=axis, cohort='MOCK_ONLY',
                    classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC', mean={'rz_count': 3},
                    n_paths=1, source={'fixture': 'MOCK'}, normalization_is_price_multiplier=False, peak_qubits=None,
                    model=None, uncertainty=None))
        select = {(v['identity']['candidate']['id'], v['axis']): v['cohort'] for v in mock}
        for v in c1:
            if v['c1']['price_kind'].endswith('_DIRECT'):
                select[v['identity']['candidate']['id'], v['axis']] = v['cohort']
        with patch.object(b.nmodel, 'analyze', side_effect=AssertionError('No N')):
            out = r.evaluate_menu(menu, shots, mock + c1, cohort_selection=select, price_policy='finite_cohort_scenario')
        item = next(v for v in out['rows'] if v['candidate_id'] == target['candidate']['id'])
        cx, cy = (next(v['mean']['rz_count'] for v in c1 if v['identity']['candidate']['id'] == target['candidate']['id']
                       and v['axis'] == a and v['classification'] != 'MODEL_PREDICTION') for a in p.AXES)
        self.assertEqual(Fraction(item['cost']['numerator'], item['cost']['denominator']),
                         r.a.exact_scalar(cx)*huge + r.a.exact_scalar(cy)*huge)
        n = next(v for v in shots if p.key(v['identity']) == p.key(target))
        self.assertEqual(json.loads(json.dumps(n))['plan']['N_X'], huge)
        self.assertFalse(out['scientific_R_adopted'] or out['full_RPE_qualified'] or out['rpe_success_qualified'])

    def test_15_C_module_is_pinned_and_publication_reader_uses_git_only(self):
        name = 'src/trottertracks/r_variable_compiled_price.py'
        self.assertEqual((ROOT/name).read_bytes(), subprocess.check_output(['git', 'show', ci.C_PLAN_COMMIT+':'+name], cwd=ROOT))
        self.assertEqual(self.intake.c.__file__, str(ROOT/ci.C_MODULE))
        self.assertFalse((ROOT/ci.C_MODULE).exists())
        with self.assertRaisesRegex(ValueError, 'repository-relative'):
            self.intake.read_published('a'*40, '/abs/run')
        real = ci._blob
        with patch.object(ci, '_blob', side_effect=lambda root, commit, path: None if path.startswith(RUN) else real(root, commit, path)):
            out = self.intake.read_published('b'*40, RUN)
        self.assertEqual((out['completion'], out['prices'], out['N_C_G']), ('NOT_PUBLISHED', [], None))
        self.assertTrue(all(v['intake_status'] == 'NOT_RECEIVED' for v in out['cells']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
