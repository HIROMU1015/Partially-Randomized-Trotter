"""Artificial/mock metadata and API tests only. No saved scientific prices."""
from copy import deepcopy
from pathlib import Path
import ast
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks.r_variable_compiled_price import describe_request, RVariableSavedPriceProvider
from check_pf_pr_rpe_pipeline import fake_shot, fixture


def toy_task(system='H4', Rs=(4, 8), q=1):
    task, _, _, _ = fixture()
    task['binding']['system'] = system
    task['configuration']['base_time'] = (.1 if system == 'H4' else 1.6)*q
    rows = []
    for R in Rs:
        c = dict(system=system, method='PR', formula='2nd' if system == 'H4' else '4th(m5_best)',
            prefix=1 if system == 'H4' else 7, T=task['configuration']['base_time'], q=q,
            delta=.1 if system == 'H4' else 1.6, R_outer=R, K=2)
        c['id'] = p.candidate_id(c)
        rows.append(dict(m=0, q_RPE=1, q_PF=q, candidate=c, structure=p.structure(c), adoption='ARTIFICIAL_METADATA_ONLY'))
    task['candidates'] = rows
    p.validate_task(task)
    return task


def mock_delegate(task, rows):
    records = []
    for row in rows:
        anchor = 4 if row['candidate']['system'] == 'H4' else 16
        for axis in p.AXES:
            value = (7 if axis == 'X' else 9) if row['candidate']['R_outer'] == anchor else None
            records.append(dict(**p.measurement(row, axis, saved_sampler=True), identity=p.identity(task, row),
                axis=axis, cohort='MOCK_COHORT', n_paths=8, mean={'rz_count': value},
                classification='DIRECT_COMPILED_FINITE_SAMPLE' if value is not None else 'UNAVAILABLE',
                expectation='FINITE_COHORT_ONLY' if value is not None else 'UNAVAILABLE',
                source={'fixture': 'MOCK_ONLY'}, model=None,
                uncertainty={'joint_covariance': None, 'total_CI': None},
                normalization_is_price_multiplier=False))
    return records


class Tests(unittest.TestCase):
    def test_R_identity_and_axis_are_never_merged(self):
        task = toy_task(q=2)
        values = [describe_request(task, r, a, pipeline=p) for r in task['candidates'] for a in p.AXES]
        self.assertEqual(len({v['join_key'] for v in values}), 4)
        self.assertEqual({v['event_count'] for v in values}, {8, 16})
        self.assertTrue(all(v['q_RPE'] == 1 and v['q_PF'] == 2 for v in values))
        self.assertEqual(values[0]['R_evidence_status'], 'FIXED_R_SAVED_LOOKUP_ONLY')
        self.assertEqual(values[2]['R_evidence_status'], 'NEW_R_CALIBRATION_REQUIRED')
        self.assertTrue(all(not v['law_probabilities_evaluated'] and not v['numerical_model_evaluated'] for v in values))

    def test_H6_reuses_exact_allocation_and_minimum_eleven(self):
        task = toy_task('H6', (11, 16, 32), q=2)
        for row in task['candidates']:
            desc = describe_request(task, row, 'X', pipeline=p)
            self.assertEqual(len(desc['allocation']), 11)
            self.assertEqual(sum(desc['allocation']), desc['R_outer'])
            self.assertTrue(all(r >= 1 for r in desc['allocation']))
            self.assertEqual(desc['event_count'], 2*desc['R_outer'])
            self.assertEqual(desc['minimum_legal_R_outer'], 11)
            self.assertEqual(desc['allocation'], list(p._schedule().allocation(p._schedule().weights('4th(m5_best)'), desc['R_outer'])))
        with self.assertRaises(ValueError): toy_task('H6', (10,))
        with self.assertRaises(ValueError): toy_task('H4', (0,))
        with self.assertRaises(ValueError): toy_task('H4', (True,))

    def test_profile_law_source_and_R_are_in_join_binding(self):
        task = toy_task(Rs=(4,));row = task['candidates'][0]
        first = describe_request(task, row, 'X', pipeline=p)['join_key']
        for section, field, value in [('profile', 'compiler', 'OTHER'), ('contract', 'sampling_law_source', 'OTHER'), ('binding', 'state', 'OTHER')]:
            other = deepcopy(task);other[section][field] = value
            self.assertNotEqual(first, describe_request(other, other['candidates'][0], 'X', pipeline=p)['join_key'])
        bad = deepcopy(row);bad['candidate']['R_outer'] = 8
        with self.assertRaises(ValueError): describe_request(task, bad, 'X', pipeline=p)

    def test_recipe_mismatch_does_not_apply_an_R_model(self):
        task = toy_task('H6', (16,));task['candidates'][0]['candidate']['prefix'] = 5
        c = task['candidates'][0]['candidate'];c['id'] = p.candidate_id(c)
        with self.assertRaisesRegex(ValueError, 'OUTSIDE_R_VARIABLE'): describe_request(task, task['candidates'][0], 'X', pipeline=p)
        task = toy_task();task['candidates'][0]['candidate']['delta'] = .2
        with self.assertRaises(ValueError): describe_request(task, task['candidates'][0], 'X', pipeline=p)

    def test_PF_has_no_R_parameter(self):
        for system in ['H4', 'H6']:
            task = toy_task(system, (4 if system == 'H4' else 16,))
            c = task['candidates'][0]['candidate'];c.update(method='B0' if system == 'H4' else 'FULL',prefix=7 if system == 'H4' else 19,R_outer=None,K=None)
            c['id'] = p.candidate_id(c);task['candidates'][0]['structure'] = p.structure(c)
            desc = describe_request(task, task['candidates'][0], 'X', pipeline=p)
            self.assertEqual(desc['R_evidence_status'], 'PF_R_NOT_APPLICABLE');self.assertEqual(desc['event_count'], 0)

    def test_fixed_factories_are_reused_and_new_R_is_null(self):
        for system in ['H4', 'H6']:
            task = toy_task(system, (4, 8) if system == 'H4' else (16, 32))
            factory_name = 'h4_saved_price_provider' if system == 'H4' else 'h6_saved_price_provider'
            reader = Mock(side_effect=AssertionError('No scientific JSON needed in mock'))
            delegate = Mock(side_effect=mock_delegate);selections={r['candidate']['id']: {'price_unit': 'cohort_mean', 'cohort': 'MOCK_ONLY'} for r in task['candidates']}
            with patch.object(p, factory_name, return_value=delegate) as factory:
                provider = RVariableSavedPriceProvider(reader, pipeline=p, selections=selections)
                records = provider(task, task['candidates']);provider(task, [])
            factory.assert_called_once_with(reader, selections=selections);reader.assert_not_called();delegate.assert_called_once()
            self.assertEqual([r['mean']['rz_count'] for r in records], [7, 9, None, None])
            self.assertTrue(all(r['uncertainty']['joint_covariance'] is None for r in records))
            self.assertTrue(all(r['expectation'] != 'QUALIFIED_LAW_EXPECTATION' for r in records))

    def test_mock_batch_run_missing_never_solves_N_or_sums_G(self):
        task = toy_task();shots = [fake_shot(task, r) for r in task['candidates']]
        no_N = Mock(side_effect=AssertionError('No N'));delegate = Mock(side_effect=mock_delegate)
        with (patch.object(p, 'h4_saved_price_provider', return_value=delegate),
              patch.object(p, 'evaluate', side_effect=AssertionError('No G')),
              patch.object(p.accounting, 'aggregate_supplied_axes', side_effect=AssertionError('No sums'))):
            provider = RVariableSavedPriceProvider(Mock(), pipeline=p)
            first = p.run_missing(task, shots, [], n_provider=no_N, c_provider=provider, price_policy='finite_cohort_scenario')
            second = p.run_missing(task, shots, first['prices'], c_provider=provider, price_policy='finite_cohort_scenario')
        no_N.assert_not_called();delegate.assert_called_once();self.assertIsNone(first['totals'])
        self.assertEqual(second['plan']['C_provider_requests'], [])

    def test_wrong_R_or_fabricated_transplant_is_rejected(self):
        task = toy_task();provider = RVariableSavedPriceProvider(Mock(), pipeline=p)
        def wrong(task, rows):
            records = mock_delegate(task, rows);records[-1]['mean']['rz_count'] = 7
            return records
        with patch.object(p, 'h4_saved_price_provider', return_value=wrong):
            with self.assertRaisesRegex(ValueError, 'TRANSPLANTED'): provider(task, task['candidates'])
        provider = RVariableSavedPriceProvider(Mock(), pipeline=p)
        def foreign(task, rows):
            records = mock_delegate(task, rows);records[0]['identity']['candidate']['R_outer'] = 8
            return records
        with patch.object(p, 'h4_saved_price_provider', return_value=foreign):
            with self.assertRaisesRegex(ValueError, 'FOREIGN_R'): provider(task, task['candidates'])

    def test_original_science_modules_unchanged(self):
        for filename in ['pf_pr_rpe_pipeline.py', 'h4_df10_price_provider.py', 'h6_df19_price_provider.py']:
            path = 'src/trottertracks/'+filename
            self.assertEqual((ROOT/path).read_bytes(), subprocess.check_output(['git', 'show', '4cb61ef4633467446122954ed6cc43462e0306a9:'+path], cwd=ROOT))
        # Existing window counter is pure; test a toy pattern, not a physical law.
        path = ROOT/'src/trotterlib/rte_connected_cluster_cost_validation.py'
        nodes = ast.parse(path.read_text()).body
        bodies = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in ('_pattern_key', '_parameter_key', '_sequence_form')]
        scope = {'Sequence': Sequence};exec(compile(ast.Module(body=bodies, type_ignores=[]), str(path), 'exec'), scope)
        form = scope['_sequence_form']((0, 2, 0, 0))
        self.assertEqual(form['k2:0,2'], 1.0);self.assertEqual(form['k3:0,2,0'], 1.0)


if __name__ == '__main__': unittest.main(verbosity=2)
