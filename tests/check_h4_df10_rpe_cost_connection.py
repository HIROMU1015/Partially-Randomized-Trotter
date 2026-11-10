"""Artificial scalars only; never load a molecular file or evaluate real N*C."""
from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_df10_rpe_cost_connection as api
from trottertracks import h4_df10_rpe_connection as old
from check_h4_df10_rpe_connection import point, prices as old_prices


def fixture(ms=(3,), *, nx=2, ny=3):
    ps = []
    for m in ms:
        toy = point(); c = toy['candidate']; c['T'] = 2.*(1 << m)
        c['id'] = 'TOY_COST_'+str(m)
        ps.append(dict(m=m, target_id='TOY_TARGET', base_time=2.,
            candidate=c, candidate_id=c['id'], binding=toy['binding'],
            q_RPE=1 << m, q_PF=c['q'], N_X=nx, N_Y=ny,
            alpha_X=.002, alpha_Y=.001, status='CONDITIONAL_ROUND_PASS',
            law_context={'id': 'TOY_DECLARED_LAW_WITH_UNQUALIFIED_SAMPLER_GAP'},
            axes={a: {'law_model': 'ideal_finite_RTE_law'} for a in api.AXIS_MAP}))
    cs = old_prices(ps)
    for c in cs:
        a = c['axis']
        c.update(law_context=deepcopy(ps[0]['law_context']),
            measurement_axis=api.AXIS_MAP[a], observable=api.OBSERVABLES[a],
            outcome_convention='P0_minus_P1', units=deepcopy(api.UNITS),
            expectation_basis='explicit_conditional_price_proxy',
            price_assumptions=['TOY finite sample mean as conditional proxy'],
            cohort='TOY_EXPLORATION')
    return ps, cs


def run(ps, cs, **kw):
    return api.account_subset(ps, cs, selected_rounds=[p['m'] for p in ps],
        expected_rounds=range(5), compiler=point()['compiler'], alpha_total=.05, **kw)


class CostTests(unittest.TestCase):
    def test_passive_import(self):
        subprocess.run([sys.executable, '-B', '-c',
            'import sys; from trottertracks import h4_df10_rpe_cost_connection; '
            'assert not any(n in sys.modules for n in ("numpy","scipy","qiskit","ffsim"))'],
            env=os.environ | {'PYTHONPATH': str(ROOT/'src')}, check=True)

    def test_partial_actual_index_and_unequal_axis_N(self):
        ps, cs = fixture(); r = run(ps, cs)
        self.assertEqual(r['subset_resources'], {'rz_count': 13.})
        self.assertEqual(r['covered_rounds'], [3]); self.assertEqual(r['missing_rounds'], [0,1,2,4])
        self.assertIsNone(r['full_rpe_resources']); self.assertFalse(r['actual_success_qualified'])

    def test_method_N_is_not_reused_across_cases(self):
        ps, cs = fixture(nx=2, ny=2); a = run(ps, cs)
        ps, cs = fixture(nx=5, ny=5); b = run(ps, cs)
        self.assertEqual(a['subset_resources']['rz_count'], 10.)
        self.assertEqual(b['subset_resources']['rz_count'], 25.)

    def test_classical_n_and_B_never_multiply_price(self):
        ps, cs = fixture(); first = run(ps, cs)
        ps[0]['normalization'] = {'B': 19., 'log_B': 3.}
        for c in cs: c['n'] = 100003
        self.assertEqual(run(ps, cs), first)

    def test_full_horizon_delegates_to_existing_axis_accountant(self):
        ps, cs = fixture((0,1), nx=2, ny=3)
        with patch.object(old, 'aggregate_supplied_axes', wraps=old.aggregate_supplied_axes) as legacy:
            r = api.account_subset(ps, cs, selected_rounds=[0,1], expected_rounds=[0,1],
                compiler=point()['compiler'], alpha_total=.05, metrics=tuple(api.UNITS))
        self.assertEqual(legacy.call_count, 1)
        self.assertEqual(r['full_rpe_resources'], {g: 26. for g in api.UNITS})
        self.assertIsNone(r['total_covariance']); self.assertIsNone(r['total_confidence_interval'])

    def test_equal_N_regression_against_legacy_scalar_accountant(self):
        ps, cs = fixture((0,1), nx=2, ny=2)
        legacy = old.existing.aggregate_totals({'same_accuracy': True},
            [dict(m=p['m'], target_id=p['target_id'], shots=2) for p in ps],
            [dict(c, target_id='TOY_TARGET', long_round_coverage=True) for c in cs],
            cost_scope=old.SCOPE, expected_scope=old.SCOPE)
        self.assertEqual(run(ps, cs, metrics=tuple(api.UNITS))['subset_resources'], legacy['totals'])

    def test_width_is_peak_or_unknown(self):
        ps, cs = fixture(); cs[1]['peak_qubits'] = 11
        self.assertEqual(run(ps, cs)['peak_qubits'], 11)
        cs[1]['peak_qubits'] = None
        self.assertIsNone(run(ps, cs)['peak_qubits'])
        with self.assertRaises(ValueError): run(ps, cs, metrics=('peak_qubits',))

    def test_signed_axis_and_unit_mismatch_rejected(self):
        for field, value in [('measurement_axis','sine'), ('observable','-Re(z_raw)'),
                             ('outcome_convention','P1_minus_P0')]:
            ps, cs = fixture(); cs[0][field] = value
            with self.assertRaises(ValueError): run(ps, cs)
        ps, cs = fixture(); cs[0]['units']['rz_count'] = 'seconds'
        with self.assertRaises(ValueError): run(ps, cs)

    def test_all_scientific_and_compiler_bindings_rejected_when_foreign(self):
        for section, key, value in [('candidate','prefix',99), ('binding','hamiltonian','FOREIGN'),
                ('binding','state','FOREIGN'), ('binding','snapshot','FOREIGN'),
                ('law_context','id','FOREIGN'), ('compiler','id','FOREIGN')]:
            ps, cs = fixture(); cs[0][section][key] = value
            with self.assertRaises(ValueError): run(ps, cs)
        ps, cs = fixture((0,1)); ps[1]['binding']['hamiltonian'] = 'FOREIGN'
        for c in cs:
            if c['m'] == 1: c['binding']['hamiltonian'] = 'FOREIGN'
        with self.assertRaises(ValueError): run(ps, cs)
        ps, cs = fixture(); cs[0]['scope'] = 'kernel_only'
        with self.assertRaises(ValueError): run(ps, cs)

    def test_missing_price_or_N_is_unavailable_never_zero(self):
        ps, cs = fixture(); cs[0]['mean']['rz_count'] = None
        self.assertIsNone(run(ps, cs)['subset_resources'])
        ps, cs = fixture(); ps[0]['N_X'] = None
        self.assertEqual(run(ps, cs)['status'], 'UNQUALIFIED_SHOTS')
        ps, cs = fixture(); self.assertEqual(run(ps, cs[:-1])['status'], 'INCOMPLETE_PRICE_COVERAGE')
        ps, cs = fixture(); ps[0]['binding']['state'] = None
        self.assertEqual(run(ps, cs)['status'], 'MISSING_SCIENTIFIC_BINDING')

    def test_cohorts_must_remain_separate_cases(self):
        ps, cs = fixture()
        with self.assertRaises(ValueError): run(ps, cs+deepcopy(cs))
        ps, cs = fixture(); cs[1]['cohort'] = 'TOY_CONFIRMATION'
        with self.assertRaises(ValueError): run(ps, cs)
        ps, cs = fixture(); cs[0]['price_assumptions'] = []
        self.assertEqual(run(ps, cs)['status'], 'UNQUALIFIED_PRICE_INTERPRETATION')

    def test_model_provenance_and_q_variables(self):
        ps, cs = fixture(); cs[0]['classification'] = 'MODEL_PREDICTION'
        self.assertEqual(run(ps, cs)['status'], 'UNQUALIFIED_MODEL_CONTEXT')
        cs[0]['model'] = dict(source='TOY_FROZEN', coverage='EXTRAPOLATION',
            binding=deepcopy(ps[0]['binding']), compiler=point()['compiler'], q_variable='q_PF', q=7)
        self.assertEqual(run(ps, cs)['subset_resources']['rz_count'], 13.)
        cs[0]['model']['q_variable'] = 'q_RPE'
        with self.assertRaises(ValueError): run(ps, cs)

    def test_full_alpha_and_round_checks(self):
        ps, cs = fixture(); ps[0]['alpha_X'] = .05
        self.assertEqual(run(ps, cs)['status'], 'FAILURE_BUDGET_EXCEEDED')
        ps, cs = fixture(); ps[0]['q_RPE'] = 16
        with self.assertRaises(ValueError): run(ps, cs)
        ps, cs = fixture(); ps[0]['candidate']['T'] = 9.
        with self.assertRaises(ValueError): run(ps, cs)

    def test_preparation_never_calls_N_or_resource_arithmetic(self):
        ps, cs = fixture((0,)); p = point(); p['candidate'] = deepcopy(ps[0]['candidate'])
        p['prices'] = [dict(candidate=deepcopy(p['candidate']), binding=deepcopy(p['binding']),
            compiler=p['compiler'], scope=old.SCOPE, classification='DIRECT_COMPILED_FINITE_SAMPLE',
            source='TOY', metrics_source='TOY', cohort='TOY_EXPLORATION', n=91, peak_qubits=None,
            mean_metrics={api.AXIS_MAP[c['axis']]: deepcopy(c['mean']) for c in cs})]
        result = dict(rows=[dict(candidate_id=ps[0]['candidate_id'], ideal_numerical_pilot=ps[0])],
            configuration={'horizon': {'round_count_M_plus_one': 5}})
        with patch.object(api, 'account_subset', side_effect=AssertionError('N*C')), \
             patch.object(old, 'assess_supplied_round', side_effect=AssertionError('N')), \
             patch('math.fsum', side_effect=AssertionError('sum')), \
             patch('math.exp', side_effect=AssertionError('B')):
            cases = api.prepare_saved_cases(result, {'points': [p]})
        self.assertEqual(cases[0]['input_status'], 'INPUTS_READY_FOR_CONDITIONAL_PROXY_ACCOUNTING')
        self.assertIsNone(cases[0]['subset_resources'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
