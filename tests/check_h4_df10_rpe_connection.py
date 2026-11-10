"""Artificial scalars only. No molecular input, signal, price or runner is read."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import h4_df10_rpe_connection as api
from trottertracks import c1_stage3_accuracy as old


def point():
    c = dict(system='H4', method='PR', T=2., q=7, prefix=1,
             formula='TOY_S2', R_outer=2, K=2, id='TOY_CANDIDATE')
    b = dict(rank=10, hamiltonian='TOY_H', snapshot='TOY_SNAPSHOT',
             state='TOY_STATE', rank_policy='TOY_DF')
    compiler = dict(id='TOY_COMPILER')
    p = dict(candidate=deepcopy(c), binding=deepcopy(b), compiler=compiler,
             scope=api.SCOPE, classification='DIRECT_COMPILED_FINITE_SAMPLE',
             n=91, source='TOY_SAVED_PRICE')
    return dict(candidate=c, binding=b, compiler=compiler, prices=[p],
        saved_signal=dict(raw=dict(real=.1, imag=.2), corrected=dict(real=.2, imag=.4),
                          reference=dict(real=.3, imag=.5), log_B=math.log(2)),
        saved_allowance=dict(u=1e-4, log_B_margin=1e-6),
        saved_finite_time_shots=dict(cosine=999, sine=777))


def plan(m=0, **changes):
    p=point(); p['candidate']['T']=math.ldexp(2.,m)
    p['candidate']['id']='TOY_CANDIDATE_'+str(m)
    p['prices'][0]['candidate']=deepcopy(p['candidate'])
    r = api.bind_saved_round(p, target_id='TOY_TARGET', m=m, base_time=2.)
    kwargs = dict(policy=dict(beta_total=.4, beta_stat=.2, alpha_X=.002,
                             alpha_Y=.001, law_model='ideal_finite_RTE_law'),
                  corrected_signal_disk=.01, normalization_error_raw=.001,
                  implementation_error_raw=0., law_error_raw=0.,
                  reference_provenance='TOY explicit reference/state disk',
                  iid_provenance='TOY independent trials', error_provenance='TOY_BOUND',
                  df_accounting='own_DF_target_no_DF_energy_shift')
    kwargs.update(changes)
    return api.assess_supplied_round(r, **kwargs)


def prices(plans):
    return [dict(m=p['m'], axis=a, candidate=deepcopy(p['candidate']),
                 binding=deepcopy(p['binding']), compiler=point()['compiler'],
                 scope=api.SCOPE, classification='DIRECT_COMPILED_FINITE_SAMPLE',
                 source='TOY_SAVED', n=91, peak_qubits=9,
                 mean={g: (2. if a=='X' else 3.) for g in old.METRICS})
            for p in plans for a in api.AXES]


class ConnectionTests(unittest.TestCase):
    def test_passive_import(self):
        subprocess.run([sys.executable, '-B', '-c',
            'import sys; from trottertracks import h4_df10_rpe_connection; '
            'assert not any(n in sys.modules for n in ["numpy","scipy","qiskit","ffsim"])'],
            check=True, env=__import__('os').environ | {'PYTHONPATH':str(ROOT/'src')})

    def test_inspection_preserves_missing_raw_and_never_solves(self):
        p = point(); p['saved_signal']['raw'] = None; saved = deepcopy(p)
        with patch('math.exp', side_effect=AssertionError('no normalization calculation')), \
             patch.object(old, '_assess_raw_signal', side_effect=AssertionError('no solve')):
            result = api.inspect_saved_point(p)
        self.assertIsNone(result['saved_signal']['raw'])
        self.assertIsNone(result['N_X']); self.assertIsNone(result['full_rpe_resources'])
        self.assertEqual(p, saved)
        result['candidate']['id']='changed'
        self.assertEqual(p, saved)

    def test_q_pf_and_q_rpe_and_time(self):
        r = api.bind_saved_round(point(), target_id='TOY', m=2, base_time=.5)
        self.assertEqual((r['q_PF'],r['q_RPE']), (7,4))
        self.assertEqual(r['binding'], point()['binding'])
        with self.assertRaises(ValueError):
            api.bind_saved_round(point(), target_id='TOY', m=2, base_time=.8)

    def test_signal_price_candidate_context(self):
        for field,value in [('candidate', {'id':'OTHER'}), ('binding',{'rank':8}),
                            ('scope','SUBCIRCUIT'), ('compiler',{'id':'FOREIGN'})]:
            p=point();p['prices'][0][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                api.inspect_saved_point(p)
        p=point();p['candidate']['method']='B3'
        with self.assertRaises(ValueError): api.inspect_saved_point(p)

    def test_existing_geometry_normalization_once_and_axis_shots(self):
        r=plan(); expected_raw=.5*.01+.001
        for a in api.AXES:
            x=r['axes'][a]
            self.assertAlmostEqual(x['raw_signal_error'],expected_raw)
            self.assertAlmostEqual(x['radius_lower'],.5-expected_raw)
            self.assertAlmostEqual(x['beta_deterministic'],math.asin(expected_raw/.5))
            epsilon=(.5-expected_raw)*math.sin(.2)/math.sqrt(2)
            helper=old._legacy('rpe_resource_accounting')._hoeffding_shots
            self.assertEqual(r['N_'+a],helper(epsilon,r['alpha_'+a]))
        self.assertNotEqual(r['N_X'],r['N_Y'])
        self.assertNotEqual(r['N_X'],point()['saved_finite_time_shots']['cosine'])
        self.assertFalse(r['full_rpe_energy_success'])

    def test_reference_and_unknown_errors_do_not_get_zero(self):
        for changes in [dict(reference_provenance=None),dict(iid_provenance=None),
                        dict(normalization_error_raw=None), dict(law_error_raw=None),
                        dict(corrected_signal_disk=None)]:
            with self.subTest(changes=changes):
                r=plan(**changes)
                self.assertEqual(r['status'],'UNRESOLVED_REFERENCE_OR_ERROR')
                self.assertIsNone(r['N_X'])

    def test_infeasible_radius_and_phase(self):
        for disk in [1.,.5]:
            r=plan(corrected_signal_disk=disk)
            self.assertEqual(r['status'],'UNQUALIFIED_ROUND')
            self.assertIsNone(r['N_X'])

    def test_full_axis_failure_union_and_duplicates(self):
        ps=[plan(0),plan(1)]
        self.assertAlmostEqual(api.failure_budget(ps,alpha_total=.01)['allocated_failure'],.006)
        self.assertFalse(api.failure_budget(ps,alpha_total=.005)['within_budget'])
        with self.assertRaises(ValueError): api.failure_budget([ps[0],ps[0]],alpha_total=.01)

    def test_axis_sum_and_no_classical_n_or_B_multiplier(self):
        ps=[plan(0),plan(1)]; cs=prices(ps)
        r=api.aggregate_supplied_axes(ps,cs,expected_rounds=[0,1],alpha_total=.01,
                                     compiler=point()['compiler'])
        expected=math.fsum(p['N_X']*2+p['N_Y']*3 for p in ps)
        self.assertEqual(r['totals']['rz_count'],expected)
        self.assertEqual(r['peak_qubits'],9)
        self.assertIsNone(r['total_confidence_interval'])
        self.assertFalse(r['full_rpe_energy_success'])

    def test_equal_N_matches_existing_aggregate(self):
        ps=[plan(0),plan(1)]
        for p in ps: p['N_Y']=p['N_X']
        cs=prices(ps)
        new=api.aggregate_supplied_axes(ps,cs,expected_rounds=[0,1],alpha_total=.01,
                                      compiler=point()['compiler'])
        old_plans=[dict(m=p['m'],target_id=p['target_id'],shots=p['N_X']) for p in ps]
        old_costs=[dict(p,target_id='TOY_TARGET',long_round_coverage=True) for p in cs]
        legacy=old.aggregate_totals(dict(same_accuracy=True),old_plans,old_costs,
                                   cost_scope=api.SCOPE,expected_scope=api.SCOPE)
        self.assertEqual(new['totals'],legacy['totals'])

    def test_missing_round_axis_metric_remains_unavailable(self):
        ps=[plan(0),plan(1)]; cs=prices(ps)
        for plans_,costs_,status in [(ps[:1],cs,'INCOMPLETE_ROUNDS'),
                                     (ps,cs[:-1],'INCOMPLETE_PRICE_COVERAGE')]:
            r=api.aggregate_supplied_axes(plans_,costs_,expected_rounds=[0,1],
                                         alpha_total=.01,compiler=point()['compiler'])
            self.assertEqual(r['status'],status);self.assertIsNone(r['totals'])
        cs[0]['mean']['cx_count']=None
        r=api.aggregate_supplied_axes(ps,cs,expected_rounds=[0,1],alpha_total=.01,
                                     compiler=point()['compiler'])
        self.assertEqual(r['status'],'MISSING_PRICE_METRIC')

    def test_frozen_model_classification_and_context(self):
        ps=[plan()];cs=prices(ps);cs[0]['classification']='MODEL_PREDICTION'
        kwargs=dict(expected_rounds=[0],alpha_total=.01,compiler=point()['compiler'])
        self.assertIsNone(api.aggregate_supplied_axes(ps,cs,**kwargs)['totals'])
        cs[0]['model']=dict(source='TOY_FROZEN_MODEL',coverage='EXTRAPOLATION',
            q_variable='q_PF',q=7,binding=deepcopy(ps[0]['binding']),compiler=point()['compiler'])
        r=api.aggregate_supplied_axes(ps,cs,**kwargs)
        self.assertEqual(r['price_origins'][0]['classification'],'MODEL_PREDICTION')
        cs[0]['model']['q_variable']='q_RPE'
        with self.assertRaises(ValueError): api.aggregate_supplied_axes(ps,cs,**kwargs)
        cs[0]['model']['q']=1;cs[0]['model']['binding']={'rank':8}
        with self.assertRaises(ValueError): api.aggregate_supplied_axes(ps,cs,**kwargs)

    def test_invalid_phase_failure_and_df_conventions(self):
        for changes in [dict(policy=dict(beta_total=2.,beta_stat=.1,alpha_X=.01,alpha_Y=.01)),
                        dict(df_accounting='DOUBLE_COUNT_DF'),dict(evidence_kind='rigorous_bound')]:
            with self.assertRaises(ValueError):plan(**changes)
        ps=[plan()]
        r=api.aggregate_supplied_axes(ps,prices(ps),expected_rounds=[0],alpha_total=.001,
                                     compiler=point()['compiler'])
        self.assertEqual(r['status'],'FAILURE_BUDGET_EXCEEDED')

    def test_common_base_time_and_no_scientific_calls(self):
        # Warm only library imports; subsequent operations must be scalar-only.
        old._legacy('rpe_resource_accounting')
        with patch('numpy.linalg.eigh', side_effect=AssertionError('eigensolve')), \
             patch('scipy.linalg.expm', side_effect=AssertionError('signal')), \
             patch('qiskit.QuantumCircuit', side_effect=AssertionError('circuit')):
            ps=[plan(0),plan(1)]
            ps[1]['base_time']=1.
            with self.assertRaises(ValueError):
                api.aggregate_supplied_axes(ps,prices(ps),expected_rounds=[0,1],
                    alpha_total=.01,compiler=point()['compiler'])

    def test_existing_decoder_on_artificial_dyadic_phases(self):
        decoder=old._legacy('rpe_four_round_phase_validation').reconstruct_rpe_phase
        r=decoder([.2,.4,.8],[1,2,4])
        self.assertAlmostEqual(r['final_phase_estimate'],.2)


if __name__=='__main__': unittest.main()
