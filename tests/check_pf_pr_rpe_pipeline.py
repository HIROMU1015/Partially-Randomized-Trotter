"""Artificial integration and read-only saved-metadata checks; no real N/C/G."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_pr_rpe_pipeline as p
import plan_pf_pr_rpe_cost as cli


def fake_shot(task,row,*,passed=True):
    lane=dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],candidate=deepcopy(row['candidate']),
        binding=deepcopy(task['binding']),base_time=task['configuration']['base_time'],target_id=task['task_id'],
        status='CONDITIONAL_ROUND_PASS' if passed else 'UNQUALIFIED_ROUND',
        N_X=(3 if p.arm(row['candidate'])=='PF' else 5)+row['m'] if passed else None,
        N_Y=(3 if p.arm(row['candidate'])=='PF' else 5)+row['m'] if passed else None,
        alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'])
    return dict(identity=p.identity(task,row),plan=lane,actual=dict(N_X=None,N_Y=None),source=dict(path='MOCK_ONLY'))


def fixture():
    config=dict(base_time=.1,beta_total=.4,beta_stat=.2,alpha_axis=.05/4,
        alpha_total_per_method=.05,energy_precision_ha=.1,energy_prior_ha=[-2.,-1.],decoder='MOCK_DECODER',
        horizon=dict(maximum_round_index_M=1,round_count_M_plus_one=2))
    binding=dict(system='TOY',hamiltonian='TOY_H',snapshot='TOY_SNAPSHOT',state='TOY_STATE',rank_policy='TOY_FIXED')
    rows=[]
    for m in (0,1):
        for alg in p.ARMS:
            c=dict(system='TOY',method='FULL' if alg=='PF' else 'PR',formula='2nd',prefix=1,T=.1*(1<<m),q=3*(1<<m),R_outer=None if alg=='PF' else 4,K=None if alg=='PF' else 2)
            c['delta']=c['T']/c['q'];c['id']=p.candidate_id(c)
            rows.append(dict(m=m,q_RPE=1<<m,q_PF=c['q'],candidate=c))
    task=p.task_from_saved(config,binding,rows,task_id='MOCK_TASK',source=dict(path='MOCK_ONLY'),contract=dict(preparation='MOCK_FIXED'))
    shots=[fake_shot(task,r) for r in task['candidates']];prices=[]
    for row in task['candidates']:
        for a in p.AXES:
            prices.append(dict(**p.measurement(row,a),identity=p.identity(task,row),axis=a,classification='DIRECT_COMPILED_DETERMINISTIC',cohort='MOCK',n_paths=99,
                mean=dict(rz_count=10. if p.arm(row['candidate'])=='PF' else 6.),source=dict(path='MOCK_ONLY'),
                expectation='DETERMINISTIC',model=None,uncertainty=None,normalization_is_price_multiplier=False,peak_qubits=3))
    select={(r['identity']['candidate']['id'],r['axis']):r['cohort'] for r in prices}
    return task,shots,prices,select


class IntegrationTests(unittest.TestCase):
    def test_plan_is_non_numerical_and_keeps_q_variables(self):
        task,shots,prices,selection=fixture()
        with patch.object(p.accounting,'aggregate_supplied_axes',side_effect=AssertionError('No sums')):
            out=p.plan(task,shots,prices)
        self.assertIsNone(out['totals']);self.assertEqual(out['science_calls'],0)
        missing=deepcopy(prices);missing[0]['mean']['rz_count']=None
        self.assertIn(missing[0]['identity']['candidate']['id'],p.plan(task,shots,missing)['unqualified_expected_C'])
        self.assertEqual(out['missing_N'],[])
        self.assertEqual(task['candidates'][0]['q_RPE'],1)
        self.assertEqual(task['candidates'][0]['q_PF'],3)

    def test_axis_costs_delegate_existing_arithmetic_without_extra_multipliers(self):
        task,shots,prices,selection=fixture()
        with patch.object(p.accounting,'aggregate_supplied_axes',wraps=p.accounting.aggregate_supplied_axes) as aggregate:
            out=p.evaluate(task,shots,prices,cohort_selection=selection)
        self.assertEqual(aggregate.call_count,2)
        self.assertEqual(out['totals'],dict(PF=dict(rz_count=140.),PR=dict(rz_count=132.)))
        self.assertFalse(out['normalization_is_price_multiplier'])
        self.assertEqual(out['comparison']['verdict'],'POINT_ESTIMATES_ONLY')
        self.assertIsNone(out['uncertainty']['covariance'])
        self.assertFalse(out['full_RPE_qualified'])

    def test_foreign_science_profile_recipe_and_double_normalization_rejected(self):
        task,shots,prices,selection=fixture()
        for change in [lambda x:x['identity']['binding'].update(state='FOREIGN'),
                lambda x:x['identity']['profile'].update(control='OTHER'),
                lambda x:x['identity']['profile'].update(wrapper='subcircuit'),
                lambda x:x['identity'].update(q_PF=1),
                lambda x:x['identity']['structure']['signed_weights'].__setitem__(0,-1.),
                lambda x:x.update(observable='-Im(z_raw)'),
                lambda x:x.update(units={'rz_count':'circuit_layers'}),
                lambda x:x.update(normalization_is_price_multiplier=True)]:
            bad=deepcopy(prices);change(bad[0])
            with self.assertRaises(ValueError):p.plan(task,shots,bad)

    def test_saved_rerun_duplicate_and_missing_round_never_sum(self):
        task,shots,prices,selection=fixture()
        with self.assertRaises(ValueError):p.plan(task,shots+[deepcopy(shots[0])],prices)
        out=p.evaluate(task,shots[:-1],prices,cohort_selection=selection)
        self.assertEqual(out['status'],'INCOMPLETE_OR_UNQUALIFIED_N');self.assertIsNone(out['totals'])
        bad=deepcopy(shots);bad[0]['plan']['alpha_X']=.025
        with self.assertRaises(ValueError):p.plan(task,bad,prices)

    def test_cohort_mean_and_model_extrapolation_have_distinct_qualification(self):
        task,shots,prices,selection=fixture()
        for price in prices:
            price.update(classification='DIRECT_COMPILED_FINITE_SAMPLE',expectation='FINITE_COHORT_ONLY')
        self.assertEqual(p.evaluate(task,shots,prices,cohort_selection=selection)['status'],'UNQUALIFIED_PRICE_EXPECTATION')
        out=p.evaluate(task,shots,prices,cohort_selection=selection,price_policy='finite_cohort_scenario')
        self.assertIsNotNone(out['totals']);self.assertFalse(out['actual_success_guaranteed'])
        never=Mock(side_effect=AssertionError('Saved cohort must be reused'))
        p.run_missing(task,shots,prices,c_provider=never,price_policy='finite_cohort_scenario');never.assert_not_called()
        prices[0].update(classification='MODEL_PREDICTION',expectation='EXPLICIT_CONDITIONAL_MODEL',
            model=dict(source=dict(path='MOCK'),coverage='mock',qualified_for_candidate=False))
        self.assertEqual(p.evaluate(task,shots,prices,cohort_selection=selection,price_policy='finite_cohort_scenario')['status'],'UNQUALIFIED_MODEL_DOMAIN')
        prices[0]['classification']='UNVALIDATED_EXTRAPOLATION'
        self.assertEqual(p.evaluate(task,shots,prices,cohort_selection=selection,price_policy='finite_cohort_scenario')['status'],'UNVALIDATED_PRICE_EXTRAPOLATION')

    def test_joint_shared_calibration_and_paired_axes_remain_correlated(self):
        task,shots,prices,selection=fixture();base=p.evaluate(task,shots,prices,cohort_selection=selection)
        n=len(base['price_order']);e=dict(labels=base['price_order'],basis='mean_price_estimator',source=dict(path='MOCK'),
            cross_round_method_complete=True,shared_calibration_included=True,covariance=[[1.]*n for _ in range(n)])
        out=p.evaluate(task,shots,prices,cohort_selection=selection,joint=e)
        self.assertEqual(out['uncertainty']['covariance'],[[196.,308.,112.],[308.,484.,176.],[112.,176.,64.]])
        self.assertIsNone(out['uncertainty']['total_confidence_interval'])
        self.assertEqual(out['comparison']['verdict'],'POINT_ESTIMATES_ONLY')
        e['shared_calibration_included']=False
        self.assertIsNone(p.evaluate(task,shots,prices,cohort_selection=selection,joint=e)['uncertainty']['covariance'])
        e['shared_calibration_included']=True;e['basis']='path_sample_covariance'
        with self.assertRaises(ValueError):p.evaluate(task,shots,prices,cohort_selection=selection,joint=e)

    def test_covariance_psd_and_complete_simultaneous_bounds(self):
        task,shots,prices,selection=fixture();base=p.evaluate(task,shots,prices,cohort_selection=selection)
        selected=sorted(prices,key=lambda x:(p.arm(x['identity']['candidate']),x['identity']['m'],x['axis']))
        e=dict(labels=base['price_order'],basis='mean_price_estimator',source=dict(path='MOCK'),
            covariance=None,simultaneous_price_bounds=[[x['mean']['rz_count']-.1,x['mean']['rz_count']+.1] for x in selected],bounds_cover_stat_model_and_law=True)
        out=p.evaluate(task,shots,prices,cohort_selection=selection,joint=e)
        self.assertEqual(out['comparison']['verdict'],'PR_LOWER_CONDITIONAL_INTERROGATION_COST')
        self.assertIsNone(out['uncertainty']['covariance']);self.assertFalse(out['actual_success_guaranteed'])
        e.update(covariance=[[-1. if i==j else 0. for j in range(8)] for i in range(8)],cross_round_method_complete=True,shared_calibration_included=True)
        with self.assertRaises(ValueError):p.evaluate(task,shots,prices,cohort_selection=selection,joint=e)

    def test_missing_batch_uses_one_call_and_reuses_cached_N(self):
        task,shots,prices,selection=fixture()
        n=Mock(return_value=shots[2:]);c=Mock(return_value=prices)
        out=p.run_missing(task,shots[:2],[],n_provider=n,c_provider=c)
        n.assert_called_once();c.assert_called_once()
        self.assertEqual(len(n.call_args.args[1]),2)
        self.assertEqual(out['plan']['missing_N'],[]);self.assertIsNone(out['totals'])
        self.assertFalse(out['automatic_retry'])
        never=Mock(side_effect=AssertionError('Cached N/C must be reused'))
        p.run_missing(task,shots,prices,n_provider=never,c_provider=never);never.assert_not_called()

    def test_batch_fail_stops_before_price_without_retry_or_scope_expansion(self):
        task,shots,prices,selection=fixture();bad=deepcopy(shots[2:]);bad[0]['plan'].update(status='UNQUALIFIED_ROUND',N_X=None,N_Y=None)
        n=Mock(return_value=bad);c=Mock(side_effect=AssertionError('No C after N fail'))
        out=p.run_missing(task,shots[:2],[],n_provider=n,c_provider=c)
        self.assertEqual(out['status'],'STOP_UNQUALIFIED_N');n.assert_called_once();c.assert_not_called()
        n.return_value=shots
        with self.assertRaises(ValueError):p.run_missing(task,shots[:2],[],n_provider=n)

    def test_one_explicit_entry_reaches_costs_using_only_mock_missing_providers(self):
        task,shots,prices,selection=fixture()
        n=Mock(return_value=shots[2:]);c=Mock(return_value=prices)
        out=p.run_fixed_evaluation(task,shots[:2],[],n_provider=n,c_provider=c,cohort_selection=selection)
        self.assertEqual(out['totals']['PF']['rz_count'],140.)
        self.assertTrue(out['conditional_on_fixed_N']);n.assert_called_once();c.assert_called_once()
        self.assertFalse(out['actual_success_guaranteed'])

    def test_static_real_carriers_only_copy_saved_N_C_and_qualifications(self):
        store=p.SavedArtifacts(ROOT)
        with patch.object(p,'evaluate',side_effect=AssertionError('No real G')),patch.object(p,'run_missing',side_effect=AssertionError('No real science')):
            h6,n6,c6,models=cli.collect(store,'H6');h4,n4,c4,models4=cli.collect(store,'H4')
            out6=p.plan(h6,n6,c6);out4=p.plan(h4,n4,c4)
        self.assertEqual(len(n6),16);self.assertEqual(len(n4),2)
        self.assertEqual(len(out6['missing_N']),10);self.assertEqual(len(out4['missing_N']),24)
        self.assertEqual(sum(x in h4['saved_signal_sources'] for x in out4['missing_N']),2)
        self.assertTrue(any(x['identity']['m']==4 and x['classification']=='DIRECT_COMPILED_DETERMINISTIC' for x in c4))
        self.assertTrue(all(x['model']['qualified_for_candidate'] is False for x in c6+c4 if x['model']))
        self.assertIsNone(out6['totals']);self.assertIsNone(out4['totals'])
        self.assertTrue(any(x['expectation']=='FINITE_COHORT_ONLY' for x in c6))

    def test_h6_future_batch_mock_shares_context_references_and_cache(self):
        from trottertracks.h6_rpe_batch_provider import H6ActionBatch
        import run_h6_df19_rpe_d2 as io
        from trottertracks import h6_df19_rpe_d2 as backend
        task,shots,prices,models=cli.collect(p.SavedArtifacts(ROOT),'H6')
        rows=[r for r in task['candidates'] if r['m']>=8]
        reuse=json.loads((ROOT/'docs/research/h6_df19_rpe_m7_preparation_20261110/PLAN.json').read_bytes())
        context=SimpleNamespace(half_steps={});factory=Mock(return_value=(context,dict(solver_energy=0.,solver_residual=0.)))
        def analyze(record,row,plan,**kwargs):
            n=fake_shot(task,row)
            return dict(candidate=row['candidate'],m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],ideal_numerical_pilot=n['plan'],actual_implementation=n['actual'])
        with patch.object(backend,'reference_pair',return_value=dict(reference=dict(real=1.,imag=0.),signal_difference=0.)) as refs, \
             patch.object(backend,'half_step_keys',return_value=[(0,1.)]), \
             patch.object(backend,'half_step',side_effect=lambda ctx,k:(k,'MOCK_CACHE')) as steps, \
             patch.object(backend,'evolve',return_value={}) as evolve,patch.object(backend,'analyze',side_effect=analyze), \
             patch.object(io,'worker_resources',return_value=dict(half_step_workers=10)):
            provider=H6ActionBatch(factory,reuse,dict(path='MOCK_ONLY'));result=provider(task,rows)
            evolve.side_effect=ValueError('MOCK_NUMERICAL_GATE')
            failed=provider(task,rows)
            self.assertTrue(all(x['plan']['N_X'] is None for x in failed))
            self.assertEqual(provider.last_run['status'],'STOP_UNQUALIFIED_N')
            evolve.side_effect=None
        self.assertEqual(factory.call_count,2);self.assertEqual(refs.call_count,10);self.assertEqual(steps.call_count,2)
        self.assertEqual(evolve.call_count,20);self.assertEqual(len(result),10)
        self.assertEqual(provider.last_run['reference_state_actions'],10)
        self.assertEqual(provider.last_run['internal_PR_exact_tail_controls'],5)
        self.assertEqual(provider.last_run['status'],'STOP_UNQUALIFIED_N')


if __name__=='__main__':unittest.main()
