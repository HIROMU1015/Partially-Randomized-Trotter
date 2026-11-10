"""B-environment integration: saved C fixtures and mock N only, no real G."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pf_pr_rpe_pipeline as p
from trottertracks import h4_df10_price_provider as api
from check_pf_pr_rpe_pipeline import fake_shot, fixture
import plan_pf_pr_rpe_cost as cli


def saved_fixture():
    task,_,_,_=cli.collect(p.SavedArtifacts(ROOT),'H4')
    task['candidates']=[r for r in task['candidates'] if
        (p.arm(r['candidate']),r['q_PF']) in [('PF',32),('PR',16),('PF',64),('PR',8)]]
    shots=[fake_shot(task,row) for row in task['candidates']]
    selections={r['candidate']['id']:dict(price_unit='cohort_mean',
        cohort=api.mq.COHORT if r['q_PF']==16 else 'CONFIRMATION')
        for r in task['candidates'] if p.arm(r['candidate'])=='PR'}
    return task,shots,selections


class H4PipelineTests(unittest.TestCase):
    def test_A_component_and_dependencies_are_exact_and_B_import_is_local(self):
        for name in ['h4_df10_price_provider.py','h4_df10_price_v1.py','h4_df10_price_v2.py',
                     'h4_df10_price_v2_independent.py','track_d_native_resources.py']:
            path='src/trottertracks/'+name
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','2a521b156:'+path],cwd=ROOT))
        self.assertEqual(Path(p.__file__).resolve(),ROOT/'src/trottertracks/pf_pr_rpe_pipeline.py')
        self.assertEqual(Path(p.accounting.__file__).resolve(),ROOT/'src/trottertracks/h4_df10_rpe_connection.py')
        provider=p.h4_saved_price_provider(p.SavedArtifacts(ROOT).read)
        self.assertIs(provider.pipeline,p)
        self.assertEqual(provider._b,{'PF':606.75,'PR':176.88888888888894})

    def test_run_missing_uses_saved_direct_once_and_keeps_nulls_unacquired(self):
        task,shots,selections=saved_fixture()
        provider=p.h4_saved_price_provider(p.SavedArtifacts(ROOT).read,selections=selections)
        callback=Mock(wraps=provider);no_N=Mock(side_effect=AssertionError('Mock N is already supplied'))
        with (patch.dict(api.mq._scope,{'predict':Mock(side_effect=AssertionError('No new price'))}),
              patch.object(api.H4PriceProvider,'predict_path',side_effect=AssertionError('No numerical entry')),
              patch.object(p,'evaluate',side_effect=AssertionError('No real G')),
              patch.object(p.accounting,'aggregate_supplied_axes',side_effect=AssertionError('No sums'))):
            out=p.run_missing(task,shots,[],n_provider=no_N,c_provider=callback)
            callback.assert_called_once();no_N.assert_not_called()
            second=p.run_missing(task,shots,out['prices'],c_provider=callback)
            callback.assert_called_once()
        self.assertEqual(len(second['prices']),8);self.assertIsNone(out['totals'])
        unknown=next(r for r in task['candidates'] if r['q_PF']==64)
        cid=unknown['candidate']['id']
        row=next(r for r in out['plan']['rows'] if r['candidate_id']==cid)
        self.assertEqual(row['C_axes'],[]);self.assertEqual(row['C_record_axes'],['X','Y'])
        self.assertIn(cid,out['plan']['missing_C']);self.assertIn(cid,out['plan']['blocked_C'])
        self.assertEqual(out['plan']['C_provider_requests'],[])
        self.assertTrue(all(c['reason']=='NO_SAVED_PRICE_FOR_SELECTOR' for c in row['price_contexts']))
        finite=[v for v in out['prices'] if v['identity']['candidate']['method']=='PR']
        self.assertTrue(all(v['expectation']=='FINITE_COHORT_ONLY' for v in finite))
        self.assertTrue(all(v['classification']=='DIRECT_COMPILED_FINITE_SAMPLE' for v in finite))
        self.assertTrue(all(v['model'] is None for v in finite))
        self.assertTrue(all(v['provider_quote']['full_RPE_price_adopted'] is False for v in finite))

    def test_saved_Mq_policy_and_shared_uncertainty_reach_B_without_promotion(self):
        task,shots,selections=saved_fixture()
        for row in task['candidates']:
            selections.setdefault(row['candidate']['id'],{})['kind']='model_prediction'
        provider=p.h4_saved_price_provider(p.SavedArtifacts(ROOT).read,selections=selections)
        prices=provider(task,task['candidates'])
        inventory=p.plan(task,shots,prices,price_policy='finite_cohort_scenario')
        for row in task['candidates']:
            context=next(v for v in inventory['rows'] if v['candidate_id']==row['candidate']['id'])
            qualified=(p.arm(row['candidate']),row['q_PF']) in [('PF',32),('PR',16)]
            self.assertEqual(bool(context['eligible_C_cohorts']),qualified)
        model=next(v for v in prices if v['identity']['candidate']['q']==16)
        self.assertEqual(model['classification'],'MODEL_PREDICTION')
        self.assertEqual(model['expectation'],'FINITE_COHORT_ONLY')
        self.assertEqual(model['model']['q_variable'],'q_PF')
        self.assertTrue(model['model']['qualified_for_candidate'])
        for value in prices:
            u=value['uncertainty']
            for field in ['coefficient_uncertainty','long_q_systematic','all_law_expectation','joint_covariance','total_CI']:
                self.assertIsNone(u[field])
            self.assertFalse(u['cross_round_independence_assumed'])
            self.assertFalse(u['observed_max_is_bound']);self.assertFalse(u['alpha_is_price_confidence'])
        paired=[v['uncertainty']['shared_calibration_group'] for v in prices if v['identity']['candidate']['q']==16]
        self.assertEqual(paired[0],paired[1]);self.assertIsNotNone(paired[0])
        self.assertEqual(model['provider_quote']['model_provenance']['calibration']['commit'],api.S1)
        self.assertEqual(model['uncertainty']['saved_validation'][0]['record']['n_paths'],33)
        no_retry=Mock(side_effect=AssertionError('Qualification needs a decision, not another lookup'))
        p.run_missing(task,shots,prices,c_provider=no_retry,price_policy='finite_cohort_scenario')
        no_retry.assert_not_called()

    def test_numeric_extrapolation_and_outside_domain_stay_blocked(self):
        task,shots,selections=saved_fixture();task['candidates']=[r for r in task['candidates'] if r['q_PF']==64]
        shots=[fake_shot(task,r) for r in task['candidates']]
        provider=p.h4_saved_price_provider(p.SavedArtifacts(ROOT).read);row=task['candidates'][0]
        prices=[]
        for axis in p.AXES:
            request=dict(binding=task['binding'],profile=task['profile'],contract=task['contract'],
                candidate=row['candidate'],q_RPE=row['q_RPE'],axis=axis,kind='model_prediction')
            # A fabricated scalar tests metadata; it is not a fresh H4 prediction.
            quote=provider._response(request,'PF',kind='model_prediction',value=7.,record={'source':{'fixture':'ARTIFICIAL_ONLY'}})
            prices.append(provider.pipeline_record(task,row,axis,quote))
        never=Mock(side_effect=AssertionError('No extrapolation retry'))
        out=p.run_missing(task,shots,prices,c_provider=never);never.assert_not_called()
        self.assertTrue(all(v['classification']=='UNVALIDATED_EXTRAPOLATION' for v in prices))
        self.assertEqual(out['plan']['rows'][0]['C_axes'],['X','Y'])
        self.assertEqual(out['plan']['rows'][0]['eligible_C_cohorts'],[])
        foreign=deepcopy(task);foreign['profile']['wrapper']='FOREIGN'
        unavailable=provider(foreign,foreign['candidates'])
        inventory=p.plan(foreign,[fake_shot(foreign,row)],unavailable)
        self.assertEqual(inventory['rows'][0]['C_axes'],[])
        self.assertTrue(all(c['validation']=='OUTSIDE_DOMAIN' for c in inventory['rows'][0]['price_contexts']))

    def test_null_values_block_evaluate_before_accounting_and_do_not_retry(self):
        task,shots,prices,selection=fixture()
        prices[0]['mean']['rz_count']=None
        with patch.object(p.accounting,'aggregate_supplied_axes',side_effect=AssertionError('No null arithmetic')):
            result=p.evaluate(task,shots,prices,cohort_selection=selection)
        self.assertEqual(result['status'],'MISSING_SELECTED_C_VALUE');self.assertIsNone(result['totals'])
        callback=Mock(side_effect=AssertionError('Null response was already observed'))
        out=p.run_missing(task,shots,prices,c_provider=callback);callback.assert_not_called()
        self.assertIn(prices[0]['identity']['candidate']['id'],out['plan']['missing_C'])

    def test_partial_axis_batch_reuses_overlap_and_replaces_only_missing_value(self):
        task,shots,prices,_=fixture()
        saved=[deepcopy(prices[0])]+deepcopy(prices[2:]);saved[0]['mean']['rz_count']=None
        callback=Mock(return_value=deepcopy(prices[:2]))
        out=p.run_missing(task,shots,saved,c_provider=callback)
        callback.assert_called_once();self.assertEqual(len(callback.call_args.args[1]),1)
        self.assertEqual(len(out['prices']),len(prices));self.assertEqual(out['plan']['missing_C'],[])
        overlap=[deepcopy(prices[0])]+deepcopy(prices[2:])
        out=p.run_missing(task,shots,overlap,c_provider=callback)
        self.assertEqual(len(out['prices']),len(prices))
        conflict=deepcopy(prices[:2]);conflict[0]['mean']['rz_count']=99.
        with self.assertRaisesRegex(ValueError,'Conflicting saved price'):
            p.run_missing(task,shots,overlap,c_provider=Mock(return_value=conflict))
        foreign=deepcopy(prices[:2]);foreign[0]['mean']['rz_count']=None
        foreign[0]['identity']['binding']['state']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'SOURCE mismatch'):
            p.run_missing(task,shots,overlap,c_provider=Mock(return_value=foreign))

    def test_paired_cohort_and_model_qualification_gaps_never_trigger_recompile(self):
        task,shots,prices,_=fixture()
        prices[0]['cohort']='DIFFERENT_X'
        never=Mock(side_effect=AssertionError('Paired-cohort mismatch requires explicit selection'))
        out=p.run_missing(task,shots,prices,c_provider=never);never.assert_not_called()
        cid=task['candidates'][0]['candidate']['id']
        self.assertIn(cid,out['plan']['blocked_C'])
        for v in prices:
            v.update(classification='MODEL_PREDICTION',expectation='FINITE_COHORT_ONLY',
                model=dict(qualified_for_candidate=False,q_variable='q_PF',q=v['identity']['q_PF'],
                    validation_source={'fixture':'MOCK'},source={'fixture':'MOCK'},coverage='MOCK'))
        out=p.run_missing(task,shots,prices,c_provider=never,price_policy='finite_cohort_scenario')
        never.assert_not_called();self.assertTrue(out['plan']['blocked_C'])


if __name__=='__main__':unittest.main(verbosity=2)
