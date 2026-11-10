"""Technical golden/mock tests only. Never predict an unsaved H6 price or G."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
B_SOURCE='cb2b1d6b4309ca7d705f397e25e27d14e8ba125a'
# Exact B code/dependencies in an isolated temporary root; no writes to B's tree.
SNAPSHOT=tempfile.TemporaryDirectory(prefix='h6_price_provider_B_fixture_')
BROOT=Path(SNAPSHOT.name)
DEPENDENCIES=('src/trottertracks/pf_pr_rpe_pipeline.py','src/trottertracks/h4_df10_rpe_connection.py',
    'src/trottertracks/c1_stage3_accuracy.py','src/trotterlib/h46_m5_plan.py',
    'scripts/h46_long_time_signal.py','scripts/h46_new2_core.py')
for name in DEPENDENCIES:
    path=BROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(subprocess.check_output(['git','show',B_SOURCE+':'+name],cwd=ROOT))
for package in ('trottertracks','trotterlib'):(BROOT/'src'/package/'__init__.py').write_text('')
provider_path='src/trottertracks/h6_df19_price_provider.py'
(BROOT/provider_path).write_bytes((ROOT/provider_path).read_bytes())
sys.path.insert(0,str(BROOT/'src'))
from trottertracks import h6_df19_price_provider as api
from trottertracks import pf_pr_rpe_pipeline as p


@lru_cache(maxsize=None)
def saved(commit,path):return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))
def read(ref):return deepcopy(saved(ref['commit'],ref['path']))


def request(provider, family='PF_FULL19', q=4, axis='X', **selection):
    pr=family=='PR_B'
    c=dict(system='H6',method='PR' if pr else 'FULL',formula='4th(m5_best)',prefix=7 if pr else 19,
        q=q,T=1.6*q,delta=1.6,R_outer=16 if pr else None,K=2 if pr else None)
    c['id']=p.candidate_id(c)
    result=dict(candidate=c,axis=axis,q_RPE=16*q,binding=provider.binding,profile=deepcopy(api.PROFILE),
        contract=dict(completion='saved_weighted_Hermitian_completion',fragment_order='saved_snapshot_order',
            state_preparation_cost=0,input_source={'fixture':'SAVED_METADATA_ONLY'},
            preparation_binding=dict(snapshot=provider.binding['snapshot'],provider_source=api.PROVIDER_SOURCE)))
    result.update(selection);return result


def task_fixture(provider, specs):
    plan=read(dict(commit='18fcd90f1614e3bf7b91bb02d8254f968e048a94',path='docs/research/h6_df19_rpe_m6_preparation_20261110/PLAN.json'))
    rows=[]
    for family,q in specs:
        r=request(provider,family,q);m={1:4,2:5,4:6,8:7}[q]
        rows.append(dict(m=m,q_RPE=1<<m,q_PF=q,candidate=r['candidate']))
    return p.task_from_saved(plan['configuration_reuse'],provider.binding,rows,task_id='H6_PROVIDER_TECHNICAL_FIXTURE',
        source={'fixture':'SAVED_METADATA_NO_SCIENCE'},contract=request(provider)['contract'])


def fake_shot(task,row):
    lane=dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],candidate=deepcopy(row['candidate']),binding=deepcopy(task['binding']),
        base_time=task['configuration']['base_time'],target_id=task['task_id'],status='CONDITIONAL_ROUND_PASS',
        N_X=3,N_Y=5,alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'])
    return dict(identity=p.identity(task,row),plan=lane,actual=dict(N_X=None,N_Y=None),source={'fixture':'MOCK_N_NO_SCIENCE'})


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.provider=api.H6AffinePriceProvider(p.SavedArtifacts(ROOT).read,pipeline=p)

    def test_frozen_coefficients_and_B_snapshot_are_exact(self):
        for name in DEPENDENCIES:self.assertEqual((BROOT/name).read_bytes(),subprocess.check_output(['git','show',B_SOURCE+':'+name],cwd=ROOT))
        frozen=read(api.ref(api.FROZEN_SOURCE,api.BASE+'Q4_FROZEN_MODELS.json'))
        self.assertEqual(self.provider.coefficients,{k:v['coefficients'] for k,v in frozen['families'].items()})
        copy=self.provider.coefficients;copy['PF_FULL19'][0][0]=0
        self.assertNotEqual(copy,self.provider.coefficients)
        self.assertEqual(api.candidate_id(request(self.provider)['candidate']),p.candidate_id(request(self.provider)['candidate']))

    def test_all_saved_catalog_prices_are_lookup_only(self):
        provider=self.provider;count=0
        with patch.object(api,'_affine',side_effect=AssertionError('No numerical prediction')):
            for record in provider._records:
                r=request(provider,'PR_B' if record['candidate']['method']=='PR' else 'PF_FULL19',record['candidate']['q'],record['axis'],
                    kind=record['kind'],price_unit=record['price_unit'],cohort=record['cohort'])
                if record['price_unit']=='path':r.update(path_index=record['path_index'],event_digest=record['event_digest'],digest_kind=record['digest_kind'])
                # Compare against the pinned raw JSON field, not merely the loader's catalog.
                source=record['source'];doc=read(source)
                if source['path'].endswith('Q_VALIDATION.json'):
                    summary=next(v for v in doc['summaries'] if all(v[k]==source[k] for k in ('family','q_PF','axis','model')))
                    field='predicted_mean' if record['kind']=='model_prediction' else 'observed_mean'
                    self.assertEqual(source['model'],'Affine' if record['kind']=='model_prediction' else 'M0')
                    expected=summary['metrics'][field]
                elif source['path'].endswith(('Q2_DIRECT.json','Q4_DIRECT.json')):
                    self.assertEqual(record['kind'],'direct_compiled')
                    raw=next(v for v in doc['rows'] if v['candidate']['id']==source['candidate_id'] and v['index']==source['index'])
                    expected=raw['prices'][source['axis']]['rz_count']
                elif record['price_unit']=='cohort_mean':
                    self.assertEqual(record['kind'],'direct_compiled')
                    expected=next(v for v in doc['summaries'] if v['family']=='PR_B' and v['axis']==source['axis'])['mean_compiled']
                else:
                    self.assertEqual(record['kind'],'direct_compiled')
                    raw=next(v for v in doc['rows'] if v['candidate_id']==source['candidate_id'] and v['path_index']==source['path_index'] and v['axis']==source['axis'])
                    expected=raw['compiled_rz']
                out=provider.quote(r);self.assertEqual(out['value'],expected);self.assertEqual(out['kind'],record['kind'])
                self.assertFalse(out['full_RPE_price_adopted']);count+=1
        # 86 q1 path/deterministic axes + 4 q1 means; 36 new direct axes + 4 means + 8 model axes.
        self.assertEqual(count,138)
        print('SAVED_CATALOG_SCALAR_MATCHES='+str(count))

    def test_q1_cohorts_and_model_miss_are_not_pooled_or_substituted(self):
        provider=self.provider
        for cohort,n in [('EXPLORATION',9),('CONFIRMATION',33)]:
            out=provider.quote(request(provider,'PR_B',1,price_unit='cohort_mean',cohort=cohort))
            source=read(api.ref(api.V1,api.V1_BASE+('DEVELOPMENT' if n==9 else 'HOLDOUT')+'.json'))
            expected=next(s for s in source['summaries'] if s['family']=='PR_B' and s['axis']=='cosine')['mean_compiled']
            self.assertEqual(out['value'],expected);self.assertEqual(out['saved_record']['n_paths'],n)
        miss=provider.quote(request(provider,q=1,kind='model_prediction'))
        self.assertIsNone(miss['value']);self.assertEqual(miss['validation'],'CALIBRATION_Q1')
        exploration=provider.quote(request(provider,'PR_B',1,price_unit='cohort_mean',cohort='EXPLORATION'))
        confirmation=provider.quote(request(provider,'PR_B',1,price_unit='cohort_mean',cohort='CONFIRMATION'))
        self.assertEqual(exploration['uncertainty']['saved_direct_mean_statistics'],provider._models['PR_B']['q1'])
        self.assertIsNone(confirmation['uncertainty']['saved_direct_mean_statistics'])
        self.assertIsNone(provider.quote(request(provider,'PR_B',4))['value'])
        self.assertIsNone(provider.quote(request(provider,'PR_B',4,price_unit='law_mean'))['value'])

    def test_path_means_and_digest_types_stay_distinct(self):
        provider=self.provider;record=next(v for v in provider._records if v['price_unit']=='path' and v['candidate']['q']==4)
        r=request(provider,'PR_B',4,price_unit='path',cohort=record['cohort'],path_index=record['path_index'],event_digest=record['event_digest'],digest_kind=record['digest_kind'])
        self.assertEqual(provider.quote(r)['value'],record['value'])
        for update in [dict(event_digest='WRONG'),dict(digest_kind='canonical_events_sha256'),dict(path_index=True),dict(ordered_events=[{'fabricated':'payload'}]),dict(kind='model_prediction')]:
            self.assertIsNone(provider.quote(dict(r,**update))['value'])

    def test_domain_profile_and_preparation_fail_closed(self):
        provider=self.provider
        mutations=[lambda r:r['binding'].update(hamiltonian='OTHER'),lambda r:r['binding'].update(snapshot='OTHER'),
            lambda r:r['binding'].update(state='OTHER'),lambda r:r['binding'].update(rank=10),
            lambda r:r['profile'].update(wrapper='subroutine'),lambda r:r['profile'].update(compiler='OTHER'),
            lambda r:r['profile'].update(optimization_level=True),lambda r:r['profile'].update(layout_method='trivial'),
            lambda r:r['contract'].update(fragment_order='sorted'),lambda r:r.update(preparation_hash='OTHER'),
            lambda r:r['contract']['preparation_binding'].update(provider_source='OTHER')]
        for mutate in mutations:
            r=request(provider);mutate(r);out=provider.quote(r);self.assertIsNone(out['value']);self.assertEqual(out['validation'],'OUTSIDE_DOMAIN')
        for fields in [dict(method='B0'),dict(prefix=5,R_outer=32),dict(formula='2nd'),dict(delta=.8,T=3.2)]:
            r=request(provider,'PR_B' if 'R_outer' in fields else 'PF_FULL19');r['candidate'].update(fields);r['candidate']['id']=p.candidate_id(r['candidate'])
            self.assertIsNone(provider.quote(r)['value'])
        for q in [True,0,-1,1.5]:
            r=request(provider);r['candidate']['q']=q
            with self.assertRaises(ValueError):provider.quote(r)

    def test_calibration_q4_and_unacquired_q_roles_without_new_H6_predictions(self):
        provider=self.provider
        with patch.object(api,'_affine',side_effect=AssertionError('No unseen H6 forecast')):
            for q,role in [(2,'CALIBRATION_Q2'),(4,'FINITE_INDEPENDENT_Q'),(8,'UNVALIDATED_EXTRAPOLATION')]:
                out=provider.quote(request(provider,q=q,kind='model_prediction'))
                self.assertEqual(out['validation'],role);self.assertEqual(out['qualified_for_candidate'],q==4)
                if q in (3,8):self.assertIsNone(out['value'])
        # Arithmetic exercised on artificial coefficients only for an unseen q.
        self.assertEqual(api._affine([[2,4],[3,6]],8,0),26)
        self.assertEqual(api._affine([[2,4],[3,6]],8,1),52)
        for family in ('PF_FULL19','PR_B'):
            for q in (2,4):
                r=request(provider,family,q,kind='model_prediction',price_unit='model_mean' if family=='PR_B' else 'deterministic')
                actual=provider.predict_affine(r)
                target=provider._models[family]['q4_mean_prediction'][0] if q==4 else provider._models[family]['q2']['mean'][0]
                self.assertAlmostEqual(actual['value'],target,places=7)
                if family=='PR_B':self.assertFalse(actual['qualified_for_candidate'])

    def test_nominal_covariance_survives_without_joint_or_law_promotion(self):
        provider=self.provider;r=request(provider,'PR_B',4,kind='model_prediction',price_unit='cohort_mean',cohort='H6_PRICE_V2_20261110_Q4_INDEPENDENT')
        out=provider.quote(r);u=out['uncertainty'];self.assertIsNotNone(u['nominal_coefficient_covariance']);self.assertIsNotNone(u['saved_q4_prediction_mean_covariance'])
        for key in ('coefficient_uncertainty','long_q_systematic','joint_covariance','all_law_expectation','unobserved_order2_price','total_CI'):self.assertIsNone(u[key])
        self.assertEqual(u['observed_pilot_orders'],{'0':768,'2':0});self.assertFalse(u['cross_round_independence_assumed'])
        paired=provider.quote(dict(r,axis='Y'));self.assertEqual(paired['uncertainty']['shared_calibration_group'],u['shared_calibration_group'])
        direct=provider.quote(request(provider,'PR_B',2,price_unit='cohort_mean',cohort='H6_PRICE_V2_20261110_Q2_CALIBRATION'))
        self.assertEqual(direct['uncertainty']['calibration_data_group'],u['shared_calibration_group'])
        self.assertIsNone(direct['uncertainty']['shared_calibration_group'])

    def test_B_early_PR_family_is_unavailable_without_fallback(self):
        provider=self.provider;template=task_fixture(provider,[('PR_B',1)])
        c=deepcopy(template['candidates'][0]['candidate']);c.update(prefix=5,R_outer=32,T=.8,delta=.8);c['id']=p.candidate_id(c)
        task=p.task_from_saved(template['configuration'],provider.binding,[dict(m=3,q_RPE=8,q_PF=1,candidate=c)],
            task_id='EARLY_PR_METADATA_ONLY',source={'fixture':'SAVED_RECIPE'},contract=template['contract'])
        with patch.object(api,'_affine',side_effect=AssertionError('No early-round extrapolation')):
            records=provider(task,task['candidates'])
        self.assertTrue(all(r['mean']['rz_count'] is None and r['provider_quote']['reason']=='OUTSIDE_MODEL_RECIPE' for r in records))
        row=p.plan(task,[fake_shot(task,task['candidates'][0])],records)['rows'][0]
        self.assertEqual(row['C_axes'],[]);self.assertEqual(row['C_record_axes'],['X','Y'])

    def test_B_run_missing_preserves_nulls_and_never_calculates_real_G(self):
        provider=api.h6_saved_price_provider(p.SavedArtifacts(ROOT).read,pipeline=p)
        task=task_fixture(provider,[('PF_FULL19',2),('PR_B',4),('PF_FULL19',8)])
        pr=task['candidates'][1];provider.selections={pr['candidate']['id']:dict(price_unit='cohort_mean',cohort='H6_PRICE_V2_20261110_Q4_INDEPENDENT')}
        callback=Mock(wraps=provider);shots=[fake_shot(task,r) for r in task['candidates']]
        with (patch.object(api,'_affine',side_effect=AssertionError('No auto prediction')),
              patch.object(p,'evaluate',side_effect=AssertionError('No G')),
              patch.object(p.accounting,'aggregate_supplied_axes',side_effect=AssertionError('No N*C'))):
            result=p.run_missing(task,shots,[],c_provider=callback);callback.assert_called_once()
            again=p.run_missing(task,shots,result['prices'],c_provider=callback);callback.assert_called_once()
        self.assertIsNone(result['totals']);unknown=again['plan']['rows'][2]
        self.assertEqual(unknown['C_axes'],[]);self.assertEqual(unknown['C_record_axes'],['X','Y'])
        self.assertIn(unknown['candidate_id'],again['plan']['blocked_C']);self.assertEqual(again['plan']['C_provider_requests'],[])
        finite=next(v for v in result['prices'] if v['identity']['candidate']['method']=='PR')
        self.assertEqual(finite['expectation'],'FINITE_COHORT_ONLY');self.assertIsNone(finite['model'])

    def test_B_finite_model_policy_and_mock_extrapolation_stay_separate(self):
        provider=api.h6_saved_price_provider(p.SavedArtifacts(ROOT).read,pipeline=p)
        task=task_fixture(provider,[('PF_FULL19',2),('PF_FULL19',4),('PR_B',4),('PF_FULL19',8)])
        for row in task['candidates']:
            s=dict(kind='model_prediction')
            if row['candidate']['method']=='PR':s.update(price_unit='cohort_mean',cohort='H6_PRICE_V2_20261110_Q4_INDEPENDENT')
            provider.selections[row['candidate']['id']]=s
        prices=provider(task,task['candidates']);shots=[fake_shot(task,r) for r in task['candidates']]
        with patch.object(p,'evaluate',side_effect=AssertionError('No G')):
            plan=p.plan(task,shots,prices,price_policy='finite_cohort_scenario')
        self.assertEqual([bool(r['eligible_C_cohorts']) for r in plan['rows']],[False,True,True,False])
        self.assertEqual(p.plan(task,shots,prices)['rows'][2]['eligible_C_cohorts'],[])
        row=task['candidates'][-1];mock_prices=[]
        for axis in p.AXES:
            r=request(provider,q=8,axis=axis,kind='model_prediction')
            with patch.object(api,'_affine',return_value=7.) as numerical:
                quote=provider.predict_affine(r);numerical.assert_called_once_with(provider._models['PF_FULL19']['coefficients'],8,list(p.AXES).index(axis))
            quote['source']={'fixture':'ARTIFICIAL_ONLY_NO_H6_PREDICTION'}
            mock_prices.append(provider.pipeline_record(task,row,axis,quote))
        self.assertTrue(all(v['classification']=='UNVALIDATED_EXTRAPOLATION' and not v['model']['qualified_for_candidate'] for v in mock_prices))
        blocked=p.plan(task,shots,mock_prices,price_policy='finite_cohort_scenario')['rows'][-1]
        self.assertEqual(blocked['eligible_C_cohorts'],[])
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):provider.pipeline_record(task,row,'Y',quote if quote['conditions']['axis']=='X' else dict(quote,conditions=dict(quote['conditions'],axis='X')))
        provider.selections[row['candidate']['id']]={'candidate':task['candidates'][0]['candidate']}
        with self.assertRaisesRegex(ValueError,'SELECTOR_CANNOT'):provider(task,[row])


if __name__=='__main__':
    try:unittest.main(verbosity=2)
    finally:SNAPSHOT.cleanup()
