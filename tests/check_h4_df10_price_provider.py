"""Passive golden-record checks and synthetic provider tests; no new H4 prices."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_df10_price_provider as api


@lru_cache(maxsize=None)
def _json(commit,path):
    return json.loads(subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))


def read(ref):
    return deepcopy(_json(ref['commit'],ref['path']))


def request(provider, method='PF', q=32, axis='X', **selection):
    c=dict(system='H4',method='PR' if method=='PR' else 'B0',formula='2nd',prefix=1 if method=='PR' else 7,
           T=.1*q,q=q,delta=.1,R_outer=4 if method=='PR' else None,K=2 if method=='PR' else None)
    c['id']=api._canonical_id()(c)
    result=dict(candidate=c,axis=axis,binding=provider.binding,profile=deepcopy(api.PROFILE),q_RPE=64,
        contract=dict(completion='saved_weighted_Hermitian_completion',fragment_order='saved_snapshot_order',
            state_preparation_cost=0,input_source={'fixture':'saved_metadata_only'},
            preparation_binding=dict(snapshot=provider.binding['snapshot'],provider_source=api.PROVIDER_SOURCE)))
    result.update(selection)
    return result


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider=api.H4PriceProvider(read)

    def test_saved_golden_prices_and_predictions_are_copied_without_counter(self):
        p=self.provider
        data=read(api.ref(api.RELEASE,api.BASE+'completion01/RESULT.json'))
        predictions=read(api.ref(api.RELEASE,api.BASE+'PREDICTIONS.json'))['predictions']
        trials={(r['candidate_id'],r['index']):r for r in data['trials']}
        with patch.dict(api.mq._scope,{'predict':lambda *a,**k: self.fail('no fresh prediction during lookup')}):
            for row in predictions:
                axis={'cosine':'X','sine':'Y'}[row['axis']]
                r=request(p,row['method'],row['q'],axis)
                if row['method']=='PR':
                    r.update(price_unit='path',cohort=api.mq.COHORT,path_index=row['path_index'],
                             event_digest=row['events_sha256'],digest_kind='canonical_events_sha256')
                direct=p.quote(r)
                self.assertEqual(direct['value'],trials[row['candidate_id'],row['path_index']]['prices'][row['axis']]['rz_count'])
                r['kind']='model_prediction';model=p.quote(r)
                self.assertEqual(model['value'],row['Mq'])
                self.assertEqual(model['validation'],'FINITE_INDEPENDENT_Q')
                self.assertFalse(model['full_RPE_price_adopted'])
                self.assertEqual(model['model_provenance']['calibration']['commit'],api.S1)
        self.assertIsNone(direct['model'])

    def test_cohort_means_remain_cohort_specific_and_not_law_expectations(self):
        p=self.provider
        for cohort,n,value in [('EXPLORATION',9,16857.11111111111),('CONFIRMATION',33,16947.78787878788)]:
            out=p.quote(request(p,'PR',8,price_unit='cohort_mean',cohort=cohort))
            self.assertEqual(out['value'],value)
            evidence=out['uncertainty']['saved_validation']
            self.assertEqual(len(evidence),1)
            self.assertEqual(evidence[0]['record']['n_paths'],n)
            self.assertEqual(evidence[0]['record']['cohort'],cohort)
            self.assertIsNone(out['uncertainty']['all_law_expectation'])
        self.assertIsNone(p.quote(request(p,'PR',16))['value'])
        law=p.quote(request(p,'PR',16,price_unit='law_mean'))
        self.assertIsNone(law['value']);self.assertEqual(law['reason'],'ALL_LAW_EXPECTATION_UNAVAILABLE')

    def test_scientific_recipe_profile_preparation_and_source_mismatches(self):
        p=self.provider
        changes=[lambda r:r['binding'].update(hamiltonian='FOREIGN'),
                 lambda r:r['binding'].update(snapshot='FOREIGN'),
                 lambda r:r['binding'].update(state='FOREIGN'),
                 lambda r:r['binding'].update(rank=12),
                 lambda r:r['profile'].update(wrapper='subcircuit'),
                 lambda r:r['profile'].update(compiler='OTHER'),
                 lambda r:r['profile'].update(optimization_level=True),
                 lambda r:r['contract']['preparation_binding'].update(provider_source='OTHER'),
                 lambda r:r.update(preparation_hash='OTHER'),
                 lambda r:r['contract'].update(fragment_order='reordered')]
        for change in changes:
            r=request(p);change(r);out=p.quote(r)
            self.assertIsNone(out['value']);self.assertFalse(out['qualified_for_candidate'])
        for field,value in [('system','H6'),('method','FULL'),('prefix',1),('delta',.2),('formula','4th')]:
            r=request(p);r['candidate'][field]=value
            if field=='delta':r['candidate']['T']=value*r['candidate']['q']
            r['candidate']['id']=api._canonical_id()(r['candidate'])
            out=p.quote(r)
            self.assertIsNone(out['value'])
            if field=='system':self.assertEqual(out['reason'],'OUTSIDE_MODEL_RECIPE')

    def test_path_identity_and_order_are_bound_before_returning_saved_price(self):
        p=deepcopy(self.provider)
        events=[{'synthetic_event':0},{'synthetic_event':1}]
        r=request(p,'PR',16,price_unit='path',cohort='ARTIFICIAL',path_index=3,
                  ordered_events=events,event_digest=api.events_digest(events),digest_kind='canonical_events_sha256')
        p._records=[dict(method='PR',q=16,axis='X',candidate_id=r['candidate']['id'],cohort='ARTIFICIAL',
            price_unit='path',path_index=3,kind='direct_compiled',value=17,source={'fixture':'ARTIFICIAL_ONLY'},
            event_digest=r['event_digest'],digest_kind='canonical_events_sha256')]
        self.assertEqual(p.quote(r)['value'],17)
        for change in [lambda x:x.update(path_index=4),lambda x:x.update(event_digest='OTHER'),
                       lambda x:x.update(ordered_events=list(reversed(events)))]:
            bad=deepcopy(r);change(bad);self.assertIsNone(p.quote(bad)['value'])

    def test_q_rpe_separation_unknowns_and_synthetic_extrapolation(self):
        p=self.provider
        self.assertEqual(p.quote(request(p,q=32))['value'],176723)
        self.assertEqual(p.quote(request(p,q=32))['conditions']['q_RPE'],64)
        r=request(p,q=64,kind='model_prediction')
        self.assertIsNone(p.quote(r)['value'])
        self.assertEqual(p.quote(r)['validation'],'UNVALIDATED_EXTRAPOLATION')
        blocks=[dict(block_id='one_body' if i==0 else 'df-fragment-'+str(i-1),basis=[],
                     diagonal=SimpleNamespace(rz=[],rzz=[],global_phase=0.)) for i in range(8)]
        descriptors=dict(binding=p.binding,preparation_hash=p.preparation_hash('PF'),blocks=blocks,
                         scalar=0.,source={'fixture':'ARTIFICIAL_EMPTY_BLOCKS_ONLY'})
        # Empty synthetic blocks exercise the existing counter, never molecular descriptors.
        out=p.predict_path(r,descriptors)
        self.assertEqual(out['kind'],'model_prediction')
        self.assertFalse(out['qualified_for_candidate'])
        self.assertEqual(out['validation'],'UNVALIDATED_EXTRAPOLATION')
        self.assertEqual(out['saved_record']['coefficient'],606.75)
        self.assertIsNone(out['uncertainty']['long_q_systematic'])
        self.assertIsNone(out['uncertainty']['total_CI'])

    def test_synthetic_XY_and_return_mutation_do_not_change_saved_catalog(self):
        p=deepcopy(self.provider);r=request(p)
        for record in p._records:
            if record['method']=='PF' and record['q']==32 and record['kind']=='direct_compiled':
                record['value']=13 if record['axis']=='X' else 29
        x=p.quote(r);self.assertEqual(x['value'],13)
        r['axis']='Y';self.assertEqual(p.quote(r)['value'],29)
        x['saved_record']['value']=999;x['conditions']['candidate']['prefix']=999
        self.assertEqual(p.quote(request(p))['value'],13)

    def test_mock_PR_numeric_entry_uses_qPF_and_frozen_PR_coefficient(self):
        p=self.provider
        events=[[dict(taylor_order=0,rotation_angle=.1+i*.01,phase={'real':1.,'imag':0.},
                      application_sequence=[dict(application_index=0,role='rotation',coefficient_sign=1,is_identity=True)])
                 for i in range(4)] for _ in range(2)]
        r=request(p,'PR',2,price_unit='path',path_index=0,ordered_events=events,
                  event_digest=api.events_digest(events),digest_kind='canonical_events_sha256',
                  event_source={'fixture':'ARTIFICIAL_EVENT_SOURCE'})
        descriptors=dict(binding=p.binding,preparation_hash=p.preparation_hash('PR'),scalar=.17,
            source={'fixture':'ARTIFICIAL_MOCK_ONLY'},blocks=[dict(block_id='one_body'),dict(block_id='df-fragment-0')])
        calls=[]
        def counter(blocks,q,scalar,T,axis,batches):
            calls.append((q,T,axis,batches));return {'structural_rz':101}
        with patch.dict(api.mq._scope,{'predict':counter}):
            out=p.predict_path(r,descriptors)
        self.assertEqual(calls,[(2,.2,'cosine',events)])
        self.assertEqual(out['value'],101+2*176.88888888888894)
        self.assertFalse(out['qualified_for_candidate'])
        self.assertNotIn('ordered_events',out['conditions'])
        bad=deepcopy(r);bad['event_digest']='other'
        with self.assertRaisesRegex(ValueError,'DIGEST_MISMATCH'):p.predict_path(bad,descriptors)
        bad=deepcopy(descriptors);bad['preparation_hash']='other'
        with self.assertRaisesRegex(ValueError,'DESCRIPTOR'):p.predict_path(r,bad)
        bad=deepcopy(r);bad['ordered_events'][0][0]['taylor_order']=4
        bad['event_digest']=api.events_digest(bad['ordered_events'])
        with self.assertRaisesRegex(ValueError,'K2_APPLICATION'):p.predict_path(bad,descriptors)
        bad=deepcopy(r);del bad['event_source']
        with self.assertRaisesRegex(ValueError,'EVENT_SOURCE_REQUIRED'):p.predict_path(bad,descriptors)


class BatchCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('H4_PROVIDER_B_WORKTREE'),'B working interface path is optional')
    def test_B_batch_uses_its_identity_measurement_and_static_plan_only(self):
        root=Path(os.environ['H4_PROVIDER_B_WORKTREE'])
        path=root/'src/trottertracks/pf_pr_rpe_pipeline.py';raw=path.read_bytes()
        module=ModuleType('trottertracks._h4_provider_B_snapshot')
        module.__package__='trottertracks';module.__file__=str(path)
        exec(compile(raw,str(path),'exec'),module.__dict__)
        p=module;store=p.SavedArtifacts(ROOT);provider=api.H4PriceProvider(store.read,pipeline=p)
        config=read(api.ref(api.S1,'docs/research/h4_df10_rpe_d1_20261110/result.json'))['configuration']
        rows=[]
        for method,q,m in [('PF',32,5),('PR',16,4),('PF',64,6),('PR',8,3)]:
            c=request(provider,method,q)['candidate']
            rows.append(dict(m=m,q_RPE=1<<m,q_PF=q,candidate=c))
        contract=request(provider)['contract']
        task=p.task_from_saved(config,provider.binding,rows,task_id='TECHNICAL_SAVED_METADATA_ONLY',
                               source={'fixture':'metadata'},contract=contract)
        provider.selections={rows[1]['candidate']['id']:dict(price_unit='cohort_mean',cohort=api.mq.COHORT),
                             rows[3]['candidate']['id']:dict(price_unit='cohort_mean',cohort='CONFIRMATION')}
        with (patch.object(p,'evaluate',side_effect=AssertionError('No G')),
              patch.object(p,'run_missing',side_effect=AssertionError('No science'))):
            prices=provider(task,task['candidates']);plan=p.plan(task,[],prices)
        self.assertEqual(len(prices),8);self.assertIsNone(plan['totals'])
        pf=[x for x in prices if x['identity']['candidate']['q']==32]
        self.assertTrue(all(x['classification']=='DIRECT_COMPILED_DETERMINISTIC' for x in pf))
        pr=[x for x in prices if x['identity']['candidate']['q']==16]
        self.assertTrue(all(x['expectation']=='FINITE_COHORT_ONLY' for x in pr))
        unknown=[x for x in prices if x['identity']['candidate']['q']==64]
        self.assertTrue(all(x['mean']['rz_count'] is None for x in unknown))
        self.assertTrue(all(x['cohort'].startswith('UNAVAILABLE:') for x in unknown))
        provider.selections[rows[0]['candidate']['id']]={'kind':'model_prediction'}
        provider.selections[rows[1]['candidate']['id']]['kind']='model_prediction'
        models=provider(task,task['candidates'][:2])
        self.assertTrue(all(x['classification']=='MODEL_PREDICTION' for x in models))
        self.assertTrue(all(x['model']['q_variable']=='q_PF' for x in models))
        self.assertTrue(all(x['model']['qualified_for_candidate'] for x in models))
        self.assertTrue(all(x['expectation']=='FINITE_COHORT_ONLY' for x in models if x['identity']['candidate']['method']=='PR'))
        r=request(provider,q=64,kind='model_prediction')
        fake=provider._response(r,'PF',kind='model_prediction',value=7.,record={'source':{'fixture':'ARTIFICIAL_QUOTE'}})
        extrapolated=provider.pipeline_record(task,task['candidates'][2],'X',fake)
        self.assertEqual(extrapolated['classification'],'UNVALIDATED_EXTRAPOLATION')
        self.assertEqual(extrapolated['expectation'],'UNQUALIFIED_MODEL')
        self.assertFalse(extrapolated['model']['qualified_for_candidate'])
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISMATCH'):
            provider.pipeline_record(task,task['candidates'][2],'Y',fake)
        provider.selections[rows[0]['candidate']['id']]={'candidate':rows[1]['candidate']}
        with self.assertRaisesRegex(ValueError,'SELECTOR_CANNOT'):
            provider(task,task['candidates'][:1])
        print('B_INTERFACE_SNAPSHOT_SHA256='+hashlib.sha256(raw).hexdigest())


if __name__=='__main__':unittest.main(verbosity=2)
