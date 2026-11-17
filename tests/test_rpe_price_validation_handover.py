"""Saved metadata and artificial migration only; no scientific acquisition."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reconcile_rpe_price_validation_store import reconcile
from trottertracks import rpe_price_validation as V
from trottertracks.pr_price_rules import GitReader, PriceRules
from test_rpe_price_validation_entry import description, candidate, ROOT


class Handover(unittest.TestCase):
    def test_normal_gateway_carries_explicit_stage_budget_without_inference_from_q3(self):
        from test_rpe_local0_search import fixture_config,request
        from trottertracks.rpe_local0_search import resolve,load,write,RequestError
        with tempfile.TemporaryDirectory()as tmp:
            config=fixture_config(Path(tmp));spec=load(config['round'])
            spec['executor']={'distributed':{'preparation':{'budgets':{'signal':0,'compile':0,'audit':0}}}}
            write(config['round'],spec);q=request(2,2);q['action']='preview'
            _,resolved,_=resolve(config,q)
            self.assertEqual(resolved['executor']['distributed']['preparation']['budgets']['compile'],0)
            q['scope']['preparation_budgets']={'compile':1}
            _,resolved,canonical=resolve(config,q)
            self.assertEqual(resolved['executor']['distributed']['preparation']['budgets']['compile'],1)
            self.assertEqual(canonical['preparation_budgets'],{'compile':1})
            self.assertFalse(resolved['science_approved'])
            q['scope']['preparation_budgets']={'compile':True}
            with self.assertRaises(RequestError):resolve(config,q)

    def test_owner_frozen_request_partial_results_and_unknown_survive_across_runs(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);source=V.ModelStore(base/'old.sqlite3');desc=description()
            source.register(desc,lambda:([],[]))
            source.change(desc['model_id'],lambda r:r.update(status='PENDING',owner_run='/old-controller/run',
                job_id='same-job',request={'request_id':'same-job','contract':'FROZEN'},
                selected_round=desc['points'][2],partial_results={'X':{'path0':'SAVED'}},
                external_job={'state':'UNKNOWN','attempt_id':'same-attempt','host_id':'server2'}))
            original=source.row(desc['model_id'])
            report=reconcile(source.path,base/'shared.sqlite3')
            self.assertEqual(report['imported'],1)
            self.assertEqual(reconcile(source.path,base/'shared.sqlite3')['retained'],1)
            for run in ('resumed','another-run'):
                service=V.Service(ROOT,base/run,dict(price_validation={'store':str(base/'shared.sqlite3')}))
                with patch.object(service,'_submit',side_effect=AssertionError('No duplicate dispatch')):
                    service.observe([candidate(desc)]);service.poll()
                self.assertEqual(service.store.row(desc['model_id']),original)
            self.assertEqual(source.row(desc['model_id']),original)

    def test_conflicting_frozen_job_is_held_without_blocking_an_unrelated_import(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();stores=[V.ModelStore(base/name)for name in ('old','new')]
            for store in stores:store.register(desc,lambda:([],[]))
            stores[0].change(desc['model_id'],lambda r:r.update(job_id='first'))
            stores[1].change(desc['model_id'],lambda r:r.update(job_id='different'))
            before=stores[1].row(desc['model_id'])
            other=deepcopy(desc);other['identity']['model_source']['commit']='other-fixed'
            other['model_id']=V.digest(other['identity']);stores[0].register(other,lambda:([],[]))
            report=reconcile(stores[0].path,stores[1].path)
            self.assertEqual(report['status'],'HOLD')
            self.assertEqual((report['held'],report['imported']),(1,1))
            self.assertEqual(report['conflicts'][0]['model_id'],desc['model_id'])
            self.assertEqual(report['conflicts'][0]['fields'],['job_id'])
            self.assertEqual(stores[1].row(desc['model_id']),before)
            self.assertEqual(stores[1].row(other['model_id']),stores[0].row(other['model_id']))

    def test_conflicting_external_attempt_is_held_and_partial_progress_is_preserved(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();old=V.ModelStore(base/'old');new=V.ModelStore(base/'new')
            for store in(old,new):store.register(desc,lambda:([],[]))
            old.change(desc['model_id'],lambda r:r.update(external_job={'job_id':'same','attempt_id':'old','state':'UNKNOWN'}))
            new.change(desc['model_id'],lambda r:r.update(external_job={'job_id':'same','attempt_id':'current','state':'COMPLETED_PENDING_TRANSFER'},partial_results={'X':{'path0':'saved'}}))
            before=new.row(desc['model_id']);report=reconcile(old.path,new.path)
            self.assertEqual(report['conflicts'][0]['fields'],['external_job.attempt_id'])
            self.assertEqual(new.row(desc['model_id']),before)

    def test_bad_model_description_holds_only_that_model_and_keeps_candidate_usable(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);bad=description();store=V.ModelStore(base/'shared')
            store.register(bad,lambda:([],[]));before=store.row(bad['model_id'])
            broken=deepcopy(bad);broken['identity']['model_source']['commit']='wrong-frozen-source'
            good=deepcopy(bad);good['identity']['model_source']['commit']='another-valid-source'
            good['model_id']=V.digest(good['identity'])
            c=candidate(good);c['price_model_validations']=[broken,good]
            prices=deepcopy(c['prices'])
            service=V.Service(ROOT,base/'run',dict(price_validation={'store':str(store.path)}))
            with patch.object(service,'resolve',return_value=([],[],None)):
                service.observe([c])
            self.assertEqual(store.row(bad['model_id']),before)
            self.assertIsNotNone(store.row(good['model_id']))
            self.assertIn(good['model_id'],service.descriptions)
            self.assertEqual(c['price_validation_issue']['model_id'],bad['model_id'])
            self.assertEqual(c['price_validation_issue']['status'],'INCOMPLETE')
            self.assertIn(bad['model_id'],service.model_holds)
            self.assertTrue(c['ready']);self.assertEqual(c['needs'],[]);self.assertEqual(c['prices'],prices)

    def test_optional_saved_qr_enrichment_uses_frozen_evidence_not_current_application(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();evidence=dict(commit='b'*40,path='SAVED.json',selector=['rows'])
            desc.update(evidence_sources=[dict(source=evidence)],evidence_points=[dict(m=5,q=32,R=33)])
            desc['points'][5].update(q=999,R=999)  # Another kappa/application is not evidence.
            point=dict(m=5,q=None,R=None,source=dict(evidence,selector=['rows',0]),
                observations=[dict(axis=axis,compiled_RZ=123,source=dict(commit='c'*40,path='RECEIPTS.json',selector=[axis]))for axis in V.AXES])
            preserved=dict(m=9,q=512,R=526,source={'unchanged':'accepted'},observations=[{'unchanged':'accepted'}])
            store=V.ModelStore(base/'shared');store.register(desc,lambda:([point,preserved],[]))
            store.change(desc['model_id'],lambda r:r.update(owner_run='/original-owner',attempt_id='same',partial_results={'X':'kept'}))
            before=store.row(desc['model_id'])
            class Reader:
                def __init__(self,*args):pass
                def read(self,source):return dict(axis=source['selector'][0],request=dict(m=5,q=32,R=33))
            for enabled in(False,True):
                settings=dict(store=str(store.path),enrich_saved_evidence_metadata=enabled)
                service=V.Service(ROOT,base/str(enabled),dict(price_validation=settings))
                with patch('trottertracks.pr_price_rules.GitReader',Reader),patch.object(service,'resolve',side_effect=AssertionError('No independent validation')):
                    service.observe([candidate(desc)])
                expected=deepcopy(before)
                if enabled:expected['validation_points'][0].update(q=32,R=33)
                self.assertEqual(store.row(desc['model_id']),expected)
                self.assertEqual(service.model_holds,set())

    def test_unproven_saved_qr_metadata_is_held_without_changing_saved_values(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();evidence=dict(commit='b'*40,path='SAVED.json',selector=['rows'])
            desc.update(evidence_sources=[dict(source=evidence)],evidence_points=[dict(m=5,q=32,R=33)])
            point=dict(m=5,q=None,R=None,source=dict(evidence,selector=['rows',0]),
                observations=[dict(axis='X',source=dict(commit='c'*40,path='RECEIPTS.json',selector=['X']))])
            store=V.ModelStore(base/'shared');store.register(desc,lambda:([point],[]));before=store.row(desc['model_id'])
            class Reader:
                def __init__(self,*args):pass
                def read(self,source):return dict(axis='X',request=dict(m=5,q=999,R=999))
            service=V.Service(ROOT,base/'run',dict(price_validation=dict(store=str(store.path),enrich_saved_evidence_metadata=True)))
            c=candidate(desc)
            with patch('trottertracks.pr_price_rules.GitReader',Reader):service.observe([c])
            self.assertEqual(store.row(desc['model_id']),before)
            self.assertIn('FROZEN_RECEIPT_METADATA_DIFFERS',c['price_validation_issue']['reason'])
            self.assertTrue(c['ready']);self.assertEqual(c['needs'],[])
            wrong=deepcopy(desc);wrong['evidence_sources']=[]
            with self.assertRaisesRegex(ValueError,'UNMATCHED_FROZEN_EVIDENCE_SOURCE'):
                V.saved_point_metadata_updates(before,wrong,Reader())

    def test_repeated_snapshot_preserves_newly_collected_metadata_and_shared_publication(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();source=V.ModelStore(base/'old')
            source.register(desc,lambda:([],[]));reconcile(source.path,base/'shared')
            current=V.ModelStore(base/'shared')
            current.change(desc['model_id'],lambda r:r.update(status='PENDING',job_id='new-owned-job',owner_run='/local/owner'))
            self.assertEqual(reconcile(source.path,current.path)['retained'],1)
            service=V.Service(ROOT,base/'other-run',dict(price_validation={'store':str(current.path),'publish_shared_models':True}))
            service.export();view=json.loads((service.run/'PRICE_MODEL_VALIDATIONS.json').read_text())
            self.assertEqual(view['model_scope'],'CONTROLLER_SHARED_MODEL_STORE')
            self.assertEqual(view['models'][0]['job_id'],'new-owned-job');self.assertEqual(view['applications'],[])

    def test_saved_df7_holdout_matches_exact_frozen_axis_price_source(self):
        reader=GitReader(ROOT)
        registry=reader.read(dict(commit='c1f95e0806f5478a1ee5ce9ecb148ce9d10461aa',path='configs/pr_price_rules/h4_saved_rules.json'))
        entry=next(e for e in registry['entries']if e['name']=='DF_LD7')
        rules=PriceRules(reader,dict(entries=[entry]));cfg=rules.entries[0]['configuration']
        quote=rules.quote(dict(configuration=cfg,implementation=entry['implementation'],policy='MODEL'))
        desc=V.describe(rules,cfg,entry['name'],quote)[0]
        points,issues=V.saved_checks(desc,reader)
        self.assertEqual(issues,[]);self.assertEqual(len(points),1)
        self.assertEqual((points[0]['m'],points[0]['q'],points[0]['R'],points[0]['independent_paths']),(3,8,1,2))
        self.assertEqual(len(points[0]['observations']),4)
        changed=deepcopy(desc);changed['identity']['coefficients']['X']['alpha']=V.enc(123)
        points,issues=V.saved_checks(changed,reader)
        self.assertEqual(points,[]);self.assertIn('DIFFERENT_FROZEN_PREDICTION_SOURCE',issues)


if __name__=='__main__':unittest.main()
