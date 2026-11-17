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

    def test_conflicting_frozen_jobs_abort_atomically(self):
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);desc=description();stores=[V.ModelStore(base/name)for name in ('old','new')]
            for store in stores:store.register(desc,lambda:([],[]))
            stores[0].change(desc['model_id'],lambda r:r.update(job_id='first'))
            stores[1].change(desc['model_id'],lambda r:r.update(job_id='different'))
            before=stores[1].row(desc['model_id'])
            with self.assertRaisesRegex(ValueError,'Model state differs'):
                reconcile(stores[0].path,stores[1].path)
            self.assertEqual(stores[1].row(desc['model_id']),before)

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
