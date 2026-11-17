"""Normal-entry acceptance: saved LD2 evidence + synthetic search; no science."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_rpe_method_optimizer as CLI
from trottertracks import rpe_price_validation as V

ROOT=Path(__file__).resolve().parents[1]


def description():
    identity=dict(family=dict(method='PR_PAULI',implementation={'lowering_id':'SYNTHETIC'}),actual_step=V.enc('1/10'),
        mode='SYNTHETIC',model_source=dict(commit='fixed',path='MODEL.json'),coefficients={'one':V.enc(2)})
    return dict(model_id=V.digest(identity),identity=identity,registry='ARTIFICIAL',calibration_sources=[],
        calibration_points=[dict(m=0)],evidence_sources=[],configuration_id='artificial',points=[
            dict(m=m,q=2**m,R=2,physical_id='p'+str(m),predicted_RZ={a:V.enc(2**m+1)for a in V.AXES})for m in range(13)])


def candidate(desc):
    return dict(configuration_id=desc['configuration_id'],method='PR_PAULI',family='PR',
        prices=[Fraction(2**m+1)for m in range(13)for _ in V.AXES],caps=[4]*13,
        price_model_validations=[desc],name='synthetic',ready=True,needs=[])


class Entry(unittest.TestCase):
    def test_calibration_only_has_one_representative_across_runs_allocations_and_concurrency(self):
        with tempfile.TemporaryDirectory()as d:
            spec=dict(price_validation=dict(store=str(Path(d)/'shared.sqlite')))
            desc=description();c=candidate(desc)
            services=[V.Service(ROOT,Path(d)/('run'+str(i)),spec)for i in range(4)]
            def request(service):
                service.observe([deepcopy(c)]);service.poll()
            with ThreadPoolExecutor(4)as pool:list(pool.map(request,services))
            record=services[0].store.row(desc['model_id'])
            self.assertEqual(record['status'],'PENDING');self.assertEqual(record['selected_round']['m'],2)
            self.assertNotIn('owner_run',record)  # a metadata-only run never steals future execution
            for service in services:
                view=json.loads((service.run/'PRICE_MODEL_VALIDATIONS.json').read_text())
                self.assertEqual(len(view['models']),1);self.assertEqual(view['models'][0]['selected_round'],record['selected_round'])
            c['validation_vector']=[100]*26
            services[0].observe([c]);services[0].poll()
            self.assertEqual(services[0].store.row(desc['model_id'])['selected_round'],record['selected_round'])

    def test_active_equivalent_job_prevents_new_request(self):
        with tempfile.TemporaryDirectory()as d:
            desc=description();service=V.Service(ROOT,d,dict(price_validation=dict(store=str(Path(d)/'db'))))
            class Reader:
                def read(self,source):return [{'model_id':desc['model_id'],'state':'RUNNING','job_id':'previous-host-job'}]
            service.settings['existing_job_sources']=[dict(commit='fixed',path='JOBS.json')]
            service.store.register(desc,lambda:service.resolve(desc,Reader()))
            service.descriptions[desc['model_id']]=(desc,candidate(desc));service.poll()
            row=service.store.row(desc['model_id']);self.assertEqual(row['status'],'PENDING')
            self.assertEqual(row['external_job']['job_id'],'previous-host-job');self.assertNotIn('selected_round',row)

    def test_identity_and_source_changes_are_not_shared(self):
        a=description();b=deepcopy(a);b['identity']['model_source']['commit']='new'
        self.assertNotEqual(V.digest(a['identity']),V.digest(b['identity']))
        b=deepcopy(a);b['identity']['actual_step']=V.enc('1/5')
        self.assertNotEqual(V.digest(a['identity']),V.digest(b['identity']))

    def test_direct_only_does_not_scan_or_reserve(self):
        with tempfile.TemporaryDirectory()as d:
            service=V.Service(ROOT,d,dict(price_validation=dict(store=str(Path(d)/'db'))))
            c=candidate(description());c.pop('price_model_validations')
            with patch.object(service,'resolve',side_effect=AssertionError('direct prices need no validation')):
                service.observe([c]);service.poll()
            self.assertEqual(json.loads((Path(d)/'PRICE_MODEL_VALIDATIONS.json').read_text())['models'],[])

    def test_frozen_dispatch_survives_uncertain_submission_without_rebuilding(self):
        from trottertracks.rpe_preparation_queue import normalize_preparation_request
        with tempfile.TemporaryDirectory()as d:
            directory=Path(d)/'run';desc=description();c=candidate(desc)
            c['generated_configuration']={'id':'artificial','spec':{},'rounds':[]}
            spec=dict(science_approved=True,executor=dict(distributed=dict(preparation=dict(budgets=dict(compile=1)))),
                price_validation=dict(store=str(Path(d)/'db'),providers=dict(PR_PAULI=dict(kind='PR_PAULI',commands=[]))))
            class Executor:
                source_manifest={'commit':'fixed','files':{}}
                calls=0
                ids=set()
                def submit_preparation(self,request,**kwargs):
                    self.calls+=1;self.ids.add(request['request_id'])
                    if self.calls==1:raise RuntimeError('connection lost after persistent reserve')
                    return request['request_id']
                def preparation_result(self,job,wait=False):return None
            ex=Executor();ex.run_dir=directory;service=V.Service(ROOT,directory,spec,ex);service.observe([c])
            task=dict(method='PR_PAULI',route=dict(kind='preparation'),preparation=dict(stage='compile',provider='PRICE_MODEL_VALIDATION',
                source=ex.source_manifest,command=['fixed.py'],inputs={},outputs=[],artificial=True))
            with patch('trottertracks.rpe_preparation_bridge.DistributedPreparationBridge.build_request',return_value=task)as build:
                with self.assertRaisesRegex(RuntimeError,'connection lost'):service.poll()
                fixed=service.store.row(desc['model_id']);self.assertIn('job_id',fixed)
                service.implementation_source='changed-after-resume';service.poll()
                self.assertEqual(build.call_count,1)
                self.assertEqual(len(ex.ids),1)
                self.assertEqual(service.store.row(desc['model_id'])['request'],fixed['request'])

    def test_saved_ld2_through_normal_executor_entry_zero_compiles_and_multiple_allocations(self):
        scope={'__name__':'fixed_price_rules','__file__':'fixed_price_rules'}
        exec(compile(subprocess.check_output(['git','show','c1f95e080:src/trottertracks/pr_price_rules.py'],cwd=ROOT),'fixed','exec'),scope)
        registry=json.loads(subprocess.check_output(['git','show','c1f95e080:configs/pr_price_rules/h4_saved_rules.json'],cwd=ROOT))
        entry=next(e for e in registry['entries']if e['name']=='PAULI_LD2')
        rules=scope['PriceRules'](scope['GitReader'](ROOT),dict(entries=[entry]));cfg=rules.entries[0]['configuration']
        quote=rules.quote(dict(configuration=cfg,implementation=entry['implementation'],policy='MODEL'))
        descs=V.describe(rules,cfg,entry['name'],quote)
        self.assertEqual(len(descs),1)
        c=dict(configuration_id=cfg['id'],method='PR_PAULI',family='PR',prices=[V.frac(r['unit_price_RZ'])for r in quote['rows']],
            caps=[48,45,42,39,36,33,29,26,23,19,16,13,9],price_model_validations=descs,ready=True,needs=[])
        class Executor:
            def __init__(self,*args,**kwargs):pass
            def finish(self):pass
        def search(*args,**kwargs):
            self.assertTrue(kwargs['context']['candidates'][0]['ready'])
            return {'PR_PAULI':dict(stop_reason='SYNTHETIC_SEARCH_ADVANCED',new_calls=0)}
        with tempfile.TemporaryDirectory()as d:
            spec=dict(methods=['PR_PAULI'],max_new_q3_per_method={'PR_PAULI':0},science_approved=False,
                price_validation=dict(store=str(Path(d)/'db')))
            with patch.object(CLI.O,'Executor',Executor),patch.object(CLI.O,'optimize',side_effect=search)as run:
                ex,optimize=CLI.make_executor(ROOT,Path(d),{}, {}, {},spec)
                first=optimize(ROOT,{},context={'candidates':[c]});c['validation_vector']=[3]*26
                second=optimize(ROOT,{},context={'candidates':[c]});ex.finish()
            self.assertEqual(first,second);self.assertEqual(run.call_count,2)
            view=json.loads((Path(d)/'PRICE_MODEL_VALIDATIONS.json').read_text());self.assertEqual(len(view['models']),1)
            model=view['models'][0];self.assertIsNotNone(model,(Path(d)/'PRICE_VALIDATION_ISSUE.json').read_text()if(Path(d)/'PRICE_VALIDATION_ISSUE.json').exists()else view);self.assertEqual(model['status'],'INDEPENDENT_MEASURED')
            m5=next(p for p in model['validation_points']if p['m']==5)
            self.assertEqual((m5['q'],m5['R'],m5['independent_paths']),(32,33,2))
            self.assertEqual([p['compiled_RZ']for p in m5['observations']],[68228,68300,68228,68300])
            self.assertEqual(V.frac(m5['signed_relative_difference']['X']),Fraction(2751307649,6167206567816))
            self.assertNotIn('job_id',model)
            m9=next(p for p in model['validation_points']if p['m']==9)
            self.assertEqual((m9['q'],m9['R'],m9['independent_paths']),(512,526,2))
            changed=deepcopy(descs[0]);changed['points'][5]['R']=999
            kept,issues=V.saved_checks(changed,rules.reader)
            self.assertEqual(next(p for p in kept if p['m']==5)['R'],33)  # measured and applied q/R stay distinct


if __name__=='__main__':unittest.main()
