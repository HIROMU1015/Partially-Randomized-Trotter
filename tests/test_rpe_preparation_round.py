"""Finite four-method preparation orchestration, artificial metadata providers."""
from copy import deepcopy
from pathlib import Path
import sys
import json
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_preparation_round as R
from trottertracks import pr_candidate_completion as PR,pf_candidate_selection as PF

METHODS=['FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF']


def candidate(method,cid,ready):
    return dict(method=method,configuration_id=cid,ready=ready,prices=[1],needs=[])


class Completion:
    def __init__(self,directory,settings):
        self.directory=Path(directory);self.settings=settings;self.state={'records':{}};self.lock=threading.RLock()
    def bind_stage_executor(self,runtime):self.runtime=runtime;return self
    def install_snapshot(self,context,problem,keys):
        routes=deepcopy(problem['execution_routes'])
        with self.lock:records=deepcopy([r for key,r in self.state['records'].items()if key in keys])
        for record in records:
            cid=record['configuration_id'];method=record['method']
            if not any(c['configuration_id']==cid for c in context['candidates']):context['candidates'].append(candidate(method,cid,True))
            context['expected'][cid]={'binding':'ARTIFICIAL_'+cid}
            routes.setdefault(method,[])
            if cid not in [r.get('cid')for r in routes[method]]:routes[method].append({'cid':cid})
        return routes,records


class PreparationRoundTests(unittest.TestCase):
    def inputs(self):
        context=dict(candidates=[candidate('FULL_PF','pf-old',True),candidate('TRUNCATED_PF','b0i-old',False)],
            expected={'pf-old':{'binding':'OLD_PF'},'b0i-old':{'binding':'OLD_B0I'}},saved=SimpleNamespace(table={}),
            generation={},facts={'hamiltonian':'ARTIFICIAL'},tau=1)
        spec=dict(methods=METHODS,science_approved=True,max_new_q3_per_method={method:3 for method in METHODS},
            initial='all',candidate_completion={'source':'ARTIFICIAL'},search_overrides={})
        problem=dict(search={'max_steps_per_method':10},methods={method:{}for method in METHODS})
        executor=SimpleNamespace(log=lambda **event:None,routes=None,expected=None)
        return context,spec,problem,executor

    def test_truncated_pf_stage_hold_leaves_ready_context_usable(self):
        routes = {'FULL_PF': [{'kind': 'artificial'}]}
        context, spec, problem, _ = self.inputs()
        spec['pf_completion'] = dict(mode='complete', entry='synthetic-entry',
                                     selected=[dict(configuration_body={})])
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                PF.PF, 'complete_selected', side_effect=PR.Held('DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:PF_COMPILE')):
            current, unchanged_routes, status = PF.prepare(ROOT, problem, context, spec, tmp, routes,
                                                            stage_executor=object())
            self.assertFalse((Path(tmp)/'pf_auto/pf_completion/RESPONSE.json').exists())
        self.assertIs(current, context)
        self.assertIs(unchanged_routes, routes)
        self.assertEqual(status['status'], 'PREPARATION_INCOMPLETE')
        self.assertIn('PF_COMPILE', status['reason'])

    def test_pf_and_pr_prepare_independently_and_each_ready_method_searches_early(self):
        context,spec,problem,executor=self.inputs();runtime=object()
        first_pr=threading.Event();pf_done=threading.Event();pf_searched=threading.Event();pr_done=threading.Event()
        calls=[];producer_contexts=[]
        def pf_prepare(root,p,c,s,run,routes,*,stage_executor):
            self.assertIs(stage_executor,runtime);producer_contexts.append(c)
            self.assertTrue(first_pr.wait(3),'PR READY was blocked behind PF preparation')
            c['candidates']=[candidate('FULL_PF','pf-new',True),candidate('TRUNCATED_PF','b0i-new',True),candidate('PR_PAULI','WRONG_PF_COPY_PR',True)]
            c['expected'].update({'pf-new':{'binding':'NEW_PF'},'b0i-new':{'binding':'NEW_B0I'},'WRONG_PF_COPY_PR':{}})
            c['pf_evidence_registry']={'ARTIFICIAL':True};c['generation']['TRUNCATED_PF']={'ARTIFICIAL':'PF_ONLY'}
            c['saved'].table={('b0i-new',(1,)):{'status':'ARTIFICIAL_SAVED'}}
            routes['TRUNCATED_PF']=[{'cid':'b0i-new'}]
            pf_done.set();return c,routes,{'status':'ARTIFICIAL_PF_READY'}
        def pr_prepare(root,p,c,s,run,*,completion,stage_executor):
            self.assertIs(completion.runtime,runtime);self.assertIs(stage_executor,runtime);producer_contexts.append(c)
            with completion.lock:completion.state['records']['pa']=dict(status='READY_FOR_SEARCH',method='PR_PAULI',configuration_id='pa',key='pa')
            self.assertTrue(pf_searched.wait(3),'PF READY was blocked behind the remaining PR preparation')
            with completion.lock:completion.state['records']['df']=dict(status='READY_FOR_SEARCH',method='PR_DF',configuration_id='df',key='df')
            c['candidates'].append(candidate('PR_DF','WRONG_PRODUCER_MUTATION',True));pr_done.set()
            return c,p['execution_routes'],completion
        def optimize(root,p,*,context,methods,initial,max_new,executor,extra_results,science_approved):
            self.assertEqual(max_new,spec['max_new_q3_per_method']);self.assertEqual(initial,'all');self.assertTrue(science_approved)
            self.assertEqual(extra_results,[{'ARTIFICIAL':'existing'}])
            calls.append((list(methods),pf_done.is_set(),pr_done.is_set()))
            if 'PR_PAULI'in methods:first_pr.set()
            if 'TRUNCATED_PF'in methods:pf_searched.set()
            return {m:{'stop_reason':'ARTIFICIAL_RANGE_END','new_calls':1}for m in methods}
        with tempfile.TemporaryDirectory()as tmp,patch.object(PR,'Completion',Completion),patch.object(PF,'prepare',pf_prepare),patch.object(PR,'prepare_for_search',pr_prepare):
            got,routes,pf_status,completion,early=R.prepare_inputs(ROOT,problem,context,spec,tmp,{m:[]for m in METHODS},executor,optimize,runtime,[{'ARTIFICIAL':'existing'}])
        self.assertEqual(set(early),set(METHODS));self.assertEqual(pf_status['status'],'ARTIFICIAL_PF_READY')
        self.assertTrue(any('PR_PAULI'in methods and not pf_ready and not pr_ready for methods,pf_ready,pr_ready in calls))
        self.assertTrue(any('TRUNCATED_PF'in methods and pf_ready and not pr_ready for methods,pf_ready,pr_ready in calls))
        ids={c['configuration_id']for c in got['candidates']}
        self.assertEqual(ids,{'pf-new','b0i-new','pa','df'})
        self.assertNotIn('WRONG_PF_COPY_PR',got['expected']);self.assertIn('pa',got['expected'])
        self.assertIsNot(producer_contexts[0]['candidates'],producer_contexts[1]['candidates'])
        self.assertIsNot(producer_contexts[0]['candidates'],got['candidates'])
        self.assertEqual(got['saved'].table,{('b0i-new',(1,)):{'status':'ARTIFICIAL_SAVED'}})
        self.assertEqual(executor.routes,routes);self.assertEqual(executor.expected,got['expected'])

    def test_fatal_early_result_suppresses_further_q3_and_drains_preparation(self):
        context,spec,problem,executor=self.inputs();runtime=object();calls=[];drained=[]
        def pf_prepare(root,p,c,s,run,routes,*,stage_executor):
            time.sleep(.04);c['candidates'].append(candidate('TRUNCATED_PF','b0i-late',True));drained.append('PF')
            return c,routes,{'status':'ARTIFICIAL_READY'}
        def pr_prepare(root,p,c,s,run,*,completion,stage_executor):
            time.sleep(.06)
            with completion.lock:completion.state['records']['late']=dict(status='READY_FOR_SEARCH',method='PR_DF',configuration_id='late',key='late')
            drained.append('PR');return c,p['execution_routes'],completion
        def optimize(root,p,**kwargs):
            calls.append(list(kwargs['methods']));return {m:{'stop_reason':'UNKNOWN_RESULT','new_calls':0}for m in kwargs['methods']}
        with tempfile.TemporaryDirectory()as tmp,patch.object(PR,'Completion',Completion),patch.object(PF,'prepare',pf_prepare),patch.object(PR,'prepare_for_search',pr_prepare):
            got,routes,pf_status,completion,early=R.prepare_inputs(ROOT,problem,context,spec,tmp,{m:[]for m in METHODS},executor,optimize,runtime,[])
        self.assertEqual(calls,[['FULL_PF']]);self.assertEqual(set(drained),{'PF','PR'})
        self.assertIn('late',{c['configuration_id']for c in got['candidates']})
        self.assertEqual(early['FULL_PF']['stop_reason'],'UNKNOWN_RESULT')

    def test_runtime_is_required_and_no_local_fallback_is_invoked(self):
        context,spec,problem,executor=self.inputs()
        with tempfile.TemporaryDirectory()as tmp,patch.object(PF,'prepare')as producer:
            with self.assertRaisesRegex(ValueError,'no local heavy fallback'):
                R.prepare_inputs(ROOT,problem,context,spec,tmp,{},executor,lambda *a,**k:None,None,[])
        producer.assert_not_called()

    def test_cli_continuation_retains_fatal_stop_when_another_method_needs_inputs(self):
        context,spec,problem,executor=self.inputs()
        executor.launched=lambda method:0
        for stop in ('UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'):
            with self.subTest(stop=stop),tempfile.TemporaryDirectory()as tmp:
                completion=Completion(tmp,{})
                result={method:dict(stop_reason=stop if method=='FULL_PF'else'NEEDS_INPUTS',
                    new_calls=0,input_requests=[{'ARTIFICIAL':'another missing candidate'}])for method in METHODS}
                with patch.object(PR,'prepare_for_search')as producer:
                    optimizer=unittest.mock.Mock(side_effect=AssertionError('Q3 after fatal STOP'))
                    got=PR.continue_selected_preparation(ROOT,problem,context,spec,Path(tmp),
                        completion,executor,result,optimizer=optimizer)
                self.assertIs(got,result);producer.assert_not_called();optimizer.assert_not_called()

    def test_pf_wave_resolves_price_when_receipt_arrives_and_replays_no_work(self):
        from trottertracks import pf_candidate_selection as selection
        from trottertracks.pf_saved_prices import receipt_locators

        def point(cid, method, prefix, width):
            return dict(configuration_id=cid,method=method,key=dict(p=prefix,formula='2nd',delta_over_h=width),
                        ready=False,needs=['PRICE_26AXIS','SIGNAL_13ROUND_PACKET'],
                        configuration_body=dict(id=cid,spec=dict(split=dict(original_PF_method='B0I' if prefix!=10 else 'FULL'))))

        points=[point('full','FULL_PF',10,'4'),point('hold','TRUNCATED_PF',7,'2'),
                point('same_family','TRUNCATED_PF',7,'1')]
        methods=['FULL_PF','TRUNCATED_PF']
        context=dict(candidates=points,catalog=dict(candidates=[]),pf_evidence_registry=dict(configurations={}))
        spec=dict(methods=methods,science_approved=True,pf_completion=dict(mode='complete',max_candidates=2,entry='fixture'))
        entered=threading.Barrier(2)
        calls=[]
        directory=None
        def complete(root,spec,run,*,stage_executor):
            cfg=spec['pf_completion']['selected'][0]['configuration_body']['id']
            calls.append(cfg)
            self.assertIn(cfg,str(run))
            if 'recheck-' not in str(run) and cfg!='same_family':entered.wait(timeout=5)
            held = cfg=='hold' and not receipt_locators(directory/'pf_auto')
            item=({'status':'PRICE_EVIDENCE_REQUIRED','reason':'fixed price plan has no admissible axes'}
                  if held else {'status':'READY_FOR_SAME_OPTIMIZER'})
            registry=dict(configurations={} if held else {cfg:dict(artificial=True)})
            return dict(registry_file=registry,response=dict(items=[item]))

        def refresh(root,problem,context,methods,registry,artificial):
            after=deepcopy(context);after['pf_evidence_registry']=registry
            for c in after['candidates']:
                if c['configuration_id'] in registry['configurations']:c.update(ready=True,needs=[])
            return after

        with tempfile.TemporaryDirectory() as tmp, patch.object(selection.PF,'complete_selected',side_effect=complete), \
             patch.object(selection,'_refresh',side_effect=refresh):
            run=Path(tmp)
            directory=run
            _,_,yielded=selection.prepare_next(ROOT,{},context,spec,run,{},object(),stop_requested=lambda:True)
            self.assertEqual(yielded['status'],'PREPARATION_WAVE_YIELDED')
            self.assertEqual(calls,[])
            after,_,first=selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertCountEqual(calls,['full','hold'])
            self.assertEqual(first['records']['hold']['status'],'PRICE_ACTION_UNAVAILABLE')
            self.assertEqual(set(after['pf_evidence_registry']['configurations']),{'full'})
            after,_,_=selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertEqual(calls[-1],'same_family')
            self.assertEqual(set(after['pf_evidence_registry']['configurations']),{'full','same_family'})
            selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertEqual(calls.count('hold'),2)  # One reconciliation of the legacy decision.
            _,_,done=selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertEqual(done['status'],'NO_INCOMPLETE_PF_CANDIDATE')
            self.assertEqual(calls.count('hold'),2)
            leaf=run/'pf_auto/candidates/full/pf_auto/pf_completion/provider_workspace'
            audit=leaf/'docs/research/distributed_selected_pf/full/PF_AUDIT.RESULT.json'
            audit.parent.mkdir(parents=True)
            audit.write_text(json.dumps(dict(status='AUDIT_PASS',prices=[
                dict(physical_id='circuit', gate='C_IMPLEMENTATION_GUARD_PASS',
                     price_evidence=dict(source=dict(commit='HEAD',path='PROJECT_MAP.md')))])))
            after,_,state=selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertEqual(state['records']['hold']['status'],'READY_RETURNED_TO_SAME_OPTIMIZER')
            self.assertTrue(all(c['ready'] for c in after['candidates']))
            selection.prepare_next(ROOT,{},context,spec,run,{},object())
            self.assertEqual(calls.count('hold'),3)

    def test_normal_preparation_resume_reconciles_existing_tasks_before_attempt_limit(self):
        with tempfile.TemporaryDirectory()as tmp:
            calls=[]
            class RecoveringCompletion:
                directory=Path(tmp);source='ARTIFICIAL_FIXED'
                state={'records':{}}
                def bind_stage_executor(self,runtime):self.stage_executor=runtime;return self
                def ingest_price_results(self):pass
                def refresh_price_hold(self,record):pass
                def price_deferred_keys(self):return {'unchanged-price'}
                def install(self,context,problem,records):
                    context['ready']=[r['key']for r in records if r['status']=='READY_FOR_SEARCH'];return {}
                def complete(self,request):
                    calls.append(deepcopy(request));record=self.state['records'][request['key']]
                    record.update(status='READY_FOR_SEARCH',configuration_id=request['key'],recoverable_preparation=False)
                    return record
            completion=RecoveringCompletion()
            for key,status,extra in [('selected','SELECTED',{}),
                    ('pending','WAITING_DISTRIBUTED_PREPARATION',{'recoverable_preparation':True}),
                    ('failed','SELECTED',{'scientific_failure':True}),('stop','STOP_NO_RETRY',{})]:
                completion.state['records'][key]=dict(key=key,method='PR_PAULI',status=status,
                    request={'key':key,'method':'PR_PAULI'},**extra)
            for key in ('new-price','unchanged-price'):
                completion.state['records'][key]=dict(key=key,method='PR_DF',status='WAITING_INPUTS',
                    signal_deferred_for_price=True,request={'key':key,'method':'PR_DF'})
            spec=dict(methods=['PR_PAULI','PR_DF'],science_approved=True,
                candidate_completion={'source':'ARTIFICIAL_FIXED','max_attempted_per_method':2})
            problem={'search':{},'methods':{'PR_PAULI':{},'PR_DF':{}}};context={}
            with patch.object(PR,'publish',return_value='ARTIFICIAL_FIXED'),patch('trottertracks.rpe_method_optimizer.optimize',side_effect=AssertionError('New selection after attempt cap')):
                got,_,_=PR.prepare_for_search(ROOT,problem,context,spec,Path(tmp),completion=completion,stage_executor=object())
                PR.prepare_for_search(ROOT,problem,context,spec,Path(tmp),completion=completion,stage_executor=object())
            self.assertCountEqual([r['key']for r in calls],['selected','pending','new-price'])
            self.assertEqual(got['ready'],['selected','pending','new-price'])
            self.assertTrue(completion.state['records']['failed']['scientific_failure'])
            self.assertEqual(completion.state['records']['stop']['status'],'STOP_NO_RETRY')

if __name__=='__main__':unittest.main()
