"""Finite four-method preparation orchestration, artificial metadata providers."""
from copy import deepcopy
from pathlib import Path
import sys
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

if __name__=='__main__':unittest.main()
