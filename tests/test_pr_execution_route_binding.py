"""Route/descriptor regressions. Synthetic workers only; no H4 numerical calls."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_candidate_completion as C, rpe_method_optimizer as O, rpe_optimizer_concurrent as Q
import check_pr_width_search as F
import run_rpe_schedule_search as E
import test_rpe_optimizer_concurrent as CF


class Reader:
    def __init__(self):self.docs={}
    def add(self,name,value):
        loc=dict(commit='ARTIFICIAL',path=name);self.docs[name]=deepcopy(value);return loc
    def read(self,loc):
        value=deepcopy(self.docs[loc['path']])
        for k in loc.get('selector',[]):value=value[k]
        return value


def saved_context(method='PR_PAULI',cid='ARTIFICIAL_LD4'):
    reader=Reader();target={'artificial_full_H_target':True};adopted={'loss':'ARTIFICIAL_EXACT_LOSS'}
    config=reader.add('config',dict(id=cid,spec=dict(L_D=4)))
    packet=reader.add('packet',{'configuration_id':cid})
    case=dict(original_configuration_id=cid,configuration=config,signal_packet=packet,
              binding_id='ARTIFICIAL_BINDING',total_configuration_id='ARTIFICIAL_TOTAL',evaluation_target=target)
    source=reader.add('case',dict(cases=[case],adopted_evaluation=adopted))
    contract=reader.add('contract',dict(cases=[dict(evaluation_target=target)],adopted_evaluation=adopted,
                                      statistical_contract={'artificial':True},options={'artificial':True}))
    point=dict(method=method,configuration_id=cid,configuration_source=config,signal_source=packet,
               case_source=source,binding_id=case['binding_id'],total_configuration_id=case['total_configuration_id'])
    candidate=dict(method=method,configuration_id=cid,ready=True,needs=[],signal_source=packet)
    context=dict(reader=reader,catalog=dict(candidates=[point]),candidates=[candidate],expected={cid:dict(
        binding_id=case['binding_id'],total_configuration_id=case['total_configuration_id'])})
    completion=C.Completion.__new__(C.Completion)
    def route(method,cid,configuration,packet,binding):
        return dict(kind='pauli_b_common_entry'if method=='PR_PAULI'else'prdf_d2_common_entry',
                    configuration_ids=[cid],contract_template=contract,configurations={cid:dict(
                        configuration=configuration,signal_packet=packet,binding_id=binding['binding_id'],
                        total_configuration_id=binding['total_configuration_id'])})
    completion.common_route=route
    return completion,context,reader


class RouteBinding(unittest.TestCase):
    def test_wildcard_does_not_cover_an_absent_template_case(self):
        reader=Reader();loc=reader.add('old',dict(inputs=dict(cases=[dict(configuration_id='other')])))
        rule=dict(kind='prdf_d2_split',configuration_ids='*',templates=[dict(loc,inputs_selector=['inputs'])])
        self.assertFalse(O.route_supports_configuration(rule,'new',reader))
        self.assertTrue(O.route_supports_configuration(rule,'other',reader))

    def test_common_route_requires_own_packet_and_bindings(self):
        rule=dict(kind='pauli_b_common_entry',configuration_ids='*',contract_template={'x':1},configurations={})
        self.assertFalse(O.route_supports_configuration(rule,'new',Reader()))

    def test_saved_catalog_routes_reuse_own_sources_and_build_both_axis_allocations(self):
        spec=importlib.util.spec_from_file_location('_route_binding_job',ROOT/'scripts/run_rpe_optimizer_q3_job.py')
        job=importlib.util.module_from_spec(spec);spec.loader.exec_module(job)
        for method in('PR_PAULI','PR_DF'):
            with self.subTest(method=method):
                comp,ctx,reader=saved_context(method);before=deepcopy(ctx['catalog']);cid=ctx['candidates'][0]['configuration_id']
                routes=comp.install_saved_routes(ctx,{method:[]});rule=routes[method][0]
                self.assertTrue(O.route_supports_configuration(rule,cid,reader))
                request=dict(route=rule,configuration_id=cid,request_id='ARTIFICIAL',N_X=list(range(1,14)),N_Y=list(range(2,15)))
                with patch.object(job,'git_json',side_effect=lambda view,loc:reader.read(dict(loc,selector=loc.get('inputs_selector',[])))):
                    case,inputs=job.build(ROOT,request)
                self.assertEqual(case['configuration'],before['candidates'][0]['configuration_source'])
                self.assertEqual(case['signal_packet'],before['candidates'][0]['signal_source'])
                self.assertEqual(case['allocation']['shots'][-1],dict(m=12,N_X=13,N_Y=14))
                self.assertEqual(inputs['adopted_evaluation'],reader.docs['contract']['adopted_evaluation'])
                self.assertEqual(ctx['catalog'],before)
                self.assertEqual(comp.install_saved_routes(ctx,routes),routes)

    def test_saved_catalog_route_rejects_conflicting_identity_target_or_contract(self):
        for mismatch in('binding','target','contract','configuration','signal','duplicate'):
            with self.subTest(mismatch=mismatch):
                comp,ctx,reader=saved_context()
                if mismatch=='binding':reader.docs['case']['cases'][0]['binding_id']='OTHER'
                if mismatch=='target':reader.docs['case']['cases'][0]['evaluation_target']={'different':True}
                if mismatch=='contract':reader.docs['case']['adopted_evaluation']={'different':True}
                if mismatch=='configuration':reader.docs['config']['id']='OTHER'
                if mismatch=='signal':ctx['candidates'][0]['signal_source']={'other':True}
                if mismatch=='duplicate':ctx['catalog']['candidates'].append(dict(ctx['catalog']['candidates'][0],binding_id='OTHER'))
                self.assertEqual(comp.install_saved_routes(ctx,{'PR_PAULI':[]}),{'PR_PAULI':[]})
                self.assertIn('execution_binding_hold',ctx['candidates'][0])

    def test_usable_saved_route_preserves_prior_job_key_without_metadata_rebinding(self):
        comp,ctx,reader=saved_context();rule=dict(kind='artificial',configuration_ids='*')
        with patch.object(comp,'common_route',side_effect=AssertionError('Do not replace usable route')):
            self.assertEqual(comp.install_saved_routes(ctx,{'PR_PAULI':[rule]}),{'PR_PAULI':[rule]})

    def test_unroutable_candidate_keeps_saved_best_and_other_candidate_advances(self):
        problem,ctx,routes=CF.fixture(methods=['PR_DF'],configurations=2,lookahead=0,steps=2)
        method='PR_DF';c0,c1=ctx['candidates'];routes[method][0]['configuration_ids']=[c1['configuration_id']]
        problem['search'].update(rule='METHOD_MINIMUM_PRIORITY_V2',kappa_neighbourhood={'outer':'ARTIFICIAL'})
        with tempfile.TemporaryDirectory()as tmp:
            ex=Q.ConcurrentExecutor(ROOT,Path(tmp),routes,{},problem,max_concurrent=2,budgets={method:1},science_approved=True)
            result=Q.optimize(ROOT,problem,context=ctx,executor=ex,methods=[method],max_new={method:1})[method]
            self.assertEqual(c0['execution_needs'],['Q3_ROUTE_CONFIGURATION_OR_TEMPLATE'])
            self.assertEqual(result['best']['configuration_id'],c0['configuration_id'])
            self.assertEqual(ex.launched(method),1)
            requests=list(Path(tmp).glob('jobs/*/*/REQUEST.json'))
            self.assertTrue(all(json.loads(p.read_text())['configuration_id']==c1['configuration_id']for p in requests))
            self.assertTrue(any(r['configuration_id']==c0['configuration_id']for r in result['input_requests']))
            self.assertNotIn(result['stop_reason'],('NO_ROUTE_ADAPTER','UNKNOWN_RESULT'))

    def test_legacy_pauli_descriptor_uses_saved_index_while_derived_keeps_envelope(self):
        for prefix in(1,2,3,5,4):
            with self.subTest(prefix=prefix),tempfile.TemporaryDirectory()as tmp:
                comp=C.Completion.__new__(C.Completion);comp.settings={};comp.state={'records':{}}
                source=({'commit':'fixed','path':'descriptor'}if prefix==4 else{'index':'fixed-index','context_sha256':'own-context'})
                prepared=dict(descriptor=dict(source=source,descriptor={'tail':'own'}))
                config=dict(id='ARTIFICIAL',spec=dict(L_D=prefix,binding={'source':'own'}))
                def call(*args,**kwargs):return dict(status='SIGNALS_READY')
                with patch.object(comp,'call',side_effect=call):comp.pauli_signals(config,prepared,Path(tmp))
                request=json.loads((Path(tmp)/'PAULI_SIGNAL_REQUEST.json').read_text())
                self.assertEqual(request['tail_descriptor']is None,prefix!=4)
                if prefix==4:self.assertEqual(request['tail_descriptor']['source'],source)
                self.assertEqual(prepared['descriptor']['source'],source)

    def test_unresolved_descriptor_refused_before_any_signal_stage(self):
        comp=C.Completion.__new__(C.Completion);comp.settings={};comp.state={'records':{}}
        config=dict(id='ARTIFICIAL',spec=dict(L_D=4,binding={}))
        for source in ({},{'commit':'fixed'}):
            with self.subTest(source=source),tempfile.TemporaryDirectory()as tmp,patch.object(comp,'call',side_effect=AssertionError('No acquisition')):
                with self.assertRaisesRegex(C.Held,'PAULI_DESCRIPTOR_SOURCE_UNRESOLVED'):
                    comp.pauli_signals(config,dict(descriptor=dict(source=source,descriptor={})),Path(tmp))

    def test_pauli_import_dependencies_checked_before_entry_invocation(self):
        import test_pr_pauli_candidate_signals as S
        W=S.W
        with tempfile.TemporaryDirectory()as tmp:
            request,state,calls,patches=S.SignalsWrapperTests().setup_fixture(tmp)
            missing=Path(request['workspace'])/'scripts/run_df_head_pauli_tail_ld3_signals.py'
            missing.unlink()
            from contextlib import ExitStack
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                with self.assertRaises(FileNotFoundError):W.run(request)
            self.assertEqual(calls,[])
            self.assertEqual(state['counter'],0)


class OrdinaryEntry(unittest.TestCase):
    def test_unbound_route_input_request_is_waiting_not_exhausted(self):
        from types import SimpleNamespace
        driver=E.WidthDriver.__new__(E.WidthDriver)
        driver.state=dict(cycle=0,points=[],families={})
        driver.a=SimpleNamespace(max_cycles=1,plan_only=False);driver.spec=dict(science_approved=True)
        driver.log=lambda **kwargs:None;driver.save=lambda:None
        driver.bootstrap_families=lambda:None;driver.plan=lambda:([],{})
        driver.complete_points=lambda:None
        driver.batch=lambda:{'PR_DF':dict(stop_reason='NO_CANDIDATE_LEFT_IN_RANGE',new_calls=0,
            input_requests=[dict(kind='CONFIGURATION_INPUTS_MISSING',needs=['Q3_ROUTE_CONFIGURATION_OR_TEMPLATE'])])}
        driver.loop()
        self.assertEqual(driver.state['stop']['reason'],'WAITING_FOR_INPUTS')

    def test_catalog_route_recovery_actual_multwidth_cli_concurrent_and_resume(self):
        recovered=[]
        class Completion(F.ArtificialCompletion):
            def install(self,context,problem,records):
                routes=super().install(context,problem,records)
                for method in('PR_PAULI','PR_DF'):
                    rules=list(routes[method]);routes[method]=[]  # reproduce catalog-only route loss
                    for rule in rules:
                        cid=rule['configuration_ids'][0]
                        comp,fixture,reader=saved_context(method,cid)
                        # Acquisition and Q3 numerics are artificial. The saved
                        # metadata reconciliation itself is the production method.
                        comp.common_route=lambda *args,r=rule:dict(deepcopy(r),contract_template=dict(commit='ARTIFICIAL',path='contract'))
                        fixture['candidates'][0]['signal_source']=fixture['catalog']['candidates'][0]['signal_source']
                        routes=comp.install_saved_routes(fixture,routes)
                        recovered.append(cid)
                return routes
        class Driver(F.ArtificialDriver):
            def completion_api(self):
                if self.completion is None:self.completion=Completion(self.run)
                return self.completion
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);args=F.WidthSearchTests().files(root)
            round=json.loads((root/'ROUND.json').read_text())
            round['executor']=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1,prefix_cache_directory='prefix_cache')
            F.write(root/'ROUND.json',round)
            with redirect_stdout(io.StringIO()):first=E.main(args,driver_factory=Driver)
            result=json.loads((first.run/'ROUND_RESULT.json').read_text())
            self.assertTrue(recovered)
            self.assertTrue(all(m['best']for m in result['methods'].values()))
            self.assertTrue(all(m['stop_reason']not in('NO_ROUTE_ADAPTER','UNKNOWN_RESULT')for m in result['methods'].values()))
            launches={str(p):p.read_bytes()for p in first.run.glob('jobs/*/*/LAUNCH.json')}
            self.assertTrue(launches);stages=first.completion_api().state['stage_calls']
            with redirect_stdout(io.StringIO()):resumed=E.main(args,driver_factory=Driver)
            self.assertEqual(launches,{str(p):p.read_bytes()for p in resumed.run.glob('jobs/*/*/LAUNCH.json')})
            self.assertEqual(resumed.completion_api().state['stage_calls'],stages)


if __name__=='__main__':unittest.main(verbosity=2)
