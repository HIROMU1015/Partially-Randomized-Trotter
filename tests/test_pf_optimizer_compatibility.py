"""Saved metadata and artificial-only compatibility with e839's RPE scheduler."""
import ast
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_method_optimizer as O,pf_optimizer_connection as PF
from trottertracks import rpe_optimizer_concurrent as C,pf_candidate_space as P
from trottertracks.saved_rpe_cost_catalog import Reader


def enc(v):
    v=Fraction(v);return dict(numerator=v.numerator,denominator=v.denominator)


def loc(path):return dict(commit='ARTIFICIAL_SOURCE',path=path)


class FixtureReader:
    root=ROOT
    def __init__(self,documents):self.documents=documents
    def commit(self,value):return value
    def read(self,source):
        value=deepcopy(self.documents[source['path']])
        for key in source.get('selector',[]):value=value[key]
        return value
    def document(self,commit,path):return deepcopy(self.documents[path])


def metadata_fixture():
    cid='ARTIFICIAL_P0';physical={m:'P0_M%d'%m for m in range(13)}
    configuration=dict(id=cid,spec=dict(L_D=0,formula='2nd',delta_pf=enc(Fraction(1,5)),h=enc(Fraction(1,10)),
        binding=dict(rank=10,snapshot='SNAPSHOT',hamiltonian='H',state='STATE'),split=dict(original_PF_method='B0I')),
        rounds=[dict(m=m,identity=dict(m=m,p=0),law_binding=dict(law='ARTIFICIAL_DETERMINISTIC',p=0))for m in range(13)])
    packet=dict(configuration_id=cid,means=[dict(m=m,identity=r['identity'],law_binding=r['law_binding'],
        physical_candidate=dict(id=physical[m]),status='SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED')for m,r in enumerate(configuration['rounds'])])
    prices=dict(axes=[dict(m=m,axis=a,physical_id=physical[m],unit_price_RZ=enc(m+1 if a=='X'else 2*(m+1)),
        kind='ARTIFICIAL_CONDITIONAL_PRICE',price_evidence=loc('PRICE_SOURCE'))for m in range(13)for a in ('X','Y')])
    entry=dict(method='TRUNCATED_PF',p=0,formula='2nd',delta_over_h='2',configuration=loc('CONFIGURATION'),
        signal_packet=loc('SIGNAL'),prices=loc('PRICES'),q3_inputs=loc('Q3'),total_configuration_id='TOTAL',binding_id='BIND',
        price_scope='ARTIFICIAL_CONDITIONAL_ONLY')
    inputs=dict(cases=[dict(configuration_id=cid,configuration=entry['configuration'],signal_packet=entry['signal_packet'],
        total_view_id='TOTAL',binding_id='BIND')])
    registry=dict(schema=PF.SCHEMA,configurations={cid:entry})
    docs=dict(CONFIGURATION=configuration,SIGNAL=packet,PRICES=prices,Q3=inputs,
              INDEX=dict(candidates=[dict(method='B0I_P7',formula='2nd',requested_delta_over_h='2',
                                          configuration_id=cid,source_bound_13round_packet=None)]))
    problem=dict(methods=dict(TRUNCATED_PF=dict(family='PF',index_method='B0I_P7',formulas=['2nd'],delta_over_h_range=['1','4'],caps=[2]*13)),
        evidence=dict(pf_candidate_index=loc('INDEX')),pf_evidence_registry=registry,generator={},accuracy=dict(epsilon_E_decimal='.1'))
    return cid,physical,docs,problem


class MetadataCompatibility(unittest.TestCase):
    def test_normal_prepare_joins_own_zero_prefix_inputs_without_changing_seed_rule(self):
        cid,physical,docs,problem=metadata_fixture();reader=FixtureReader(docs)
        facts=dict(fragment_count=10,snapshot_sha256='SNAPSHOT',hamiltonian_hash='H',state_hash='STATE')
        generate=lambda caps,prices,budget,rule:dict(shots=list(caps),conditional_cost=O.cost(caps,prices))
        with patch.object(O,'hamiltonian_facts',return_value=facts), \
             patch.object(O,'load_catalog',return_value=(reader,dict(comparison={}),dict(candidates=[]))), \
             patch.object(O,'check_comparison'),patch.object(O,'pinned_function',return_value=generate), \
             patch.object(PF,'physical_ids',return_value=physical), \
             patch.object(PF,'complete_selected',side_effect=AssertionError('No acquisition in prepare')):
            context=O.prepare(ROOT,problem,generate_configurations=False)
        candidate=context['candidates'][0]
        self.assertEqual(candidate['configuration_id'],cid);self.assertEqual(candidate['key']['p'],0)
        self.assertTrue(candidate['ready']);self.assertEqual(candidate['prices'][:2],[Fraction(1),Fraction(2)])
        self.assertEqual(context['expected'][cid]['total_configuration_id'],'TOTAL')
        proposals,_=O.proposals([candidate],{},Fraction(1,100),dict(seed_rules=['B_PRICE'],large_step_headroom='9/10'),generate)
        self.assertEqual(proposals[0]['origin']['kind'],'GENERATOR_SEED')

    def test_requested_prefixes_are_metadata_missing_before_registry_hydration(self):
        _,_,docs,problem=metadata_fixture();problem['methods']['TRUNCATED_PF']['p_values']=[0,7,8,9,10]
        result=[]
        PF.enumerate_prefixes(ROOT,FixtureReader(docs),problem,result)
        self.assertEqual({c['key']['p']for c in result},{0,7,8,9})
        self.assertEqual(len({c['configuration_id']for c in result}),4)
        self.assertTrue(all(not c['ready'] and c['prices']is None for c in result))
        self.assertTrue(all(c['priority_evidence']['usage']=='PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION'for c in result))
        full=P.configure(ROOT,'B0I',10,'2nd','2')
        self.assertEqual(full,P.configure(ROOT,'FULL_PF',10,'2nd','2'))

    def test_saved_h4_registry_is_source_join_only_no_provider_call(self):
        reader=Reader(ROOT)
        registry=PF.load_registry(ROOT,reader,dict(commit='36ad751d4',path='docs/research/d_pf_on_demand_evidence_20261116/C_RESULT_JOIN.json'))
        problem=json.loads((ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json').read_text())
        result=[]
        with patch.object(P.C,'execute',side_effect=AssertionError('No H4 signal execution')), \
             patch.object(PF,'complete_selected',side_effect=AssertionError('No provider call')):
            expected=PF.hydrate(ROOT,reader,problem,result,registry)
        self.assertEqual(len(result),2)
        self.assertTrue(all(c['ready'] and len(c['prices'])==26 for c in result))
        self.assertEqual({c['key']['p']for c in result},{7,10})
        self.assertTrue(all(expected[c['configuration_id']]['binding_id']for c in result))

    def test_price_schedule_and_law_mismatch_are_rejected(self):
        _,physical,docs,problem=metadata_fixture()
        docs['SIGNAL']['means'][0]['law_binding']={'OTHER_LAW':True}
        with patch.object(PF,'physical_ids',return_value=physical),self.assertRaises(ValueError):
            PF.hydrate(ROOT,FixtureReader(docs),problem,[],problem['pf_evidence_registry'])

    def test_artificial_registry_is_explicit_and_catalog_default_rejects_it(self):
        _,physical,docs,problem=metadata_fixture();registry=problem['pf_evidence_registry'];registry['artificial']=True
        with self.assertRaises(ValueError):PF.load_registry(ROOT,FixtureReader(docs),registry)
        accepted=PF.load_registry(ROOT,FixtureReader(docs),registry,allow_artificial=True)
        with self.assertRaises(ValueError):PF.hydrate(ROOT,FixtureReader(docs),problem,[],accepted)
        result=[]
        with patch.object(PF,'physical_ids',return_value=physical):
            PF.hydrate(ROOT,FixtureReader(docs),problem,result,accepted,allow_artificial=True)
        self.assertTrue(result[0]['ready']);self.assertTrue(accepted['artificial'])

    def test_artificial_registry_only_uses_artificial_method_routes(self):
        _,_,_,problem=metadata_fixture();registry=problem['pf_evidence_registry']
        routes=dict(TRUNCATED_PF=[dict(kind='artificial',configuration_ids='*')])
        with self.assertRaises(ValueError):PF.registry_routes(routes,registry)
        registry['artificial']=True
        self.assertEqual(PF.registry_routes(routes,registry),routes)
        self.assertNotIn('templates',routes['TRUNCATED_PF'][0])
        mixed=dict(TRUNCATED_PF=routes['TRUNCATED_PF']+[dict(kind='pf_b_pf6',templates=[])])
        with self.assertRaises(ValueError):PF.registry_routes(mixed,registry)

    def test_metadata_plan_mode_needs_no_science_and_resumes_without_second_provider(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);entry=root/'fixture_entry.py'
            entry.write_text("import argparse,json,pathlib\n"
                "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');p.add_argument('--mode');a=p.parse_args()\n"
                "r=json.load(open(a.request));d=pathlib.Path(a.output).parent;c=d/'CALLS';c.write_text(str(int(c.read_text())+1)if c.exists()else'1')\n"
                "json.dump({'artificial':True,'mode':a.mode,'selected':r['selected'],'science_calls':0},open(a.output,'w'))\n")
            completion=dict(entry=str(entry),interpreter=sys.executable,mode='plan',selected=[dict(method='TRUNCATED_PF',p=8,formula='2nd',delta_over_h='2')],
                            registry_file=str(root/'registry.json'),output_root=str(root/'output'))
            spec=dict(methods=['TRUNCATED_PF'],science_approved=False,pf_completion=completion)
            first=PF.complete_selected(root,spec,root/'run');second=PF.complete_selected(root,spec,root/'run')
            self.assertEqual(first,second);self.assertEqual(first['mode'],'plan')
            self.assertEqual(first['response']['science_calls'],0)
            self.assertEqual((root/'run/pf_completion/CALLS').read_text(),'1')
            with self.assertRaises(ValueError):
                PF.complete_selected(root,dict(spec,pf_completion=dict(completion,mode='complete')),root/'no_science')
            with self.assertRaises(ValueError):
                PF.complete_selected(root,dict(spec,pf_completion=dict(completion,mode='other')),root/'invalid')

    def test_e839_shot_selection_functions_remain_unchanged(self):
        original=ast.parse(subprocess.check_output(['git','show','e839b0d3f:src/trottertracks/rpe_method_optimizer.py'],cwd=ROOT))
        current=ast.parse((ROOT/'src/trottertracks/rpe_method_optimizer.py').read_bytes())
        for name in ('proposals','method_state','run_method','run_by_configuration'):
            a=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name==name)
            b=next(n for n in current.body if isinstance(n,ast.FunctionDef)and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b),name)


class PriorOwnerCompatibility(unittest.TestCase):
    proposal=dict(configuration_id='A',vector=tuple([2]*26),cost=Fraction(1),origin=dict(kind='ARTIFICIAL'))
    rule=dict(kind='artificial',configuration_ids='*',interpreter=sys.executable,view_commit='ARTIFICIAL_SOURCE')

    def run_case(self,concurrent,running,old_rule=None,new_rule=None,documents=None,expected='IN_PROGRESS',completed=False):
        old_rule=deepcopy(old_rule or self.rule);new_rule=deepcopy(new_rule or old_rule)
        problem=dict(accuracy=dict(ARTIFICIAL=True),target=dict(energy_ha=0),rpe=dict(M=12))
        expected_ids=dict(A=dict(total_configuration_id='TOTAL',binding_id='BIND'))
        with tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);prior=O.Executor(ROOT,base/'B',dict(M=[old_rule]),expected_ids,problem).for_method('M')
            job=prior.job_dir(self.proposal);job.mkdir(parents=True)
            (job/'REQUEST.json').write_text(json.dumps(prior.request(self.proposal,job,old_rule),default=str))
            (job/'LAUNCH.json').write_text(json.dumps(dict(pid=123456789)))
            if completed:
                (job/'OUTCOME.json').write_text(json.dumps(dict(status=O.PASS,MSE_lower=enc(Fraction(1,1000)),MSE_upper=enc(Fraction(1,1000)),
                    identity=dict(configuration_id='A',allocation=[dict(m=m,N_X=2,N_Y=2)for m in range(13)],
                                  total_configuration_id='TOTAL',binding_id='BIND',contract_matches=True))))
            hashes=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in job.iterdir()}
            before=hashes();cls=C.ConcurrentExecutor if concurrent else O.Executor
            kwargs=dict(budgets={'M':0},science_approved=False)if concurrent else{}
            root=cls(ROOT,base/'D',dict(M=[new_rule]),expected_ids,problem,prior_run_dirs=[base/'B'],**kwargs)
            view=root.for_method('M')
            if documents:view.pf_template_reader=FixtureReader(documents)
            try:
                with patch.object(view,'alive',return_value=running), \
                     patch.object(view,'run_adapter',side_effect=AssertionError('No execute/collect on prior owner')):
                    self.assertTrue(view.has_job(self.proposal))
                    view.prefetch([self.proposal])
                    answer=view.answer(self.proposal,allow_new=True)
                    self.assertEqual(answer['status'],expected)
                    self.assertEqual(answer['calls'],0)
                    self.assertEqual(answer['source'],'PRIOR_ROUND_JOB')
                    if expected in ('IN_PROGRESS','STALE_JOB'):
                        self.assertTrue(answer['owner_review_required'])
                self.assertFalse(view.job_dir(self.proposal).exists())
                self.assertEqual(hashes(),before)
                self.assertEqual(view.launched('M'),0)
                if concurrent:self.assertEqual(root.metrics()['new_real_route_launches'],0)
            finally:root.finish()

    def test_legacy_and_concurrent_prior_running_dead_are_read_only(self):
        for concurrent in (False,True):
            for running in (False,True):
                with self.subTest(concurrent=concurrent,running=running):
                    self.run_case(concurrent,running,expected='IN_PROGRESS'if running else'STALE_JOB')

    def test_completed_prior_observation_is_reused_without_mutation(self):
        for concurrent in (False,True):
            with self.subTest(concurrent=concurrent):
                self.run_case(concurrent,False,expected=O.PASS,completed=True)

    def test_unrelated_pf_template_preserves_owner_changed_target_holds(self):
        old=dict(kind='pf_b_pf6',configuration_ids='*',view='FIXED_VIEW',view_commit='WORKER_SOURCE',
                 interpreter=sys.executable,templates=[dict(loc('TARGET'),inputs_selector=['inputs'])])
        unrelated=dict(old,templates=[loc('UNRELATED')]+old['templates'])
        changed=dict(old,templates=[loc('CHANGED_TARGET')])
        documents=dict(TARGET=dict(inputs=dict(cases=[dict(configuration_id='A')])),
                       UNRELATED=dict(cases=[dict(configuration_id='OTHER')]),CHANGED_TARGET=dict(cases=[dict(configuration_id='A')]))
        for concurrent in (False,True):
            with self.subTest(concurrent=concurrent):
                self.run_case(concurrent,True,old,unrelated,documents)
                self.run_case(concurrent,True,old,changed,documents,expected='STALE_JOB')
                self.run_case(concurrent,False,old,changed,documents,expected='STALE_JOB',completed=True)


if __name__=='__main__':unittest.main(verbosity=2)
