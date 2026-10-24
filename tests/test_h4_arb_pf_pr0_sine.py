"""Artificial interval/projection/authority tests; no real H4 propagation."""
import copy
from fractions import Fraction as Q
import math
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
runner=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pf_pr0_sine.py'));A,C,B=runner['A'],runner['C'],runner['B']
F=runpy.run_path(str(ROOT/'tests/test_h4_arb_pr1_sine.py'));N=F['N'];backend=B._backend()


def contract(source='TOY_SOURCE'):
    return dict(kind='H4_PF_PR0_SINE_ARB_ACTIVE_V1',base_commit=A.BASE_COMMIT,SOURCE=source,
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,cases=list(A.KEYS),axes=['sine'],max_state_actions=9,
        precision_schedule_bits=list(A.PRECISIONS),targets={key:A.case_contract(key,source) for key in A.KEYS},
        forbidden_calls=A.FORBIDDEN,mandatory_STOP=True,automatic_next_stage=False,formal_certificate_accepted=False,
        seal_sha256='TOY',approval_sha256='TOY')


def authority_fixture(base):
    ns=base/'run01';ns.mkdir();seal=dict(SOURCE='TOY_SOURCE',bound_manifests={'approval.md':'TOY'})
    C.save(base/'seal.json',seal);c=contract();c['seal_sha256']=C.sha(base/'seal.json');C.save(base/'active_contract.json',c)
    g=dict(kind='H4_PF_PR0_SINE_ARB_ONE_SHOT',namespace=str(ns),SOURCE='TOY_SOURCE',one_shot=True,
        contract_sha256=C.sha(base/'active_contract.json'),seal_sha256=C.sha(base/'seal.json'),approval_sha256='TOY')
    g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
    return ns,c,g


class PairSineTests(unittest.TestCase):
    def test_PF_whole_saved_imag_interval_not_real_or_midpoint(self):
        result=dict(implemented_law=dict(PF=dict(corrected_encoded_mean=dict(real=dict(lower='10',upper='20'),imag=dict(lower='-1',upper='3')))),
            state_norm=dict(lower='2',upper='2'))
        record=A.quote_pf_interval(result,Q(2))
        self.assertEqual(record['normalized_interval_exact'],['-1/2','3/2'])
        self.assertEqual(record['saved_raw_interval'],dict(lower='-1',upper='3'));self.assertEqual(record['new_C_target_state_actions'],0)
        _,rho=B._residual_interval(backend.real(0),dict(zip(('lower','upper'),record['normalized_interval_exact'])),B_saved_exact='1',budget_exact='1',bits=64)
        self.assertEqual(rho['budget_relation'],'UNDETERMINED')
        self.assertGreaterEqual(Q(rho['interval']['upper']),Q(3,2))

    def test_PF_norm_excluding_raw_and_reversed_imag_rejected(self):
        result=dict(implemented_law=dict(PF=dict(corrected_encoded_mean=dict(imag=dict(lower='1',upper='2')))),state_norm=dict(lower='2',upper='2'))
        with self.assertRaises(ValueError):A.quote_pf_interval(result,1)
        result['implemented_law']['PF']['corrected_encoded_mean']['imag']=dict(lower='2',upper='1')
        with self.assertRaises(ValueError):A.quote_pf_interval(result,2)

    def test_PR0_imag_uses_same_unitary_dense_reference_tiny_systems(self):
        for n in (2,3):
            model,join,_=F['F']['fixture'](n,orders=(0,0,0,0))
            p=A.PR_PLAN.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PAIR');raw=F['F']['raw'](n)
            reference=F['dense_imag'](model,join,raw)
            for bits in (64,96,128):
                signal,r=A.PR_PLAN._selected_unitary_enclosure(p,raw,bits)
                self.assertTrue(signal.contains(reference));self.assertEqual(r['product_applications'],0)
                self.assertFalse(r['normalization_B_applied_to_target']);self.assertFalse(r['PR_law_mean_used'])

    def test_native_Sdg_positive_imag_and_PR0_B_once(self):
        model,join,_=F['F']['fixture'](orders=(0,),q=1,r=1,blocks=False,scalar=(.25,),components=[dict(label='IZ',sign=1,coefficient_abs=.2)])
        p=A.PR_PLAN.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PAIR');raw=F['F']['raw'](2)
        def h():return [N.gate('rz',2,theta=math.pi/2),N.gate('sx',2),N.gate('rz',2,theta=math.pi/2)]
        gates=[*h(),N.gate('rz',0,theta=.125),N.gate('cx',2,0),N.gate('rz',0,theta=-.125),N.gate('cx',2,0),
            N.gate('rz',2,theta=.25),N.gate('rz',2,theta=-math.pi/2),*h()]
        program=N.program(3,gates,phase=.125,axis='sine');_,target=A.PR_PLAN._selected_unitary_enclosure(p,raw,64)
        _,compiled=B._native_unitary_enclosure(program,raw,64)
        rho=A.CASES['PR0_sine'].reduce_pair(target,compiled,bits=64,budget_exact='1/1000000',scale_exact='3/2')
        self.assertEqual(rho['budget_relation'],'UPPER_WITHIN_BUDGET');self.assertEqual(rho['B_saved_exact'],'3/2')
        wrong=copy.deepcopy(program);wrong['instructions'][8]['params_exact']=[str(Q(math.pi/2))]
        _,bad=B._native_unitary_enclosure(wrong,raw,64)
        self.assertEqual(A.CASES['PR0_sine'].reduce_pair(target,bad,bits=64,budget_exact='1/1000000',scale_exact='3/2')['budget_relation'],'LOWER_EXCEEDS_BUDGET')

    def test_two_target_contract_limits_axis_budget_and_target_set(self):
        A.validate_contract(contract())
        for kind in ('axis','case','total','PF_limit','PR0_limit','budget','PF_scale','grant'):
            c=contract()
            if kind=='axis':c['axes']=['cosine']
            if kind=='case':c['cases']=['PR1_sine']
            if kind=='total':c['max_state_actions']=10
            if kind=='PF_limit':c['targets']['PF_sine']['max_state_actions']=4
            if kind=='PR0_limit':c['targets']['PR0_sine']['max_state_actions']=7
            if kind=='budget':c['targets']['PF_sine']['rho_budget_exact']=A.PF_OLD.GUARD
            if kind=='PF_scale':c['targets']['PF_sine']['B_saved_exact']='2'
            if kind=='grant':c['grant_valid']=False
            with self.subTest(kind=kind),self.assertRaises(ValueError):A.validate_contract(c)

    def test_each_target_schedule_stops_independently(self):
        pf_seen=[];pr_seen=[]
        def pf(bits):pf_seen.append(bits);return dict(relation='UPPER_WITHIN_BUDGET',preferred_width_target_met=False)
        def pr(bits):pr_seen.append(bits);return dict(relation='NONFINITE' if bits==64 else 'LOWER_EXCEEDS_BUDGET')
        self.assertEqual(A.CASES['PF_sine'].finite_schedule(pf)['relation'],'UPPER_WITHIN_BUDGET')
        self.assertEqual(A.CASES['PR0_sine'].finite_schedule(pr)['relation'],'LOWER_EXCEEDS_BUDGET')
        self.assertEqual(pf_seen,[64]);self.assertEqual(pr_seen,[64,96])

    def test_precision_exhaustion_and_unexpected_failure_no_retry(self):
        for key in A.KEYS:
            seen=[]
            def unresolved(bits):seen.append(bits);return dict(relation='UNDETERMINED')
            self.assertEqual(A.CASES[key].finite_schedule(unresolved)['status'],'INCONCLUSIVE_PRECISION_EXHAUSTED')
            self.assertEqual(seen,[64,96,128])
            def fail(_):raise RuntimeError('TOY_UNEXPECTED')
            with self.assertRaises(RuntimeError):A.CASES[key].finite_schedule(fail)

    def test_call_caps_nonrefundable_one_each_no_PF_target_key(self):
        for key in A.KEYS:
            with tempfile.TemporaryDirectory() as folder:
                b=C.CallBudget(Path(folder),A.CASES[key].LIMITS);b.take('compiled_64')
                with self.assertRaises(ValueError):b.take('compiled_64')
                if key=='PF_sine':
                    with self.assertRaises(ValueError):b.take('target_64')
                else:b.take('target_64')
                self.assertEqual(C.read(Path(folder)/'call_ledger.json')['used']['compiled_64'],1)

    def test_one_parent_grant_consumption_and_STOP_gate(self):
        env=runner['controller']()
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);ns,c,g=authority_fixture(base)
            with patch.dict(env['check'].__globals__,BASE=base,verify=lambda _:None):
                env['check'](consume=True)
                with self.assertRaises(FileExistsError):env['check'](consume=True)
                env['check']();C.save(ns/'STOP.json',dict(status='TOY_STOP'))
                with self.assertRaises(ValueError):env['check']()

    def test_delegation_requires_consumed_parent_and_cannot_authorize_PR1(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);ns,c,g=authority_fixture(base);(ns/'PF_sine').mkdir()
            with patch.object(A,'BASE',base):
                with self.assertRaises(FileNotFoundError):A.authorize_case(ns,'PF_sine',c['targets']['PF_sine'])
                C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id']));C.save(ns/'WORKER_CLAIM.json',dict(pid=1))
                proof=A.authorize_case(ns,'PF_sine',c['targets']['PF_sine'])
                self.assertTrue(proof['delegation_only']);self.assertFalse(proof['new_grant']);self.assertFalse(proof['new_consumption'])
                with self.assertRaises(ValueError):A.authorize_case(ns,'PR1_sine',c['targets']['PF_sine'])
                C.save(ns/'PF_sine'/'STOP.json',dict(status='TOY_STOP'))
                with self.assertRaises(ValueError):A.authorize_case(ns,'PF_sine',c['targets']['PF_sine'])

    def test_parallel_case_process_native_join_and_delegation_no_science(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);ns,c,g=authority_fixture(base);C.save(ns/'CONSUMED.json',dict(grant_id=g['grant_id']));C.save(ns/'WORKER_CLAIM.json',dict(pid=1))
            def fake_run(directory,case,budget,verify):
                pr=case['target_key']=='PR0_sine';budget.take('compiled_64')
                if pr:budget.take('target_64')
                C.save(directory/'result_summary.json',dict(state_action_attempts=2 if pr else 1,successful_state_actions=2 if pr else 1,residual_reductions=0,tiers=[]))
            cases={key:SimpleNamespace(**{**vars(a),'run':fake_run}) for key,a in A.CASES.items()}
            with patch.object(A,'BASE',base),patch.object(A,'CASES',cases):A.run(ns,c,lambda:None)
            r=C.read(ns/'result_summary.json');self.assertEqual(r['state_action_attempts'],3)
            for key in A.KEYS:
                receipt=C.read(ns/(key+'_native_exit.json'));self.assertEqual(receipt['exit_code'],0);self.assertTrue(receipt['native_wait'] and receipt['reaped'])
                self.assertTrue((ns/key/'PARENT_AUTHORIZATION.json').is_file());self.assertFalse((ns/key/'CONSUMED.json').exists())

    def test_reused_native_witness_on_tiny_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'native'
            code="import runpy,sys; from pathlib import Path; r=runpy.run_path('scripts/run_h4_arb_pf_pr0_sine.py')['controller']();sys.exit(r['native_witness'](Path(sys.argv[1]),[sys.executable,'-B','-c','print(1)']))"
            result=subprocess.run([sys.executable,'-B','-c',code,str(directory)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr);receipt=C.read(directory/'worker_native_exit.json')
            self.assertEqual(receipt['exit_code'],0);self.assertTrue(receipt['reaped']);self.assertEqual(receipt['scope'],'NEW_H4_PF_PR0_SINE_ARB_WORKER_ONLY')


if __name__=='__main__':unittest.main(verbosity=2)
