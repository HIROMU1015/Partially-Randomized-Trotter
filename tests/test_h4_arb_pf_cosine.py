"""Artificial production/authority fixtures only; no real H4 numerical call."""
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
runner=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pf_cosine.py'))
A,C,B=runner['A'],runner['C'],runner['B']


def contract():
    return dict(kind='H4_PF_COSINE_ARB_ACTIVE_V1',base_commit=A.BASE_COMMIT,SOURCE='TOY_SOURCE',
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,target_key='PF0_cosine',cases=['pauli_jw_PF'],axes=['cosine'],
        precision_schedule_bits=list(A.PRECISIONS),limits=A.LIMITS,max_state_actions=3,compiled_sha256=A.QPY_SHA,
        fingerprint=A.QPY_FP,state_hash=A.STATE_HASH,rho_budget_exact=A.GUARD,B_saved_exact='1',forbidden_calls=A.FORBIDDEN,
        mandatory_STOP=True,automatic_next_stage=False,formal_certificate_accepted=False,seal_sha256='TOY',approval_sha256='TOY')


class ProductionTests(unittest.TestCase):
    def test_decisive_64_stops_even_when_advisory_width_not_met(self):
        seen=[]
        def evaluate(bits):seen.append(bits);return dict(relation='UPPER_WITHIN_BUDGET',width_target_met=False)
        result=A.finite_schedule(evaluate)
        self.assertEqual(seen,[64]);self.assertEqual(result['relation'],'UPPER_WITHIN_BUDGET')

    def test_straddle_escalates_once_then_stops_on_exceeds(self):
        seen=[]
        def evaluate(bits):seen.append(bits);return dict(relation='UNDETERMINED' if bits==64 else 'LOWER_EXCEEDS_BUDGET')
        self.assertEqual(A.finite_schedule(evaluate)['relation'],'LOWER_EXCEEDS_BUDGET')
        self.assertEqual(seen,[64,96])

    def test_nonfinite_and_straddle_stop_inconclusive_after_128(self):
        seen=[]
        def evaluate(bits):seen.append(bits);return dict(relation='NONFINITE' if bits==64 else 'UNDETERMINED')
        self.assertEqual(A.finite_schedule(evaluate)['status'],'INCONCLUSIVE_PRECISION_EXHAUSTED')
        self.assertEqual(seen,[64,96,128])

    def test_unexpected_failure_does_not_silently_escalate(self):
        seen=[]
        def evaluate(bits):seen.append(bits);raise RuntimeError('TOY_INPUT_FAILURE')
        with self.assertRaises(RuntimeError):A.finite_schedule(evaluate)
        self.assertEqual(seen,[64])

    def test_changed_contract_scope_scale_or_precision_rejected(self):
        A.validate_contract(contract())
        for key,value in [('axes',['sine']),('cases',['pauli_jw_PR']),('B_saved_exact','2'),('precision_schedule_bits',[128]),
                          ('max_state_actions',4),('rho_budget_exact','1'),('execution_authorized',False),('formal_certificate_accepted',True)]:
            c=copy.deepcopy(contract());c[key]=value
            with self.assertRaises(ValueError):A.validate_contract(c)

    def test_same_shared_kernel_artificial_and_production_paths(self):
        spec=runpy.run_path(str(ROOT/'tests/test_h4_arb_bridge.py'))
        p=spec['program'](3,[spec['gate']('sx',2),spec['gate']('rz',2,theta=.125),spec['gate']('cx',2,0),spec['gate']('sx',2)],phase=.3)
        raw=[[.25,.125],[.5,-.25],[.375,0.],[1.,.5]]
        expected,_=B.evaluate_artificial(p,raw,bits=64,fixture_id='TOY_NATIVE')
        progress=[]
        actual,record=B._native_unitary_enclosure(p,raw,64,progress=lambda done,total:progress.append((done,total)))
        self.assertEqual(B._backend().endpoints(actual),B._backend().endpoints(expected))
        self.assertEqual(record['real_H4_state_actions'],0)
        self.assertEqual(progress[-1],(4,4))

    def test_direct_interval_difference_budget_branches_no_acceptance(self):
        backend=B._backend()
        for target,budget,relation in [('0','1','LOWER_EXCEEDS_BUDGET'),('1','1','UPPER_WITHIN_BUDGET')]:
            _,r=B._residual_interval(backend.real(1),dict(lower=target,upper=target),B_saved_exact='2',budget_exact=budget,bits=64)
            self.assertEqual(r['budget_relation'],relation);self.assertFalse(r['formal_certificate_accepted']);self.assertFalse(r['LOCAL_BUDGET_MET'])
        _,r=B._residual_interval(backend.real(1),dict(lower='0',upper='1'),B_saved_exact='2',budget_exact='1',bits=64)
        self.assertEqual(r['budget_relation'],'UNDETERMINED')

    def test_nonrefundable_call_reserved_only_once(self):
        with tempfile.TemporaryDirectory() as folder:
            ns=Path(folder);budget=C.CallBudget(ns,A.LIMITS);budget.take('compiled_64')
            with self.assertRaises(ValueError):budget.take('compiled_64')
            used=C.read(ns/'call_ledger.json')['used'];self.assertEqual(used['compiled_64'],1)
            self.assertEqual(sum(used.values()),1)

    def test_grant_consumption_namespace_and_STOP_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);ns=base/'run01';ns.mkdir()
            seal=dict(SOURCE='TOY_SOURCE',bound_manifests={'approval.md':'TOY'})
            C.save(base/'seal.json',seal);c=contract();c['seal_sha256']=C.sha(base/'seal.json');C.save(base/'active_contract.json',c)
            g=dict(kind='H4_PF_COSINE_ARB_ONE_SHOT',namespace=str(ns),SOURCE='TOY_SOURCE',one_shot=True,
                contract_sha256=C.sha(base/'active_contract.json'),seal_sha256=C.sha(base/'seal.json'),approval_sha256='TOY')
            g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
            env=runner['check'].__globals__
            with patch.dict(env,BASE=base,verify=lambda _:None):
                runner['check'](consume=True)
                with self.assertRaises(FileExistsError):runner['check'](consume=True)
                runner['check']()
                C.save(ns/'STOP.json',dict(status='TOY_STOP'))
                with self.assertRaisesRegex(ValueError,'ALREADY_STOPPED'):runner['check']()

    def test_native_wait_reap_and_scope_label_on_artificial_process(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'native'
            code="import runpy,sys; from pathlib import Path; r=runpy.run_path('scripts/run_h4_arb_pf_cosine.py'); sys.exit(r['native_witness'](Path(sys.argv[1]),[sys.executable,'-B','-c','print(123)']))"
            r=subprocess.run([sys.executable,'-B','-c',code,str(directory)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            native=C.read(directory/'worker_native_exit.json');stop=C.read(directory/'STOP.json')
            self.assertEqual(native['exit_code'],0);self.assertTrue(native['reaped']);self.assertTrue(stop['all_owned_reaped'])
            self.assertEqual(native['scope'],'NEW_H4_PF_COSINE_ARB_WORKER_ONLY')


if __name__=='__main__':unittest.main(verbosity=2)
