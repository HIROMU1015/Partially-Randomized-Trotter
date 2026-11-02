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
runner=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pr0_cosine.py'))
A,C,B=runner['A'],runner['C'],runner['B']


def contract():
    return dict(kind='H4_PR0_COSINE_ARB_ACTIVE_V1',base_commit=A.BASE_COMMIT,SOURCE='TOY_SOURCE',
        execution_authorized=True,dispatch_enabled=True,grant_valid=True,target_key='PR0_cosine',cases=['pauli_jw_PR'],axes=['cosine'],trajectory_id=A.P.TID,
        precision_schedule_bits=list(A.PRECISIONS),limits=A.LIMITS,max_state_actions=6,compiled_sha256=A.QPY_SHA,
        fingerprint=A.QPY_FP,state_hash=A.STATE_HASH,rho_budget_exact=A.GUARD,B_saved_exact=A.SCALE,forbidden_calls=A.FORBIDDEN,
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
        for key,value in [('axes',['sine']),('cases',['pauli_jw_PF']),('B_saved_exact','2'),('precision_schedule_bits',[128]),
                          ('max_state_actions',4),('rho_budget_exact','1'),('execution_authorized',False),('formal_certificate_accepted',True)]:
            c=copy.deepcopy(contract());c[key]=value
            with self.assertRaises(ValueError):A.validate_contract(c)

    def test_shared_selected_kernel_same_as_guarded_artificial_interface(self):
        fixture=runpy.run_path(str(ROOT/'tests/test_h4_arb_pr0_preparation.py'))
        _,_,p=fixture['fixture'](3);raw=fixture['raw'](3)
        expected,old=A.P.evaluate_target_artificial(p,raw,bits=64,fixture_id='TOY_PR0')
        seen=[];actual,new=A.P._selected_unitary_enclosure(p,raw,64,progress=lambda done,total:seen.append((done,total)))
        self.assertEqual(old,new);self.assertEqual(B._backend().endpoints(expected),B._backend().endpoints(actual))
        self.assertEqual(seen[-1],(2,2));self.assertEqual(new['real_H4_state_actions'],0)

    def test_parallel_artificial_pair_native_exits_common_norm_and_finite_scope(self):
        fixture=runpy.run_path(str(ROOT/'tests/test_h4_arb_pr0_preparation.py'))
        _,_,plan=fixture['fixture'](2);raw=fixture['raw'](2)
        native=runpy.run_path(str(ROOT/'tests/test_h4_arb_bridge.py'))
        program=native['program'](3,[native['gate']('sx',2),native['gate']('rz',2,theta=.125)])
        with tempfile.TemporaryDirectory() as folder:
            ns=Path(folder);budget=C.CallBudget(ns,A.LIMITS)
            pair=A.action_pair(ns,plan,program,raw,64,'TOY_SOURCE',budget)
            self.assertEqual(pair['target']['record']['raw_norm_exact'],pair['compiled']['record']['raw_norm_exact'])
            self.assertTrue(all(v['status']=='FINITE' for v in pair.values()))
            for role in ('target','compiled'):
                claim=C.read(ns/(role+'_64_worker_claim.json'));receipt=C.read(ns/(role+'_64_native_exit.json'))
                self.assertEqual(claim['pid'],receipt['pid']);self.assertTrue(receipt['reaped']);self.assertEqual(receipt['exit_code'],0)
            used=C.read(ns/'call_ledger.json')['used'];self.assertEqual(used['target_64'],1);self.assertEqual(used['compiled_64'],1)
            with self.assertRaises(ValueError):budget.take('target_64')

    def test_whole_intervals_difference_scales_B_once_and_preserves_raw_ball(self):
        target=dict(interval=dict(lower='1/4',upper='1/4'))
        compiled=dict(interval=dict(lower='1/2',upper='1/2'))
        rho=A.reduce_pair(target,compiled,bits=64,budget_exact='1',scale_exact='3/2')
        from fractions import Fraction as Q
        self.assertLessEqual(Q(rho['interval']['lower']),Q(3,8));self.assertGreaterEqual(Q(rho['interval']['upper']),Q(3,8))
        self.assertLess(Q(rho['interval']['upper']),Q(1,2));self.assertTrue(rho['B_applied_exactly_once'])
        self.assertFalse(rho['LOCAL_BUDGET_MET']);self.assertFalse(rho['formal_certificate_accepted'])

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
            g=dict(kind='H4_PR0_COSINE_ARB_ONE_SHOT',namespace=str(ns),SOURCE='TOY_SOURCE',one_shot=True,
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
            code="import runpy,sys; from pathlib import Path; r=runpy.run_path('scripts/run_h4_arb_pr0_cosine.py'); sys.exit(r['native_witness'](Path(sys.argv[1]),[sys.executable,'-B','-c','print(123)']))"
            r=subprocess.run([sys.executable,'-B','-c',code,str(directory)],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            native=C.read(directory/'worker_native_exit.json');stop=C.read(directory/'STOP.json')
            self.assertEqual(native['exit_code'],0);self.assertTrue(native['reaped']);self.assertTrue(stop['all_owned_reaped'])
            self.assertEqual(native['scope'],'NEW_H4_PR0_COSINE_ARB_WORKER_ONLY')

    def test_new_runtime_pins_actual_files_and_records_inherited_differences(self):
        data=runner['runtime']()
        self.assertGreater(data['reused_numeric_files'],0)
        C.hashes(data['files'])
        for p,change in data['prior_runtime_differences'].items():
            self.assertEqual(data['files'][p],change['current_sha256'])
            self.assertNotEqual(change['old_sha256'],change['current_sha256'])
            self.assertNotIn('/flint/',p);self.assertNotIn('/python_flint.libs/',p)


if __name__=='__main__':unittest.main(verbosity=2)
