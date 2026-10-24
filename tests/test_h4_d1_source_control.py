"""Artificial SOURCE-D1 authorization, QPY and stop-order checks only."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['d1_private']=pkg
from d1_private import h4_d1_source_control as D
C=D.C


def active():
    plan=C.read(D.DESIGN/'contract_DRAFT.json')
    return dict(plan,kind=D.KIND,execution_authorized=True,criterion_adopted=True,
                dispatch_enabled=True,grant_valid=True,call_limits=D.LIMITS,
                automatic_retry=False,D2_D3_authorized=False,N_G_authorized=False,
                science_SOURCE='TOY_SOURCE')


class SourceControlTests(unittest.TestCase):
    def test_draft_and_incomplete_adoption_rejected(self):
        with self.assertRaises(ValueError):D.scope(C.read(D.DESIGN/'contract_DRAFT.json'))
        c=active();D.scope(c)
        for key in ('criterion_adopted','execution_authorized','dispatch_enabled','grant_valid'):
            changed=copy.deepcopy(c);changed[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):D.scope(changed)

    def test_fixed_scope_and_phase_ownership_cannot_change(self):
        c=active()
        for key,value in [('maximum_cost_transpile_calls',10),('maximum_full_input_build_calls',1),
                          ('D2_D3_authorized',True),('N_G_authorized',True),('automatic_retry',True)]:
            changed=copy.deepcopy(c);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):D.scope(changed)
        for key,field,value in [('fixed','q',9),('ablation','event','drop phase'),
                               ('comparisons','interaction','clip to zero')]:
            changed=copy.deepcopy(c);changed[key][field]=value
            with self.subTest(key=key),self.assertRaises(ValueError):D.scope(changed)

    def test_no_full_build_or_refund_budget(self):
        with tempfile.TemporaryDirectory() as td:
            budget=C.CallBudget(td,D.LIMITS)
            with self.assertRaises(ValueError):budget.take('full_input_build')
            for _ in range(4):budget.take('full_QPY_validation')
            with self.assertRaises(ValueError):budget.take('full_QPY_validation')
            for _ in range(9):budget.take('cost_transpile')
            with self.assertRaises(ValueError):budget.take('cost_transpile')
            with self.assertRaises(ValueError):budget.take('sampling')

    def test_missing_authority_never_imports_quantum(self):
        previous={k for k in sys.modules if k.startswith('qiskit')}
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
            with self.assertRaises(FileNotFoundError):D.authority()
        self.assertEqual(previous,{k for k in sys.modules if k.startswith('qiskit')})

    def test_grant_consumed_once_stop_and_approval_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D,'verify_seal'):
            ns=Path(td)/'run01';ns.mkdir();(Path(td)/'approval.json').write_text('{}')
            c=active();s=dict(SOURCE='TOY_SOURCE',approval_sha256=C.sha(Path(td)/'approval.json'))
            C.save(Path(td)/'seal.json',s)
            c.update(seal_sha256=C.sha(Path(td)/'seal.json'),approval_sha256=s['approval_sha256'])
            C.save(Path(td)/'active_contract.json',c)
            g=dict(kind='H4_D1_SOURCE_SCIENCE_ONE_SHOT',SOURCE='TOY_SOURCE',namespace=str(ns),
                contract_sha256=C.sha(Path(td)/'active_contract.json'),seal_sha256=C.sha(Path(td)/'seal.json'),
                approval_sha256=s['approval_sha256'],call_limits=D.LIMITS,one_shot=True,automatic_retry=False)
            g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
            D.authority(consume=True)
            with self.assertRaises(ValueError):D.authority(consume=True)
            C.save(ns/'ENGINE_STOP.json',dict(status='TOY_STOP'))
            with self.assertRaises(ValueError):D.authority(task=True)
        # Real verify_seal checks approval SHA even on task mode; no real input.
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D.D,'relative_hashes'):
            (Path(td)/'approval.json').write_text('changed')
            s=dict(kind='D1_SOURCE_REUSE_QPY_SEAL_V1',python_sha256=C.sha(sys.executable),
                   code_files={},input_files={},approval_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'APPROVAL'):D.verify_seal(s,runtime=False,committed=False)

    def run_pipeline(self,*,stage_A_fails=False,price_matches=True):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D.D,'record_resources'):
            ns=Path(td)/'run01';ns.mkdir();c=active();calls=[]
            def a_gate(contract,namespace):
                if stage_A_fails:raise ValueError('TOY_UNRESOLVED_PHASE')
                C.save(namespace/'stage_A.json',dict(status='PASS_FULL4_ADOPTED_SOURCE_CORRESPONDENCE'))
            def fake_batch(namespace,kind,variants):
                calls.append((kind,list(variants)))
                if kind=='compile':
                    for variant in variants:
                        folder=namespace/'variants'/variant;folder.mkdir(exist_ok=True)
                        data=dict(RZ=15,CX=15,old_RZ_matches=price_matches)
                        C.save(folder/'metrics.json',data)
            with patch.object(D,'authority',return_value=(c,{},{})),patch.object(D,'stage_A',side_effect=a_gate),patch.object(D,'batch',side_effect=fake_batch):
                if stage_A_fails:
                    with self.assertRaises(ValueError):D.worker()
                else:self.assertEqual(D.worker(),0)
            return calls,C.read(ns/'ENGINE_STOP.json')

    def test_unresolved_stage_A_never_compiles(self):
        calls,stop=self.run_pipeline(stage_A_fails=True)
        self.assertFalse(calls);self.assertEqual(stop['status'],'SOURCE_D1_FAILED_NO_RETRY')
        self.assertEqual(stop['stage'],'A')

    def test_first_price_mismatch_stops_before_further_calls(self):
        calls,stop=self.run_pipeline(price_matches=False)
        self.assertEqual(calls,[('compile',[D.FULL[0]])])
        self.assertEqual(stop['stage'],'B');self.assertEqual(stop['stage_C'],'NOT_EXECUTED')

    def test_full4_precede_five_isolation_targets(self):
        calls,stop=self.run_pipeline()
        self.assertEqual(calls[:4],[('compile',[v]) for v in D.FULL])
        self.assertEqual(calls[4][0],'build');self.assertEqual(calls[5][0],'compile')
        self.assertEqual(calls[4][1],calls[5][1]);self.assertEqual(len(calls[5][1]),5)
        self.assertEqual(stop['cost_compiles'],9)

    def toy_saved(self,td):
        from test_h4_d1_adapter import A,toy
        profile=active()['fixed']['compiler'];D.require_compiler(A.api(),profile)
        with self.assertRaisesRegex(ValueError,'COMPILER_PROFILE'):
            D.require_compiler(A.api(),dict(profile,optimization_level=2))
        model,join=toy('pauli_jw','PR');built=A.build(model,'PR',join)
        origin=Path(td)/'origin';origin.mkdir()
        data=A.save_input(built,origin);data.update(SOURCE='TOY_OLD')
        C.save(origin/'input.json',data)
        expected=dict(path=str(origin/'input.qpy'),sha256=C.sha(origin/'input.qpy'),
            D1_input_fingerprint=data['input_fingerprint'],legacy_input_fingerprint='f'*64)
        return A,expected

    def test_saved_QPY_reused_without_rebuild_and_historical_fail_kept(self):
        with tempfile.TemporaryDirectory() as td:
            A,expected=self.toy_saved(td);folder=Path(td)/'new';folder.mkdir()
            with patch.object(A,'build',side_effect=AssertionError('NO_FULL_BUILD')):
                result=D.adopt_QPY({},expected,folder,'TOY_NEW',A.api())
            self.assertTrue(result['current_D1_fingerprint_matches'])
            self.assertFalse(result['historical_fingerprint_matches'])
            self.assertEqual(result['historical_status'],'LEGACY_FINGERPRINT_MISMATCH')
            self.assertEqual(result['full_input_build_calls'],0)
            self.assertEqual(C.sha(folder/'input.qpy'),expected['sha256'])

    def test_QPY_current_integrity_failure_keeps_evidence_and_stops(self):
        with tempfile.TemporaryDirectory() as td:
            A,expected=self.toy_saved(td);expected['D1_input_fingerprint']='0'*64
            folder=Path(td)/'new';folder.mkdir()
            with self.assertRaisesRegex(ValueError,'CURRENT_QPY_FINGERPRINT_MISMATCH'):
                D.adopt_QPY({},expected,folder,'TOY_NEW',A.api())
            self.assertFalse(C.read(folder/'input.json')['current_D1_fingerprint_matches'])
            self.assertFalse((folder/'input.qpy').exists())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceControlTests))
    output=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
                scope='ARTIFICIAL_QPY_AND_AUTHORITY_ONLY_NO_REAL_H4_OR_COST_COMPILE')
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(output,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
