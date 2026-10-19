"""Artificial circuits/exact scalars only. Never restore a molecular state."""
from fractions import Fraction
import json,math,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
from unittest.mock import patch
from qiskit import QuantumCircuit,qpy
import mpmath

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h6_m5_phase_ledger_core as core
import h6_m5_phase_ledger_contract as c
import run_h6_m5_phase_ledger as runner
import inspect_h6_m5_phase_structure as structure


def toy_extractions():
    wrapper=QuantumCircuit(13,1);wrapper.p(.6,0);wrapper.measure(12,0)
    compiled=QuantumCircuit(13,1);compiled.global_phase=.3;compiled.measure(12,0)
    return core.extract(wrapper,'WRAPPER'),core.extract(compiled,'COMPILED')


class ExactLedgerTests(unittest.TestCase):
    def test_exact_sum_preserves_small_addend(self):
        values=[1e16,1.,-1e16]
        self.assertEqual(sum(map(core.exact_float,values),Fraction(0)),1)
        naive=0.
        for value in values:naive+=value
        self.assertEqual(naive,0.)  # Python3.12 sum itself has improved compensation.
        for bad in (float('nan'),float('inf'),True,1):
            with self.assertRaises(ValueError):core.exact_float(bad)

    def test_rational_roundtrip_and_noncanonical_refusal(self):
        f=Fraction(-113,128);self.assertEqual(core.restore_rational(core.rational_payload(f)),f)
        for bad in (dict(numerator='2',denominator='2'),dict(numerator='1',denominator='0')):
            with self.assertRaises(ValueError):core.restore_rational(bad)

    def test_exact_scalar_cancellation_and_no_acceptance(self):
        w,d=toy_extractions();r=core.evaluate(w,d,dict(ordinary=0.,extended=0.))
        self.assertEqual(core.restore_rational(r['B_exact']),0)
        self.assertTrue(all(v['branch_integer']=='0' and v['principal_delta_rad']=='0.0' for v in r['evaluations']))
        self.assertIsNone(r['physical_match_threshold']);self.assertIsNone(r['causal_classification'])
        self.assertFalse(r['acceptance_override']);self.assertTrue(r['old_STOP_remains_STOP'])

    def test_small_signed_phase_and_large_modulo(self):
        for f in (Fraction.from_float(1e-9),Fraction.from_float(-1e-9),Fraction(1),Fraction(-1)):
            r=core.one_precision(f,120)
            mp=mpmath.mp.clone();mp.dps=120;delta=mp.mpf(r['principal_delta_rad']);period=2*mp.pi/8192
            self.assertTrue(-period/2<=delta<period/2)
            self.assertLess(abs(mp.mpf(f.numerator)/f.denominator-int(r['branch_integer'])*period-delta),mp.mpf('1e-115'))

    def test_fixed_half_open_branch_near_boundary(self):
        below=math.pi/8192;above=math.nextafter(below,math.inf)
        self.assertEqual(core.one_precision(Fraction.from_float(below),120)['branch_integer'],'0')
        self.assertEqual(core.one_precision(Fraction.from_float(above),120)['branch_integer'],'1')

    def test_unstable_precision_or_branch_stops(self):
        w,d=toy_extractions();real=core.one_precision
        def unstable(base,digits):
            r=real(base,digits)
            if digits==80:r['branch_integer']='1'
            return r
        with patch.object(core,'one_precision',side_effect=unstable):
            with self.assertRaisesRegex(ValueError,'NOT_STABLE'):core.evaluate(w,d,dict(ordinary=0.,extended=0.))

    def test_saved_phase_difference_uses_exact_binary64(self):
        w,d=toy_extractions();r=core.evaluate(w,d,dict(ordinary=1e-9,extended=-1e-9))
        self.assertEqual(r['comparisons']['ordinary']['saved_common_phase_binary64_hex'],(1e-9).hex())
        self.assertTrue(r['comparisons']['ordinary']['ledger_minus_saved_common_phase_rad'].startswith('-'))
        self.assertFalse(r['branch_selected_to_fit_saved_phase'])


class AuthorizationTests(unittest.TestCase):
    def test_draft_refuses_before_seals_and_cli(self):
        with patch.object(c,'verify_seals',side_effect=AssertionError('late')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_SCALAR'):c.authorize({'authorized':False,'grant':{'valid':False}})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';p.write_text(json.dumps(dict(authorized=False,grant={'valid':False})))
            for mode in ('--execute','--supervise'):
                result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h6_m5_phase_ledger.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_SCALAR',result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[p])

    def test_scope_and_old_grant_cannot_expand(self):
        self.assertEqual(c.scope()['internal_decimal_precisions'],[80,120])
        self.assertEqual(c.scope()['new_statevector_replays'],0)
        cfg=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,grant=dict(valid=True,approval_reference='ARTIFICIAL',scope_id='OLD_REPLAY_GRANT'))
        with self.assertRaisesRegex(ValueError,'GRANT_EXACT_LEDGER_SCOPE'):c.authorize(cfg)

    def test_external_supervisor_reuses_code_with_private_globals(self):
        self.assertIs(runner.supervise.__code__,runner.observed.supervise.__code__)
        self.assertIs(runner.supervise.__globals__['c'],c)
        self.assertEqual(runner.supervise.__globals__['__file__'],str(ROOT/'scripts/run_h6_m5_phase_ledger.py'))
        self.assertIsNot(runner.observed.supervise.__globals__['c'],c)

    def test_no_replay_or_compile_calls(self):
        import ast
        for rel in ('scripts/h6_m5_phase_ledger_core.py','scripts/run_h6_m5_phase_ledger.py'):
            for node in ast.walk(ast.parse((ROOT/rel).read_text())):
                if isinstance(node,ast.Call):
                    name=getattr(node.func,'id',getattr(node.func,'attr',''))
                    self.assertNotIn(name,('evolve','transpile','to_matrix','load_vectors','simulate_statevector','decompose','normalize'))


class SpawnTests(unittest.TestCase):
    def run_toy(self,mismatch=False):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'run';inputs={};expected={}
            for kind in ('WRAPPER','COMPILED'):
                qc=QuantumCircuit(13,1)
                if kind=='WRAPPER':qc.p(.6,0)
                else:qc.global_phase=.3
                qc.measure(12,0)
                p=base/(kind+'.qpy')
                with p.open('wb') as f:qpy.dump(qc,f)
                inputs[kind.lower()+'_qpy']=c.seal(p,kind)
                expected[kind]=structure.summarize(qc,kind,13)
            if mismatch:expected['WRAPPER']['ordered_metadata_sha256']='0'*64
            p=base/'static.json';c.save(p,dict(circuits=expected));inputs['static_metadata']=c.seal(p,'static')
            p=base/'saved.json';c.save(p,{mode:dict(common_phase=0.) for mode in ('ordinary','extended')});inputs['saved_report']=c.seal(p,'saved')
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,runtime={'python':sys.executable},grant={'valid':True},
                run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'marker'),inputs=inputs,
                sealed_files=[dict(path=str(Path(mpmath.__file__).resolve()))],resources={'per_worker_RSS_proxy_bytes':1})
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(runner.observed,'fresh_resources',return_value={'workers':2}):
                code=runner.execute(cfg)
                with self.assertRaisesRegex(ValueError,'UNUSED_OUTPUT_ROOT'):runner.execute(cfg)
            witness=c.read(root/'completion_witness.json')
            return code,witness,c.read(root/'scalar_ledger_result.json') if (root/'scalar_ledger_result.json').exists() else None

    def test_real_scalar_path_two_workers_reaped(self):
        code,witness,result=self.run_toy()
        self.assertEqual(code,0);self.assertTrue(witness['all_reaped']);self.assertEqual(len(witness['native_workers']),2)
        self.assertEqual(witness['stage_entries'],['EXTRACT_COMPILED','EXTRACT_WRAPPER','SCALAR_LEDGER'])
        self.assertEqual(core.restore_rational(result['B_exact']),0);self.assertFalse(result['acceptance_override'])

    def test_metadata_mismatch_stops_before_scalar_ledger(self):
        code,witness,result=self.run_toy(True)
        self.assertEqual(code,1);self.assertTrue(witness['all_reaped']);self.assertIsNone(result)
        self.assertNotIn('SCALAR_LEDGER',witness['stage_entries']);self.assertFalse(witness['retry'])


if __name__=='__main__':unittest.main()
