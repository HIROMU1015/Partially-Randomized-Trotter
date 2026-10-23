"""Artificial authority, finite budget and owned native-process audit tests."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('controller_under_test',ROOT/'src/trotterlib/h4_certificate_control.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
CLI=ROOT/'scripts/run_h4_numerical_evidence.py'

def contract():
    return dict(cases=['pauli_jw_PF','pauli_jw_PR','df_native_PF','df_native_PR'],axes=['cosine','sine'],epsilon_sig=[.05,.01,.005,.001,.0005],precision_schedule_bits=[64,96,128],total_call_budget_max=dict(c.LIMITS,**{k:0 for k in c.ZERO}),fixed_model=dict(T=.8,q=8,r=4,K=4,product_formula='S2',full_dimension=256),mandatory_STOP=True,automatic_retry=False,execution_authorized=False,grant_valid=False,dispatch_enabled=False)

class Control(unittest.TestCase):
    def test_inactive_contract(self):c.validate_contract(contract())
    def test_no_grant_dispatch(self):
        with tempfile.TemporaryDirectory() as t:
            c.save(Path(t)/'contract.json',contract())
            with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):c.authorize(t)
            self.assertFalse((Path(t)/'CONSUMED.json').exists())
    def test_no_grant_issue(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);c.save(p/'contract.json',contract())
            with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):c.issue_grant(p/'contract.json',p/'absent',p/'not_created')
            self.assertFalse((p/'not_created').exists())
    def test_three_flags_required(self):
        for name in ['execution_authorized','grant_valid','dispatch_enabled']:
            d=contract();d.update(execution_authorized=True,grant_valid=True,dispatch_enabled=True);d[name]=False
            with self.assertRaisesRegex(ValueError,'NOT_AUTHORIZED'):c.validate_contract(d,active=True)
    def test_explicit_approval_required(self):
        d=contract();d.update(execution_authorized=True,grant_valid=True,dispatch_enabled=True)
        with self.assertRaisesRegex(ValueError,'APPROVAL'):c.validate_contract(d,active=True)
    def test_call_contract_unchanged(self):
        d=contract();d['total_call_budget_max']['reference_signal_enclosures']=7
        with self.assertRaisesRegex(ValueError,'CALL_CONTRACT'):c.validate_contract(d)
    def test_forbidden_science(self):
        d=contract();d['total_call_budget_max']['SCF']=1
        with self.assertRaisesRegex(ValueError,'FORBIDDEN'):c.validate_contract(d)
    def test_no_retry_contract(self):
        d=contract();d['automatic_retry']=True
        with self.assertRaisesRegex(ValueError,'STOP_CONTRACT'):c.validate_contract(d)
    def test_precision_contract(self):
        d=contract();d['precision_schedule_bits'].append(256)
        with self.assertRaisesRegex(ValueError,'PRECISION'):c.validate_contract(d)
    def test_budget_nonrefund(self):
        with tempfile.TemporaryDirectory() as t:
            b=c.CallBudget(t,{'x':1});b.take('x')
            with self.assertRaisesRegex(ValueError,'EXHAUSTED'):b.take('x')
            self.assertEqual(c.read(Path(t)/'call_ledger.json')['used']['x'],1)
    def test_zero_budget_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'EXHAUSTED'):c.CallBudget(t,{'SCF':0}).take('SCF')
    def test_unknown_budget_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'FORBIDDEN'):c.CallBudget(t).take('extra')
    def test_budget_change_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            c.CallBudget(t,{'x':1}).take('x')
            with self.assertRaisesRegex(ValueError,'CHANGED'):c.CallBudget(t,{'x':2}).take('x')
    def test_exclusive_claim(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'claim';c.save(p,{'used':True})
            with self.assertRaises(FileExistsError):c.save(p,{'used':False})
            self.assertTrue(c.read(p)['used'])
    def test_sealed_routing_match(self):
        seal=dict(routing={'legacy_root':'/frozen/root'},historical_scientific_SOURCE='OLD',input_manifest_sha256='input')
        c.validate_routing(dict(legacy_root='/frozen/root',historical_scientific_SOURCE='OLD',input_manifest_sha256='input'),seal)
    def test_sealed_routing_mismatch(self):
        with self.assertRaisesRegex(ValueError,'ROUTING_MISMATCH'):
            c.validate_routing(dict(legacy_root='/other/root'),dict(routing={'legacy_root':'/frozen/root'}))
    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input';p.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):c.hashes({str(p):'bad'})
    def test_missing_hash_input(self):
        with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):c.hashes({'/nonexistent/artificial-input':'bad'})
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input';p.write_bytes(b'x');link=Path(t)/'link';link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'HASH_MISMATCH'):c.hashes({str(link):c.sha(p)})
    def test_width_only_advancement(self):
        seen=[]
        def evaluate(bits):seen.append(bits);return {'status':'WIDTH_INSUFFICIENT' if bits==64 else 'TARGET_MET'}
        self.assertEqual(c.precision_schedule(evaluate)['status'],'TARGET_MET');self.assertEqual(seen,[64,96])
    def test_no_promotion_after_exception(self):
        seen=[]
        def evaluate(bits):seen.append(bits);raise ValueError('SOURCE_MISMATCH')
        with self.assertRaisesRegex(ValueError,'SOURCE_MISMATCH'):c.precision_schedule(evaluate)
        self.assertEqual(seen,[64])
    def test_no_promotion_model_unresolved(self):
        seen=[];c.precision_schedule(lambda bits:(seen.append(bits) or dict(status='MODEL_UNRESOLVED')));self.assertEqual(seen,[64])
    def test_no_promotion_target_impossible(self):
        seen=[];c.precision_schedule(lambda bits:(seen.append(bits) or dict(status='TARGET_NOT_MET')));self.assertEqual(seen,[64])
    def test_finite_exhaustion(self):
        seen=[];out=c.precision_schedule(lambda bits:(seen.append(bits) or dict(status='WIDTH_INSUFFICIENT')));self.assertEqual(seen,[64,96,128]);self.assertEqual(out['status'],'UNDETERMINED_PRECISION_EXHAUSTED')
    def test_worker_denied_before_import(self):
        with tempfile.TemporaryDirectory() as t:
            c.save(Path(t)/'contract.json',contract());p=subprocess.run([sys.executable,'-B',str(CLI),'--worker',t,'--representation','df_native'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertIn('C_EXECUTION_NOT_AUTHORIZED',p.stderr);self.assertNotIn('flint',p.stderr)
    def test_corrupt_contract_refused(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t)/'contract.json').write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):c.authorize(t)
            self.assertFalse((Path(t)/'CONSUMED.json').exists())
    def test_parallel_reservations_never_exceed_budget(self):
        with tempfile.TemporaryDirectory() as t:
            code='import importlib.util; s=importlib.util.spec_from_file_location("c",'+repr(str(ROOT/'src/trotterlib/h4_certificate_control.py'))+');c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c.CallBudget('+repr(t)+',{"x":6}).take("x")'
            processes=[subprocess.Popen([sys.executable,'-B','-c',code],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL) for _ in range(10)]
            statuses=[p.wait(timeout=10) for p in processes]
            self.assertEqual(statuses.count(0),6);self.assertEqual(c.read(Path(t)/'call_ledger.json')['used']['x'],6)
    def test_duplicate_or_stopped_authority(self):
        # Artificial mocked manifest/approval, no science grant issuance or call.
        with tempfile.TemporaryDirectory() as t:
            ns=Path(t);d=contract();d['output_namespace']=str(ns);g=dict(kind='C_SCIENCE_ONE_SHOT',namespace=str(ns),contract_sha256='contract',seal_sha256='seal');g['grant_id']=c.digest(g);d['technical_seal_sha256']='seal'
            for name,obj in [('contract.json',d),('grant.json',g),('seal.json',{})]:c.save(ns/name,obj)
            with patch.object(c,'validate_contract',side_effect=lambda x,**kw:x),patch.object(c,'verify_seal'),patch.object(c,'validate_routing'),patch.object(c,'sha',side_effect=lambda p:'contract' if Path(p).name=='contract.json' else 'seal'):
                c.authorize(ns)
                with self.assertRaisesRegex(ValueError,'ALREADY_CONSUMED'):c.authorize(ns)
                (ns/'CONSUMED.json').unlink();c.save(ns/'STOP.json',{'status':'ARTIFICIAL_STOP'})
                with self.assertRaisesRegex(ValueError,'STOPPED'):c.authorize(ns)

class Native(unittest.TestCase):
    def witness(self,path,program):
        command=[sys.executable,'-B','-c',program]
        return subprocess.Popen([sys.executable,'-B',str(CLI),'--witness',str(path),'--command-json',json.dumps(command)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    def finish(self,p):
        out,err=p.communicate(timeout=15);return p.returncode,err
    def test_native_zero_durable(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';p=self.witness(d,'pass');code,err=self.finish(p);self.assertEqual(code,0,err);self.assertEqual(c.read(d/'worker_native_exit.json')['exit_code'],0);self.assertTrue(c.read(d/'STOP.json')['all_owned_reaped'])
    def test_native_failure(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';p=self.witness(d,'raise RuntimeError("artificial failure")');code,err=self.finish(p);self.assertNotEqual(code,0);self.assertEqual(c.read(d/'worker_native_exit.json')['exit_code'],1);self.assertEqual(c.read(d/'STOP.json')['status'],'FAILED_NO_RETRY')
    def test_native_signal(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';p=self.witness(d,'import os,signal;os.kill(os.getpid(),signal.SIGKILL)');code,err=self.finish(p);self.assertNotEqual(code,0);self.assertEqual(c.read(d/'worker_native_exit.json')['exit_code'],-9)
    def test_descendant_reap(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';program='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"]);time.sleep(.1)'
            p=self.witness(d,program);code,err=self.finish(p);stop=c.read(d/'STOP.json');self.assertTrue(stop['all_owned_reaped']);self.assertGreaterEqual(len(stop['reaped_pids']),2)
    def test_supervisor_loss_keeps_native_record(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';pidfile=Path(t)/'witness_pid'
            cmd=[sys.executable,'-B',str(CLI),'--witness',str(d),'--command-json',json.dumps([sys.executable,'-B','-c','import time;time.sleep(.3)'])]
            program='import subprocess,time,pathlib; p=subprocess.Popen('+repr(cmd)+',start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);pathlib.Path('+repr(str(pidfile))+').write_text(str(p.pid));time.sleep(30)'
            supervisor=subprocess.Popen([sys.executable,'-B','-c',program])
            try:
                deadline=time.monotonic()+5
                while not (d/'worker_identity.json').exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue((d/'worker_identity.json').exists())
                supervisor.kill();supervisor.wait(timeout=5)
                while not (d/'STOP.json').exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(c.read(d/'worker_native_exit.json')['exit_code'],0)
                self.assertTrue(c.read(d/'STOP.json')['all_owned_reaped'])
            finally:
                if supervisor.poll() is None:supervisor.kill();supervisor.wait()
    def test_duplicate_native_namespace(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'worker';p=self.witness(d,'pass');self.assertEqual(self.finish(p)[0],0);before=c.sha(d/'worker_native_exit.json');p=self.witness(d,'raise Exception()');self.assertNotEqual(self.finish(p)[0],0);self.assertEqual(c.sha(d/'worker_native_exit.json'),before)

if __name__=='__main__':unittest.main()
