"""Artificial data/process tests only. Never loads a molecular input."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pf_direct_compile_contract as c
import pf_direct_execution as old
import pf_direct_retry_contract as r
from pf_direct_retry_support import array_record,scalar_value
from observe_pf_direct_retry216 import observe_command
from run_pf_direct_execution import run


class LoaderTests(unittest.TestCase):
    def test_saved_array_format(self):
        import numpy as np
        samples=[np.array(1.25),np.array(2,dtype=np.int64),np.arange(12).reshape(3,4)[:,::2],
                 np.asfortranarray(np.arange(12).reshape(3,4)),np.array([1+2j]),np.array([],dtype='>f8')]
        for sample in samples:
            with self.subTest(shape=sample.shape,dtype=str(sample.dtype)):
                contiguous=np.ascontiguousarray(sample)
                self.assertEqual(array_record(sample,np),dict(shape=list(contiguous.shape),dtype=contiguous.dtype.str,
                    sha256=hashlib.sha256(contiguous.tobytes()).hexdigest()))
        self.assertEqual(array_record(samples[0],np)['shape'],[1])

    def test_object_array_rejected(self):
        import numpy as np
        with self.assertRaisesRegex(ValueError,'OBJECT_ARRAY'):array_record(np.array([{}],dtype=object),np)

    def test_saved_scalar_formats_and_once_only(self):
        for receipt,expected in [({'C':2.},3.25),({'scalar':7.},7.),
                ({'policy':'REUSED_SAVED_P21_FULL_FOCK_SCALAR','receipt':{'C':2.}},3.25)]:
            before=deepcopy(receipt)
            self.assertEqual(scalar_value(1.25,receipt),expected)
            self.assertEqual(receipt,before)

    def test_invalid_scalar_schemas(self):
        for receipt in [None,{}, {'C':True},{'C':float('nan')},{'scalar':float('inf')},
                {'policy':'other','receipt':{'C':1.}}, {'policy':'REUSED_SAVED_P21_FULL_FOCK_SCALAR','receipt':[]}]:
            with self.subTest(receipt=receipt):
                with self.assertRaises((ValueError,TypeError)):scalar_value(1.,receipt)


class ScopeTests(unittest.TestCase):
    def test_exact_scope_and_reuse(self):
        _,rows=r.expected_tasks()
        self.assertEqual(len(rows),216)
        self.assertEqual({s:sum(v['system']==s for v in rows) for s in {v['system'] for v in rows}},
            dict(H2_DF3=72,H4_DF10=122,H8_DF28=22))
        accepted=c.read_json(ROOT/r.OLD_PACKET/'accepted_measurements.json')['rows']
        self.assertEqual(len(accepted),572)
        self.assertFalse({v['task_id'] for v in rows}&{v['task_id'] for v in accepted})
        caps=c.call_caps(rows)
        self.assertEqual((caps['native_evolution'],caps['control_probe'],caps['wrapper_compile']),(108,432,216))
        for key in c.FORBIDDEN_CALLS:self.assertEqual(caps[key],0)

    def test_half_pair_rejected(self):
        proposals,rows=r.expected_tasks()
        with self.assertRaisesRegex(ValueError,'PAIRED_SCOPE'):c.select_pairs(proposals,[rows[0]['task_id']])

    def test_no_grant_before_scope_or_science(self):
        with patch.object(r,'expected_tasks',side_effect=AssertionError('scope accessed')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_RETRY216_GRANT_REQUIRED'):r.authorize({}, {})

    def test_publication_is_not_science(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_RETRY216_GRANT_REQUIRED'):
            r.authorize({},dict(action='PUBLICATION',approved_by_user=True,runtime_authorization=True))

    def test_metadata_import_does_not_load_scientific_modules(self):
        command="import sys;sys.path.insert(0,'scripts');import pf_direct_retry_contract,launch_pf_direct_retry216;assert not any(x in sys.modules for x in ('numpy','qiskit','pyscf','trotterlib'))"
        subprocess.run([sys.executable,'-B','-c',command],cwd=ROOT,check=True)


class NativeProcessTests(unittest.TestCase):
    def test_observer_real_exit_and_heartbeat(self):
        for code in (0,7):
            with tempfile.TemporaryDirectory(prefix='pf-retry-artificial-') as directory:
                witness=observe_command([sys.executable,'-B','-c',f'import time;time.sleep(.08);raise SystemExit({code})'],
                    directory,'fake',heartbeat_seconds=.01)
                self.assertEqual(witness['exit_code'],code)
                self.assertTrue(witness['reaped'])
                self.assertEqual(witness['cleanup_errors'],[])
                self.assertTrue((Path(directory)/'observer_heartbeat.jsonl').is_file())
                with self.assertRaises(ChildProcessError):os.waitpid(witness['pid'],os.WNOHANG)

    def test_detached_child_survives_launcher_exit(self):
        with tempfile.TemporaryDirectory(prefix='pf-retry-detached-artificial-') as directory:
            path=Path(directory);child=path/'child.py';done=path/'done.json';observation=path/'observed'
            child.write_text("import json,os,time\nfrom pathlib import Path\ntime.sleep(.3)\nPath("+repr(str(done))+
                ").write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),sid=os.getsid(0))))\n")
            command="import sys,os,json;sys.path.insert(0,'scripts');from launch_pf_direct_retry216 import launch_detached;print(json.dumps(dict(launcher=os.getpid(),sid=os.getsid(0),child=launch_detached("+repr([sys.executable,'-B',str(child)])+","+repr(str(observation))+"))),flush=True)"
            result=subprocess.run([sys.executable,'-B','-c',command],cwd=ROOT,capture_output=True,text=True,check=True)
            launch=json.loads(result.stdout)
            deadline=time.monotonic()+5
            while not done.exists() and time.monotonic()<deadline:time.sleep(.02)
            info=json.loads(done.read_text())
            self.assertEqual(info['pid'],launch['child'])
            self.assertEqual(info['sid'],info['pid'])
            self.assertNotEqual(info['sid'],launch['sid'])
            self.assertNotEqual(info['ppid'],launch['launcher'])

    def fake_coordinator(self,exitcode):
        with tempfile.TemporaryDirectory(prefix='pf-retry-coordinator-artificial-') as directory:
            private=Path(directory);seal=private/'sealed';seal.mkdir();worker=private/'fake_worker.py'
            pair=[dict(system='H2_DF3',candidate_id='B0_p1_q1_2nd',axis=axis,
                       task_id='H2_DF3:B0_p1_q1_2nd:'+axis) for axis in c.AXES]
            spec=dict(SOURCE='f'*40,selected_task_ids=[v['task_id'] for v in pair],call_caps=c.call_caps(pair),
                output_root=str(private/'run01/fake'),missing_saved_oracles=[],systems={'H2_DF3':dict(python=sys.executable)})
            old.save(seal/'fake_spec.json',spec);old.save(seal/'fake_grant.json',{})
            worker.write_text('''import argparse,json,sys
from pathlib import Path
sys.path.insert(0,"'''+str(ROOT/'scripts')+'''")
import pf_direct_compile_contract as c
p=argparse.ArgumentParser()
for name in ('spec','grant','system','candidate','output'):p.add_argument('--'+name)
x=p.parse_args();out=Path(x.output);out.mkdir()
if '''+str(exitcode)+''':raise SystemExit('''+str(exitcode)+''')
rows=[]
for axis in c.AXES:
 rows.append(dict(task_id=x.system+':'+x.candidate+':'+axis,axis=axis,compiler_hash=c.COMPILER,
  fingerprint_before='ARTIFICIAL',fingerprint_after='ARTIFICIAL',compiled_input_fingerprint='ARTIFICIAL',
  RZ=2,CX=1,RZ_depth=2,CX_depth=1,depth=3,size=3,posttranspile_gate_counts=dict(rz=2,cx=1),
  native_exit_code=None,reaped=False,compile_seconds=0.))
pair=[dict(system=x.system,candidate_id=x.candidate,axis=a) for a in c.AXES]
(out/'result.json').write_text(json.dumps(dict(rows=rows,calls=c.call_caps(pair))))
''')
            api=SimpleNamespace(GROUPS=('fake',),PRIVATE=private,ROOT=ROOT,authorize=lambda *_:(pair,pair),
                save=old.save,append=old.append,sha=old.sha,candidate_pairs=old.candidate_pairs,
                gate=lambda *_:dict(status='PASS'),resource_snapshot=lambda:dict(cpu_slots=2,effective_available_memory=1024),
                choose_admissions=old.choose_admissions)
            run(seal,execution_api=api,worker_script=str(worker),estimator=lambda _:1)
            terminal=c.read_json(private/'run01/terminal.json')
            summary=terminal['contracts']['fake']
            self.assertEqual(summary['accepted_axes'],2 if exitcode==0 else 0)
            self.assertEqual(summary['stopped_or_unperformed_axes'],0 if exitcode==0 else 2)
            self.assertTrue(summary['all_workers_reaped']);self.assertEqual(terminal['active_owned_workers'],0)
            output=Path(spec['output_root'])/'H2_DF3__B0_p1_q1_2nd'
            witness=c.read_json(output/'owned_worker_exit.json')
            self.assertEqual(witness['exit_code'],exitcode)
            self.assertTrue(witness['start_ticks'])
            with self.assertRaises(ChildProcessError):os.waitpid(witness['pid'],os.WNOHANG)
            events=[json.loads(v) for v in (Path(spec['output_root'])/'ledger.jsonl').read_text().splitlines()]
            self.assertEqual([v['event'] for v in events],['BEFORE_WORKER_RESERVED','WORKER_STARTED','NATIVE_REAP_AND_ACCEPTANCE'])

    def test_fake_coordinator_accepts_only_native_success(self):self.fake_coordinator(0)
    def test_fake_coordinator_stops_failed_child_without_retry(self):self.fake_coordinator(3)


if __name__=='__main__':unittest.main()
