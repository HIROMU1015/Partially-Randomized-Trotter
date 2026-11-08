"""Artificial inheritance, exact numerical-path reuse, and remainder budgets."""
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import run_track_d_c8_1_completion as runner
boot=runner.reused.frozen.c2.old();boot.bootstrap(Path(os.environ["TRACK_D_ORIGIN"]))
from trottertracks import track_d_c8_1_completion as adapter
from trottertracks import track_d_c7_3_acceleration as fast
from trottertracks import c1_h6_state_action as primitive
from threadpoolctl import threadpool_limits
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

oldtests=boot.load_file("completion_old_fixture",ROOT/"tests/check_track_d_c8_1_handoff.py")


class CompletionTests(unittest.TestCase):
    def repair_fixture(self,directory):
        directory.mkdir()
        def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
        old=directory.parent/"old"
        bindings=[dict(path=str(p.relative_to(old)),sha256=adapter.evidence.sha(p)) for p in sorted(old.rglob('*')) if p.is_file()]
        write(directory/"SOURCE.json",dict(execution_source=adapter.REPAIR_SOURCE,fixed_inputs=dict(members=bindings)))
        write(directory/"result.json",dict(completed=4,pass_count=4,prior_preserved=True))
        write(directory/"STOP.json",dict(mandatory_stop=True));write(directory/"payload_index.json",dict(members=[]))
        for arm in adapter.ARMS:
            for axis in adapter.AXES:
                name=arm+"_"+axis;write(directory/(name+"_check.json"),dict(status="PASS"))
                p=directory/"local"/name;p.mkdir(parents=True)
                (p/"observable.npz").write_bytes(b"ARTIFICIAL NOT ARRAY")
                write(p/"runtime_private.json",dict(peak_RSS_KiB=1))

    def test_hash_inheritance_and_copies_preserve_prior_stops(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/"old";oldtests.HandoffTests().fixture(old)
            repair=root/"repair";self.repair_fixture(repair)
            packet=adapter.inheritance(old,repair);before=(repair/"STOP.json").read_bytes()
            adapter.verify(old,repair,packet)
            self.assertEqual((packet['new_sampling'],packet['new_raw_limit'],packet['new_compile_limit'],packet['new_postcheck_limit']),(0,7,28,28))
            out=root/"new";out.mkdir();adapter.inherit_repair(repair,out,packet)
            self.assertEqual((repair/"STOP.json").read_bytes(),before)
            self.assertEqual((out/"local/q64_sample00/reference_X_observable.npz").read_bytes(),b"ARTIFICIAL NOT ARRAY")
            (repair/"local/reference_X/observable.npz").write_bytes(b"CHANGED")
            with self.assertRaises(ValueError):adapter.verify(old,repair,packet)

    def test_failed_prior_post_is_not_inherited(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t)/"old";oldtests.HandoffTests().fixture(old)
            repair=Path(t)/"repair";self.repair_fixture(repair)
            (repair/"reference_X_check.json").write_text(json.dumps(dict(status="FAIL")))
            with self.assertRaises(ValueError):adapter.inheritance(old,repair)

    def test_repaired_pass_must_bind_to_same_compiled_qpy(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t)/"old";oldtests.HandoffTests().fixture(old)
            repair=Path(t)/"repair";self.repair_fixture(repair)
            (old/"local/q64_sample00/circuits/reference_X_compiled.qpy").write_bytes(b"DIFFERENT")
            with self.assertRaisesRegex(ValueError,'REPAIR_PASS_BOUND_TO_DIFFERENT_INPUT'):
                adapter.inheritance(old,repair)

    def test_progress_observes_all_columns_without_changing_values(self):
        qc=QuantumCircuit(3);qc.rz(.17,0);qc.sx(1);qc.cx(1,2);qc.global_phase=.123
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"progress.json"
            engine=adapter.install(2,path,existing_install=fast.install)
            self.assertFalse(engine.backend.options.fusion_enable)
            with threadpool_limits(limits=1):
                actual=primitive.StateActions(2).apply(qc,np.eye(8,dtype=complex))
                target=Operator(qc).data.T
            np.testing.assert_allclose(actual,target,atol=1e-12,rtol=0)
            row=json.loads(path.read_text());self.assertEqual((row['completed_columns'],row['total_columns']),(8,8))
            self.assertEqual(row['status'],'PROPAGATION_COMPLETE')
        adapter.PROGRESS=None

    def test_compile_memory_uses_completed_p0_peaks_not_raw_history(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old/local/q64_sample00';old.mkdir(parents=True)
            (old/'precompile_checks.json').write_text(json.dumps(dict(peak_RSS_KiB=100)))
            for i,value in enumerate((100,10,11,12)):
                (old/(str(i)+'_compile_stage.json')).write_text(json.dumps(dict(peak_RSS_KiB=value)))
            for name,value in [('reference_X',500),('reference_Y',499),('symmetric_directional_X',20),('symmetric_directional_Y',19)]:
                p=root/'repair/local'/name;p.mkdir(parents=True)
                (p/'runtime_private.json').write_text(json.dumps(dict(peak_RSS_KiB=value)))
            with patch.object(runner.reused.frozen.old.frozen,'memory_available',return_value=10**12):
                row=runner.resources(root/'old',root/'repair')
            self.assertEqual(row['compile_memory_estimate'],24*1024)
            self.assertEqual(row['raw_memory_estimate'],200*1024)
            self.assertEqual(row['post_memory_estimate'],1000*1024)

    def test_post_memory_uses_its_arm_and_preserves_compile_reservation(self):
        pending=[dict(stage='post',method='reference',memory_bytes=999),
                 dict(stage='post',method='symmetric_directional',memory_bytes=999),
                 dict(stage='compile',method='reference',memory_bytes=7)]
        def existing(allocator,cpu,memory,tasks,workers,active):return tasks
        result=adapter.allocation(existing,None,64,10000,pending,64,0,
            dict(reference_post_memory_estimate=100,direct_post_memory_estimate=10))
        self.assertEqual([p['memory_bytes'] for p in result],[100,10,7])

    def mock_pipeline(self,fail_gate=False):
        reused=runner.reused;calls=[];pairs=[];stop=Event()
        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,func,*args):
                f=Future()
                if func==reused.frozen.raw_worker:
                    sid=args[0]['sample_id'];calls.append((sid,'raw',args[3]));f.set_result((sid,dict(status='PASS',peak_RSS_KiB=1)))
                else:
                    sid,task,stage,_=args;calls.append((sid,stage,task[4]));f.set_result((sid,task[2]+'_'+task[3],stage,dict(status='PASS')))
                return f
        def paired(directory,q):
            pairs.append(directory.name)
            if fail_gate:raise ValueError('ARTIFICIAL_FINAL_GATE_FAIL')
        events=[dict(sample_id=f'q64_sample{i:02d}') for i in range(8)]
        raw={'q64_sample00':dict(status='PASS')}
        compiled={'q64_sample00':{m+'_'+a:dict(status='PASS') for m in adapter.ARMS for a in adapter.AXES}}
        resources=dict(numerical_threads=64,memory=dict(MemAvailable=10**12),raw_memory_estimate=1,post_memory_estimate=1,compile_memory_estimate=1)
        with patch.object(reused,'ProcessPoolExecutor',Pool),patch.object(reused,'write'),patch.object(reused.frozen.old.frozen,'memory_available',return_value=10**12),patch.object(reused.frozen.previous,'paired_postcheck',side_effect=paired):
            if fail_gate:
                with self.assertRaises(ValueError):reused.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),raw,compiled,set())
            else:reused.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),raw,compiled,set())
        return calls,pairs

    def test_sample00_final_gate_then_exact7raw28compile28post(self):
        calls,pairs=self.mock_pipeline()
        self.assertTrue(all(sid!='q64_sample00' for sid,_,_ in calls))
        self.assertEqual(sum(stage=='raw' for _,stage,_ in calls),7)
        self.assertEqual(sum(stage=='compile' for _,stage,_ in calls),28)
        self.assertEqual(sum(stage=='post' for _,stage,_ in calls),28)
        self.assertTrue(all(cpu==1 for _,stage,cpu in calls if stage=='compile'))
        self.assertEqual(len(pairs),8);self.assertEqual(pairs[0],'q64_sample00')

    def test_failed_sample00_final_gate_releases_no_task(self):
        calls,pairs=self.mock_pipeline(True)
        self.assertEqual(calls,[]);self.assertEqual(pairs,['q64_sample00'])


if __name__=='__main__':unittest.main()
