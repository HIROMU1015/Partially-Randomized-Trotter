"""Synthetic stopped-file and mocked stage-dispatch checks; no scientific run."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_c7_3_accelerated as runner


class HandoffTests(unittest.TestCase):
    def fixture(self,old):
        def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
        ids=[f'q32_sample{i:02d}' for i in range(8)]
        write(old/'SOURCE.json',dict(execution_source=runner.OLD_SOURCE))
        write(old/'result.json',dict(source_commit=runner.OLD_SOURCE,status='STOP_FAILURE',planned_ids=ids,
            canonical_occurrence_sampling_calls_started=256,compile_calls_started=0,source_input_preserved=False))
        write(old/'STOP.json',dict(mandatory_stop=True));write(old/'failure.json',dict(error_type='KeyboardInterrupt'))
        for name in ['seed_plan.json','frozen_model.json']:write(old/name,dict(artificial=True))
        events=[]
        for sid in ids:
            d=old/'local'/sid;write(d/'precompile_checks.json',dict(status='PASS'))
            write(d/'structural_features_raw.json',dict(artificial=True))
            write(d/'conditions.json',dict(artificial=True));write(d/'fusion.json',dict(artificial=True))
            (d/'references.npz').write_bytes(b'ARTIFICIAL ONLY')
            event=old/'local/events'/(sid+'.json');write(event,dict(artificial=True))
            for i in range(10):
                p=d/'circuits'/f'artificial{i}_raw.qpy';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'NOT A QUANTUM PAYLOAD')
            events.append(dict(sample_id=sid,record_sha256=runner.sha(event)))
        write(old/'event_bindings.json',events)
        return events

    def test_inheritance_seals_raw_once_keeps_old_stop_and_detects_byte_changes(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t);self.fixture(old);before=(old/'STOP.json').read_bytes()
            packet=runner.inheritance_plan(old)
            self.assertEqual(packet['compile_remaining'],32);self.assertEqual(packet['sampling_remaining'],0)
            self.assertEqual(packet['raw_remaining'],[]);self.assertFalse(packet['original_source_input_preserved_field'])
            runner.verify_inheritance(old,packet)
            self.assertEqual((old/'STOP.json').read_bytes(),before)
            p=old/'local/q32_sample00/circuits/artificial0_raw.qpy';p.write_bytes(b'changed')
            with self.assertRaises(ValueError):runner.verify_inheritance(old,packet)

    def test_no_handoff_after_scientific_failure_or_started_incomplete_wrapper(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t);self.fixture(old);d=old/'local/q32_sample00'
            (d/'reference_X_compile.json').write_text(json.dumps(dict(status='STARTED',compile_calls=1)))
            with self.assertRaises(ValueError):runner.inheritance_plan(old)
            (d/'reference_X_compile.json').unlink()
            (d/'precompile_checks.json').write_text(json.dumps(dict(status='FAIL')))
            with self.assertRaises(ValueError):runner.inheritance_plan(old)

    def test_allocator_compile_one_thread_and_post_native_parallel_budget(self):
        tasks=[dict(stage='compile',sample_id='a',method='reference',axis='X',memory_bytes=1),
               dict(stage='post',sample_id='b',method='symmetric_directional',axis='X',memory_bytes=1)]
        result=runner.allocation(8,100,tasks,8,0)
        self.assertEqual([n for _,n in result],[1,7])
        self.assertIs(result[0][0],tasks[0]);self.assertIs(result[1][0],tasks[1])

    def test_mocked_pipeline_exactly32_compile32_post_without_raw_or_resampling(self):
        class Pool:
            calls=[]
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,*args):
                self.calls.append(args);sid,task,stage,stop=args;f=Future()
                f.set_result((sid,task[2]+'_'+task[3],stage,dict(status='PASS')));return f
        compiled={};events=[dict(sample_id=f'q32_sample{i:02d}') for i in range(8)]
        resources=dict(numerical_threads=64,memory={'MemAvailable':10000},
            reference_task_memory_estimate_bytes=1,direct_task_memory_estimate_bytes=1)
        stop=S(is_set=lambda:False,set=lambda:None)
        with tempfile.TemporaryDirectory() as t,patch.object(runner,'ProcessPoolExecutor',Pool),patch.object(runner.frozen,'memory_available',return_value=10000):
            runner.pipeline(events,dict(wrappers_reused=[]),Path(t),Path(t),resources,stop,Path(t),compiled)
        self.assertEqual(len(Pool.calls),64);self.assertEqual(sum(map(len,compiled.values())),32)
        done=set()
        for sid,task,stage,_ in Pool.calls:
            key=(sid,task[2],task[3])
            if stage=='compile':self.assertNotIn(key,done);done.add(key);self.assertEqual(task[4],1)
            else:self.assertIn(key,done)

    def test_stage_compile_calls_existing_compiler_once_and_rejects_repeat(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'circuits').mkdir();(d/'precompile_checks.json').write_text(json.dumps(dict(status='PASS')))
            calls=[]
            def compile_one(task):
                calls.append(task)
                (d/'reference_X_compile.json').write_text(json.dumps(dict(status='PASS',compile_calls=1)))
                (d/'circuits/reference_X_compiled.qpy').write_bytes(b'artificial')
            stub=S(bootstrap=lambda origin:None,compile_one=compile_one)
            stop=S(is_set=lambda:False,set=lambda:None)
            with patch.object(runner.frozen.c2,'old',return_value=stub):
                first=runner.stage_worker('a',(d,d,'reference','X',1),'compile',stop)
                second=runner.stage_worker('a',(d,d,'reference','X',1),'compile',stop)
            self.assertEqual(first[-1]['status'],'PASS');self.assertEqual(second[-1]['status'],'FAIL')
            self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
