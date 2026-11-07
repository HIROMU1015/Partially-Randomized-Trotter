"""Synthetic stage reservation and drained-file inheritance; zero H4 calls."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_track_d_c8_1_accelerated as runner
h=runner.handoff


class HandoffTests(unittest.TestCase):
    def fixture(self,old):
        def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
        write(old/'SOURCE.json',dict(execution_source=h.OLD_SOURCE))
        write(old/'result.json',dict(source_commit=h.OLD_SOURCE,source_input_preserved=True,
            canonical_occurrences_started=512,events_sealed=8,status='STOP_FAILURE',compile_calls_started=4))
        write(old/'STOP.json',dict(mandatory_stop=True));write(old/'failure.json',dict(error_type='KeyboardInterrupt'))
        for name in ('seed_plan.json','frozen_models.json'):write(old/name,dict(artificial=True))
        events=[]
        for i in range(8):
            sid=f'q64_sample{i:02d}';p=old/'local/events'/(sid+'.json');write(p,dict(artificial=True,sample=i))
            events.append(dict(sample_id=sid,record_sha256=h.sha(p)))
        write(old/'event_bindings.json',events);(old/'local/car_context.npz').write_bytes(b'ARTIFICIAL')
        d=old/'local/q64_sample00';write(d/'precompile_checks.json',dict(status='PASS'))
        for name in ('conditions.json','fusion.json','structural_features_raw.json'):write(d/name,dict(artificial=True))
        (d/'references.npz').write_bytes(b'ARTIFICIAL')
        for m in h.METHODS:
            for a in h.AXES:
                name=m+'_'+a;write(d/(name+'_compile.json'),dict(status='PASS',compile_calls=1))
                for stage in ('raw','compiled'):
                    p=d/'circuits'/(name+'_'+stage+'.qpy');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'ARTIFICIAL NOT CIRCUIT')

    def test_compile_reserves_one_and_post_gets_idle_budget(self):
        tasks=[dict(stage='compile',sample_id='a',method='reference',axis='X',memory_bytes=1),
            dict(stage='post',sample_id='b',method='symmetric_directional',axis='X',memory_bytes=1)]
        result=h.allocation(runner.frozen.old.frozen.task_allocation,64,100,tasks,64,0)
        self.assertEqual([threads for task,threads in result],[1,63])
        self.assertIs(result[0][0],tasks[0]);self.assertEqual(result[0][0]['method'],'reference')

    def test_four_compiles_reserve_four_not64(self):
        tasks=[dict(stage='compile',sample_id='a',method=m,axis=a,memory_bytes=1) for m in h.METHODS for a in h.AXES]
        result=h.allocation(runner.frozen.old.frozen.task_allocation,64,100,tasks,64,0)
        self.assertEqual([v for _,v in result],[1]*4)

    def test_raw_and_post_share_remainder_and_memory_admission(self):
        tasks=[dict(stage=stage,sample_id='a',method='reference',axis='X',memory_bytes=10) for stage in ('raw','post')]
        result=h.allocation(runner.frozen.old.frozen.task_allocation,7,100,tasks,7,0)
        self.assertEqual([v for _,v in result],[4,3])
        self.assertEqual(len(h.allocation(runner.frozen.old.frozen.task_allocation,7,10,tasks,7,0)),1)

    def test_reuses_raw1_compile4_no_sampling_and_preserves_old_stop(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t);self.fixture(old);before=(old/'STOP.json').read_bytes();packet=h.inheritance_plan(old)
            self.assertEqual(packet['raw_reused'],['q64_sample00']);self.assertEqual(len(packet['raw_remaining']),7)
            self.assertEqual(packet['compile_remaining'],28);self.assertEqual(packet['postcheck_remaining'],32)
            self.assertEqual(packet['sampling_remaining'],0);h.verify(old,packet)
            self.assertEqual((old/'STOP.json').read_bytes(),before)
            p=old/'local/events/q64_sample07.json';p.write_text('{}')
            with self.assertRaises(ValueError):h.verify(old,packet)

    def test_completed_prior_can_seal_while_compiles_drain_but_science_cannot_handoff(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t);self.fixture(old)
            (old/'STOP.json').unlink()
            for p in (old/'local/q64_sample00').glob('*_compile.json'):
                p.write_text(json.dumps(dict(status='STARTED',compile_calls=1)))
            prior=h.prior_plan(old);h.verify(old,prior)
            self.assertEqual(prior['original_compile_tasks'],4)
            self.assertTrue(all('_compiled.qpy' not in r['path'] for r in prior['members']))
            with self.assertRaises(FileNotFoundError):h.inheritance_plan(old)

    def test_refuses_incomplete_compile_or_scientific_failure(self):
        with tempfile.TemporaryDirectory() as t:
            old=Path(t);self.fixture(old);marker=old/'local/q64_sample00/reference_X_compile.json'
            marker.write_text(json.dumps(dict(status='STARTED',compile_calls=1)))
            with self.assertRaises(ValueError):h.inheritance_plan(old)
            marker.write_text(json.dumps(dict(status='PASS',compile_calls=1)))
            (old/'local/q64_sample00/reference_X_check.json').write_text(json.dumps(dict(status='FAIL')))
            with self.assertRaises(ValueError):h.inheritance_plan(old)

    def mock_pipeline(self,fail=False):
        calls=[];stop=Event()
        class Pool:
            def __init__(self,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,func,*args):
                f=Future()
                if func==runner.frozen.raw_worker:
                    sid=args[0]['sample_id'];calls.append((sid,'raw',args[3]));f.set_result((sid,dict(status='PASS',peak_RSS_KiB=1)))
                else:
                    sid,task,stage,_=args;calls.append((sid,stage,task[4]));status='FAIL' if fail and sid=='q64_sample00' else 'PASS'
                    f.set_result((sid,task[2]+'_'+task[3],stage,dict(status=status)))
                return f
        events=[dict(sample_id=f'q64_sample{i:02d}') for i in range(8)]
        raw={'q64_sample00':dict(status='PASS')};compiled={};done={(events[0]['sample_id'],m,a) for m in h.METHODS for a in h.AXES}
        resources=dict(numerical_threads=64,memory=dict(MemAvailable=10**12),raw_memory_estimate=10000,compile_memory_estimate=10000,post_memory_estimate=10000)
        with patch.object(runner,'ProcessPoolExecutor',Pool),patch.object(runner,'write'),patch.object(runner.frozen.old.frozen,'memory_available',return_value=10**12),patch.object(runner.frozen.previous,'paired_postcheck'):
            if fail:
                with self.assertRaises(ValueError):runner.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),raw,compiled,done)
            else:runner.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),raw,compiled,done)
        return calls,raw,compiled

    def test_mocked_handoff_7raw_28compile_32post_gate_and_cpu(self):
        calls,raw,compiled=self.mock_pipeline()
        self.assertEqual(sum(v[1]=='raw' for v in calls),7)
        self.assertEqual(sum(v[1]=='compile' for v in calls),28)
        self.assertEqual(sum(v[1]=='post' for v in calls),32)
        self.assertTrue(all(v[2]==1 for v in calls if v[1]=='compile'))
        first_other=next(i for i,v in enumerate(calls) if v[0]!='q64_sample00')
        self.assertEqual(first_other,4);self.assertTrue(all(v[1]=='post' for v in calls[:4]))
        self.assertEqual(len(raw),8);self.assertEqual(sum(map(len,compiled.values())),32)

    def test_sample00_failure_blocks_all_remaining_raw_compile(self):
        calls,_,_=self.mock_pipeline(fail=True)
        self.assertTrue(all(v[0]=='q64_sample00' and v[1]=='post' for v in calls))


if __name__=='__main__':unittest.main()
