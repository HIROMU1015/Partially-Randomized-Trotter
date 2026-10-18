"""Artificial overhead regression plus the original 36-case contract matrix."""
from pathlib import Path
import os
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

import test_g6_w2c_b3_parallel_candidate as Legacy
import b3_parallel_artificial_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_parallel_overhead as O


class OptimizedMatrix(Legacy.ParallelCandidate):
    def setUp(self):
        super().setUp()
        context=O.install('coalesced','cached')
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)


class ObserverAndProtocol(unittest.TestCase):
    def test_free_variable_self_still_excludes_meter_methods(self):
        def factory(self):
            def inner():return self
            return inner
        function=factory(O.D.Meter(O.D.clock))
        for cls in (O.ScopedObserver,O.CachedObserver):
            observer=cls([__file__])
            frame=NS(f_code=function.__code__,f_locals={'self':O.D.Meter(O.D.clock)})
            observer.hook(frame,'call',None);observer.hook(frame,'return',None)
            self.assertFalse(observer.frames);self.assertEqual(observer.meter.rows,{})

    def test_distinct_code_objects_with_one_label_merge_counts(self):
        rows=[]
        for cls in (O.ScopedObserver,O.CachedObserver):
            ticks=iter([(1,2),(4,6),(10,20),(16,28)])
            observer=cls([__file__],lambda:next(ticks))
            for _ in range(2):
                namespace={};exec(compile('def equal(): pass',__file__,'exec'),namespace)
                frame=NS(f_code=namespace['equal'].__code__,f_locals={})
                observer.hook(frame,'call',None);observer.hook(frame,'return',None)
            rows.append(observer.meter.rows)
        self.assertEqual(rows[0],rows[1])
        self.assertEqual(len(rows[1]),1);self.assertEqual(next(iter(rows[1].values()))['calls'],2)

    def test_unacknowledged_write_on_worker_death_keeps_reserved_credit(self):
        original=F.P.ReservedFile.write
        def exit_after_write(file,data):
            size=original(file,data)
            if Path(file.file.name).name=='correctness_baseline_ordinary_A.jsonl':
                os._exit(91)
            return size
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
            root=Path(temp);s=F.P.Settings(profile_n=16,workload='light')
            c,l,a=O.prepare(F,root,s,'coalesced','cached')
            with patch.object(F.P.ReservedFile,'write',exit_after_write):
                with self.assertRaises(F.P.BundleFailure) as caught:
                    O.invoke(F,root,c,l,a,s,transport='coalesced',observer='cached')
            report=caught.exception.report
            self.assertGreaterEqual(report['states']['TERMINATED_WITH_UNKNOWN_RESULT'],1)
            raw=root/'artificial-output/correctness_baseline_ordinary_A.jsonl'
            self.assertEqual(len(F.read_stream(raw)['rows']),1)
            self.assertGreaterEqual(report['output']['reserved']['result']-report['output']['observed']['result'],raw.stat().st_size)
            self.assertEqual(l.row(c['scope'])[4],'STOPPED')

    def test_prior_commits_are_recorded_before_next_reservation_denial(self):
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
            root=Path(temp);s=F.P.Settings(workers=1,profile_n=16,workload='light',call_limit=3)
            c,l,a=O.prepare(F,root,s,'coalesced','cached')
            with self.assertRaises(F.P.BundleFailure) as caught:
                O.invoke(F,root,c,l,a,s,transport='coalesced',observer='cached')
            report=caught.exception.report
            self.assertEqual(report['reserved'],3)
            self.assertEqual(report['states']['COMPLETED'],3)
            state=(root/'artificial-output/trial_states.bin').read_bytes()
            self.assertEqual(state[:3],bytes([F.P.CODE['COMPLETED']])*3)

    def test_exact_nested_exclusive_costs_and_counts_with_fake_clock(self):
        path=str(Path(__file__).resolve())
        namespace={};exec(compile('def outer(): pass\ndef inner(): pass',path,'exec'),namespace)
        frames=[NS(f_code=namespace[x].__code__,f_locals={}) for x in ('outer','inner')]
        values=[]
        for cls in (O.ScopedObserver,O.CachedObserver):
            ticks=iter([(10,20),(30,50),(50,80),(90,140)])
            observer=cls([path],lambda:next(ticks))
            for frame,event in [(frames[0],'call'),(frames[1],'call'),(frames[1],'return'),(frames[0],'return')]:
                observer.hook(frame,event,None)
            self.assertFalse(observer.frames)
            values.append(observer.meter.rows)
        self.assertEqual(values[0],values[1])

    def test_code_identity_not_code_equality(self):
        namespaces=[]
        for name in ('watched.py','unwatched.py'):
            n={};exec(compile('def equal(): pass',name,'exec'),n);namespaces.append(n)
        observer=O.CachedObserver([str(Path('watched.py').resolve())])
        # Use absolute compile paths, matching actual runtime selection.
        for name,n in zip(('watched.py','unwatched.py'),namespaces):
            exec(compile('def equal(): pass',str(Path(name).resolve()),'exec'),n)
            frame=NS(f_code=n['equal'].__code__,f_locals={})
            observer.hook(frame,'call',None);observer.hook(frame,'return',None)
        self.assertEqual(len(observer.meter.rows),1)
        self.assertEqual(next(iter(observer.meter.rows.values()))['calls'],1)

    def test_parent_messages_fall_without_reducing_calls_or_bytes(self):
        results=[]
        for transport in ('legacy','coalesced'):
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
                root=Path(temp);s=F.P.Settings(profile_n=16,workload='light')
                c,l,a=O.prepare(F,root,s,transport,'cached')
                r=O.invoke(F,root,c,l,a,s,transport=transport,observer='cached')
                self.assertTrue(r['complete']);self.assertEqual(r['reserved'],768)
                self.assertEqual(r['output']['observed'],r['output']['reserved'])
                results.append(r)
        self.assertEqual(sum(results[0]['parent_messages'].values()),3864)
        self.assertEqual(sum(results[1]['parent_messages'].values()),2376)
        self.assertEqual(results[1]['parent_messages'].get('WRITE_ACK',0),0)
        self.assertEqual(results[1]['state_update_io_bytes'],results[0]['state_update_io_bytes'])

    def test_scientific_function_call_counts_equal_observers(self):
        signatures=[]
        for transport,observer in [('legacy','legacy'),('legacy','cached'),('coalesced','cached')]:
            with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
                root=Path(temp);s=F.P.Settings(profile_n=16,workload='light')
                c,l,a=O.prepare(F,root,s,transport,observer)
                r=O.invoke(F,root,c,l,a,s,transport=transport,observer=observer)
                signatures.append([{k:v['calls'] for k,v in p['fine'].items()} for p in r['pairs'] if p['fine']])
        self.assertEqual(signatures[0],signatures[1])
        def core(rows):
            return [{k:v for k,v in row.items() if k.startswith(('g6_w2c_b3_driver.py:','b3_parallel_artificial_fixture.py:'))
                     and ':delta.<locals>.' not in k} for row in rows]
        self.assertEqual(core(signatures[0]),core(signatures[2]))

    def test_flush_inside_emit_is_not_completed(self):
        original=F.D.Stream
        class EveryRow(original):
            def __init__(self,*args,**kwargs):
                kwargs['flush_rows']=1;super().__init__(*args,**kwargs)
        with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as temp:
            root=Path(temp);s=F.P.Settings(profile_n=16,workload='light',faults=((0,0,'flush',0),))
            c,l,a=O.prepare(F,root,s,'coalesced','cached')
            with patch.object(F.D,'Stream',EveryRow):
                with self.assertRaises(F.P.BundleFailure) as caught:
                    O.invoke(F,root,c,l,a,s,transport='coalesced',observer='cached')
            row=next(x for x in caught.exception.report['pairs'] if x['task']['pair_index']==0)
            f=row['failure']['trial_failure_provenance']
            self.assertEqual(sum(f['completed_prefix'].values()),0)
            self.assertFalse(f['current_attempt']['stream_emit_returned'])
            self.assertEqual(len(F.read_stream(root/'artificial-output/correctness_baseline_ordinary_A.jsonl')['rows']),1)


if __name__=='__main__':unittest.main()
