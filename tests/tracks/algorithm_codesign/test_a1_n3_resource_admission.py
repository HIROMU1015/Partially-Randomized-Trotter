"""Synthetic tick tables and mocks only; no real inputs, RNG or manager job."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from trottertracks.algorithm_codesign import a1_n3_resource_admission as r
from trottertracks.algorithm_codesign import a1_n3_pilot_activation as a

class AdmissionTests(unittest.TestCase):
    def test_only_allowed_cpus_and_idle_not_iowait(self):
        before={0:(0,)*8,1:(0,)*8};after={0:(20,0,0,30,50,0,0,0),1:(0,0,0,100,0,0,0,0)}
        n,d=r.available_from_ticks(before,after,{0});self.assertEqual(n,0);self.assertEqual(d['idle_core_equivalents_exact'],'3/10')
        n,d=r.available_from_ticks(before,after,{0,1});self.assertEqual(n,1);self.assertEqual(d['idle_core_equivalents_exact'],'13/10')

    def test_no_arbitrary_worker_cap(self):
        b={i:(0,)*8 for i in range(128)};e={i:(0,0,0,100,0,0,0,0) for i in b}
        self.assertEqual(r.available_from_ticks(b,e,set(b))[0],128)

    def test_negative_missing_and_zero_ticks_rejected(self):
        for after in ({},{0:(0,)*8},{0:(-1,0,0,1,0,0,0,0)}):
            with self.assertRaises(ValueError):r.available_from_ticks({0:(0,)*8},after,{0})

    def test_wait_then_admit_and_raw_ticks_saved(self):
        ticks=iter([{0:(0,)*8},{0:(100,0,0,0,0,0,0,0)},{0:(200,0,0,0,0,0,0,0)},
                    {0:(200,0,0,0,0,0,0,0)},{0:(200,0,0,100,0,0,0,0)},{0:(200,0,0,200,0,0,0,0)}])
        with TemporaryDirectory() as tmp,patch.object(r.obs,'memory_snapshot',return_value={}):
            n=r.wait_for_resources(tmp,lambda:False,read=lambda:next(ticks),affinity=lambda:{0},sleep=lambda _:None,clock=lambda:0)
            self.assertEqual(n,1)
            rows=[json.loads(x) for x in (Path(tmp)/'RESOURCE_ADMISSION.jsonl').read_text().splitlines()]
            self.assertEqual([x['event'] for x in rows],['CPU_OBSERVATION','CPU_OBSERVATION','WAITING','CPU_OBSERVATION','CPU_OBSERVATION','ADMITTED'])
            self.assertIn('before',rows[0]['cpus'][0])

    def test_timeout_and_stop(self):
        counter=iter(range(0,100,10));k=[0]
        def busy():k[0]+=100;return {0:(k[0],0,0,0,0,0,0,0)}
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'expired'):
                r.wait_for_resources(tmp,lambda:False,wait_seconds=1,read=busy,affinity=lambda:{0},sleep=lambda _:None,clock=lambda:next(counter))
            with self.assertRaisesRegex(RuntimeError,'STOP'):
                r.wait_for_resources(tmp,lambda:True)

    def test_preinput_admission_precedes_load_and_rng(self):
        import inspect
        s=inspect.getsource(a.parent)
        self.assertLess(s.index('admission.wait_for_resources'),s.index('load_certified_services'))
        self.assertLess(s.index('load_certified_services'),s.index('pcg64_streams'))
        self.assertNotIn('c.available_workers()',s)
        self.assertEqual(a.ATTEMPT,2);self.assertTrue(str(a.OUTPUT).endswith('run02'));self.assertEqual(a.SEED,20261108)

    def test_timeout_reaches_durable_stop_without_binding_or_rng(self):
        from contextlib import nullcontext
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/a.OUTPUT;out.mkdir(parents=True)
            spec=dict(source_commit='toy',manager_SOURCE='toy')
            with patch.object(a,'read_spec',return_value=(root,spec)),patch.object(a,'verify',return_value={}),patch.object(a.c,'forbidden_science',return_value=nullcontext()),patch.object(a.managed,'connect_legacy',return_value=nullcontext()),patch.object(a.admission,'wait_for_resources',side_effect=RuntimeError('finite pre-input resource wait expired')),patch.object(a.c.n3,'load_certified_services') as load,patch.object(a.c.n3,'pcg64_streams') as rng,patch.object(a.c,'reconcile',return_value={}):
                self.assertEqual(a.parent('unused'),1)
            load.assert_not_called();rng.assert_not_called()
            self.assertTrue((out/'STOP_REQUESTED.json').exists())
            self.assertEqual(json.loads((out/'INVOCATION.json').read_text())['attempt'],2)

    def test_affinity_change_rejected(self):
        ticks=iter([{0:(0,)*8},{0:(0,0,0,100,0,0,0,0)}]);cpus=iter([{0},{0,1}])
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'affinity changed'):
                r.wait_for_resources(tmp,lambda:False,read=lambda:next(ticks),affinity=lambda:next(cpus),sleep=lambda _:None)

if __name__=='__main__':unittest.main()
