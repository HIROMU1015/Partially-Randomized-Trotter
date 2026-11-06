"""Synthetic lifecycle/adapter tests; real cron is a separate qualification."""
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import a1_n3_managed_runtime as m
from trottertracks.algorithm_codesign import a1_n3_cron_manager as c


def identity(value):return value
def failure():raise ValueError('invented')


class RuntimeTests(unittest.TestCase):
    def test_receipt_collision_does_not_replace_original(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'receipt';m.durable_new(p,{'first':True})
            with self.assertRaises(FileExistsError):m.durable_new(p,{'first':False})
            self.assertTrue(json.loads(p.read_text())['first'])

    def test_stop_is_durable_before_drain_and_worker_code_collected(self):
        with TemporaryDirectory() as d:
            r=m.Runtime(d);pool=m.ManagedPool(r,max_workers=1,mp_context=mp.get_context('fork'))
            self.assertEqual(pool.submit(identity,7).result(),7)
            r.request_stop('invented');pool.shutdown()
            stop=json.loads((Path(d)/'STOP_REQUESTED.json').read_text())
            rows=[json.loads(x) for x in (Path(d)/'OBS_parent.jsonl').read_text().splitlines()]
            self.assertTrue(all(stop['monotonic_ns']<x['monotonic_ns'] for x in rows if x['event']=='POOL_DRAIN_START'))
            codes=json.loads((Path(d)/'WORKER_EXIT_CODES.json').read_text())
            self.assertTrue(all(x['exit_code']==0 for x in codes.values()))

    def test_worker_exception_stops_submission_without_retry(self):
        with TemporaryDirectory() as d:
            r=m.Runtime(d);pool=m.ManagedPool(r,max_workers=1,mp_context=mp.get_context('fork'))
            with self.assertRaises(ValueError):pool.submit(failure).result()
            pool.shutdown()
            self.assertTrue(r.stop.is_set())
            with self.assertRaisesRegex(RuntimeError,'STOP'):pool.submit(identity,1)

    def test_data_classification_distinguishes_missing_price_and_bit(self):
        with TemporaryDirectory() as d:
            r=m.Runtime(d);p=Path(d)/'worker_toy.jsonl'
            rows=[{'event':'WORKER_STARTED','ticket':'a'},{'event':'BIT_SAVED','ticket':'a'},
                  {'event':'WORKER_STARTED','ticket':'b'}]
            p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            states=r.classify()
            self.assertEqual(states['a'],'BIT_DURABLE_PRICE_MISSING')
            self.assertEqual(states['b'],'ATTEMPT_NO_DURABLE_BIT')

    def test_duplicate_bit_is_not_silently_deduplicated(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'worker_toy.jsonl';p.write_text('{"event":"BIT_SAVED","ticket":"a"}\n'*2)
            with self.assertRaisesRegex(ValueError,'duplicate'):m.Runtime(d).classify()

    def test_owned_signal_refuses_pid_reuse(self):
        with patch.object(m.old,'process_identity',return_value={'pid':5,'start_ticks':9}):
            self.assertEqual(m.owned_signal({'pid':5,'start_ticks':8},signal.SIGTERM),'IDENTITY_GONE_OR_CHANGED')

    def test_adapter_preserves_arguments_result_and_restores_functions(self):
        class Engine:
            def selected_circuit(self,*args,**kwargs):return args,kwargs
            probability=selected_circuit
        kernel=NS(SavedEngine=Engine,born_probability=lambda x:x)
        compiler=NS(transpile_and_measure_cost=lambda x:x)
        sentinel=object();source=NS(kernel=kernel,module=lambda name:compiler,ProcessPoolExecutor=sentinel,_STOP_EVENT=None)
        original=Engine.selected_circuit;value=object()
        with TemporaryDirectory() as d:
            with m.connect_legacy(source,m.Runtime(d)):
                args,kw=Engine().selected_circuit(value,phase='kept')
                self.assertIs(args[0],value);self.assertEqual(kw,{'phase':'kept'})
                self.assertIs(kernel.born_probability(value),value)
            self.assertIs(source.ProcessPoolExecutor,sentinel)
            self.assertIs(Engine.selected_circuit,original)


class CronGuards(unittest.TestCase):
    def test_existing_crontab_is_protected(self):
        with TemporaryDirectory() as d,patch.object(c,'own_crontab',return_value='existing\n'):
            with self.assertRaisesRegex(RuntimeError,'protected'):c.submit_qualification(d,Path(d)/'out')

    def test_foreign_change_is_not_overwritten_on_cleanup(self):
        with patch.object(c,'own_crontab',return_value='other\n'),patch.object(c.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'changed'):c.remove_own_entry({'installed_crontab':'ours\n'})
            run.assert_not_called()

    def test_synthetic_only_dispatch_rejects_science_spec(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'spec';p.write_text(json.dumps({'kind':'science','science_authorized':True,'output':d,'root':d}))
            with self.assertRaisesRegex(ValueError,'artificial'):c.dispatch(p)

    def test_consumed_dispatch_never_starts_child(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'spec';p.write_text(json.dumps({'kind':'artificial_qualification','science_authorized':False,'output':d,'root':d}))
            (Path(d)/'CONSUMED.json').write_text('{}')
            with patch.object(c.subprocess,'Popen') as child:self.assertEqual(c.dispatch(p),72)
            child.assert_not_called()

    def test_import_does_not_load_science_modules(self):
        code='import sys; from trottertracks.algorithm_codesign import a1_n3_managed_runtime,a1_n3_cron_manager; assert not any(x in sys.modules for x in ("numpy","qiskit","openfermion","trottertracks.algorithm_codesign.a1_n3_campaign"))'
        subprocess.run([sys.executable,'-B','-c',code],check=True)


if __name__=='__main__':unittest.main()
