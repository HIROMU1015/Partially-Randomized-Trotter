"""Mock batches only: no molecular loader, normalizer, mean or reference call."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('signal_caller',ROOT/'scripts/run_pr_rpe_first_signal.py')
caller=importlib.util.module_from_spec(spec);spec.loader.exec_module(caller)


class CallerTests(unittest.TestCase):
    def run_mock(self,status):
        calls=[]
        class Batch:
            def __init__(self,config,physical,**kwargs):
                calls.append(config['spec']['L_D'])
                self.status=status if len(calls)==1 else caller.COMPLETE
                self.config=config
            def run(self,out):
                return dict(status=self.status,records=[],configuration_id=self.config['id'])
        provider=SimpleNamespace(SignalBatch=Batch,
            load_h4_context=lambda *a:(_ for _ in ()).throw(AssertionError('real arrays')),
            saved_h4_reference=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('real reference')))
        configs=[dict(id=str(p),spec=dict(L_D=p),rounds=[]) for p in (1,2)]
        with tempfile.TemporaryDirectory() as tmp:
            out=caller.run_batches(Path(tmp),dict(signal_source={'fixture':'MOCK'}),configs,
                SimpleNamespace(validate=lambda c:None),lambda v:v,None,provider)
        return calls,out

    def test_complete_menu_starts_both_once(self):
        calls,out=self.run_mock(caller.COMPLETE)
        self.assertEqual(calls,[1,2]);self.assertEqual(len(out),2)

    def test_numerical_failure_never_starts_LD2(self):
        calls,out=self.run_mock('STOP_NUMERICAL_FAILURE')
        self.assertEqual(calls,[1]);self.assertEqual(len(out),1)

    def test_input_failure_never_starts_LD2(self):
        calls,out=self.run_mock('STOP_INPUT_OR_SOURCE_FAILURE')
        self.assertEqual(calls,[1]);self.assertEqual(len(out),1)

    def test_existing_output_rejected_before_any_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):caller.execute(Path(tmp))

    def test_completed_first_batch_retained_if_second_constructor_raises(self):
        class Batch:
            def __init__(self,c,rows,**kwargs):
                if c['spec']['L_D']==2:raise ValueError('mock metadata mismatch')
            def run(self,out):return dict(status=caller.COMPLETE,records=[])
        provider=SimpleNamespace(SignalBatch=Batch)
        configs=[dict(id=str(p),spec=dict(L_D=p),rounds=[]) for p in (1,2)]
        sink=[]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                caller.run_batches(Path(tmp),dict(signal_source={}),configs,
                    SimpleNamespace(validate=lambda c:None),lambda v:v,None,provider,outcomes=sink)
        self.assertEqual([r['label'] for r in sink],['LD1'])


if __name__=='__main__':unittest.main()
