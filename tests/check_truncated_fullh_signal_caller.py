"""Only the new batch glue: artificial producer/reader, no real inputs."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('caller', ROOT / 'scripts/run_truncated_fullh_signals.py')
caller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(caller)


class Tests(unittest.TestCase):
    def run_batch(self, status):
        plans = [(name, dict(configuration=dict(id=name), pending=dict(requests=[dict(
            m=0, physical_candidate=dict(id='ARTIFICIAL_' + name), identity=dict(id=name), law_binding={})])))
            for name in ('FIRST', 'SECOND')]
        calls = []

        def producer(root, plan, out, **kwargs):
            calls.append(plan['configuration']['id'])
            self.assertEqual(kwargs['saved'], ())
            out.mkdir()
            return dict(configuration_id=plan['configuration']['id'], target={}, source={'fixture': True},
                means=[dict(m=0, raw={} if status == caller.COMPLETE else None, status=status)],
                producer_result=dict(status=status, new_mean_evaluations=1, saved_copies=0))

        with tempfile.TemporaryDirectory() as td, patch.object(caller.I, 'produce_signals', producer):
            result = caller.run_schedules(ROOT, plans, Path(td), None, None)
        return result, calls

    def test_complete_starts_second_and_copies_only(self):
        result, calls = self.run_batch(caller.COMPLETE)
        self.assertEqual(calls, ['FIRST', 'SECOND'])
        self.assertEqual(result['new_mean_evaluations'], 2)
        self.assertIsNone(result['G'])

    def test_first_failure_stops_next_without_retry(self):
        result, calls = self.run_batch('STOP_NUMERICAL_UNRESOLVED')
        self.assertEqual(calls, ['FIRST'])
        self.assertEqual(result['status'], 'STOP_UNKNOWN')
        self.assertEqual(result['schedules'][1]['status'], 'NOT_STARTED')
        self.assertEqual(result['schedules'][1]['coverage'][0]['status'], 'NOT_STARTED')

    def test_reference_reader_copies_original_pair_no_solver(self):
        locator = dict(commit='ARTIFICIAL_SOURCE', path='ARTIFICIAL_PATH')
        shared = dict(binding=dict(state='ARTIFICIAL_STATE'), independent_reference_pairs=dict(source=locator))
        descriptor = dict(m=0, T=1, binding=shared['binding'],
                          source=dict(locator, selector=['rounds', 0, 'diagnostics', 'reference_pair']))
        pair = dict(reference=dict(real=.2, imag=.3), independent_reference=dict(real=.2, imag=.3), signal_difference=0.)
        saved = dict(binding=shared['binding'], rounds=[dict(m=0, physical_candidate=dict(T=1),
                                                           diagnostics=dict(reference_pair=pair))])
        api = SimpleNamespace(B=SimpleNamespace(a=SimpleNamespace(exact_scalar=lambda v: v)))
        with patch.object(caller, 'read', return_value=saved):
            reader = caller.reference_reader(ROOT, shared, api)
        copied = reader(descriptor)
        self.assertEqual(copied['reference'], pair['reference'])
        copied['reference']['real'] = 99
        self.assertEqual(pair['reference']['real'], .2)
        wrong = deepcopy(descriptor)
        wrong['source']['commit'] = 'OTHER'
        with self.assertRaises(ValueError):
            reader(wrong)


if __name__ == '__main__':
    unittest.main()
