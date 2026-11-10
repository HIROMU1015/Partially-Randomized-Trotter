"""Non-scientific schema/mock checks. No arrays, operators, or simulation."""
from pathlib import Path
import builtins
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'scripts/run_calpha_h4_step1_mean.py'


def load_module():
    spec=importlib.util.spec_from_file_location('calpha_step1_contract_target',SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.module=load_module()
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.review=self.directory/'review.md'
        self.review.write_text('mock scientific-spec document; no matrices')

    def test_import_has_no_scientific_imports(self):
        original=builtins.__import__
        def checked(name,*args,**kwargs):
            if name.split('.')[0] in ('numpy','scipy','qiskit','openfermion','mpmath','trotterlib'):
                raise AssertionError('scientific import during schema test:'+name)
            return original(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=checked):
            load_module()

    def test_round_window_rejects_extra_missing_duplicate_or_reordered(self):
        self.module.validate_rounds([{'m':0},{'m':17}])
        for values in ((0,),(17,0),(0,1,17),(0,17,17)):
            with self.subTest(values=values),self.assertRaises(ValueError):
                self.module.validate_rounds([{'m':m} for m in values])

    def test_disk_outside_domain_is_not_clamped(self):
        result=self.module.disk_diagnostic(0.,0.,2.,1.)
        self.assertFalse(result['positive_radius'])
        self.assertIsNone(result['phase_bound_rad'])
        self.assertIsNone(result['finite_bit_native_implementation_error'])
        self.assertIn('NOT_CERTIFIED',result['qualification'])

    def test_mock_result_serializes_exactly_two_points_and_stops(self):
        callback=Mock(return_value=dict(status='MOCK_ONLY',rounds=[{'m':0},{'m':17}]))
        with patch.object(self.module,'verify_old_evidence',return_value={'status':'mock'}):
            result=self.module.run('unopened_private_path',self.directory/'output',self.review,evaluator=callback)
        callback.assert_called_once()
        self.assertTrue(result['mandatory_stop'])
        self.assertEqual(result['fixed_conditions']['rounds'],[0,17])
        saved=json.loads((self.directory/'output/result.json').read_text())
        self.assertEqual(saved['status'],'MOCK_ONLY')
        self.assertEqual(saved['rounds'],[{'m':0},{'m':17}])

    def test_existing_output_refuses_before_callback(self):
        output=self.directory/'existing';output.mkdir()
        callback=Mock()
        with self.assertRaises(ValueError):
            self.module.run('unopened',output,self.review,evaluator=callback)
        callback.assert_not_called()

    def test_missing_spec_refuses_before_callback(self):
        callback=Mock()
        with self.assertRaises(ValueError):
            self.module.run('unopened',self.directory/'output',self.directory/'missing.md',evaluator=callback)
        callback.assert_not_called()

    def test_callback_failure_persists_stop_without_retry(self):
        callback=Mock(side_effect=ValueError('mock input mismatch'))
        with patch.object(self.module,'verify_old_evidence',return_value={'status':'mock'}):
            with self.assertRaisesRegex(ValueError,'mock input mismatch'):
                self.module.run('unopened',self.directory/'output',self.review,evaluator=callback)
        callback.assert_called_once()
        result=json.loads((self.directory/'output/result.json').read_text())
        self.assertEqual(result['status'],'TECHNICAL_STOP_EXCEPTION')
        self.assertTrue(result['mandatory_stop'])

    def test_extra_point_from_callback_is_rejected_and_recorded(self):
        callback=Mock(return_value={'rounds':[{'m':0},{'m':1},{'m':17}]})
        with patch.object(self.module,'verify_old_evidence',return_value={'status':'mock'}):
            with self.assertRaisesRegex(ValueError,'ONLY_M0_M17'):
                self.module.run('unopened',self.directory/'output',self.review,evaluator=callback)
        result=json.loads((self.directory/'output/result.json').read_text())
        self.assertEqual(result['status'],'TECHNICAL_STOP_EXCEPTION')
        self.assertNotIn('rounds',result)


if __name__=='__main__':
    unittest.main()
