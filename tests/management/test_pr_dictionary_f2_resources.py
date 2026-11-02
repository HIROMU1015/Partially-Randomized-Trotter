"""Resource-only routing checks using artificial records and mocked science."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import f2_execution as execution
from trottertracks.pr_dictionary_validation import f2_resources as limited
from trottertracks.pr_dictionary_validation.f2_plan import ARMS,counts


class LimitedResourceTests(unittest.TestCase):
    def tearDown(self):
        for name in ('numpy','qiskit','mpmath','trotterlib'):
            self.assertNotIn(name,sys.modules)

    def test_only_statistical_calls_are_removed_from_limited_inventory(self):
        full=counts(); limited_counts=limited.resource_counts()
        for key in full:
            self.assertEqual(limited_counts[key],0 if key in (
                'shot_qualifications','conditional_selector','known_four_search') else full[key])
        self.assertEqual(limited_counts['full_wrapper_build_requests'],13676)
        self.assertEqual(limited_counts['reference_precision_evaluations'],27356)

    def test_foreign_grants_are_rejected_before_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet=Path(tmp)
            for stage in ('F1','F2'):
                (packet/'authorization.json').write_text(json.dumps(dict(stage=stage,
                    authority='USER_EXPLICIT_'+stage,status='APPROVED_BOUND_ONE_SHOT')))
                with self.assertRaises(PermissionError):
                    limited.verify_packet(packet,ROOT,packet/'output')

    def test_original_qualified_entry_refuses_limited_contract(self):
        with self.assertRaises(PermissionError):
            execution.execute_f2(dict(contract=dict(comparison=limited.COMPARISON)))

    def test_resource_summary_never_calls_statistics_or_zero_fills_N(self):
        # Values are artificial; the shared measurement pipeline is mocked.
        from fractions import Fraction
        rusage=SimpleNamespace(ru_utime=0,ru_stime=0)
        row=dict(compile_seconds=1,operator=dict(max_entry=0),signal_error=0)
        measured=dict(compiled=[row],built=[dict(raw_operator=dict(max_entry=0),raw_signal_error=0)],
            groups=[('mock',('mock_X',))],means={a:dict(max_entry=0) for a in ARMS},
            phis={a:{axis:dict(rz_count=Fraction(3,2)) for axis in ('X','Y')} for a in ARMS},
            timings={},cpu=0,start=0,children=rusage)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(execution,'measure_f2_circuits',return_value=measured) as shared, \
                patch.object(execution,'check_source'), \
                patch('trottertracks.pr_dictionary_validation.f2_statistics.qualified_shots',
                      side_effect=AssertionError('Statistics outside scope')):
            v=dict(contract=dict(comparison=limited.COMPARISON),output=Path(tmp))
            result=limited.execute_resources(v)
            shared.assert_called_once_with(v)
            self.assertIsNone(result['N']);self.assertIsNone(result['weighted_costs'])
            self.assertFalse(result['same_accuracy_total_resource_advantage_claim'])
            self.assertTrue(result['mandatory_stop'])


if __name__=='__main__':unittest.main()
