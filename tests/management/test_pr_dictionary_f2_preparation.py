"""Artificial metadata/mocks only. No quantum, reference or interval evaluation."""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import f2_plan as plan
from trottertracks.pr_dictionary_validation import f2_execution as execution
from trottertracks.pr_dictionary_validation import f2_statistics as statistics
from trottertracks.pr_dictionary_validation import f2_preflight as preflight


class PreparationTests(unittest.TestCase):
    def tearDown(self):
        for name in ('qiskit','numpy','mpmath','trotterlib'):
            self.assertNotIn(name,sys.modules)

    def test_saved_B_integer_descriptors_include_diagnostics_and_both_precisions(self):
        saved=json.loads((ROOT/'docs/research/pr_algorithm_improvement_phase2d_binding_execution_20261023/F1_COUNT_DESCRIPTORS.json').read_text())
        self.assertEqual(plan.LENGTHS,saved['L'])
        q=saved['F2_deferred_descriptors']['Q']
        self.assertEqual(q,13674)
        c=plan.counts()
        self.assertEqual(c['full_wrapper_build_requests'],q+2)
        self.assertEqual(c['reference_precision_evaluations'],27356)
        jobs=list(plan.iter_jobs());self.assertEqual(len(jobs),6838)
        ids=[plan.case_id(*job,axis) for job in jobs for axis in ('X','Y')]
        self.assertEqual(len(set(ids)),13676)
        self.assertEqual(c['quantum_measurement_draws'],0)

    def test_only_identical_complete_IR_keys_are_grouped_and_all_aliases_retained(self):
        rows=[dict(id='b',compile_key='full_ir_A'),dict(id='a',compile_key='full_ir_A'),
              dict(id='c',compile_key='different_angle_or_phase_IR')]
        groups=dict(plan.compile_groups(rows))
        self.assertEqual(groups['full_ir_A'],('a','b'))
        self.assertEqual(groups['different_angle_or_phase_IR'],('c',))
        with self.assertRaises(ValueError):plan.compile_groups(rows+[rows[0]])

    def test_candidate_policy_and_unqualified_provider_never_produce_N(self):
        with self.assertRaises(PermissionError):statistics.qualified_shots('2',dict(status='CANDIDATE_NOT_ADOPTED'))
        # Artificial values test budget routing, not the F2 numerical candidate.
        s=dict(status='USER_ADOPTED',eta_stat='1/2',alpha='1/4',finite_bit_allowance='1/4',
            alpha_scope='PER_ARM_COMPLEX_SIGNAL',sampling_qualification=dict(
                status='EXTERNALLY_APPROVED_CONDITIONS',evidence='mock_only',
                fresh_iid_per_hadamard_shot=True,independent_bounded_outcomes=True,
                range_bound_justified=True,finite_bit_allowance_justified=True),
            interval_qualification=dict(status='UNQUALIFIED'))
        self.assertEqual(statistics.require_adopted_statistics(s),(Fraction(1,4),Fraction(1,4)))
        with self.assertRaises(PermissionError):statistics.qualified_shots('2',s)
        s['alpha_scope']='SIMULTANEOUS_FOUR_ARMS'
        self.assertEqual(statistics.require_adopted_statistics(s),(Fraction(1,4),Fraction(1,16)))
        s['finite_bit_allowance']='1/2'
        with self.assertRaises(ValueError):statistics.require_adopted_statistics(s)

    def test_exact_resource_aggregation_uses_both_axes_and_does_not_subtract_diagnostic(self):
        weights={a:(Fraction(1,4),Fraction(3,4)) for a in plan.ARMS};rows=[]
        for a in plan.ARMS:
            for i in range(2):
                for j in range(2):
                    for axis in ('X','Y'):
                        rows.append(dict(id=plan.case_id(a,i,j,axis),arm=a,axis=axis,i=i,j=j,
                            metrics=dict(rz_count=10+i+2*j,cx_count=4)))
        rows.append(dict(id='diagnostic_X',arm='diagnostic',metrics=dict(rz_count=999)))
        result=execution.aggregate_metrics(rows,weights)
        for a in plan.ARMS:
            for axis in ('X','Y'):
                self.assertEqual(result[a][axis],dict(rz_count=Fraction(49,4),cx_count=Fraction(4)))
        with self.assertRaises(ValueError):execution.aggregate_metrics(rows[1:],weights)
        with self.assertRaises(ValueError):execution.aggregate_metrics(rows+[rows[0]],weights)

    def test_worker_dispatch_journals_one_mock_attempt_and_stops_after_failure(self):
        old=execution._WORKER
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);(out/'attempts').mkdir()
                build=Mock(return_value=['mock_record'])
                execution._WORKER=SimpleNamespace(out=out,build=build)
                self.assertEqual(execution.build_job(('SS',0,0)),['mock_record'])
                build.assert_called_once_with(('SS',0,0))
                build.side_effect=RuntimeError('mock_technical_failure')
                with self.assertRaises(RuntimeError):execution.build_job(('SS',0,1))
                self.assertTrue((out/'STOP_REQUESTED.json').exists())
                with self.assertRaises(RuntimeError):execution.build_job(('SS',1,0))
                self.assertEqual(build.call_count,2)
                journal=list((out/'attempts').glob('*.jsonl'))
                self.assertEqual(len(journal),2)
        finally:execution._WORKER=old

    def test_F1_grant_is_rejected_before_input_read_or_science_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'authorization.json').write_text(json.dumps(dict(
                stage='F1',authority='USER_EXPLICIT_F1',status='APPROVED_BOUND_ONE_SHOT')))
            with self.assertRaises(PermissionError):preflight.verify_packet(p,ROOT,p/'output')
            self.assertFalse((p/'output').exists())
            self.assertEqual(len(list(p.iterdir())),1)


if __name__=='__main__':unittest.main()
