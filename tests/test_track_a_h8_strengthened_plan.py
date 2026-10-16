"""Metadata guard failure tests; no molecular input, providers, or compiler."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h8_plan_check', ROOT/'scripts/check_track_a_h8_strengthened_plan.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class MetadataGuards(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((check.BUNDLE/'execution_plan_UNAPPROVED_v1.json').read_text())

    def rejects(self, change):
        plan = copy.deepcopy(self.plan)
        change(plan)
        with self.assertRaises(ValueError):
            check.validate(plan)

    def test_proposal_metadata(self):
        self.assertEqual(check.validate(self.plan)['combined_deterministic'], 184)

    def test_existing_candidate_cannot_be_rescheduled(self):
        self.rejects(lambda p: p['registrations']['additional_deterministic'].append(
            p['registrations']['existing_deterministic'][0]))

    def test_missing_B0I_S4_not_accepted(self):
        self.rejects(lambda p: p['registrations']['additional_deterministic'].pop())

    def test_approval_flag_is_not_execution_permission(self):
        self.rejects(lambda p: p['authorization'].update(approved=True))

    def test_unsealed_command_not_accepted(self):
        self.rejects(lambda p: p['authorization'].update(absolute_launch_command='run'))

    def test_compile_cannot_enter_phase_A(self):
        self.rejects(lambda p: p['phase_A']['call_caps'].update(compile=1))

    def test_parallel_compile_violates_immediate_stop(self):
        self.rejects(lambda p: p['phase_B2'].update(compile_concurrency=2))

    def test_threshold_denominator_or_boundary_cannot_change(self):
        self.rejects(lambda p: p['phase_B2'].update(
            stop_relative_error='abs(predicted-measured)/predicted > 1/50'))

    def test_phase_transition_requires_approval(self):
        self.rejects(lambda p: p['phase_C'].update(separate_explicit_approval_required=False))

    def test_V2_fit_conditions_cannot_be_relaxed(self):
        self.rejects(lambda p: p['V2'].update(conditions_relaxed=True))


if __name__ == '__main__':
    unittest.main()
