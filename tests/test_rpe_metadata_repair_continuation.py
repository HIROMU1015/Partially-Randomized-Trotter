"""Artificial metadata-failure continuation preserves the scientific boundary."""
import hashlib
import json
import unittest

import test_rpe_search_continuation as base
from trottertracks.rpe_release_registry import fingerprint
from trottertracks.rpe_search_continuation import verify


class MetadataRepairTests(unittest.TestCase):
    # Reuse the synthetic filesystem setup without rerunning the other class's tests.
    def setUp(self):
        base.ContinuationTests.setUp(self)
        self.round['continuation']['kind'] = 'OPERATIVE_METADATA_REPAIR'
        self.error = dict(reason='RUNTIME_EXCEPTION', error_type='CalledProcessError',
                          message="Command scripts/check_pf_preparation_owner.py --output ORIGINAL_OWNER_CHECK.json failed")
        self.save('KAPPA_STATE.json', {'stop': self.error})
        self.bind_error()
        (self.original/'ROUND_RESULT.json').unlink()
        job = self.original/'jobs/PR_PAULI/b'
        job.mkdir(parents=True)
        self.save('jobs/PR_PAULI/b/OUTCOME.json', {'status': 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'})
        self.old['max_new_q3_per_method']['PR_PAULI'] = 'UNLIMITED'
        self.old['executor']['distributed']['preparation']['budgets']['compile'] = 'UNLIMITED'
        self.round['max_new_q3_per_method']['PR_PAULI'] = 'UNLIMITED'
        self.round['executor']['distributed']['preparation']['budgets']['compile'] = 'UNLIMITED'
        self.save('ROUND.requested.json', self.old)
        self.save('APPLIED_SCOPE.json', dict(code_source='old', signature='sig',
                  round_requested_sha256=fingerprint(self.old), problem_sha256=fingerprint(self.problem)))

    save = base.ContinuationTests.save

    def bind_error(self):
        row = dict(self.error, phase='DRIVER', event='RUNTIME_EXCEPTION')
        self.save('RUNTIME_ERRORS.jsonl', row)
        raw = (self.original/'RUNTIME_ERRORS.jsonl').read_bytes()
        self.round['continuation']['parent_error_log_sha256'] = hashlib.sha256(raw).hexdigest()

    def test_metadata_continuation_preserves_unlimited_and_deducts_integer(self):
        before = {p: p.read_bytes() for p in self.original.rglob('*') if p.is_file()}
        result = verify(self.successor, self.problem, self.round)
        self.assertEqual(result['remaining_q3'], {'PR_PAULI': 'UNLIMITED'})
        self.assertEqual(result['remaining'], {'signal': 0, 'compile': 'UNLIMITED', 'audit': 1})
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_metadata_continuation_rejects_changed_failure_log(self):
        (self.original/'RUNTIME_ERRORS.jsonl').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'saved metadata'):
            verify(self.successor, self.problem, self.round)

    def test_metadata_continuation_does_not_allow_scientific_runtime_failures(self):
        self.error['message'] = 'Command scripts/run_scientific_calculation.py failed'
        self.save('KAPPA_STATE.json', {'stop': self.error})
        self.bind_error()
        with self.assertRaisesRegex(ValueError, 'saved metadata'):
            verify(self.successor, self.problem, self.round)

    def test_metadata_continuation_preserves_explicit_and_scientific_stops(self):
        self.save('STOP_REQUESTED.json', {'reason': 'USER_REQUESTED_STOP'})
        with self.assertRaisesRegex(ValueError, 'explicit STOP'):
            verify(self.successor, self.problem, self.round)
        (self.original/'STOP_REQUESTED.json').unlink()
        self.save('jobs/PR_PAULI/b/OUTCOME.json', {'status': 'UNKNOWN_RESULT'})
        with self.assertRaisesRegex(ValueError, 'scientific STOP'):
            verify(self.successor, self.problem, self.round)

    def test_metadata_continuation_requires_ended_relay_and_pbs(self):
        (self.original/'distributed/relay').mkdir()
        self.save('distributed/relay/STATUS.json', {'stopped': False})
        with self.assertRaisesRegex(ValueError, 'relay'):
            verify(self.successor, self.problem, self.round)
        self.save('distributed/relay/STATUS.json', {'stopped': True})
        (self.original/'distributed/pbs').mkdir()
        self.save('distributed/pbs/STATE.json', {'allocations': {'slot': {
            'status': 'R', 'observed': {'job_id': '17.pbs'}}}})
        with self.assertRaisesRegex(ValueError, 'terminal parent PBS'):
            verify(self.successor, self.problem, self.round)
        self.round['continuation']['terminal_pbs_jobs'] = {'17.pbs': 'F'}
        verify(self.successor, self.problem, self.round)
        self.assertEqual(json.loads((self.original/'distributed/pbs/STATE.json').read_text())['allocations']['slot']['status'], 'R')

    def test_metadata_continuation_cannot_reset_finite_budget(self):
        self.round['executor']['distributed']['preparation']['budgets']['signal'] = 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            verify(self.successor, self.problem, self.round)
