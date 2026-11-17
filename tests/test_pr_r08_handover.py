"""Artificial metadata only: R08 handover never launches scientific leaves."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('r08_handover', Path(__file__).resolve().parents[1] / 'scripts/resume_pr_r08_after_drain.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


class HandoverTests(unittest.TestCase):
    def test_terminal_keys_failures_and_ambiguous_selected_preserved(self):
        requests = [dict(method='PR_DF', design=dict(L_D=i, delta_pf=.1)) for i in (1, 2, 3)]
        keys = [H.request_key(r) for r in requests]
        state = dict(cycle=3, round_id='R08', stop=dict(reason='RUNTIME_EXCEPTION'),
                     points=[dict(status='COMPLETION', completion_request=r) for r in requests])
        completion = dict(source='old', records={
            keys[0]: dict(status='READY_FOR_SEARCH', request=requests[0], key=keys[0], configuration_id='one'),
            keys[1]: dict(status='HELD_WITH_REASON', request=requests[1], key=keys[1], reason='STAGE_FAILED:SIGNAL'),
            keys[2]: dict(status='SELECTED', request=requests[2], key=keys[2])})
        original = deepcopy((state, completion))
        new, imported, held = H.reconcile(state, completion, dict(code_source='old'))
        self.assertEqual((state, completion), original)
        self.assertEqual([r['status'] for r in new['points']], ['REGISTERED', 'HELD', 'HELD'])
        self.assertEqual(new['resumed_from']['previous_stop'], state['stop'])
        self.assertEqual(imported['records'][keys[1]]['reason'], 'STAGE_FAILED:SIGNAL')
        self.assertEqual(set(imported['records']), set(keys))
        self.assertEqual(held, [keys[2]])
        self.assertEqual(new['cycle'], 3)

    def test_historical_request_key_and_signed_zero_not_aliased(self):
        request = dict(method='PR_DF', design=dict(delta_pf=-0.0))
        completed = dict(source='old', records={'historical':dict(status='READY_FOR_SEARCH', request=request,
                                                                  key='historical', configuration_id='exact')})
        state = dict(points=[dict(status='COMPLETION', completion_request=request),
                             dict(status='COMPLETION', completion_request=dict(method='PR_DF', design=dict(delta_pf=0.0)))])
        result, _, _ = H.reconcile(state, completed, {})
        self.assertEqual(result['points'][0]['completion']['key'], 'historical')
        self.assertEqual(result['points'][1]['status'], 'COMPLETION')

    def test_q3_missing_or_failed_return_is_not_automatic_retry(self):
        with tempfile.TemporaryDirectory() as t:
            run = Path(t)
            job = run / 'jobs/PR_DF/job'
            job.mkdir(parents=True)
            H.write(job / 'REQUEST.json', {})
            H.write(job / 'LAUNCH.json', {})
            self.assertEqual(H.q3_preflight(run)['unresolved'][0]['reason'], 'OWNED_Q3_WITHOUT_OUTCOME')
            H.write(job / 'OUTCOME.json', dict(status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'))
            H.write(job / 'PROCESS_RETURN.json', dict(returncode=1))
            self.assertEqual(H.q3_preflight(run)['unresolved'][0]['reason'], 'PRIOR_Q3_PROCESS_RETURN_NOT_CLEAN')
            H.write(job / 'PROCESS_RETURN.json', dict(returncode=0))
            self.assertEqual(H.q3_preflight(run)['unresolved'], [])


if __name__ == '__main__':
    unittest.main()
