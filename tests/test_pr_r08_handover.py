"""Artificial metadata only: R08 handover never launches scientific leaves."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
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

    def test_existing_configuration_guard_holds_only_affected_point(self):
        from types import SimpleNamespace
        def check(point, record):
            if record['key'] == 'bad':
                raise ValueError('DIFFERENT_SPLIT_LAMBDA_SOURCE')
        driver = SimpleNamespace(state=dict(points=[
            dict(family='f', kappa='1', status='REGISTERED', completion=dict(key='bad')),
            dict(family='g', kappa='2', status='REGISTERED', completion=dict(key='good'))]),
            check_completed_configuration=check)
        records = {key:dict(key=key, status='READY_FOR_SEARCH') for key in ('bad', 'good')}
        held = H.validate_ready_points(driver, records)
        self.assertEqual([p['status'] for p in driver.state['points']], ['HELD', 'REGISTERED'])
        self.assertEqual(held[0]['reason'], 'COMPLETED_CONFIGURATION_MISMATCH:DIFFERENT_SPLIT_LAMBDA_SOURCE')
        self.assertEqual(records['bad']['status'], 'READY_FOR_SEARCH')

    def test_previous_handover_identity_and_terminal_status(self):
        with tempfile.TemporaryDirectory() as t:
            study = Path(t).resolve()
            run = study / 'run02'
            plan = dict(previous_handover_wrapper=True, old_code_source='previous-code',
                        old_input_commit='previous-input')
            H.write(run / 'CONTROLLER_PID.json', dict(status='CONTROLLER_START',
                source='previous-code', input_commit='previous-input'))
            terminal = dict(status='CONTROLLER_RETURN', run_directory=str(run), returncode=-2,
                            stop=dict(reason='RUNTIME_EXCEPTION', error_type='KeyboardInterrupt'))
            status = study / 'HANDOVER_STATUS.json'
            H.write(status, terminal)
            self.assertEqual(H.previous_controller_return(plan, run), terminal)
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                H.previous_controller_return(dict(plan, old_input_commit='other-input'), run)
            H.write(status, dict(terminal, status='FIXED_SOURCE_RUNNING'))
            with self.assertRaisesRegex(RuntimeError, 'terminal'):
                H.previous_controller_return(plan, run)

    def test_preparation_boundary_and_controller_only_signal(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t).resolve()
            old = root / 'old' / 'run02'
            study = root / 'new'
            H.write(old / 'KAPPA_STATE.json', dict(cycle=12))
            H.write(old / 'completion/STATE.json', dict(records={'a': dict(status='SELECTED')}))
            # The progress files use one JSON object per line.
            (old / 'ROUND_STAGES.jsonl').write_text(json.dumps(dict(event='START',
                phase='WIDTH_READY_DURING_PREPARATION')) + '\n')
            progress = old / 'DRIVER_PROGRESS.jsonl'
            progress.write_text(json.dumps(dict(cycle=12, event='PREPARATION_WAVE_START')) + '\n')
            self.assertFalse(H.preparation_wave_drained(old))
            progress.write_text(json.dumps(dict(cycle=12, event='PREPARATION_WAVE_END')) + '\n')
            self.assertFalse(H.preparation_wave_drained(old))
            H.write(old / 'completion/STATE.json', dict(records={'a': dict(status='READY_FOR_SEARCH')}))
            self.assertTrue(H.preparation_wave_drained(old))
            child_command = [sys.executable, '-c', 'import time; time.sleep(30)']
            child = subprocess.Popen(child_command, cwd=root, stderr=subprocess.DEVNULL)
            try:
                H.write(old / 'CONTROLLER_PID.json', dict(pid=child.pid, command=child_command,
                                                         source='previous-code', input_commit='previous-input'))
                H.write(old.parent / 'HANDOVER_STATUS.json', dict(status='FIXED_SOURCE_RUNNING'))
                plan = dict(old_pid=child.pid, old_root=str(root), old_code_source='previous-code',
                            old_input_commit='previous-input', new_code_source='new-code')
                status = study / 'HANDOVER_STATUS.json'
                H.request_drain_at_boundary(plan, study, old, status)
                self.assertIn(child.wait(timeout=5), (-signal.SIGINT, 130))
                request = study / 'DRAIN_REQUEST.json'
                saved = request.read_bytes()
                self.assertEqual(H.read(request)['worker_signals'], 0)
                H.request_drain_at_boundary(plan, study, old, status)
                self.assertEqual(request.read_bytes(), saved)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()

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
