"""Artificial process control and inactive recovery-contract checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import pf_common_run_contract as c
import prepare_h6_pf_cost_recovery as recovery
import run_detached_local_job as detached


def inactive_fixture():
    row = lambda formula, method, prefix, q: dict(system='H6', formula=formula, method=method, prefix=prefix, q=q)
    rows = [row('4th', 'FULL', 19, 8)]
    rows += [row('4th(m5_best)', m, p, 4) for p in (10, 12, 13, 14) for m in ('B0', 'B0I')]
    rows += [row('4th(m5_best)', 'B0I', 6, 8)]
    rows += [row('4th(m5_best)', m, p, 8) for p in (8, 9, 10, 12, 13, 14) for m in ('B0', 'B0I')]
    return dict(status='UNAPPROVED_NOT_A_GRANT', execution_authorized=False,
        additional_budget_approved=False, new_grants_created=0, execution_command=None,
        retry=False, resume=False, expected22_ids=sorted(c.cid(r) for r in rows),
        fresh_budget=dict(candidates=rows, exceeds_fixed_internal_budget=True,
            fresh_caps=dict(native_evolution=22, native_piece=1520, wrapper_compile=44, wrapper_build=44)),
        stages=dict(ACCOUNT=dict(requires_new_COST_success=True)), seals=[])


class InactiveContractTests(unittest.TestCase):
    def test_exact_fixture_is_inactive_and_has_original_piece_count(self):
        p = inactive_fixture()
        self.assertTrue(recovery.validate_inactive(p))
        self.assertEqual(sum(r['q']*c.PIECES[r['formula']] for r in p['fresh_budget']['candidates']), 1520)

    def test_activation_grant_or_command_is_rejected(self):
        for key, value in [('execution_authorized', True), ('additional_budget_approved', True),
                           ('new_grants_created', 1), ('execution_command', ['python']),
                           ('resume', True), ('retry', True)]:
            with self.subTest(key=key):
                p = inactive_fixture(); p[key] = value
                with self.assertRaises(ValueError):
                    recovery.validate_inactive(p)

    def test_candidate_substitution_and_budget_edits_are_rejected(self):
        for field, value in [('prefix', 18), ('q', 16), ('system', 'H4')]:
            p = inactive_fixture(); p['fresh_budget']['candidates'][1][field] = value
            with self.assertRaises(ValueError): recovery.validate_inactive(p)
        p = inactive_fixture(); p['fresh_budget']['candidates'][1] = p['fresh_budget']['candidates'][2]
        with self.assertRaises(ValueError): recovery.validate_inactive(p)
        p = inactive_fixture(); p['fresh_budget']['fresh_caps']['native_piece'] += 1
        with self.assertRaises(ValueError): recovery.validate_inactive(p)

    def test_changed_old_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old_STOP.json'; path.write_text('original')
            p = inactive_fixture(); p['seals'] = [c.seal(path, 'old_STOP')]
            self.assertTrue(recovery.verify_seals(p))
            path.write_text('modified')
            with self.assertRaises(ValueError): recovery.verify_seals(p)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.affinity = os.sched_getaffinity(0)

    def tearDown(self):
        os.sched_setaffinity(0, self.affinity)
        self.tmp.cleanup()

    def request(self, command):
        data = dict(command=command, cwd=str(self.root), env={},
            CPU_affinity=sorted(self.affinity), source_sha256=hashlib.sha256(Path(detached.__file__).read_bytes()).hexdigest())
        detached.write(self.root/'request.json', data)
        return hashlib.sha256((self.root/'request.json').read_bytes()).hexdigest()

    def test_child_success_marks_stopped_heartbeat_and_cannot_repeat(self):
        sha = self.request([sys.executable, '-c', 'print("artificial")'])
        self.assertEqual(detached.worker(self.root, sha), 0)
        self.assertFalse(json.loads((self.root/'heartbeat.json').read_text())['running'])
        receipt = (self.root/'exit_receipt.json').read_bytes()
        self.assertTrue(detached.status(self.root)['child_reaped'])
        self.assertTrue(detached.status(self.root)['scientific_stage_success_must_be_checked_separately'])
        with self.assertRaises(FileExistsError): detached.worker(self.root, sha)
        self.assertEqual((self.root/'exit_receipt.json').read_bytes(), receipt)
        self.assertFalse((self.root/'STOP.json').exists())

    def test_nonzero_child_is_stopped_without_retry(self):
        sha = self.request([sys.executable, '-c', 'raise SystemExit(7)'])
        self.assertEqual(detached.worker(self.root, sha), 7)
        s = detached.status(self.root)
        self.assertEqual(s['reason'], 'CHILD_NONZERO_EXIT')
        self.assertEqual(s['returncode'], 7)
        self.assertFalse(s['automatic_retry'])

    def test_spawn_failure_is_recorded_before_any_child(self):
        sha = self.request([str(self.root/'missing-executable')])
        with self.assertRaises(FileNotFoundError): detached.worker(self.root, sha)
        s = detached.status(self.root)
        self.assertFalse(s['child_launched'])
        self.assertFalse(s['scientific_stage_success'])
        self.assertFalse((self.root/'exit_receipt.json').exists())

    def test_changed_request_is_rejected_and_stopped(self):
        sha = self.request([sys.executable, '-c', 'pass'])
        (self.root/'request.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Request changed'): detached.worker(self.root, sha)
        self.assertFalse(detached.status(self.root)['child_launched'])

    def test_absent_or_reused_supervisor_pid_is_never_success(self):
        detached.write(self.root/'worker_entered.json', dict(pid=os.getpid(), process_identity=dict(pid=os.getpid(), start_ticks='-1')))
        self.assertEqual(detached.status(self.root)['status'], 'PROCESS_ABSENT_NO_SUCCESS_RECEIPT')
        self.assertFalse(detached.status(self.root)['scientific_stage_success'])

    def test_sigterm_reaps_only_its_child_and_preserves_unrelated_child(self):
        sha = self.request([sys.executable, '-c', 'import time; time.sleep(30)'])
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        supervisor = subprocess.Popen([sys.executable, '-B', str(SCRIPTS/'run_detached_local_job.py'),
            'worker', '--root', str(self.root), '--request-sha', sha], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            limit = time.monotonic()+5
            while not (self.root/'child_started.json').exists() and time.monotonic() < limit:
                time.sleep(.02)
            self.assertTrue((self.root/'child_started.json').exists())
            supervisor.send_signal(signal.SIGTERM)
            supervisor.wait(timeout=10)
            s = detached.status(self.root)
            self.assertEqual(s['signal'], signal.SIGTERM)
            self.assertTrue(s['child_reaped'])
            self.assertFalse(s['automatic_retry'])
            self.assertIsNone(unrelated.poll())
            child = json.loads((self.root/'child_started.json').read_text())
            self.assertNotEqual(detached.process_identity(child['pid']), child['process_identity'])
        finally:
            if supervisor.poll() is None:
                supervisor.terminate(); supervisor.wait(timeout=10)
            unrelated.terminate(); unrelated.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
