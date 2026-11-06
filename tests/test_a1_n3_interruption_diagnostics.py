"""Invented journals and disposable owned children only; no science inputs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import a1_n3_interruption_diagnostics as audit
from trottertracks.algorithm_codesign import a1_n3_lifecycle_observer as observer


def toy():
    domain = dict(method='A1', axis='Y', stage='train', cell='2')
    parent = [dict(event='TRIAL_PLANNED', ticket='toy', domain=domain, trial=0),
              dict(event='PATH_FIXED', ticket='toy', domain=domain, generation_seconds=0)]
    worker = [dict(event='WORKER_STARTED', ticket='toy', domain=domain, trial=0),
              dict(event='CALL', ticket='toy', name='exact_engine'),
              dict(event='CALL', ticket='toy', name='selected_full_build')]
    return parent, worker, domain


class JournalTests(unittest.TestCase):
    def test_last_entry_does_not_prove_build_failed_or_still_running(self):
        parent, worker, _ = toy()
        result = audit.inspect_records(parent, {'toy.jsonl': worker})
        row = result['unfinished'][0]
        self.assertEqual(row['status'], 'ENGINE_ATTEMPT_NO_DURABLE_BIT')
        self.assertIn('UNKNOWN', row['last_call_return'])
        self.assertEqual(row['last_logged_call'], 'selected_full_build')

    def test_price_requires_durable_bit(self):
        parent, worker, _ = toy()
        worker.append(dict(event='PRICE_SAVED', ticket='toy', record={}))
        with self.assertRaisesRegex(ValueError, 'before bit'):
            audit.inspect_records(parent, {'toy.jsonl': worker})

    def test_price_record_mismatch_is_rejected(self):
        parent, worker, domain = toy()
        worker.extend([dict(event='BIT_SAVED', ticket='toy', record={'bit':0}, domain=domain, engine_seconds=1),
                       dict(event='PRICE_SAVED', ticket='toy', record={'bit':1})])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            audit.inspect_records(parent, {'toy.jsonl': worker})

    def test_duplicate_bit_is_rejected(self):
        parent, worker, domain = toy()
        bit = dict(event='BIT_SAVED', ticket='toy', record={}, domain=domain, engine_seconds=1)
        with self.assertRaisesRegex(ValueError, 'duplicate saved bit'):
            audit.inspect_records(parent, {'toy.jsonl': worker+[bit, bit]})

    def test_saved_bit_survives_absent_price_and_failed_worker(self):
        parent, worker, domain = toy()
        worker.extend([dict(event='BIT_SAVED', ticket='toy', record={'bit':0}, domain=domain, engine_seconds=1),
                       dict(event='WORKER_STOP', ticket='toy', reason='invented price failure')])
        result = audit.inspect_records(parent, {'toy.jsonl': worker})
        self.assertEqual(result['bits'], 1)
        self.assertEqual(result['prices'], 0)
        self.assertEqual(result['unfinished'][0]['status'], 'BIT_DURABLE_PRICE_MISSING')

    def test_truncated_tail_is_retained_as_unknown(self):
        rows, tail = audit.parse_records(b'{"event":"toy"}\n{"event":')
        self.assertEqual(len(rows), 1)
        self.assertEqual(tail['status'], 'UNKNOWN')
        self.assertEqual(tail['bytes'], 9)

    def test_malformed_middle_is_not_silently_skipped(self):
        with self.assertRaisesRegex(ValueError, 'malformed'):
            audit.parse_records(b'{bad}\n{"event":"toy"}\n')

    def test_frozen_reconcile_reuse_on_invented_records(self):
        parent, worker, domain = toy()
        with TemporaryDirectory() as directory:
            out = Path(directory)
            for name, rows in [('JOURNAL.jsonl',parent), ('worker_toy.jsonl',worker)]:
                (out/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            source = Path(__file__).resolve().parents[1]/'src/trottertracks/algorithm_codesign/a1_n3_campaign.py'
            audit.frozen_reconcile(source, out)
            self.assertEqual(json.loads((out/'CALL_COUNTS.json').read_text())['bits_saved'],0)
            self.assertEqual(json.loads((out/'ACCOUNTING.json').read_text())['methods']['A1']['bits_saved'],0)

    def test_missing_parent_identity_rejected(self):
        _, worker, _ = toy()
        with self.assertRaisesRegex(ValueError, 'identity'):
            audit.inspect_records([], {'toy.jsonl':worker})


class LifecycleTests(unittest.TestCase):
    def test_unknown_wait_status_stays_unknown(self):
        self.assertEqual(observer.wait_status(None)['status'], 'UNKNOWN')

    def test_exit137_is_not_inferred_to_be_sigkill(self):
        self.assertEqual(observer.wait_status(137)['status'], 'EXITED')
        self.assertIsNone(observer.wait_status(137)['signal'])

    def test_owned_child_exit_and_clock_identity(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)/'child'
            result=observer.observe_child([sys.executable,'-c','raise SystemExit(7)'],out)
            self.assertEqual(result['exit_code'],7)
            rows=[json.loads(line) for line in (out/'LIFECYCLE.jsonl').read_text().splitlines()]
            self.assertTrue(all('utc' in row and 'monotonic_ns' in row for row in rows))
            self.assertEqual(rows[-1]['event'],'CHILD_WAIT_STATUS')

    def test_owned_child_sigterm_and_sigkill(self):
        for number in (signal.SIGTERM,signal.SIGKILL):
            with self.subTest(signal=number), TemporaryDirectory() as directory:
                result=observer.observe_child([sys.executable,'-c',
                    'import os; os.kill(os.getpid(), '+str(int(number))+')'],Path(directory)/'child')
                self.assertEqual(result['signal'],number)
                self.assertEqual(result['status'],'SIGNALED')
                self.assertEqual(result['sender'],'UNKNOWN')

    def test_heartbeat_does_not_timeout_or_retry_child(self):
        with TemporaryDirectory() as directory:
            out=Path(directory)/'child'
            result=observer.observe_child([sys.executable,'-c','import time; time.sleep(0.06)'],out,poll_seconds=0.01)
            rows=[json.loads(line) for line in (out/'LIFECYCLE.jsonl').read_text().splitlines()]
            self.assertEqual(result['exit_code'],0)
            self.assertEqual(sum(row['event']=='CHILD_START' for row in rows),1)
            self.assertTrue(any(row['event']=='HEARTBEAT' for row in rows))

    def test_collision_starts_no_child(self):
        with TemporaryDirectory() as directory, patch.object(subprocess,'Popen') as spawn:
            with self.assertRaises(FileExistsError):
                observer.observe_child(['never'],directory)
            spawn.assert_not_called()

    def test_call_return_and_exception_never_log_result_or_erase_bit(self):
        with TemporaryDirectory() as directory:
            log=Path(directory)/'events'
            observer.event(log,'BIT_SAVED',ticket='toy',bit=0)
            with observer.traced_call(log,'toy','build'):
                secret={'state':'never logged','probability':'never logged'}
            with self.assertRaises(ValueError),observer.traced_call(log,'toy','price'):
                raise ValueError('invented failure')
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows],['BIT_SAVED','CALL_START','CALL_RETURN','CALL_START','CALL_EXCEPTION'])
            self.assertNotIn('probability',log.read_text())
            self.assertNotIn('state',log.read_text())

    def test_missing_process_or_memory_is_unknown(self):
        with TemporaryDirectory() as directory:
            self.assertEqual(observer.process_identity(123,Path(directory))['identity'],'UNKNOWN')
            self.assertEqual(observer.memory_snapshot(Path(directory),Path(directory))['cgroup_path'],'UNKNOWN')

    def test_child_cgroup_and_ancestors_not_monitor_cgroup(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);proc=root/'proc';group=root/'cgroup'
            (proc/'self').mkdir(parents=True)
            (proc/'self/cgroup').write_text('0::/monitor\n')
            (group/'monitor').mkdir(parents=True)
            (group/'job').mkdir()
            (group/'job/memory.max').write_text('300\n')
            (group/'monitor/memory.max').write_text('100\n')
            (group/'memory.max').write_text('900\n')
            data=observer.memory_snapshot(group,proc,cgroup_record='0::/job\n')
            self.assertEqual(data['memory.max'],'300\n')
            self.assertEqual(data['ancestor_cgroups'][str(group)]['memory.max'],'900\n')

    def test_no_science_modules_imported_in_fresh_interpreter(self):
        code='import sys; from trottertracks.algorithm_codesign import a1_n3_interruption_diagnostics,a1_n3_lifecycle_observer; assert not any(x in sys.modules for x in ("numpy","qiskit","openfermion","trottertracks.algorithm_codesign.a1_n3_campaign"))'
        subprocess.run([sys.executable,'-B','-c',code],check=True)


if __name__ == '__main__':
    unittest.main()
