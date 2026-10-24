"""Approval, call accounting and failure preservation without science imports."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import rustiq_execution as execution


class ExecutionManagementTests(unittest.TestCase):
    def tearDown(self):
        for name in ('numpy', 'qiskit', 'mpmath', 'trotterlib'):
            self.assertNotIn(name, sys.modules)

    def test_foreign_approval_is_rejected_before_source_or_science(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            execution.write_json(root/'approval.json', dict(stage='F1', approved=True))
            with patch.object(execution.subprocess, 'check_output', side_effect=AssertionError('No git')), \
                    self.assertRaises(PermissionError):
                execution.verify_packet(root, root, root, root/'output')

    def test_bound_source_and_input_and_fresh_output_are_required(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'journals').mkdir()
            (root/'input').write_bytes(b'artificial input')
            contract = dict(cases=[dict(id=x) for x in execution.CASE_IDS],
                source_files={}, input_files={'input': execution.file_sha(root/'input')})
            (root/execution.CONTRACT).parent.mkdir(parents=True)
            execution.write_json(root/execution.CONTRACT, contract)
            sha = execution.file_sha(root/execution.CONTRACT)
            approval = dict(stage='C4_Q13_LIMITED_V1', authority='USER_EXPLICIT_C4_EXECUTION',
                approved=True, contract_sha256=sha, source_commit='synthetic-commit',
                output=str(root/'output'), execution_files={})
            execution.write_json(root/'approval.json', approval)
            def git(cmd, **kwargs):
                return 'synthetic-commit\n' if cmd[1] == 'rev-parse' else b''
            with patch.object(execution, 'CONTRACT_SHA', sha), \
                    patch.object(execution.subprocess, 'check_output', side_effect=git):
                verified = execution.verify_packet(root, root, root, root/'output')
                self.assertEqual(verified['inputs']['input'], b'artificial input')
                (root/'input').write_bytes(b'drift')
                with self.assertRaises(ValueError):
                    execution.verify_packet(root, root, root, root/'output')
                (root/'output').mkdir()
                with self.assertRaises(PermissionError):
                    execution.verify_packet(root, root, root, root/'output')

    def test_journal_counts_completed_requests_and_preserves_failed_attempt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'journals').mkdir()
            journal = execution.Journal(root, 'artificial')
            self.assertEqual(journal.call('mock', 'one', lambda: 'token'), 'token')
            self.assertEqual(execution.journal_counts(root), ({'mock': 1}, 2))
            def fail():
                raise TypeError('artificial technical interruption')
            with self.assertRaises(TypeError):
                journal.call('mock', 'two', fail)
            self.assertTrue((root/'STOP.json').exists())
            self.assertIn('artificial technical interruption', journal.path.read_text())
            with self.assertRaises(RuntimeError):
                journal.call('mock', 'no-new-dependency', lambda: None)
            self.assertEqual(len(journal.path.read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
