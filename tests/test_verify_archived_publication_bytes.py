"""Failure and scope checks for the read-only fixed-publication byte helper."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_archived_publication_bytes.py'
spec = importlib.util.spec_from_file_location('archived_publication_bytes', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ArchivedPublicationBytesTest(unittest.TestCase):
    def fixture(self, payload=b'fixed SOURCE\n', extra=None):
        case = dict(checker.CASES['b3-api-preflight'])
        path = 'scripts/example.py'
        row = dict(path=path, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                   url=checker.fixed_url(case['content_commit'], path))
        rows = [row] + (extra or [])
        receipt = dict(content_commit=case['content_commit'], file_count=len(rows), files=rows)
        raw = json.dumps(receipt).encode()
        case['receipt_blob'] = checker.blob_oid(raw)
        return case, row, raw

    def test_changed_source_bytes_fail(self):
        case, row, receipt = self.fixture()
        with patch.dict(checker.CASES, fixture=case), patch.object(checker, 'read_file', side_effect=[receipt, b'changed SOURCE\n']):
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                checker.verify('fixture')

    def test_missing_public_file_fails(self):
        case, row, receipt = self.fixture()
        with patch.dict(checker.CASES, fixture=case), patch.object(checker, 'read_file', side_effect=[receipt, OSError('404')]):
            with self.assertRaises(OSError):
                checker.verify('fixture')

    def test_same_path_at_another_commit_is_rejected(self):
        case, row, raw = self.fixture()
        receipt = json.loads(raw)
        receipt['files'][0]['url'] = checker.fixed_url('1' * 40, row['path'])
        with self.assertRaisesRegex(ValueError, 'fixed content snapshot'):
            checker.select_rows(receipt, case)

    def test_changed_original_receipt_is_rejected(self):
        case, row, raw = self.fixture()
        with patch.dict(checker.CASES, fixture=case), patch.object(checker, 'read_file', return_value=raw + b' '):
            with self.assertRaisesRegex(ValueError, 'receipt blob mismatch'):
                checker.verify('fixture')

    def test_raw_and_bundle_parts_are_reported_without_reading(self):
        case = checker.CASES['b3-api-preflight']
        excluded = [dict(path=p, bytes=1, sha256=hashlib.sha256(b'x').hexdigest(),
                         url=checker.fixed_url(case['content_commit'], p))
                    for p in ['raw/trajectory.json', 'history.bundle.part000']]
        case, row, receipt = self.fixture(extra=excluded)
        with patch.dict(checker.CASES, fixture=case), patch.object(checker, 'read_file', side_effect=[receipt, b'fixed SOURCE\n']) as reader:
            result = checker.verify('fixture')
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(result['checked_listed_files'], 1)
        self.assertEqual(result['skipped_files'], [r['path'] for r in excluded])
        self.assertEqual(result['original_receipt_files'], 3)

    def test_empty_verification_scope_is_not_a_pass(self):
        case, row, raw = self.fixture()
        receipt = json.loads(raw)
        receipt['files'] = []
        receipt['file_count'] = 0
        with self.assertRaisesRegex(ValueError, 'No report'):
            checker.select_rows(receipt, case)


if __name__ == '__main__':
    unittest.main()
