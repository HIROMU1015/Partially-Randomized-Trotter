"""Meaningful grant boundaries without molecular operations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('migration', Path(__file__).parents[1]/'src/trotterlib/h246_migration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class GrantTests(unittest.TestCase):
    def test_unapproved_refused_before_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'EXPLICIT'):
                m.claim(Path(tmp)/'marker', {'valid': True})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_consumed_grant_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'marker'
            grant = dict(valid=True, approved_by_user=True, retry=False)
            m.claim(p, grant)
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                m.claim(p, grant)
            self.assertEqual(p.read_bytes(), original)

    def test_modified_budget_refused_before_source_io(self):
        with self.assertRaisesRegex(ValueError, 'EXACT_NEW_GENERATION_GRANT'):
            m.verify_generation('/absent', dict(call_budget={'compile':0}, SOURCE='a'*40),
                dict(valid=True, approved_by_user=True, retry=False, plan_digest='wrong'))
