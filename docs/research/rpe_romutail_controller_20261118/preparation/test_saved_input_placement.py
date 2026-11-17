"""Synthetic file integrity checks; no scientific import or execution."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('saved_placement', ROOT / 'scripts/prepare_romutail_preparation_inputs.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class PlacementTests(unittest.TestCase):
    def fixture(self, base, raw=b'SYNTHETIC CONFIG'):
        expected = hashlib.sha256(b'SYNTHETIC CONFIG').hexdigest()
        tasks = [dict(method='PR_DF', preparation=dict(dry_check_only=True,
                 required_paths=dict(config=dict(kind='file', sha256=expected))))]
        profile = dict(host_id='synthetic', preparation_profiles=[dict(profile={})])
        package = base / 'inputs.tar'
        with tarfile.open(package, 'w') as archive:
            member = tarfile.TarInfo('config/config.json')
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        return tasks, profile, package

    def test_correct_package_maps_host_path_and_preserves_original_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tasks, profile, package = self.fixture(base)
            result = M.prepare(tasks, profile, package, base, base / 'owned')
            self.assertEqual(result['files_verified'], 1)
            self.assertNotIn('preparation_paths', profile)
            self.assertEqual((base / 'owned/runtime/config/config.json').read_bytes(), b'SYNTHETIC CONFIG')

    def test_changed_saved_config_is_rejected_before_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tasks, profile, package = self.fixture(base, b'DIFFERENT')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                M.prepare(tasks, profile, package, base, base / 'owned')
            self.assertFalse((base / 'owned/PROFILE.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
