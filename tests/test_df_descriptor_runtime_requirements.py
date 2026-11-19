"""Synthetic dependency preflight; no saved inputs or scientific work."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from trottertracks import rpe_preparation_bridge as B
from trottertracks import rpe_preparation_worker as W


class DescriptorRuntime(unittest.TestCase):
    def test_descriptor_probe_includes_public_package_imports_only_for_provider(self):
        with tempfile.TemporaryDirectory() as temp:
            script = Path(temp) / 'leaf.py'
            script.write_text('# synthetic fixed descriptor\n')
            configured = {'packages': {'numpy': '1.26.4'}, 'imports': ['json']}
            runtime, _ = B.contract_runtime([], script, configured, 'DF_DESCRIPTOR')
            self.assertEqual(runtime['packages'], configured['packages'])
            self.assertEqual(runtime['imports'], ['json', 'matplotlib.pyplot', 'openfermionpyscf'])
            self.assertEqual(configured['imports'], ['json'])
            ordinary, _ = B.contract_runtime([], script, configured, 'DF_SIGNALS')
            self.assertEqual(ordinary, configured)

    def test_worker_runtime_preflight_rejects_missing_plotting_before_leaf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Shadow only the absent package; the probe really imports in a
            # subprocess, as it does during the worker dry-check.
            (root / 'matplotlib.py').write_text(
                "raise ModuleNotFoundError(\"No module named 'matplotlib'\")\n")
            (root / 'leaf.py').write_text('# synthetic leaf never executed\n')
            runtime, _ = B.contract_runtime([], root / 'leaf.py', {}, 'DF_DESCRIPTOR')
            with patch.dict(os.environ, {'PYTHONPATH': str(root)}):
                with self.assertRaisesRegex(ValueError, '(?s)runtime/package probe failed:.*matplotlib'):
                    W._runtime({'interpreter': sys.executable}, runtime)
            self.assertFalse((root / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
