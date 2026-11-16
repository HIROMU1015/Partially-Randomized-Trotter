"""Descriptor bridge tests: JSON metadata and artificial signed fragments only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
package = ModuleType('trotterlib'); package.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules.setdefault('trotterlib', package)
from trottertracks import pauli_split_descriptor as D


class DescriptorTests(unittest.TestCase):
    def test_fragment_range_comes_from_snapshot_shapes(self):
        receipt = dict(arrays=dict(lambdas=dict(shape=[7]), g_matrices=dict(shape=[7, 3, 3])), rank=7)
        self.assertEqual(D.fragment_count(receipt), 7)
        for invalid in ({}, dict(receipt, rank=8), dict(arrays=dict(
                lambdas=dict(shape=[7]), g_matrices=dict(shape=[6, 3, 3])))):
            with self.assertRaises(ValueError):
                D.fragment_count(invalid)

    def test_fixed_numerical_leaf_matches_saved_artificial_descriptor(self):
        # The two tail fragments have I/Z1/Z2/ZZ coefficients
        # .375,-.25,.125,-.25 and -.5,.5,0,0 respectively.
        ham = SimpleNamespace(n_blocks=3, lambdas=np.array([5., 1., -1.]),
            g_matrices=[np.eye(2), np.diag([1., -.5]), np.diag([1., 0.])])
        saved = dict(scalar=-.125, lambda_R=1.125)
        with patch.object(np, 'load', side_effect=AssertionError('No saved H arrays')):
            value = D.derive_descriptor(ROOT, ham, 1)
        self.assertEqual(value, saved)
        self.assertEqual(D.saved_descriptor(dict(prefixes={'1': saved}), 1,
            dict(commit='ARTIFICIAL', path='context.json'))['descriptor'], value)
        for p in (0, 3, True):
            with self.assertRaises(ValueError):
                D.derive_descriptor(ROOT, ham, p)

    def test_missing_descriptor_preserves_control_context_and_restores_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)/'context.json'
            payload = dict(prefixes={'1': dict(scalar=.4, lambda_R=2.),
                                     '2': dict(scalar=.2, lambda_R=1.)})
            raw = json.dumps(payload).encode(); context.write_bytes(raw)
            receipt = Path(tmp)/'receipt.json'
            receipt.write_text(json.dumps(dict(arrays=dict(lambdas=dict(shape=[5]),
                g_matrices=dict(shape=[5, 2, 2])))))
            base = dict(context_source=dict(path=str(context), sha256=hashlib.sha256(raw).hexdigest()),
                inputs={'H4_tol_receipt.json': dict(path=str(receipt))},
                saved_tail_descriptors=deepcopy(payload['prefixes']), binding=dict(system='ARTIFICIAL'))
            descriptor = dict(scalar=.025, lambda_R=.5)
            source = dict(commit='ARTIFICIAL', path='LD4_descriptor.json')
            prior = D.L.p0_metadata
            with patch.object(D.L.p0, 'metadata', return_value=deepcopy(base)), \
                    patch.object(np, 'load', side_effect=AssertionError('No arrays')):
                missing = D.saved_descriptor(payload, 4, source)
                self.assertEqual(missing['status'], 'DESCRIPTOR_DERIVATION_REQUIRED')
                self.assertIn('L_D=4', missing['missing_inputs'][0])
                with self.assertRaisesRegex(RuntimeError, 'planned'):
                    with D.stage1_metadata(ROOT, 4, descriptor, source):
                        expanded = D.L.p0_metadata(ROOT, (2, 4))
                        self.assertEqual(expanded['saved_tail_descriptors']['2'], payload['prefixes']['2'])
                        self.assertEqual(expanded['saved_tail_descriptors']['4'], descriptor)
                        self.assertEqual(expanded['tail_descriptor_sources']['4'], source)
                        with self.assertRaisesRegex(ValueError, 'ANOTHER_SPLIT'):
                            D.L.p0_metadata(ROOT, (2, 3))
                        raise RuntimeError('planned')
                self.assertIs(D.L.p0_metadata, prior)
                with self.assertRaisesRegex(ValueError, 'DIFFERS_FROM_SAVED'):
                    D.metadata_with_descriptor(ROOT, 1, descriptor, source)
                with self.assertRaisesRegex(ValueError, 'SOURCE'):
                    D.metadata_with_descriptor(ROOT, 4, descriptor, None)
            self.assertEqual(context.read_bytes(), raw)
            self.assertEqual(base['saved_tail_descriptors'], payload['prefixes'])


if __name__ == '__main__':
    unittest.main()
