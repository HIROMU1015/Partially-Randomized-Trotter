"""Small synthetic checks of ID selection and literal transcription primitives."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    'saved_integration', Path(__file__).parents[1] / 'scripts/integrate_pauli_h2_h4_h6_saved_results.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SavedIntegrationChecks(unittest.TestCase):
    def test_boundary_classes_and_invalid_prefix(self):
        self.assertEqual(MODULE.classify('0', '3'), 'FULLY_RANDOMIZED_RTE')
        self.assertEqual(MODULE.classify('1', '3'), 'HYBRID_PR')
        self.assertEqual(MODULE.classify('3', '3'), 'FULL_RETENTION_NO_NONTRIVIAL_RANDOMIZED_TAIL')
        with self.assertRaises(ValueError):
            MODULE.classify('4', '3')

    def test_join_does_not_choose_minimum_or_accept_duplicate(self):
        rows = [dict(candidate='nominee', G='999/3'), dict(candidate='challenger', G='1/3')]
        self.assertEqual(MODULE.one(rows, candidate='nominee')['G'], '999/3')
        with self.assertRaises(ValueError):
            MODULE.one(rows + rows[:1], candidate='nominee')
        with self.assertRaises(ValueError):
            MODULE.one(rows, candidate='missing')

    def test_json_numeric_tokens_are_not_rounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.json'
            path.write_text('{"decimal": 0.123456789012345678901, "ratio": "7/3", "count": 12}')
            result = MODULE.quoted_json(path)
        self.assertEqual(result, dict(decimal='0.123456789012345678901', ratio='7/3', count='12'))


if __name__ == '__main__':
    unittest.main()
