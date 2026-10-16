"""No-run metadata/guard tests. Never import/evaluate bounds or analysis kernels."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_v1 import control, spec


class MetadataOnly(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((ROOT / spec.ART / 'execution_contract_candidate.json').read_text())

    def test_candidate_valid_but_not_adopted(self):
        control.validate_contract(self.c)
        with self.assertRaises(control.ExecutionClosed):
            control.validate_contract(self.c, require_adopted=True)

    def test_no_scientific_defaults(self):
        self.assertTrue(all(v is None for v in self.c['science'].values()))
        self.assertFalse(self.c['adopted'])
        self.assertFalse(self.c['coverage_model_adopted'])
        self.assertFalse(self.c['contract_grants_execution'])

    def test_candidate_closes_before_authority_input_or_git(self):
        with patch.object(control, 'git', side_effect=AssertionError('git reached')):
            with self.assertRaises(control.ExecutionClosed):
                control.authorize(ROOT, spec.ART + '/execution_contract_candidate.json', None, None)

    def test_flags_alone_never_complete_contract(self):
        c = copy.deepcopy(self.c)
        c.update(adopted=True, coverage_model_adopted=True,
                 status='ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL')
        with self.assertRaises(control.ExecutionClosed):
            control.validate_contract(c, require_adopted=True)

    def test_immutable_conditions(self):
        for k in ['model', 'grid', 'input', 'direction', 'resources', 'shot_formula']:
            c = copy.deepcopy(self.c)
            c[k] = None
            with self.subTest(k=k), self.assertRaises(control.ExecutionClosed):
                control.validate_contract(c)

    def test_closed_private_permit(self):
        with self.assertRaises(control.ExecutionClosed):
            control.require_permit(None)
        with self.assertRaises(control.ExecutionClosed):
            control.run(ROOT, None)

    def test_paths(self):
        for p in ['../escape', '/tmp/out', 'a//b', 'a/./b', 'a\\b', 'a/../b']:
            with self.subTest(p=p), self.assertRaises(control.ExecutionClosed):
                control.relative(p)

    def test_duplicate_and_nonfinite_json_tokens(self):
        for raw in ['{"a":null,"a":null}', '{"a":NaN}', '{"a":Infinity}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                control.read_json(raw)

    def test_source_is_ast_parseable_without_bytecode_compile(self):
        for name in spec.SOURCE_FILES:
            if name.endswith('.py'):
                with self.subTest(name=name):
                    ast.parse((ROOT / name).read_text())

    def test_only_fixed_identifiers_enumerated(self):
        identifiers = spec.keys()
        self.assertEqual(len(identifiers), 108)
        self.assertEqual(len(set(identifiers)), 108)
        self.assertEqual(sum(k[3] == '0' for k in identifiers), 36)

    def test_no_arithmetic_engine_imported(self):
        for suffix in ['bounds', 'analysis']:
            self.assertNotIn('trottertracks.algorithm_codesign.m1s_saved_signal_v1.' + suffix, sys.modules)

    def test_result_schema_metadata(self):
        schema = json.loads((ROOT / spec.ART / 'result_schema.json').read_text())
        self.assertEqual(schema['properties']['rows']['minItems'], 108)
        self.assertEqual(schema['properties']['comparisons']['maxItems'], 48)
        self.assertEqual(schema['properties']['scientific_GO']['type'], 'null')


if __name__ == '__main__':
    unittest.main()
