"""Artificial metadata fixtures only: no molecular imports, RNG or circuits.

Reuse the exact optional-export guard and flag expression from fixed H8 code.
The scientific loader itself is never imported or called.
Run: python3 -B <this file>
"""
import ast
import copy
import hashlib
import subprocess
from types import SimpleNamespace
import unittest

CARRIER = '0c49a88134066177816feca2741f36f6b88d23ba'
SOURCE = '745a674546f2cb6258526fbfde1d9a00601689f8'
LOADER_SHA256 = '89d6fc478692cf1f0b52a7acdd00d1a05fc42449e961247ac09c5668a7d4e02b'

def git(ref, path):
    return subprocess.check_output(['git', '--no-optional-locks', 'show', ref + ':' + path])

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

class FlagSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = git(CARRIER, 'scripts/h8_ab_native.py')
        require(hashlib.sha256(raw).hexdigest() == LOADER_SHA256, 'FIXED_LOADER_BYTES')
        tree = ast.parse(raw)
        load = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load')
        guard = next(n for n in load.body if isinstance(n, ast.For)
                     and ast.unparse(n.iter) == 'preps.items()')
        cls.guard_code = compile(ast.Module(body=[guard], type_ignores=[]), '<fixed export guard>', 'exec')
        flag = next(n.value for n in ast.walk(load) if isinstance(n, ast.keyword)
                    and n.arg == 'saved_exact_export_match')
        cls.flag_code = compile(ast.Expression(body=flag), '<fixed flag expression>', 'eval')
        tree = ast.parse(git(SOURCE, 'scripts/h8_pr_confirmation.py'))
        identity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_identity')
        scope = {'c': SimpleNamespace(require=require)}
        exec(compile(ast.Module(body=[identity], type_ignores=[]), '<fixed native identity>', 'exec'), scope)
        cls.normalize_identity = staticmethod(scope['native_identity'])

    def check_guard(self, prefix, export, observed):
        calls = []
        def fake_export(_):
            calls.append(prefix)
            return observed
        scope = dict(preps={prefix: object()}, export=export,
                     preparation_export=fake_export, c=SimpleNamespace(require=require))
        exec(self.guard_code, scope)
        return calls

    def flag(self, prefix, export):
        return eval(self.flag_code, {'p': prefix, 'export': export})

    def test_missing_prefix_is_false_without_comparison(self):
        export = {'faithful:p7': {'metadata': 'unchanged'}}
        self.assertEqual(self.check_guard(8, export, {'metadata': 'arbitrary fixture'}), [])
        self.assertFalse(self.flag(8, export))

    def test_present_equal_prefix_is_true(self):
        export = {'faithful:p7': {'metadata': 'same'}}
        self.assertEqual(self.check_guard(7, export, {'metadata': 'same'}), [7])
        self.assertTrue(self.flag(7, export))

    def test_present_mismatch_raises_instead_of_saving_false(self):
        export = {'faithful:p8': {'metadata': 'expected'}}
        with self.assertRaisesRegex(ValueError, 'EXACT_SAVED_PREFIX_COMPONENT_BASIS_BINDING'):
            self.check_guard(8, export, {'metadata': 'different'})
        self.assertTrue(self.flag(8, export))

    def test_missing_key_does_not_mutate_export(self):
        export = {'faithful:p28': {'metadata': ['saved full blocks']}}
        before = copy.deepcopy(export)
        self.check_guard(8, export, {'metadata': 'fixture'})
        self.assertEqual(export, before)

    def test_matching_new_prefix_requires_no_old_prefix_assumption(self):
        export = {'faithful:p8': {'metadata': 'same'}}
        self.assertEqual(self.check_guard(8, export, {'metadata': 'same'}), [8])
        self.assertTrue(self.flag(8, export))

    def test_native_identity_list_tuple_are_equal_after_normalization(self):
        self.assertEqual(self.normalize_identity([1, 2, 3, 4]),
                         self.normalize_identity((1, 2, 3, 4)))

    def test_native_identity_still_rejects_invalid_metadata(self):
        for value in ([1, 2, 3], [1, 2, 3, True], '1234'):
            with self.assertRaises(ValueError):
                self.normalize_identity(value)

if __name__ == '__main__':
    unittest.main(verbosity=2)
