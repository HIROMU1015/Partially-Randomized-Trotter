"""Artificial IDs only; compare with the original, uncached implementation."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import pf_pr_rpe_pipeline as p


def uncached(candidate):
    path = ROOT/'scripts/h46_long_time_signal.py'
    node = next(n for n in ast.parse(path.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'cid')
    scope = dict(json=json, hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['cid'](candidate)


def candidates():
    for system in ('TOY', 'ARTIFICIAL_SECOND_SYSTEM'):
        for method in ('FULL', 'B0', 'B0I', 'PR'):
            for formula in ('2nd', '4th', '4th(m5_best)'):
                for q in (1, 3, 4096):
                    yield dict(system=system, method=method, formula=formula, prefix=2,
                               T=.1, q=q, R_outer=7 if method=='PR' else None,
                               K=2 if method=='PR' else None, delta=.1/q)


class CandidateCacheTests(unittest.TestCase):
    def test_fixed_namespace_parser_keeps_original_code_and_global_ast(self):
        # An isolated loader namespace, as used by C's published B API.
        module=ModuleType('fixed_candidate_fixture')
        source=(ROOT/'scripts/h46_long_time_signal.py').read_bytes()
        module.__dict__.update(Path=Path,ast=ast,json=json,hashlib=hashlib,__file__=p.__file__)
        import subprocess
        old=subprocess.check_output(['git','show',
            'ce0517de6e7b58cf4bf4c69f99f998d86099c406:src/trottertracks/pf_pr_rpe_pipeline.py'],cwd=ROOT)
        fn=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='candidate_id')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'fixed_candidate_fixture','exec'),module.__dict__)
        original=module.candidate_id;parse=ast.parse
        p.cache_legacy_candidate_parser(module)
        for candidate in candidates():self.assertEqual(module.candidate_id(candidate),uncached(candidate))
        self.assertIs(original,module.candidate_id);self.assertIs(parse,ast.parse)
        self.assertEqual(module.ast.parse.cache_info().misses,1)
        self.assertEqual(ast.dump(module.ast.parse(source)),ast.dump(parse(source)))
        self.assertEqual(ast.dump(module.ast.parse(b'def cid(c): return 1')),ast.dump(parse('def cid(c): return 1')))
        self.assertEqual(module.ast.parse.cache_info().currsize,1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir()
            module.__file__=str(root/'src/trottertracks/pipeline.py')
            path=root/'scripts/h46_long_time_signal.py'
            path.write_text("def cid(c): return 'one'\n")
            self.assertEqual(module.candidate_id({}),'one')
            path.write_text("def cid(c): return 'two'\n")
            self.assertEqual(module.candidate_id({}),'two')
            path.write_text('def cid(:')
            with self.assertRaises(SyntaxError):module.candidate_id({})

    def test_same_ids_for_original_rules_and_all_synthetic_inputs(self):
        for candidate in candidates():
            self.assertEqual(p.candidate_id(candidate), uncached(candidate))

    def test_mutation_and_invalid_input_are_still_evaluated(self):
        candidate = next(candidates())
        before = p.candidate_id(candidate)
        for field, value in dict(system='NEW_TOY', method='PR', formula='4th',
                                 prefix=3, T=.2, q=2, R_outer=9, K=4).items():
            changed = deepcopy(candidate); changed[field] = value
            self.assertEqual(p.candidate_id(changed), uncached(changed))
            self.assertNotEqual(before, p.candidate_id(changed))
        candidate.pop('q')
        with self.assertRaises(KeyError): p.candidate_id(candidate)

    def test_source_change_with_identical_stat_and_invalid_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'scripts').mkdir()
            path = root/'scripts/h46_long_time_signal.py'
            path.write_text("def cid(candidate): return 'one'\n")
            with patch.object(p, '__file__', str(root/'src/trottertracks/pipeline.py')):
                self.assertEqual(p.candidate_id({}), 'one'); stat = path.stat()
                path.write_text("def cid(candidate): return 'two'\n")
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                self.assertEqual(p.candidate_id({}), 'two')
                path.write_text('def cid(:')
                with self.assertRaises(SyntaxError): p.candidate_id({})

    def test_cache_only_holds_parser_and_is_bounded(self):
        p._candidate_id_parser.cache_clear()
        for candidate in candidates(): p.candidate_id(candidate)
        info = p._candidate_id_parser.cache_info()
        self.assertEqual((info.misses, info.currsize), (1, 1))
        self.assertGreater(info.hits, 1)


if __name__ == '__main__': unittest.main(verbosity=2)
