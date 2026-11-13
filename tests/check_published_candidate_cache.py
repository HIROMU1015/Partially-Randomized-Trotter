"""Actual fixed-B loader and artificial IDs only; no physical context/kernel."""
import ast
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import baseline_signal_pipeline as C
from check_candidate_id_cache import candidates,uncached


class PublishedCacheTests(unittest.TestCase):
    def test_fixed_code_and_process_wide_ast_are_preserved(self):
        parser=ast.parse;api=C.published_api(str(ROOT));module=api.B.p
        self.assertIs(parser,ast.parse)
        self.assertEqual(module.candidate_id.__code__.co_filename,
                         C.B_SOURCE+':src/trottertracks/pf_pr_rpe_pipeline.py')
        self.assertIsNot(module.ast,ast)

    def test_actual_private_API_preserves_ids_and_checks_every_input(self):
        module=C.published_api(str(ROOT)).B.p
        module.ast.parse.cache_clear()
        for candidate in candidates():self.assertEqual(module.candidate_id(candidate),uncached(candidate))
        self.assertEqual(module.ast.parse.cache_info().misses,1)
        with self.assertRaises(KeyError):module.candidate_id({})

    def test_private_namespaces_have_separate_parsers(self):
        left=C.published_api(str(ROOT)).B.p
        right=C.published_api(str(ROOT)+'/.' ).B.p
        self.assertIsNot(left,right)
        self.assertIsNot(left.ast.parse,right.ast.parse)


if __name__=='__main__':unittest.main(verbosity=2)
