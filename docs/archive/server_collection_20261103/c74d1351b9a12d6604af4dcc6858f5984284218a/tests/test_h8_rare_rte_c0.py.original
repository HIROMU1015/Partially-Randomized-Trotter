"""Only fixed toy values; no H8 files, probabilities, circuits or providers loaded."""
import importlib.util
from itertools import product
from fractions import Fraction
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('c0_mock', ROOT/'scripts/h8_rare_rte_c0_mock.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ConditionalLawTests(unittest.TestCase):
    def test_exact_count_law_matches_exhaustive_toy_words(self):
        for stratum in m.STRATA:
            raw = {}
            for word in product((0,2,4), repeat=3):
                n2, n4 = word.count(2), word.count(4)
                if m.member(n2,n4,stratum):
                    w = Fraction(1)
                    for k in word: w *= m.TOY_P[(0,2,4).index(k)]
                    raw[(n2,n4)] = raw.get((n2,n4), Fraction(0)) + w
            total = sum(raw.values())
            self.assertEqual(dict(m.toy_count_law(stratum)), {k:v/total for k,v in raw.items()})
    def test_multi_not_fixed_to_two(self):
        self.assertIn((3,0), dict(m.toy_count_law('S2_MULTI')))
    def test_s4_includes_multiple_and_mixed_orders(self):
        law = dict(m.toy_count_law('S4_PRESENT'))
        self.assertIn((0,3),law); self.assertIn((1,1),law)
    def test_single_uniform_position_support(self):
        seen = {m.toy_order_word(1,0,random.Random(i)) for i in range(40)}
        self.assertEqual(seen, m.toy_words(1,0))
    def test_rotation_and_products_have_correct_count(self):
        for e in m.toy_components((0,2,4)):
            self.assertEqual(len(e['indices']), e['order']+1)
    def test_unknown_stratum_refused(self):
        with self.assertRaises(ValueError): m.toy_count_law('IGNORED_TAIL')


class IsolationTests(unittest.TestCase):
    def test_paired_event_identity(self):
        events=m.toy_components((0,2,0))
        pair={a:dict(events=events) for a in ('cosine','sine')}
        self.assertIs(pair['cosine']['events'],pair['sine']['events'])
    def test_cache_separates_axis_and_profile(self):
        self.assertNotEqual(m.toy_identity(),m.toy_identity(axis='sine'))
        self.assertNotEqual(m.toy_identity(),m.toy_identity(compiler='TOY_B'))
        self.assertNotEqual(m.toy_identity(),m.toy_identity(events='TOY_EVENT_B'))
    def test_registry_domains_separate_conditions_and_strata(self):
        ids={m.toy_identity(candidate=c,stratum=s,replica=i) for c in ('TOY_A','TOY_B') for s in m.STRATA for i in range(3)}
        self.assertEqual(len(ids),24)
    def test_nonrefundable_reservation_prevents_double_attempt(self):
        with tempfile.TemporaryDirectory(prefix='h8_c0_toy_') as d:
            m.toy_reserve(d,'X')
            with self.assertRaises(FileExistsError): m.toy_reserve(d,'X')
    def test_nested_parallelism_accounts_for_both_axes(self):
        self.assertEqual(m.toy_admission(12,20,20),(6,2))
        self.assertEqual(m.toy_admission(1,20,20),(1,1))
        self.assertEqual(m.toy_admission(12,2,20),(2,2))
    def test_no_available_memory_defers_new_tasks(self):
        self.assertEqual(m.toy_admission(12,0,20)[0],0)
    def test_native_exit_zero_reaped(self):
        r=m.toy_native_child(0)
        self.assertTrue(r['reaped'] and r['parent_payload_unchanged']);self.assertEqual(r['exit_code'],0)
    def test_native_failure_reaped_without_retry(self):
        r=m.toy_native_child(7)
        self.assertTrue(r['reaped']);self.assertEqual((r['exit_code'],r['attempts']),(7,1))
    def test_scientific_entry_always_refused(self):
        with self.assertRaises(PermissionError):m.refuse_scientific_execution(approved=True)


if __name__=='__main__': unittest.main(verbosity=2)
