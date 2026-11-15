"""Small synthetic checks for exact subset arithmetic and reversal directions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_pf6_partial_arithmetic',HERE/'analyze.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def artificial(prices):
    return [dict(label=label,m=m,axis=a,N=2 if a=='X'else 3,C=c,status='ACCEPTED_DIRECT')
            for label,c in prices.items()for m in M.K for a in ('X','Y')]


class PartialTests(unittest.TestCase):
    def test_axis_products_integer_sum_and_pair_direction(self):
        terms,sums,pairs=M.aggregate(artificial({'a':7,'b':11}),['a','b'])
        self.assertEqual([r['K_i']for r in sums],[280,440])
        self.assertEqual([r['total']for r in sums[0]['rounds']],[35]*8)
        self.assertEqual((pairs[0]['i'],pairs[0]['j'],pairs[0]['K_j_minus_K_i']),('a','b',160))
        self.assertEqual(pairs[0]['equal_condition'],dict(operator='=',threshold=160))
        self.assertEqual(pairs[0]['i_cheaper_condition']['operator'],'<')
        self.assertEqual(pairs[0]['i_more_expensive_condition']['operator'],'>')
        for delta,order in [(159,-1),(160,0),(161,1)]:
            diff=280-440+delta;self.assertEqual((diff>0)-(diff<0),order)
        self.assertTrue(all(r['full_13_round_cost']is None for r in sums))
        self.assertEqual(len(terms),32)

    def test_input_order_and_equal_known_cost(self):
        _,_,pairs=M.aggregate(artificial({'a':11,'b':7}),['a','b'])
        self.assertEqual((pairs[0]['i'],pairs[0]['j']),('b','a'))
        _,_,pairs=M.aggregate(artificial({'a':7,'b':7}),['a','b'])
        self.assertEqual(pairs[0]['K_j_minus_K_i'],0)

    def test_missing_duplicate_null_and_fractional_counts_rejected(self):
        rows=artificial({'a':1})
        for bad in (rows[:-1],rows+[deepcopy(rows[0])]):
            with self.assertRaisesRegex(ValueError,'Missing/duplicate'):M.aggregate(bad,['a'])
        for value in (None,1.5,True,dict(numerator=3,denominator=2)):
            bad=deepcopy(rows);bad[0]['C']=value
            with self.assertRaises(ValueError):M.aggregate(bad,['a'])

    def test_huge_integers_do_not_pass_through_binary64(self):
        rows=artificial({'a':2**80+1})
        for row in rows:row['N']=2**90+3
        terms,sums,_=M.aggregate(rows,['a'])
        self.assertEqual(terms[0]['N_times_C'],(2**80+1)*(2**90+3))
        self.assertEqual(sums[0]['K_i'],16*(2**80+1)*(2**90+3))

    def test_96_fixed_metadata_rows_preserve_Q3_partial_status(self):
        d=json.loads((HERE/'INPUTS.json').read_text())
        self.assertEqual(d['issues'],[])
        self.assertEqual(len([r for r in d['axes']if r['status']=='ACCEPTED_DIRECT']),96)
        self.assertEqual(len([r for r in d['axes']if r['status']=='UNOBSERVED_ROUND']),60)
        self.assertTrue(all(c['original_API_return_available']is False for c in d['candidates']))
        self.assertTrue(all(c['original_configuration_id']!=c['total_configuration_id']for c in d['candidates']))
        self.assertTrue(all(r['price_generation_candidate_id']==r['original_candidate_id']
                            for r in d['axes']if r['status']=='ACCEPTED_DIRECT'))
        self.assertTrue(d['G']is None and d['winner']is None and d['sufficient_N']is None)


if __name__=='__main__':unittest.main(verbosity=2)
