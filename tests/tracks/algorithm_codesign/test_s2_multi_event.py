"""Invented literal gates/scalars only; no saved scientific input."""
from fractions import Fraction as F
import unittest
from trottertracks.algorithm_codesign.s2_multi_event import pair_table, positive_t_prices, evaluate_law
from trottertracks.algorithm_codesign.rte_development.boundary import simplify, native_price


def bindings():
    return [dict(index=i, event=dict(coefficient=c, proposal=p, weight=c/p, phase_i_power=i),
                 native_ir=ir, native_cost=native_price(ir, {}))
            for i, c, p, ir in [(0, F(2), F(1, 3), [('T', 0), ('H', 1)]),
                                 (1, F(1), F(2, 3), [('H', 1), ('T', 0)])]]


class PairAdapter(unittest.TestCase):
    def test_identity_order_coefficients_and_prices(self):
        bs = bindings(); rows = pair_table(bs, {})
        self.assertEqual([(r['i'], r['j']) for r in rows], [(0,0),(0,1),(1,0),(1,1)])
        for r in rows:
            a, b = bs[r['i']], bs[r['j']]
            self.assertEqual(r['B'], native_price(simplify(a['native_ir']+b['native_ir']), {}))
            self.assertEqual(r['proposal']*r['weight'], r['coefficient'])
        self.assertEqual(rows[1]['cancellation_depth'], 1)
        self.assertEqual(bs[1]['event']['phase_i_power'], 1)

    def test_identity_and_weight_tamper_stop(self):
        bs=bindings(); bs[1]['index']=0
        with self.assertRaises(ValueError): pair_table(bs,{})
        bs=bindings(); bs[0]['event']['weight']=F(1)
        with self.assertRaises(ValueError): pair_table(bs,{})

    def test_missing_cache_and_price_mismatch_stop(self):
        bs=bindings(); bs[0]['native_ir']=[('R',0,'1/7',1)]
        with self.assertRaises(KeyError): pair_table(bs,{})
        bs=bindings(); bs[0]['native_cost']['T']=99
        with self.assertRaises(ValueError): pair_table(bs,{})

    def test_zero_t_stops_without_floor(self):
        bs=bindings(); bs[0]['native_ir']=[('H',0)]; bs[0]['native_cost']=native_price([('H',0)],{})
        with self.assertRaisesRegex(ValueError,'ZERO_T'): positive_t_prices(pair_table(bs,{}))

    def test_all_laws_same_coefficients_and_finite_weights(self):
        rows=pair_table(bindings(),{}); positive_t_prices(rows)
        for name in ('P0','qA','qB'):
            out=evaluate_law(name,rows,F(1,10),F(1,20))
            self.assertEqual(sum(e['proposal'] for e in out['events']),1)
            for e,r in zip(out['events'],rows):
                self.assertEqual(e['proposal']*e['weight'],r['coefficient'])
                if name!='P0': self.assertEqual((e['proposal']*2**160).denominator,1)
            for c in out['cells'].values():
                self.assertEqual(c['full_known_expected_by_axis']['Y']['1Q']-c['full_known_expected_by_axis']['X']['1Q'],1)
                self.assertEqual(c['known_two_axis_cost']['T'],2*out['N_per_axis']*c['body_expected']['T'])
                self.assertEqual(c['unknown_common_h_slope'],2*out['N_per_axis'])

    def test_bias_exhaustion_stop(self):
        with self.assertRaises(ValueError): evaluate_law('P0',pair_table(bindings(),{}),F(0),F(1,20))


if __name__=='__main__': unittest.main()
