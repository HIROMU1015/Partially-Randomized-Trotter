"""Small metadata/arithmetic fixtures; no scientific package imports."""
from copy import deepcopy
from fractions import Fraction
import unittest

import build_comparison as B


def ratio(f):
    return dict(numerator=f.numerator, denominator=f.denominator)


def fixture():
    prices = dict(configuration_id=B.CONFIG, rows=[])
    joined = dict(configuration_id=B.CONFIG, total_view_id='TOY_VIEW', binding_id='TOY_BINDING',
                  model_status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', physical_rows=[],
                  allocation=dict(shots=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(B.SHOTS)]))
    for m in range(13):
        q, T = 2**m, .1*2**m
        physical = dict(id='TOY_'+str(m), method='B0I', prefix=7, formula='2nd')
        joined['physical_rows'].append(dict(m=m, q_PF=q, T=ratio(Fraction.from_float(T)),
                                           actual_delta_pf=ratio(Fraction.from_float(.1)),
                                           physical_candidate=physical))
        for axis in ('X', 'Y'):
            direct = m < 5
            prices['rows'].append(dict(configuration_id=B.CONFIG, physical_id=physical['id'],
                original_canonical_id='TOY', m=m, axis=axis, T=T, q=q, actual_delta=.1,
                T_binary64_hex=T.hex(), actual_delta_binary64_hex=(.1).hex(), physical_candidate=physical,
                status='ACCEPTED_DIRECT' if direct else 'NOT_ACQUIRED',
                compiled_RZ=5513*q+307 if direct else None,
                receipt={'TOY': axis} if direct else None, sidecar={'TOY': axis} if direct else None,
                gate_result=dict(C_guard='PASS', original_action_gates='PASS') if direct else {}))
    summary = dict(case=dict(configuration_id=B.CONFIG, allocation=joined['allocation']),
                   status=joined['model_status'], coarse_model_status=joined['model_status'],
                   comparison_source=dict(coarse_conditional_cost_RZ=435481090,
                                          coarse_cost_applies_to='S2_16h_over_7_ONLY'),
                   fine_MSE=['SAVED_FINE'], coarse_MSE=['SAVED_COARSE'], accuracy_comparison={'quote': 'TOY'})
    return prices, joined, summary


class ComparisonFixtures(unittest.TestCase):
    def test_origins_exact_arithmetic_and_no_input_mutation(self):
        args = fixture()
        before = deepcopy(args)
        table, cost = B.build(*args)
        self.assertEqual(args, before)
        self.assertEqual(table['counts']['direct'], 10)
        self.assertEqual(sum(r['origin'] == 'ADOPTED_RULE_EXTRAPOLATION' for r in table['rows']), 16)
        self.assertTrue(all(r['direct_receipt'] is None for r in table['rows'][10:]))
        self.assertEqual(cost['total_RZ'], cost['direct_part_RZ']+cost['extrapolated_part_RZ'])
        self.assertEqual(sum(r['contribution_RZ'] for r in cost['per_axis_contributions']), cost['total_RZ'])
        self.assertEqual(cost['Q3_saved_quote']['fine_MSE_Ha2'], ['SAVED_FINE'])
        self.assertIsNone(cost['G_qualified'])

    def test_individual_axis_and_origin_rejections(self):
        for index, key, value in [(1, 'compiled_RZ', 1), (3, 'axis', 'X'),
                                  (11, 'status', 'ACCEPTED_DIRECT'), (1, 'receipt', None)]:
            with self.subTest(index=index, key=key):
                args = fixture()
                args[0]['rows'][index][key] = value
                with self.assertRaises(ValueError):
                    B.build(*args)

    def test_physical_schedule_and_fixed_allocation_rejections(self):
        for field, value in [('q', 9), ('actual_delta', .2), ('physical_id', 'WRONG'),
                             ('T_binary64_hex', (.3).hex())]:
            with self.subTest(field=field):
                args = fixture()
                args[0]['rows'][2][field] = value
                with self.assertRaises(ValueError):
                    B.build(*args)
        args = fixture()
        args[1]['allocation']['shots'][0]['N_X'] += 1
        with self.assertRaises(ValueError):
            B.build(*args)


if __name__ == '__main__':
    unittest.main()
