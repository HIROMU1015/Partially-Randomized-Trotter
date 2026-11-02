"""Invented rational scalar checks only; no scientific fixture or runner."""
from fractions import Fraction as F
from pathlib import Path
import importlib.util
import itertools
import unittest

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('d1_saved',ROOT/'scripts/tracks/algorithm_codesign/analyze_d1_saved_resources.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def row(name,prices,slope,workspace=1):
    return dict(steps=1,arm=name,expected_native_intercept=dict(zip(M.METRICS,prices)),
                prep_readout_slope=slope,workspace_beyond_system=workspace)


class SavedArithmetic(unittest.TestCase):
    def test_two_factor_symmetric_decomposition(self):
        v=M.product_allocation(dict(a=2,b=3),dict(a=4,b=7))
        self.assertEqual(v['contributions'],dict(a=10,b=12));self.assertEqual(v['total_difference'],22)
    def test_four_factor_matches_permutation_average(self):
        before=dict(a=F(2),b=F(3,5),c=F(7),d=F(1,2));after=dict(a=F(4),b=F(1,5),c=F(5),d=F(3,2))
        found={k:F(0) for k in before}
        def product(v):
            out=F(1)
            for x in v.values():out*=x
            return out
        permutations=list(itertools.permutations(before))
        for order in permutations:
            current=dict(before)
            for key in order:
                old=product(current);current[key]=after[key];found[key]+=product(current)-old
        expected={k:v/len(permutations) for k,v in found.items()}
        self.assertEqual(M.product_allocation(before,after)['contributions'],expected)
    def test_unchanged_factor_zero_and_reversal(self):
        a=dict(x=2,y=3,z=4);b=dict(x=2,y=1,z=8)
        forward=M.product_allocation(a,b);backward=M.product_allocation(b,a)
        self.assertEqual(forward['contributions']['x'],0)
        self.assertEqual(forward['contributions'],{k:-v for k,v in backward['contributions'].items()})
    def test_budget_component_conservation(self):
        # Invented endpoints, a made-up rational log constant and valid ceiling residues.
        log=F(3,2)
        b0=dict(remaining='1',M2_upper='2',range_upper='3',Taylor_product_upper='1/8',implementation_product_upper='1/8',N_per_axis=12)
        # s increases by 1/2; bias decreases by 1/2.
        b1=dict(remaining='3/2',M2_upper='9/2',range_upper='3',Taylor_product_upper='0',implementation_product_upper='0',N_per_axis=10)
        v=M.budget_decomposition({'budget':b0},{'budget':b1},log)
        self.assertEqual(sum(v['contributions'].values()),-2)
        self.assertEqual(v['contributions']['Taylor_margin'],v['contributions']['implementation_margin'])
    def test_closed_half_line(self):
        self.assertEqual(M.interval_leq(6,-2),dict(empty=False,lo=3,hi=None))
        self.assertEqual(M.interval_leq(-6,2),dict(empty=False,lo=0,hi=3))
        self.assertTrue(M.interval_leq(6,2)['empty']);self.assertTrue(M.interval_leq(6,0)['empty'])
    def test_endpoint_tie_and_interval_orders(self):
        a=row('a',[1,1,1],3);b=row('b',[5,5,5],1)
        v=M.sensitivity([a,b],'T')
        self.assertEqual(v['all_exact_nonnegative_crossings'],[0,2])
        self.assertEqual(v['endpoint_rankings'][1]['ranking_tie_groups'],[['r1:a','r1:b']])
        self.assertEqual(v['open_interval_rankings'][0]['minimum_envelope_arms'],['r1:a'])
        self.assertEqual(v['open_interval_rankings'][1]['minimum_envelope_arms'],['r1:b'])
    def test_no_scalarization_pareto(self):
        a=row('a',[1,9,10],2);b=row('b',[9,1,10],2);c=row('c',[10,10,20],2)
        self.assertEqual(M.pareto_zero([a,b,c]),['r1:a','r1:b'])
    def test_independent_h_boxes(self):
        a=row('a',[1,8,8],3);b=row('b',[5,5,10],1)
        regions=M.pareto_regions([a,b])['directed_dominance_regions']
        found=next(v for v in regions if v['dominator']=='r1:b')
        self.assertEqual(found['coordinate_box']['T']['lo'],2)
        self.assertEqual(found['coordinate_box']['1Q']['lo'],1)
    def test_all_equal_h_pairs(self):
        v=M.pareto_regions([row('a',[1,1,1],2),row('b',[1,1,1],2)])
        self.assertFalse(v['directed_dominance_regions']);self.assertEqual(len(v['equal_all_h_pairs']),2)
    def test_equality_only_is_not_strict_dominance(self):
        a=row('a',[1,1,1],2);b=row('b',[1,1,1],1)
        v=M.pareto_regions([a,b])['directed_dominance_regions']
        self.assertFalse(any(r['dominator']=='r1:a' for r in v))
    def test_workspace_strict_dominance_keeps_quantum_tie(self):
        a=row('a',[1,1,1],1,workspace=0);b=row('b',[1,1,1],1,workspace=1)
        v=M.pareto_regions([a,b])['directed_dominance_regions']
        self.assertEqual(len(v),1);self.assertIsNone(v[0]['exclude_all_coordinate_equality_point'])
    def test_shared_log_constant(self):
        self.assertEqual(M.exact_log_upper(1),0)
        self.assertAlmostEqual(float(M.exact_log_upper(2)),0.6931471805599453,places=14)


if __name__=='__main__':unittest.main()
