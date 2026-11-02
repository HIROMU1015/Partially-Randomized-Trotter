"""Bounded stdlib checks for posthoc arithmetic and observed-grid semantics."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('integration',ROOT/'scripts/resource_applicability/integrate_h4_b0i_manuscript.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('figures',ROOT/'scripts/resource_applicability/plot_h4_b0i_manuscript.py')
p=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(p)


class Arithmetic(unittest.TestCase):
    def test_nonpositive_headroom_is_ineligible(self):
        e=.05;b={'cosine':e/math.sqrt(2),'sine':0.}
        self.assertIsNone(m.counts(b,1.,e));b['cosine']+=1e-8;self.assertIsNone(m.counts(b,1.,e))

    def test_ceil_and_normalization(self):
        c,h=m.counts(dict(cosine=0.,sine=0.),2.,.05)
        self.assertEqual(c,[math.ceil(2*4/h[0]**2*math.log(80))]*2)

    def test_pairing_preserves_negative_covariance(self):
        bias=dict(cosine=0.,sine=0.);pairs=[[1,3],[3,1]]*16
        r=m.evaluate(bias,1.,pairs,.1)
        self.assertEqual(r['paired_SE'],0.)
        self.assertEqual(r['G'],r['N_total']*2)

    def test_pairing_positive_covariance_exact_work(self):
        r=m.evaluate(dict(cosine=.01,sine=.005),1.,[[1,1],[3,3]]*16,.1)
        self.assertAlmostEqual(r['paired_SE'],r['N_total']/math.sqrt(31))
        self.assertEqual(r['G'],r['N_total']*2)

    def test_effective_cost_can_change_without_axis_cost_change(self):
        pairs=[[10,1]]
        a=m.evaluate(dict(cosine=.01,sine=0.),1.,pairs,.1)
        b=m.evaluate(dict(cosine=0.,sine=.01),1.,pairs,.1)
        self.assertEqual(a['C_cos'],b['C_cos']);self.assertEqual(a['C_sin'],b['C_sin'])
        self.assertNotEqual(a['C_effective'],b['C_effective'])
        ratios=m.factor_ratio(a,b)
        self.assertAlmostEqual(ratios['ratio_shots']*ratios['ratio_effective_RZ'],ratios['ratio_work'])

    def test_ineligible_fixed_configuration_has_no_zero_cost(self):
        self.assertIsNone(m.evaluate(dict(cosine=.2,sine=0.),1.,[[10,1]],.05))

    def test_grid_regions_bracket_reentrant_winners(self):
        rows=[dict(epsilon=e,winner=w) for e,w in ((.01,'B2'),(.02,'I'),(.05,'I'),(.1,'B2'))]
        runs=m.grid_segments(rows,lambda r:r['winner'])
        self.assertEqual([r['points'] for r in runs],[1,2,1])
        self.assertEqual(runs[1]['previous_grid_epsilon'],.01)
        self.assertEqual(runs[1]['next_grid_epsilon'],.1)
        self.assertFalse(any(r['continuous_boundary_certified'] for r in runs))

    def test_nonuniform_precision_cells_use_log_midpoints(self):
        xs=[.01,.049,.05,.052,.1];edges=p.log_edges(xs)
        self.assertEqual(len(edges),len(xs)+1)
        self.assertTrue(all(a<b for a,b in zip(edges,edges[1:])))
        self.assertAlmostEqual(edges[2],math.sqrt(.049*.05))


if __name__=='__main__':unittest.main()
