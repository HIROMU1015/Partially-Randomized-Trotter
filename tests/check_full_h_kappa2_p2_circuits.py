"""Toy Full-H only. Reuse P-1 fixtures/gates, normal events, fake compiler. No H4 input."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import check_full_h_kappa2_p1_circuits as T
import run_full_h_kappa2_p2 as P


class Tests(unittest.TestCase):
    def setUp(self): T.Tests.setUp(self)

    def test_same_P1_order2_gate_XY_and_compiler_restoration(self):
        row = T.ORDER2_ROW; seeds = [P.P1.seed_function()(row, P.P1.COHORT, 0, g, 0) for g in range(row['q'])]
        native = self.full.setup(); before = native.compile_cached
        out = P.acquire_path(self.full, row, P.P1.COHORT, 0, seeds, compile_cached=self.fake_compile)
        self.assertEqual(out['checks']['scalar_event_phases']['gate'], P.P1.PHASE_GATE_ID)
        self.assertEqual(out['xy_phase_gate']['status'], 'PASS')
        self.assertTrue(out['xy_phase_gate']['same_controlled_evolution'])
        self.assertTrue(out['checks']['parity']['replaced_controlled_zz'] > 0)
        self.assertEqual(len(self.compiles), 2)
        self.assertIs(native.compile_cached, before)
        self.assertTrue(out['checks']['scalar_event_phases']['event_phases'])

    def test_paired_seed_requests_equal_individually_sampled_groups(self):
        native = self.full.setup(); row = T.TOY_ROW
        seeds = [P.P1.seed_function()(row, P.P1.COHORT, 0, g, 0) for g in range(row['q'])]
        parent_row = dict(row, q=1, T=row['T'] / 2, s=row['R_outer'], delta=row['T'] / 2)
        parents = []
        from unittest.mock import patch
        for seed in seeds:
            with patch.object(native, 'trial_seed', return_value=seed):
                requests, actual_seeds, laws = native.sample_requests(parent_row, self.ctx.preparations[0], 'TOY_PARENT', 0)
            parents.append(dict(events=P.event_projection([r.rte_occurrence.events for r in requests]), seeds=actual_seeds))
        out = P.acquire_path(self.full, row, P.P1.COHORT, 0, seeds, compile_cached=self.fake_compile)
        self.assertEqual(P.check_composition(out, parents)['status'], 'PAIRED_EVENTS_IDENTICAL')


if __name__ == '__main__': unittest.main(verbosity=2)
