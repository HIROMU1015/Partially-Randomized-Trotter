"""Toy Full-H only. Reuse P-1 fixtures/gates, normal events, fake compiler. No H4 input."""
from pathlib import Path
import json
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

    def test_paired_saved_JSON_events_without_sampler_and_with_new_XY_compiles(self):
        native = self.full.setup(); row = T.ORDER2_ROW
        seeds = [P.P1.seed_function()(row, P.P1.COHORT, 0, g, 0) for g in range(row['q'])]
        parent_row = dict(row, q=1, T=row['T'] / 2, s=row['R_outer'], delta=row['T'] / 2)
        parents = []
        from unittest.mock import patch
        for i, seed in enumerate(seeds):
            parent = P.acquire_path(self.full,parent_row,'TOY_PARENT',i,[seed],compile_cached=self.fake_compile)
            parent['receipt_source'] = dict(fixture='ARTIFICIAL_PARENT',index=i)
            parents.append(parent)
        parents = json.loads(json.dumps(parents))
        self.compiles.clear()
        with patch.object(native,'sample_requests',side_effect=AssertionError('PARENT_RESAMPLED')) as sampler, \
             patch.object(type(self.ctx.preparations[0].rte_preparation),'sample_events',side_effect=AssertionError('RNG_CALLED')) as draws:
            out = P.acquire_path(self.full, row, P.P1.COHORT, 0, seeds, compile_cached=self.fake_compile,parents=parents)
            sampler.assert_not_called(); draws.assert_not_called()
        self.assertEqual(len(self.compiles),2)
        self.assertEqual(out['sampling'],'SAVED_PARENT_EVENTS_NO_DRAW')
        self.assertEqual(out['parents'],[p['receipt_source'] for p in parents])
        self.assertEqual(P.check_composition(out, parents)['status'], 'PAIRED_EVENTS_IDENTICAL')
        self.assertTrue(out['checks']['scalar_event_phases']['event_phases'])
        from copy import deepcopy
        for kind in ('scalar','phase','seed'):
            bad=deepcopy(parents)
            if kind=='scalar':bad[0]['trial']['scalar']+=1
            elif kind=='phase':bad[0]['events'][0][0]['phase']['real']*=-1
            else:bad[0]['seeds'][0]+=1
            with patch.object(native,'compose',side_effect=AssertionError('BUILD_BEFORE_PARENT_CHECK')) as build:
                with self.assertRaises(ValueError):
                    P.acquire_path(self.full,row,P.P1.COHORT,0,seeds,compile_cached=self.fake_compile,parents=bad)
                build.assert_not_called()


if __name__ == '__main__': unittest.main(verbosity=2)
