"""Toy 4-mode B1 checks with original B0 fixture, full-space oracle and fake compile. No H4."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import check_pauli_ld1_m0_pilot_circuits as T
import run_pauli_b1_prices as P


class Tests(unittest.TestCase):
    def setUp(self):T.Tests.setUp(self)

    def test_frozen_B0_default_result_is_identical_on_toy(self):
        import types
        old=types.ModuleType('_frozen_B0');old.__file__=str(ROOT/P.B0.CALLER)
        exec(compile(P.B0.blob('00f74f86d16cc6878e1d4c5977eb9abd9ce9c076',P.B0.CALLER),'frozen_B0','exec'),old.__dict__)
        old.connection=P.B0.connection  # same unchanged default emitter path, in the toy provider process
        before=old.pilot_core(self.native,self.ctx,self.d,self.req,self.ops,self.blocks,cohort='TOY',compile_cached=self.fake)
        after=P.B0.pilot_core(self.native,self.ctx,self.d,self.req,self.ops,self.blocks,cohort='TOY',compile_cached=self.fake)
        self.assertEqual(before,after)

    def test_paired_steps_retain_labels_products_phase_and_scalar_once(self):
        parents=[]
        for i,seed in enumerate((11,12)):
            r=P.acquire_path(self.native,self.ctx,self.d,self.req,cohort='TOY',index=i,seeds=[seed],
                             operators=self.ops,blocks=self.blocks,compile_cached=self.fake)
            parents.append(dict(result=r))
        req=dict(self.req,q=2,T=float(self.req['T'])*2)
        r=P.acquire_path(self.native,self.ctx,self.d,req,cohort='TOY_PAIR',index=0,seeds=[11,12],
                         operators=self.ops,blocks=self.blocks,compile_cached=self.fake)
        self.assertEqual(set(r['fingerprints']), {'cosine','sine'})
        self.assertTrue(all(f['measurement'] for f in r['fingerprints'].values()))
        # Simulate the actual saved-JSON boundary (tuples -> arrays).
        loaded=json.loads(json.dumps(parents))
        self.assertEqual(P.check_composition(r,loaded)['status'],'PAIRED_STEP_METADATA_IDENTICAL')
        for step in r['event_metadata']:
            self.assertIn('aggregate_source_global_phase',step['D2'])
            self.assertTrue(all(e['selected_labels'] for e in step['events']))
        bad=deepcopy(loaded);bad.reverse()
        with self.assertRaises(ValueError):P.check_composition(r,bad)
        bad=deepcopy(loaded);bad[0]['result']['scalar']+=.1
        with self.assertRaises(ValueError):P.check_composition(r,bad)

    def test_prefix_mismatch_fails_before_build(self):
        req=dict(self.req,prefix=2)
        with self.assertRaises(ValueError):P.acquire_path(self.native,self.ctx,self.d,req,cohort='TOY',index=0,seeds=[11],
                                                       operators=self.ops,blocks=self.blocks,compile_cached=self.fake)
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main(verbosity=2)
