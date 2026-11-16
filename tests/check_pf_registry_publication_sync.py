"""Published D registry and normal C reconciliation only; no scientific calls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from trottertracks.saved_rpe_cost_catalog import Reader,frac,ref
from trottertracks import pf_optimizer_connection as PF
import sync_saved_rpe_publications as S

ROOT=Path(__file__).resolve().parents[1]
D='352539e2749989b4ddc3ab74c31f486982bfe265'
BASE='docs/research/d_pf_on_demand_evidence_20261116'
REGISTRY=ref(D,BASE+'/C_RESULT_JOIN.json')
NEW='9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981'
P7='dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0'
SEED=ref('bd78ae252f92f7c84b6af6ae2a3bad202a5bd1bb','docs/research/h4_saved_rpe_cost_catalog_20261116/INPUTS.json')


class Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=Reader(ROOT);cls.seed=cls.r.read(SEED)

    def test_published_helper_is_reused_and_normal_discovery_finds_registry(self):
        saved=subprocess.check_output(['git','show',D+':src/trottertracks/pf_optimizer_connection.py'],cwd=ROOT)
        self.assertEqual(saved,(ROOT/'src/trottertracks/pf_optimizer_connection.py').read_bytes())
        found=S.discover(self.r,{'codex/d-p7-pf-on-demand-evidence-20261116':D})
        hits=[x for x in found.values()if x['source']['path']==REGISTRY['path']]
        self.assertEqual(len(hits),1)
        self.assertEqual(S.git(ROOT,'rev-parse',D+':'+REGISTRY['path']),hits[0]['blob'])
        self.assertTrue(S.schema_supported(self.r.read(hits[0]['source']),REGISTRY['path']))

    def test_saved_registry_adds_only_new_q3_waiting_point_and_is_idempotent(self):
        source=dict(source=REGISTRY,blob=S.git(ROOT,'rev-parse',D+':'+REGISTRY['path']),branches=['public-D'])
        # complete_selected is D's acquisition hook; the C reader must never call it.
        with patch.object(PF,'complete_selected',side_effect=AssertionError('No scientific completion')):
            before,_,_=S.reconcile(self.r,self.seed,{},{} )
            out,records,accepted=S.reconcile(self.r,self.seed,{'registry':source},{})
        self.assertEqual(records['registry']['status'],'INTEGRATED')
        self.assertEqual(len(out['candidates']),len(before['candidates'])+1)
        point=next(r for r in out['candidates']if r['configuration_id']==NEW)
        self.assertEqual(point['status'],'Q3_WAITING')
        self.assertEqual(point['signal_rounds_available'],13);self.assertEqual(len(point['price_axes']),26)
        self.assertEqual(sum(p['price_kind'].startswith('DIRECT')for p in point['price_axes']),10)
        self.assertTrue(all(point[k]is None for k in ('q3_status','conditional_cost','allocation','N_X','N_Y')))
        self.assertTrue(all(p['N']is None for p in point['price_axes']))
        self.assertEqual(point['allocation_proposal_status'],'TEMPLATE_ONLY_NOT_EVALUATED')
        self.assertTrue(point['prepared_for_optimizer'])
        old=[r for r in before['candidates']if r['configuration_id']==P7]
        current=[r for r in out['candidates']if r['configuration_id']==P7]
        self.assertEqual(old,current)
        self.assertTrue(any(frac(r['conditional_cost'])==372035410 for r in current if r['conditional_cost']is not None))
        self.assertEqual(before['method_minima'],out['method_minima'])
        self.assertEqual(len([r for r in out['signal_only_history']if r['configuration_id']==P7]),1)
        again,_,_=S.reconcile(self.r,dict(self.seed,sources=self.seed['sources']+accepted),{'registry':source},{})
        self.assertEqual(out,again)
        self.assertTrue(all(out[k]==0 for k in ('new_signal','new_sampling','new_compile','new_price_fit','new_Q3')))

    def test_published_zero_q3_return_is_reference_only(self):
        loc=ref(D,BASE+'/optimizer_round02/ROUND_RESULT.json')
        out,records,_=S.reconcile(self.r,self.seed,{'metadata':dict(source=loc,blob=S.git(ROOT,'rev-parse',D+':'+loc['path']),branches=['D'])},{})
        self.assertEqual(records['metadata']['status'],'REFERENCE_ONLY')
        self.assertIn('no evaluated Q3',records['metadata']['reason'])
        self.assertFalse(any(r['configuration_id']==NEW for r in out['candidates']))


if __name__=='__main__':unittest.main(verbosity=2)
