"""Metadata-only saved fixtures and synthetic evidence; no scientific leaves."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_rpe_rmse_s0_v2 as V


class ConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        store=V.p.SavedArtifacts(ROOT)
        cls.original=store.read(dict(commit=V.S0,path=V.S0_DIR+'CONFIGURATION.json'))
        cls.packet=store.read(dict(commit=V.S0,path=V.S0_DIR+'N_HANDOFF.json'))
        cls.connected=V.connect(cls.original,cls.packet,root=ROOT)
        cls.inventory=V.saved_inventory(ROOT)

    def test_canonical_V2_ids_physical_all13_old_packet_unchanged(self):
        c,n=deepcopy(self.original),deepcopy(self.packet)
        out=V.connect(c,n,root=ROOT);v2=out['configuration']
        self.assertEqual(c,self.original);self.assertEqual(n,self.packet)
        self.assertEqual(v2['schema'],'PR_RPE_RMSE_V2')
        self.assertNotEqual(v2['id'],c['B_configuration']['id'])
        self.assertEqual(v2['spec']['split']['deterministic_scope'],'PREFIX')
        self.assertEqual(len(v2['rounds']),13)
        for old,new in zip(c['B_configuration']['rounds'],v2['rounds']):
            self.assertEqual(old['physical_candidate'],new['physical_candidate'])
            self.assertEqual(old['structure'],new['structure'])
            self.assertEqual(old['round_law_definition'],new['round_law_definition'])
        self.assertEqual(out['N_packet']['A_original_packet'],n['A_original_packet'])
        self.assertFalse(out['N_packet']['connection_provenance']['legacy_relabelled'])

    def test_nominal_copy_only_11_nulls_sufficient_and_RMSE_null(self):
        n=self.connected['N_packet']
        for r in n['rounds']:
            nominal={0:64,6:38}.get(r['m'])
            self.assertEqual((r['nominal_N_X'],r['nominal_N_Y']),(nominal,nominal))
            self.assertIsNone(r['N_X']);self.assertIsNone(r['N_Y'])
            if nominal is None:self.assertEqual(r['model_status'],'NOT_EVALUATED')
        self.assertIsNone(n['rmse_evidence']['rmse_bound_ha'])
        self.assertFalse(n['rmse_evidence']['covers_state_target_alias_prior_decoder_all_rounds'])

    def test_exact_kappa_and_nonpower_of_two_R_preserved(self):
        v2=self.connected['configuration'];old=self.original['inputs']
        self.assertEqual(v2['spec']['A_input']['kappa_paper'],old['kappa_paper'])
        self.assertIsInstance(old['kappa_paper'],str)
        self.assertEqual([r['R_outer'] for r in v2['rounds']],[2,4,8,16,32,64,128,255,510,1020,2040,4080,8160])
        self.assertEqual(v2['rounds'][12]['structure']['event_count'],33423360)

    def test_legacy_M_kappa_SOURCE_or_nominal_modification_rejected(self):
        for key in ('M','kappa','SOURCE','law','nominal'):
            c,n=deepcopy(self.original),deepcopy(self.packet)
            if key=='M':c['inputs']['M']=11
            elif key=='kappa':c['inputs']['kappa_paper']='24.305274577756407'
            elif key=='SOURCE':c['source']['commit']='FOREIGN'
            elif key=='law':n['rounds'][0]['law_binding']['definition']['foreign']=True
            else:n['rounds'][1]['nominal_N_X']=1
            with self.subTest(key=key),self.assertRaises(ValueError):V.connect(c,n,root=ROOT)

    def test_V2_canonical_mismatch_schema_relabel_and_foreign_source_rejected(self):
        for key in ('M','kappa','law','source','candidate','relabel'):
            c=deepcopy(self.connected['configuration'])
            if key=='M':c['spec']['M']=11
            elif key=='kappa':c['spec']['A_input']['kappa_paper']='1'
            elif key=='law':c['rounds'][0]['round_law_definition']['foreign']=True
            elif key=='source':c['rounds'][0]['law_binding']['definition']['source']={'foreign':True}
            elif key=='candidate':c['rounds'][7]['physical_candidate']['R_outer']=256
            else:c=deepcopy(self.original['B_configuration']);c['schema']=V.SCHEMA
            with self.subTest(key=key),self.assertRaises(ValueError):
                V.connect(self.original,self.packet,root=ROOT,canonical_v2=c)

    def test_exact_saved_coverage_and_missing_scalar_not_zero(self):
        c=V.coverage(self.connected,self.inventory,root=ROOT)
        self.assertEqual(c['additional_saved_mean_rounds'],[1,3,4])
        self.assertEqual(c['missing_mean_rounds'],[2,5,7,8,9,10,11,12])
        self.assertEqual(c['missing_reference_pair_rounds'],[3,4])
        for m in (3,4):
            r=c['rounds'][m];self.assertIn('exact_tail_norm_error',r['missing_error_fields'])
            self.assertIsNone(r['source_gate_evidence'][0]['observed_error'])
            self.assertFalse(r['scientific_gate_rechecked_now'])

    def test_artificial_wrong_binding_law_or_candidate_not_reused(self):
        for key in ('binding','law','candidate'):
            i=deepcopy(self.inventory)
            for v in i['means']:
                if v['candidate']['id']!=self.connected['configuration']['rounds'][1]['physical_candidate']['id']:continue
                if key=='binding':v['binding']['state']='SYNTHETIC_FOREIGN'
                elif key=='law':v['law_implementation_source']={'synthetic':'FOREIGN_SOURCE'}
                else:v['candidate']['R_outer']=8
            out=V.coverage(self.connected,i,root=ROOT)['rounds'][1]
            self.assertFalse(out['saved_mean_present']);self.assertTrue(out['rejected_same_id_sources'])

    def test_saved_FAIL_retained_not_auto_retried_or_qualified(self):
        i=deepcopy(self.inventory)
        for r in i['means']:
            if r['candidate']['id']==self.connected['configuration']['rounds'][1]['physical_candidate']['id']:
                r['saved_status']='SYNTHETIC_NUMERICAL_UNRESOLVED_STOP'
        r=V.coverage(self.connected,i,root=ROOT)['rounds'][1]
        self.assertTrue(r['blocking_saved_statuses']);self.assertIsNone(r['sufficient_N_X'])

    def test_no_scientific_leaf_invoked_and_B_planner_stays_pending(self):
        with (patch.object(V.A,'nominal_axis_shots',side_effect=AssertionError('new N')),
             patch.object(V.A,'existing_finite_mean',side_effect=AssertionError('new mean')),
             patch.object(V.legacy,'supplied_disk_geometry',side_effect=AssertionError('new geometry'))):
            out=V.build(ROOT)
        bp=out['B_evidence_plan']
        self.assertIsNone(bp['G']);self.assertIsNone(bp['winner'])
        self.assertFalse(bp['comparison_complete']);self.assertFalse(bp['scientific_execution_performed'])
        self.assertEqual(out['science_calls'],0)
        self.assertFalse(any(name in sys.modules for name in ('numpy','scipy','qiskit')))

    def test_protected_S0_science_and_publication_blobs_unchanged(self):
        for name in ('CONFIGURATION.json','N_HANDOFF.json','result.json','SOURCE_INDEX.json'):
            path=V.S0_DIR+name
            blob=subprocess.check_output(['git','show',V.S0+':'+path],cwd=ROOT)
            self.assertEqual((ROOT/path).read_bytes(),blob)
        path='scripts/run_pr_rpe_rmse_s0.py'
        self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',V.S0_SCIENCE+':'+path],cwd=ROOT))


if __name__=='__main__':unittest.main()
