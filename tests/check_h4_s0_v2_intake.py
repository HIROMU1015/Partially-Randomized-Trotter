"""Saved S0 metadata and mock packets only. No science/compile/price/G calls."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import h4_s0_v2_intake as H
from trottertracks import mean_compiled_price as C


class S0IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = H.p.SavedArtifacts(ROOT)
        cls.saved = cls.store.read(H.A_CONFIG); cls.original = cls.store.read(H.A_PACKET)
        # Raising on numerical entry points also protects the saved-fixture test.
        with (patch.object(H.I.A,'evaluate',side_effect=AssertionError('No RMSE evaluation')),
              patch.object(H.I.A,'nominal_axis_shots',side_effect=AssertionError('No N')),
              patch.object(C,'full_wrapper_mean',side_effect=AssertionError('No mean C')),
              patch.object(C,'combine_mean',side_effect=AssertionError('No model')),
              patch.object(H.B,'cost',side_effect=AssertionError('No G')),
              patch.object(H.p,'run_missing',side_effect=AssertionError('No scientific provider'))):
            cls.result = H.prepare(ROOT)
        cls.config = cls.result['configuration']

    def test_canonical_fresh_V2_and_all_13_physical_rounds(self):
        config = self.config; mapping = self.result['identity_map']
        self.assertEqual(config['id'],'587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911')
        self.assertNotEqual(mapping['V1_configuration_id'],config['id'])
        self.assertEqual(len(mapping['rows']),13)
        self.assertTrue(all(r['physical_schedule_match'] for r in mapping['rows']))
        self.assertEqual([r['R_outer'] for r in config['rounds']], [2,4,8,16,32,64,128,255,510,1020,2040,4080,8160])
        self.assertEqual(H.B.a.exact_scalar(config['spec']['kappa']['value']), Fraction(self.saved['conditions']['kappa_paper_exact']))
        self.assertEqual(config['spec']['split']['deterministic_scope'],'PREFIX')
        self.assertEqual(config['spec']['A_input']['design']['PF'],dict(method='B0',prefix=7))

    def test_only_two_nominal_diagnostics_original_values_status_and_source_preserved(self):
        original = self.result['original_S0_handoff']
        self.assertEqual(original,self.original)
        self.assertEqual(original['status'],'UNQUALIFIED')
        for row in self.result['identity_map']['rows']:
            d = row['S0_diagnostic']; m = row['m']
            self.assertEqual(d['nominal_N_X'],64 if m==0 else 38 if m==6 else None)
            self.assertEqual(d['nominal_N_X'],d['nominal_N_Y'])
            self.assertFalse(d['sufficient_N_available']); self.assertIsNone(d['global_RMSE_bound'])
        self.assertTrue(all(r['N_X'] is None and r['N_Y'] is None for r in original['rounds']))
        self.assertIsNone(original['rmse_evidence']['rmse_bound_ha'])

    def test_A_SOURCE_identity_law_nominal_and_qualification_mismatch_rejected(self):
        mutations=[lambda n:n['source'].update(commit='FOREIGN'),
            lambda n:n['rounds'][0]['identity']['binding'].update(state='FOREIGN'),
            lambda n:n['rounds'][0]['law_binding'].update(kappa={'value':1}),
            lambda n:n['rounds'][1].update(nominal_N_X=64),
            lambda n:n['rounds'][0].update(N_X=64),lambda n:n.update(status='QUALIFIED')]
        for change in mutations:
            n=deepcopy(self.original);change(n)
            with self.assertRaises(ValueError):H.canonical(self.saved,n)
        bad=deepcopy(self.saved);bad['B_configuration']['rounds'][6]['R_outer']=4
        with self.assertRaises(ValueError):H.canonical(bad,self.original)

    def mock_incoming_V2_packet(self):
        # Test-only incoming publication simulation, never emitted as A evidence.
        packet=deepcopy(self.original);packet['configuration_id']=self.config['id']
        packet['fixture']='MOCK_INCOMING_PACKET_NOT_AN_A_PUBLICATION'
        for request,row in zip(self.config['rounds'],packet['rounds']):
            row['identity']=deepcopy(request['identity']);row['law_binding']=deepcopy(request['law_binding'])
        return packet

    def test_future_A_packet_full_canonical_identity_required_without_producing_real_packet(self):
        with self.assertRaisesRegex(ValueError,'V2_CONFIGURATION'):H.validate_A_v2_packet(self.config,self.original,self.original)
        mock=self.mock_incoming_V2_packet()
        self.assertTrue(H.validate_A_v2_packet(self.config,mock,self.original))
        for change in (lambda n:n['source'].update(path='FOREIGN'),
                       lambda n:n['rounds'][6]['identity']['profile'].update(compiler='FOREIGN'),
                       lambda n:n['rounds'][6].update(N_X=38),
                       lambda n:n['rmse_evidence'].update(rmse_bound_ha=0)):
            bad=deepcopy(mock);change(bad)
            with self.assertRaises(ValueError):H.validate_A_v2_packet(self.config,bad,self.original)

    def test_26_axis_plan_keeps_all_sufficient_N_C_and_G_missing(self):
        plan=self.result['evidence_plan'];rows=plan['configurations'][0]['rows']
        self.assertEqual(len(rows),26);self.assertFalse(plan['new_A_V2_packet_created'])
        self.assertTrue(all(r['N']['availability']=='MISSING' and r['C']['availability']=='MISSING' for r in rows))
        self.assertTrue(all(not r['cost_evaluation_possible'] and r['G'] is None for r in rows))
        self.assertTrue(all(not r['C_passive_lookup']['value_acquired'] for r in rows))
        self.assertIsNone(plan['G']);self.assertIsNone(plan['winner']);self.assertFalse(plan['comparison_complete'])
        self.assertTrue(all(any(w['owner']=='Codex A' for w in r['next_work']) and
                            any(w['owner']=='Codex C' for w in r['next_work']) for r in rows))

    def test_historical_round_R_menu_not_a_common_kappa_schedule(self):
        rows=self.result['evidence_plan']['configurations'][0]['rows']
        self.assertEqual([r['saved_evidence']['historical_physical_recipe_present'] for r in rows[::2]], [True]*7+[False]*6)
        self.assertTrue(all(not r['saved_evidence']['full_V2_configuration_match'] for r in rows))
        self.assertTrue(all(not r['saved_evidence']['reusable_as_V2_sufficient_N'] for r in rows))
        self.assertTrue(all(not r['saved_evidence']['reusable_as_V2_price'] for r in rows))
        for r in rows:
            hp=r['saved_evidence']['saved_HP_N']
            if hp and hp['availability']=='AVAILABLE':self.assertEqual(hp['RMSE_qualification'],'UNQUALIFIED')

    def test_C1_same_q_different_R_model_false_and_finite_cohort_not_reused(self):
        rows=self.result['evidence_plan']['configurations'][0]['rows']
        for row in [r for r in rows if r['m'] in (3,4)]:
            near=row['saved_evidence']['C1_same_q_cells'];self.assertEqual(len(near),2)
            self.assertTrue(all(not r['physical_match'] and r['model_qualified_for_candidate'] is False for r in near))
            self.assertTrue(all(r['expectation']=='FINITE_COHORT_ONLY' for r in near))
            self.assertTrue(all('R_outer' in r['differing_fields'] for r in near))

    def test_C_historical_SOURCE_and_profile_mismatch_rejected(self):
        prev=self.store.read(H.B_HISTORY);prices=self.store.read(H.C_HISTORY)
        prices['contexts']['H4']['binding']['snapshot']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'HISTORICAL_H_STATE_SOURCE'):
            H.historical(self.config,self.saved['inputs']['design'],prev,prices)
        prices=self.store.read(H.C_HISTORY);prices['contexts']['H4']['profile']['compiler']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'HISTORICAL_COMPILER'):
            H.historical(self.config,self.saved['inputs']['design'],prev,prices)

    def test_C_requests_include_m0_m6_event_law_full_wrapper_without_mean(self):
        requests=self.result['price_requests'];self.assertEqual(len(requests),26)
        for m,q,R,events in [(0,1,2,2),(6,64,128,8192)]:
            rows=[r for r in requests if r['m']==m];self.assertEqual({r['axis'] for r in rows},{'X','Y'})
            for r in rows:
                self.assertEqual((r['q_PF'],r['R_outer'],r['event_count']),(q,R,events))
                self.assertEqual(r['identity']['profile']['wrapper'],'full_Hadamard_with_measurement')
                self.assertEqual(r['identity']['profile']['P'],0);self.assertIsNone(r['requested_mean'])
                self.assertIsNone(r['actual_price_law']);self.assertIsNone(r['law_distance_qualification'])

    def test_four_method_coverage_is_not_complete_and_existing_SOURCES_unchanged(self):
        plan=self.result['evidence_plan']
        self.assertEqual(set(plan['method_coverage']),set(H.B.COMPARISON_METHODS))
        self.assertTrue(plan['method_coverage']['PR']['present'])
        self.assertTrue(all(not plan['method_coverage'][k]['present'] for k in ('FULL_PF','TRUNCATED_PF','FULL_H_RTE')))
        self.assertFalse(plan['requested_methods_have_ready_evidence'])
        for path,commit in [('src/trottertracks/bv2_mean_price.py',H.C_COMMIT),
                            ('src/trottertracks/pr_rpe_rmse_controller.py','6a9d7fd36b687e58aecbed1278c6d7927d6827d6')]:
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            r=subprocess.run([sys.executable,'-B','scripts/prepare_h4_s0_v2.py','--out',tmp],cwd=ROOT,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(r.returncode,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
