"""Metadata and synthetic mutation checks; no new scientific computation."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from trottertracks import price_model_validation as V

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m3_admission',ROOT/'scripts/admit_pf_m3_saved_prices.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)


class Metadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(V,'fit',side_effect=AssertionError('No fit')),\
             patch.object(V,'predict',side_effect=AssertionError('No prediction')),\
             patch.object(F.M.PN,'paper_cost',side_effect=AssertionError('No G')):
            cls.packet=F.prepare(ROOT)

    def guarded(self,packet):
        with patch.object(F.G,'_document',return_value=packet['proofs']),\
             patch.object(V,'fit',side_effect=AssertionError('No fit')),\
             patch.object(V,'predict',side_effect=AssertionError('No prediction')):
            return F.bind(ROOT,packet,'a'*40)

    def test_twelve_distinct_axis_locators(self):
        rows=self.packet['rows'];self.assertEqual(len(rows),12)
        self.assertTrue(all(r['m']==3 and r['status']=='PENDING_C_GUARD' for r in rows))
        for pid in {r['physical_id'] for r in rows}:
            rr=[r for r in rows if r['physical_id']==pid]
            self.assertEqual({r['axis'] for r in rr},{'X','Y'})
            self.assertNotEqual(rr[0]['origin']['fingerprint'],rr[1]['origin']['fingerprint'])
            self.assertNotEqual(rr[0]['price_source'],rr[1]['price_source'])

    def test_original_missing_fields_not_invented_or_reassigned(self):
        for r in self.packet['rows']:
            self.assertIsNone(r['original_cohort']);self.assertIsNone(r['legacy_canonical_id'])
            self.assertIsNone(r['model_group']);self.assertEqual(r['role'],F.ROLE)
        m5=[r for r in self.packet['rows'] if r['physical_id']=='1b3aa083ff69842d8f31']
        self.assertTrue(all(r['legacy_trial_id']=='M5_MIN_e847f177d7169dd3' for r in m5))
        self.assertTrue(all('csv_selector' in r['price_source'] for r in m5))
        self.assertTrue(all(p['cohort'] is None for p in self.packet['prices']))

    def test_current_guard_acceptance_and_36_unchanged_references(self):
        result=self.guarded(self.packet)
        self.assertEqual(result['counts'],dict(m3_requested=12,m3_accepted=12,m3_held=0,
            unchanged_m4_m5_calibration=24,unchanged_m6_holdout=12,total_accepted_direct_axes=48))
        self.assertEqual(len(result['guard']),12)
        self.assertTrue(all(r['status']=='ACCEPTED_SAVED_DIRECT_POINT' for r in result['rows']))

    def test_axis_gate_failure_does_not_fill_from_other_axis(self):
        packet=deepcopy(self.packet);packet['sides'][1]['implementation']['resolved_scalar']=999
        result=self.guarded(packet)
        self.assertEqual(result['counts']['m3_accepted'],11)
        self.assertEqual(result['rows'][1]['status'],'HELD_C_GUARD')
        self.assertEqual(result['rows'][0]['status'],'ACCEPTED_SAVED_DIRECT_POINT')
        self.assertTrue(result['rows'][1]['missing'])

    def test_original_axis_source_scalar_or_candidate_mismatch(self):
        row=self.packet['rows'][0];saved=F.read(ROOT,F.D)['saved_m3'][0]
        context=F.M.PN.I.price_context(self.packet['templates'][row['configuration_id']]['plan'],3)
        for field,value in [('fingerprint','other-axis'),('compiler','other-profile'),
                            ('scalar',999),('generation_SOURCE','f'*40)]:
            origin=deepcopy(row['origin']);origin[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                F.check_axis(row,saved,origin,context,row['origin']['scalar'])
        bad=deepcopy(row);bad['physical_candidate']['q']=99
        with self.assertRaises(ValueError):F.check_axis(bad,saved,row['origin'],context,row['origin']['scalar'])

    def test_direct_point_not_model_calibration_or_holdout(self):
        result=self.guarded(self.packet)
        self.assertFalse(result['model_refit']);self.assertEqual(result['predictions'],0)
        self.assertEqual(result['compile_calls'],0);self.assertIsNone(result['G']);self.assertIsNone(result['N'])
        for p in result['prices']:
            self.assertEqual(p['classification'],'DIRECT_COMPILED_DETERMINISTIC')
            self.assertEqual(p['expectation'],'DETERMINISTIC')
            self.assertEqual(p['diagnostic']['record_role'],F.ROLE)
            self.assertIsNone(p['diagnostic']['model_group'])

    def test_protected_models_and_cell_plan_bytes_preserved(self):
        for commit,path in [('12372fa6b53c878dc5dd124e036572ba0cd82061',
            'docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json'),
            ('43c15bfa34a22f313e55b824223b200332b7f23f',
             'docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json')]:
            current=F.C.blob(ROOT,'HEAD',path)
            self.assertEqual(F.C.blob(ROOT,commit,path),current)


if __name__=='__main__':unittest.main(verbosity=2)
