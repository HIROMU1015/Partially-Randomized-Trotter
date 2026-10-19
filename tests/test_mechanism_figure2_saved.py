"""Artificial metadata tests for H4 P lineage and saved-token plotting."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_mechanism_paper import fixture,_schema as m
spec=importlib.util.spec_from_file_location('figure2_saved_test',ROOT/'scripts/preview_mechanism_figure2.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def tables():
    result=m.build_tables(**fixture())
    for row in result['main_table']:
        if row['contrast_id'] in m.CONFIRMATION_IDS:row['PR_cohort']='INDEPENDENT_FIXED_CONFIRMATION_N48'
    for name in ['core_axes','explanation_factors','display_plan']:
        for row in result[name]:
            if row['contrast_id'] in m.CONFIRMATION_IDS:row['PR_cohort']='INDEPENDENT_FIXED_CONFIRMATION_N48'
    return result


def p_fixture(main):
    h4=[r for r in main if r['system']=='H4'];numbers=(0,1,2,3,4,3,5,6,5,6)
    comps=[]
    for row,i in zip(h4,numbers):
        row['PR_id']='FAKE_P_'+str(i)
        comps.append(dict(endpoint=dict(cell_id=row['PR_id'],epsilon=float(row['epsilon']),role='PRIMARY',PF_id=row['PF_id'],PF_G_exact=row['G_PF']),
            status='REPRODUCED',population_mean_certified=False,G_exact='123/7',paired_SE=1.25))
    for i,eps in [(7,.001),(7,.0005),(8,.001),(8,.0005)]:
        comps.append(dict(endpoint=dict(cell_id='FAKE_P_'+str(i),epsilon=eps,role='DIAGNOSTIC',PF_id='FAKE_PF',PF_G_exact='100'),
            status='REPRODUCED',population_mean_certified=False,G_exact='123/7',paired_SE=1.25))
    n={str('FAKE_P_'+str(i)):7 for i in range(9)}
    p=dict(SOURCE=m.H4_P_SOURCE,scientific_result_sha256='FAKE_RAW_SHA',n_fixed=n,comparisons=comps,
        old_exploration_samples_pooled=False,population_mean_certified=False)
    audit=dict(SOURCE=m.H4_P_SOURCE,scientific_result_sha256='FAKE_RAW_SHA',status='COMPLETE_MANDATORY_STOP',
        primary_comparisons=10,diagnostic_comparisons=4,n_fixed=n,actual_calls=dict(trajectory=4185,wrapper_compile=8370))
    return p,audit


class PProvenanceTests(unittest.TestCase):
    def test_P_is_separate_numeric_cohort_and_history_retained(self):
        main=tables()['main_table'];p,a=p_fixture(main);before=copy.deepcopy(main)
        out,separate=m.add_h4_p_provenance(main,p,a)
        self.assertEqual(main,before);self.assertEqual(len(separate),14)
        for original,row in zip(before,out):
            self.assertTrue(all(row[k]==v for k,v in original.items()))
            if row['system']=='H4':
                self.assertEqual(row['H4_new_provenance'],'D_COMPLETE_P_PENDING')
                self.assertEqual(row['H4_latest_provenance'],'D_AND_P_COMPLETE_LEGACY_DF12')
                self.assertEqual(row['H4_stage'],'H4_PRE_D_P_VALIDATION')
        self.assertTrue(all(r['PR_G_exact']=='123/7' for r in separate))
        self.assertTrue(all(r['R_G']=='03/04' for r in out))
    def test_source_and_raw_SHA_mismatch_rejected(self):
        main=tables()['main_table'];p,a=p_fixture(main)
        p['SOURCE']='OTHER'
        with self.assertRaisesRegex(ValueError,'H4_P_SOURCE'):m.add_h4_p_provenance(main,p,a)
        p['SOURCE']=m.H4_P_SOURCE;a['scientific_result_sha256']='OTHER'
        with self.assertRaisesRegex(ValueError,'H4_P_RAW_RESULT_SHA'):m.add_h4_p_provenance(main,p,a)
    def test_P_pooling_and_false_completion_rejected(self):
        main=tables()['main_table'];p,a=p_fixture(main);p['old_exploration_samples_pooled']=True
        with self.assertRaisesRegex(ValueError,'H4_P_COHORT'):m.add_h4_p_provenance(main,p,a)
        p['old_exploration_samples_pooled']=False;p['comparisons'][0]['status']='FAILED'
        with self.assertRaisesRegex(ValueError,'H4_P_SAVED_STATUS'):m.add_h4_p_provenance(main,p,a)
    def test_P_cannot_replace_original_PF(self):
        main=tables()['main_table'];p,a=p_fixture(main);p['comparisons'][0]['endpoint']['PF_G_exact']='999'
        with self.assertRaisesRegex(ValueError,'H4_P_FIXED_PF_JOIN'):m.add_h4_p_provenance(main,p,a)


class FigureTests(unittest.TestCase):
    def build(self,data):return m.figure_bundle(data['main_table'],data['core_axes'],data['explanation_factors'],data['display_plan'])
    def test_fixed_18_plus_two_preserves_saved_strings(self):
        data=tables();before=copy.deepcopy(data);result=self.build(data)
        self.assertEqual((len(result['rows']),len(result['examples'])),(18,2));self.assertEqual(data,before)
        self.assertEqual(result['rows'][0]['R_C_axis']['cosine'],'03/04')
        self.assertEqual(result['examples'][0]['R_G'],'03/04')
        self.assertIn('missing',result['missing_annotation'])
    def test_A82_cannot_be_completed_or_zero_filled(self):
        data=tables();next(r for r in data['main_table'] if r['contrast_id']==m.MISSING_ID)['explanation_status']='EXPLANATION_COMPLETE'
        with self.assertRaisesRegex(ValueError,'KEEP_A82_MISSING'):self.build(data)
        data=tables();r=copy.deepcopy(data['explanation_factors'][0]);r['contrast_id']=m.MISSING_ID;data['explanation_factors'].append(r)
        with self.assertRaisesRegex(ValueError,'FIGURE_64_62'):self.build(data)
    def test_cohort_or_model_relabeling_rejected(self):
        data=tables();data['core_axes'][0]['PF_cost_class']='OTHER'
        with self.assertRaisesRegex(ValueError,'FIGURE_AXIS_METADATA'):self.build(data)
        data=tables();data['main_table'][-1]['n']='32'
        with self.assertRaises(ValueError):self.build(data)
    def test_render_SHA_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);names=('main_table.csv','core_axes.csv','explanation_factors.csv','display_plan.csv')
            for name in names:(root/name).write_text('fake\n')
            binding=dict(scope='SAVED_FIGURE2_COORDINATES_ONLY',core_commit='2bf5d83293b3a47f672805fae956ff2e50df2878',
                explanation_commit='0edd62403c35c5ee18b2a2a152158faca5c2ebd8',files_sha256={n:'0'*64 for n in names})
            (root/'render_binding.json').write_text(json.dumps(binding))
            with self.assertRaisesRegex(ValueError,'FIGURE_INPUT_SHA_MISMATCH'):plot.saved_bundle(root)


if __name__=='__main__':unittest.main()
