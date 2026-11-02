"""Artificial schema, non-arithmetic transfer, provenance and preview tests."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
_spec=importlib.util.spec_from_file_location('mechanism_paper_test_schema',ROOT/'src/trotterlib/mechanism_paper.py')
_schema=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_schema)
AXES,CORE_COLUMNS,FACTOR_COLUMNS=_schema.AXES,_schema.CORE_COLUMNS,_schema.FACTOR_COLUMNS
H4_D_COMMIT,TranscriptionError,build_tables=_schema.H4_D_COMMIT,_schema.TranscriptionError,_schema.build_tables


def fixture():
    data=dict(metadata=[],core_ratios=[],core_totals=[],core_axes=[],comparison_status=[],factors=[],h4_winners=[])
    for system in ('H4','H6','H8'):
        for epsilon in ('0.05','0.01','0.005','0.001','0.0005'):
            for role in ('NEW_PR','OLD_PR'):
                cid='A82_OLD_PR' if (system,epsilon,role)==('H8','0.05','OLD_PR') else system+'_'+epsilon+'_'+role
                data['metadata'].append(dict(contrast_id=cid,system=system,H4_stage='H4_PRE_D_P_VALIDATION' if system=='H4' else 'NOT_APPLICABLE',
                    epsilon=epsilon,PR_id='FAKE_PR',PF_id='FAKE_PF',PR_role=role,n='8' if role=='NEW_PR' else '32',
                    PR_cohort='ARTIFICIAL_'+role,cohorts_pooled='False',
                    PF_cost_class='VALIDATED_MODEL_ESTIMATE' if system=='H6' and epsilon in ('0.01','0.005','0.001') else 'DIRECT_MEASURED'))
    for cid,epsilon in [('H8_CONFIRM_1','0.05'),('H8_CONFIRM_3','0.0005')]:
        data['metadata'].append(dict(contrast_id=cid,system='H8',H4_stage='NOT_APPLICABLE',epsilon=epsilon,PR_id='FAKE_PR',
            PF_id='FAKE_PF',PR_role='FIXED_CONFIRMATION',n='48',PR_cohort='ARTIFICIAL_INDEPENDENT_N48',cohorts_pooled='False',PF_cost_class='DIRECT_MEASURED'))
    for meta in data['metadata']:
        core={k:meta[k] for k in ('contrast_id','system','H4_stage','PF_cost_class')}
        core.update({k:'03/04' for k in CORE_COLUMNS})
        core.update(G_PF='100',saved_SE='1.2300e+02',core_status='CORE_COMPLETE',explanation_status='OLD_UNRESOLVED',population_conclusion='UNRESOLVED')
        data['core_ratios'].append(dict(core));data['core_totals'].append(dict(core))
        status=dict(meta,core_reused_without_recalculation='True',
            explanation_status='EXPLANATORY_INPUT_MISSING' if meta['contrast_id']=='A82_OLD_PR' else 'EXPLANATION_COMPLETE')
        data['comparison_status'].append(status)
        for axis in AXES:
            data['core_axes'].append(dict(contrast_id=meta['contrast_id'],system=meta['system'],axis=axis,
                R_N_axis='05/04',R_C_axis='03/04',w_PF='1/2',weighted_term='15/32'))
            if meta['contrast_id']!='A82_OLD_PR':
                data['factors'].append(dict(meta,axis=axis,**{k:'1.0000000000001' for k in FACTOR_COLUMNS},
                    **{'log_'+k:'0.0000000000001' for k in FACTOR_COLUMNS}))
    for mode in ('USUAL_SIGNED','EMPIRICAL_NONCANCELLATION'):
        for epsilon in ('0.05','0.01','0.005','0.001','0.0005'):
            data['h4_winners'].append(dict(bias_mode=mode,epsilon=epsilon,saved_subset_winner_changed=False,
                best=dict(cell_id='FAKE_PF',G_exact='100')))
    return data


class TransferTests(unittest.TestCase):
    def setUp(self): self.data=fixture()
    def build(self): return build_tables(**self.data)
    def rejects(self,code):
        with self.assertRaisesRegex(TranscriptionError,code):self.build()

    def test_exact_text_and_SE_are_not_normalized(self):
        row=self.build()['main_table'][0]
        self.assertEqual(row['R_G'],'03/04');self.assertEqual(row['saved_SE'],'1.2300e+02')

    def test_counts_and_original_order(self):
        result=self.build()
        self.assertEqual(result['counts']['contrasts'],32)
        self.assertEqual(result['counts']['axes'],64)
        self.assertEqual(result['counts']['explanation_factor_axes'],62)
        self.assertEqual([r['contrast_id'] for r in result['main_table']],[r['contrast_id'] for r in self.data['metadata']])

    def test_input_not_mutated(self):
        prior=copy.deepcopy(self.data);self.build();self.assertEqual(prior,self.data)

    def test_H4_overlay_keeps_historical_stage_and_status(self):
        row=self.build()['main_table'][0]
        self.assertEqual(row['H4_stage'],'H4_PRE_D_P_VALIDATION')
        self.assertEqual(row['H4_new_provenance'],'D_COMPLETE_P_PENDING')
        self.assertEqual(row['H4_D_result_commit'],H4_D_COMMIT)
        self.assertEqual(row['historical_core_explanation_status'],'OLD_UNRESOLVED')

    def test_missing_A82_not_filled_after_metadata_recovery(self):
        result=self.build();row=next(r for r in result['main_table'] if r['contrast_id']=='A82_OLD_PR')
        self.assertEqual(row['explanation_status'],'EXPLANATORY_INPUT_MISSING')
        self.assertFalse(any(r['contrast_id']=='A82_OLD_PR' for r in result['explanation_factors']))

    def test_model_and_direct_H6_retained(self):
        rows=[r for r in self.build()['main_table'] if r['system']=='H6']
        self.assertEqual({r['PF_cost_class'] for r in rows},{'DIRECT_MEASURED','VALIDATED_MODEL_ESTIMATE'})

    def test_fixed_display_rule_and_independent_n48(self):
        plan=self.build()['display_plan']
        self.assertEqual(sum(r['display_group']=='MAIN_FIXED_EPSILON' for r in plan),18)
        self.assertEqual(sum(r['display_group']=='H8_INDEPENDENT_N48_EXAMPLE' for r in plan),2)
        self.assertTrue(all(r['n']=='48' for r in plan if r['display_group']=='H8_INDEPENDENT_N48_EXAMPLE'))

    def test_duplicate_id_rejected(self):
        self.data['metadata'][1]['contrast_id']=self.data['metadata'][0]['contrast_id'];self.rejects('FIXED_32')
    def test_reordering_rejected(self):
        self.data['core_totals'].reverse();self.rejects('FIXED_ROW_ORDER')
    def test_saved_numeric_difference_rejected(self):
        self.data['core_totals'][0]['R_G']='3/4';self.rejects('SAVED_VALUE_MISMATCH')
    def test_missing_axis_rejected(self):
        self.data['core_axes'].pop();self.rejects('EXACT_64')
    def test_duplicate_axis_rejected(self):
        self.data['core_axes'].append(dict(self.data['core_axes'][0]));self.rejects('DUPLICATE')
    def test_unexpected_explanation_missing_rejected(self):
        self.data['factors'].pop();self.rejects('EXACT_62')
    def test_implicit_A82_completion_rejected(self):
        next(r for r in self.data['comparison_status'] if r['contrast_id']=='A82_OLD_PR')['explanation_status']='EXPLANATION_COMPLETE';self.rejects('HISTORICAL_EXPLANATION')
    def test_candidate_mismatch_rejected(self):
        self.data['factors'][0]['PR_id']='OTHER';self.rejects('FACTOR_METADATA')
    def test_model_label_change_rejected(self):
        self.data['core_ratios'][0]['PF_cost_class']='VALIDATED_MODEL_ESTIMATE';self.rejects('PF_CLASS')
    def test_cohort_pooling_rejected(self):
        self.data['metadata'][0]['cohorts_pooled']='True';self.rejects('COHORT_POOLING')
    def test_n48_relabeling_rejected(self):
        self.data['metadata'][-1]['n']='32';self.rejects('N48_COHORT')
    def test_H4_winner_reselection_rejected(self):
        self.data['h4_winners'][0]['saved_subset_winner_changed']=True;self.rejects('H4_SAVED_D')
    def test_H4_cost_substitution_rejected(self):
        self.data['h4_winners'][0]['best']['G_exact']='101';self.rejects('H4_D_SAVED')
    def test_no_scientific_imports_in_transfer_module(self):
        tree=ast.parse((ROOT/'src/trotterlib/mechanism_paper.py').read_text())
        names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):names.append(node.module)
        self.assertEqual(set(names),{'collections','copy'})


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('figure2_preview',ROOT/'scripts/preview_mechanism_figure2.py')
        cls.preview=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.preview)

    def test_real_data_route_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'forbidden'
            with self.assertRaisesRegex(ValueError,'REAL_EVIDENCE'):
                self.preview.render_synthetic({'origin':'SAVED_REAL_VALUES'},path)
            self.assertFalse(path.exists())

    def test_fixture_has_explicit_badge_and_two_independent_examples(self):
        data=self.preview.synthetic_fixture()
        self.assertEqual(data['origin'],'SYNTHETIC_ONLY')
        self.assertEqual(len(data['examples']),2)
        self.assertIn('gap retained',data['missing_annotation'])

    def test_existing_output_rejected_before_plot_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                self.preview.render_synthetic(self.preview.synthetic_fixture(),tmp)


class SourceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('figure2_prepare',ROOT/'scripts/prepare_mechanism_figure2.py')
        cls.prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.prepare)

    def test_SHA_mismatch_stops_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'fake.csv';data.write_text('fake\n1\n')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(mode='SAVED_VALUE_TRANSCRIPTION_ONLY',
                inputs={'metadata':dict(path=str(data),sha256='0'*64,format='csv')})))
            with self.assertRaisesRegex(ValueError,'INPUT_SHA_MISMATCH'):
                self.prepare.prepare(manifest,root/'output')
            self.assertFalse((root/'output').exists())

    def test_non_transcription_mode_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'manifest.json';manifest.write_text('{"mode":"REANALYZE"}')
            with self.assertRaisesRegex(ValueError,'TRANSCRIPTION_MODE'):
                self.prepare.read_sources(manifest)


if __name__=='__main__': unittest.main()
