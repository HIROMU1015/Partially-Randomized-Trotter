"""Metadata-only published fixtures and artificial C-S data. No real S3 calls."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import rpe_s3_intake as T
from trottertracks import rpe_d_evidence_intake as D
from trottertracks import pr_rpe_rmse_evidence as E
from trottertracks.bv2_mean_price import V2MeanPriceLookup

B,p=T.B,T.p
MOCK=dict(fixture='SYNTHETIC_ONLY')


class S3IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v,cls.store=T.load_published(ROOT)

    def accept_a(self,v=None):
        v=v or self.v
        return T.accept_a(v['config'],v['packet'],v['previous'],v['result'],v['physical'],store=self.store)

    def test_01_all_thirteen_saved_nominals_new_scope_not_sufficient_N(self):
        out=self.accept_a()
        self.assertTrue(out['accepted']);self.assertEqual(out['nominal_N_XY'],T.NOMINAL)
        self.assertEqual(out['raw_mean_rounds'],13);self.assertEqual(out['nominal_axis_cells'],26)
        self.assertIsNone(out['global_RMSE_bound']);self.assertIsNone(out['sufficient_N'])
        for m in (0,1,6):self.assertEqual(self.v['packet']['rounds'][m],self.v['previous']['rounds'][m])
        with self.assertRaises(ValueError):
            T.M.validate_m1_packet(self.v['config'],self.v['packet'],self.v['previous'],numerical_source=T.NUMERICAL,adapter_source=T.ADAPTER)

    def test_02_completion_scope_prior_SOURCE_identity_law_profile_and_request_rejected(self):
        changes=[lambda v:v['packet']['diagnostic_scope'].update(preserved_rounds=[0,6]),
            lambda v:v['packet']['source'].update(commit='BAD'),
            lambda v:v['packet']['rounds'][3]['identity']['profile'].update(compiler='OTHER'),
            lambda v:v['packet']['rounds'][2]['identity']['binding'].update(state='OTHER'),
            lambda v:v['packet']['rounds'][2]['law_binding'].update(kappa=None),
            lambda v:v['packet']['rounds'][2].update(source_request_id='OTHER'),
            lambda v:v['packet']['rounds'][0]['diagnostic_source'].update(commit='OTHER')]
        for change in changes:
            values=deepcopy(self.v);change(values)
            with self.assertRaises(ValueError):self.accept_a(values)

    def test_03_old_values_N_bound_and_nominal_integer_precision(self):
        for change in (lambda v:v['packet']['rounds'][1].update(nominal_N_X=61),
                lambda v:v['packet']['rounds'][2].update(N_X=56),
                lambda v:v['packet']['rmse_evidence'].update(rmse_bound_ha=0)):
            values=deepcopy(self.v);change(values)
            with self.assertRaises(ValueError):self.accept_a(values)
        packet=deepcopy(self.v['packet']);packet['rounds'][2]['nominal_N_X']=10**80+1
        self.assertTrue(T.M.validate_completion_packet(self.v['config'],packet,self.v['previous'],numerical_source=T.NUMERICAL,
            adapter_source=T.ADAPTER,previous_source=T.PREVIOUS))
        packet['rounds'][2]['nominal_N_X']=float(10**80+1)
        with self.assertRaises(ValueError):T.M.validate_completion_packet(self.v['config'],packet,self.v['previous'],
            numerical_source=T.NUMERICAL,adapter_source=T.ADAPTER,previous_source=T.PREVIOUS)

    def test_04_original_mean_norm_missing_source_gate_and_target_qualification_preserved(self):
        self.accept_a()
        for m in (3,4):
            d=self.v['physical']['rounds'][m]['diagnostics']
            self.assertIsNone(d['norm_evidence']['observed_error'])
            self.assertEqual(d['norm_evidence']['kind'],'SOURCE_GATE_LIMIT_NOT_OBSERVED')
        for change in (lambda v:v['physical']['rounds'][3]['diagnostics']['norm_evidence'].update(observed_error=0),
                lambda v:v['physical']['rounds'][2]['diagnostics']['signal']['raw'].update(real=0),
                lambda v:v['physical']['target'].update(ground_state_certified=True),
                lambda v:v['physical']['target'].update(reference_solver_energy_ha=0)):
            values=deepcopy(self.v);change(values)
            with self.assertRaises(ValueError):self.accept_a(values)

    def test_05_s3_complete_inputs_are_not_scientific_ready_and_no_probabilities_generated(self):
        with patch.object(T.S,'simulate',side_effect=AssertionError('No H4 Monte Carlo')), \
             patch.object(T.S,'trial_budget',side_effect=AssertionError('No real trial budget')), \
             patch.object(T.S.A,'decode_supplied_coordinates',side_effect=AssertionError('No real decode')):
            ready=T.prepare_statistics(self.v['config'],self.v['packet'],self.v['physical'],self.v['models'])
        self.assertTrue(ready['input_complete']);self.assertEqual(len(ready['rows']),13)
        self.assertFalse(ready['scientific_execution_ready']);self.assertFalse(ready['S3_executed'])
        self.assertFalse(ready['probabilities_evaluated']);self.assertFalse(ready['physical_qualification_complete'])
        self.assertFalse(ready['target']['truth_proxy_adopted']);self.assertIsNone(ready['target']['energy_ha'])
        self.assertIsNone(ready['S3_conditions']['simulation_repetitions']);self.assertIsNone(ready['global_RMSE_bound'])
        self.assertEqual(ready['rows'][3]['mean']['norm_evidence']['observed_error'],None)

    def test_06_K0_PASS_four_K1_values_and_intervals_structural_only(self):
        out=T.accept_k1(self.v['config'],self.v['k0'],self.v['k1'])
        self.assertTrue(out['complete_four_cell_values']);self.assertEqual(out['compiled_price_records'],[])
        self.assertEqual(out['mean_S_by_cell'],{'0/X':1743.6682557187337,'0/Y':1743.6682557187337,
            '6/X':1764659.275564426,'6/Y':1764659.275564426})
        for got,saved in zip(out['diagnostic']['rows'],self.v['k1']['values']):
            self.assertEqual(got['uncertainty']['numerical_interval'],saved['numerical_enclosure'])
        self.assertIsNone(out['mean_compiled_C']);self.assertFalse(out['scientific_price_qualification'])

    def test_07_K0_failure_axis_law_source_identity_and_compiled_promotion_rejected(self):
        for change in (lambda a,b:a.update(status='UNKNOWN'),lambda a,b:b.update(source='OTHER'),
                lambda a,b:b['values'][0]['law_binding'].update(kappa=None),
                lambda a,b:b['values'][0].update(axis='Z'),
                lambda a,b:b.update(mean_compiled_C=1743),
                lambda a,b:b['values'][0].update(numerical_enclosure=[0,1])):
            k0,k1=deepcopy(self.v['k0']),deepcopy(self.v['k1']);change(k0,k1)
            with self.assertRaises(ValueError):T.accept_k1(self.v['config'],k0,k1)

    def row(self,m=0,axis='X'):
        # All numbers for residual arithmetic are synthetic, not C's real means.
        r=self.v['config']['rounds'][m]
        return dict(m=m,axis=axis,identity=deepcopy(r['identity']),law_binding=deepcopy(r['law_binding']),
            physical_candidate_id=r['physical_candidate']['id'],profile=deepcopy(self.v['config']['spec']['profile']),
            source=MOCK,quantity=T.M.STRUCTURAL,mean_S=100,law_context={'law':'MOCK_IDEAL'},
            checks={'K0':'PASS_SYNTHETIC'},uncertainty={'numerical_interval':[100,100]})

    def residual(self,row,mean=Fraction(-3,2),law='MOCK_IDEAL'):
        candidate=row['physical_candidate_id'];compiler=row['profile']['compiler']
        return dict(m=row['m'],axis=row['axis'],identity=deepcopy(row['identity']),law_binding=deepcopy(row['law_binding']),
            physical_candidate_id=candidate,profile=deepcopy(row['profile']),quantity=T.R.MEAN_D,target='C_MINUS_S',
            mode='PAIRED_DIRECT_FINITE_COHORT',coverage='FULL_WRAPPER',mean_D=mean,role='DEVELOPMENT',
            source=MOCK,law_context={'law':law},uncertainty={'systematic':None},expectation='FINITE_COHORT_ONLY',n_paths=1,
            paired_paths=[dict(index=0,event_digest='MOCK',S_event_digest='MOCK',C_event_digest='MOCK',
                structural_source=MOCK,compiled_source=MOCK,axis=row['axis'],candidate_id=candidate,compiler=compiler,forced_conditional=False)])

    def test_08_D_interface_exact_artificial_sum_is_always_unqualified(self):
        row=self.row();r=self.residual(row)
        out=T.residual_price(self.v['config'],row,r,cohort='MOCK_PAIRED',source=MOCK)
        self.assertEqual(out['component']['mean'],Fraction(197,2));self.assertFalse(out['qualified_for_compiled_price'])
        self.assertEqual(out['price_record']['price_status'],'UNQUALIFIED_MODEL')
        self.assertEqual(out['component']['components']['residual']['mode'],'PAIRED_DIRECT_FINITE_COHORT')
        self.assertIsNone(out['component']['uncertainty']['law_distance'])

    def test_09_paired_paths_axis_profile_full_wrapper_and_double_counting(self):
        row=self.row();r=self.residual(row)
        for change in (lambda v:v['paired_paths'][0].update(C_event_digest='DIFFERENT'),
                lambda v:v['paired_paths'][0].update(axis='Y'),
                lambda v:v['paired_paths'][0].update(compiler='OTHER'),
                lambda v:v['paired_paths'][0].update(forced_conditional=True),
                lambda v:v.update(quantity=T.R.MEAN_C),lambda v:v.update(axis='Y'),
                lambda v:v['profile'].update(compiler='OTHER')):
            bad=deepcopy(r);change(bad)
            with self.assertRaises(ValueError):T.residual_price(self.v['config'],row,bad,cohort='MOCK',source=MOCK)
        with self.assertRaises(ValueError):T.residual_price(self.v['config'],row,[r,r],cohort='MOCK',source=MOCK)
        r['coverage']='INTERNAL_RTE_ONLY'
        self.assertIsNone(T.residual_price(self.v['config'],row,r,cohort='MOCK',source=MOCK)['price_record']['mean']['rz_count'])

    def test_10_missing_residual_and_mismatched_ideal_sampler_law_do_not_supply_price(self):
        row=self.row()
        for residual in (self.residual(row,None),self.residual(row,law='PCG64_CDF')):
            result=T.residual_price(self.v['config'],row,residual,cohort='MOCK',source=MOCK)
            self.assertIsNone(result['price_record']['mean']['rz_count'])
        mismatch=T.residual_price(self.v['config'],row,self.residual(row,law='PCG64_CDF'),cohort='MOCK',source=MOCK)
        self.assertIn('S_AND_D_LAW_DIFFER_OR_UNDECLARED',mismatch['component']['reasons'])
        absent=T.residual_price(self.v['config'],row,None,cohort='MOCK_MISSING',source=MOCK)
        self.assertIsNone(absent['component']['components']['residual'])
        self.assertIsNone(absent['price_record']['mean']['rz_count'])

    def test_11_provider_assessment_planner_reuse_with_G_null(self):
        row=self.row();out=T.residual_price(self.v['config'],row,self.residual(row),cohort='MOCK',source=MOCK)
        price=out['price_record'];cid=row['identity']['candidate']['id']
        V2MeanPriceLookup(self.v['config'],[price],integration=T.H.I)  # Passive record input only.
        with patch.object(B,'cost',side_effect=AssertionError('No real G')):
            assessment=B.assess(self.v['config'],self.v['packet'],[price],cohorts={(cid,'X'):'MOCK'})
            plan=E.plan([dict(configuration=self.v['config'],N_packet=self.v['packet'])],price_policy='qualified_expectation')
        self.assertFalse(assessment['eligible']);self.assertIn('C_UNQUALIFIED_MODEL',assessment['reasons'])
        self.assertIsNone(assessment['cost']);self.assertIsNone(plan['G']);self.assertIsNone(plan['winner'])
        structural=dict(price,quantity=T.M.STRUCTURAL)
        with self.assertRaises(ValueError):V2MeanPriceLookup(self.v['config'],[structural],integration=T.H.I)

    def test_12_Full_PF_coverage_preserved_and_science_never_called(self):
        fixed=self.store.read(dict(commit='170047ef6e7a72aef5f4c6ff9c73f942097a694e',
            path='docs/research/rpe_decoder_statistics_20261113/metadata01/D_RECEIPTS_ACCEPTANCE.json'))
        self.assertEqual(fixed['totals']['X'],{'EXACT_METADATA_MATCH':73,'MISSING_IN_CHECKED_SCOPE':63})
        self.assertEqual(fixed['previous_totals']['X']['CONDITIONAL_REUSE'],17)
        self.assertFalse(fixed['candidates_adopted_as_menu']);self.assertIsNone(fixed['G'])
        with patch.object(T.S,'simulate',side_effect=AssertionError('No H4 S3')), \
             patch.object(T.S.A,'evaluate',side_effect=AssertionError('No A N/mean evaluation')), \
             patch.object(T.R,'cohort_summary',side_effect=AssertionError('No real residual statistics')):
            self.accept_a();T.accept_k1(self.v['config'],self.v['k0'],self.v['k1'])
            T.prepare_statistics(self.v['config'],self.v['packet'],self.v['physical'],self.v['models'])

    def test_13_readiness_CLI_has_no_scientific_execution_entry(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import prepare_rpe_s3_intake as CLI
        with patch.object(T.S,'simulate',side_effect=AssertionError('No H4 S3')), \
             patch.object(T.S,'trial_budget',side_effect=AssertionError('No real budget calculation')), \
             patch.object(B,'cost',side_effect=AssertionError('No G')), \
             patch.object(T.S.A,'evaluate',side_effect=AssertionError('No nominal N recomputation')), \
             patch.object(T.R,'cohort_summary',side_effect=AssertionError('No residual statistics')):
            result=CLI.build()
        ready=result['S3_READINESS']
        self.assertTrue(ready['input_complete']);self.assertFalse(ready['scientific_execution_ready'])
        self.assertFalse(ready['S3_executed']);self.assertEqual(result['C_STRUCTURAL_ACCEPTANCE']['compiled_price_records'],[])
        self.assertIsNone(result['EVIDENCE_PLAN']['G'])
        self.assertTrue(all(r['mean']['rz_count'] is None for r in result['D_RESIDUAL_INTERFACE']['price_records']))


if __name__=='__main__':unittest.main(verbosity=2)
