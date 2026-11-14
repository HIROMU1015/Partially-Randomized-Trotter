"""Metadata/synthetic checks only; no real scientific values are evaluated."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h4_alignment_builder',ROOT/'scripts/build_h4_candidate_alignment.py')
H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)


class Alignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=H.build(str(ROOT))

    def rows(self,name):return [r for r in self.data['rows'] if r['candidate']==name]

    def test_fourteen_schedules_and_all_round_axes(self):
        self.assertEqual(H.validate(self.data)['round_axis_rows'],364)
        for c in self.data['candidates']:
            self.assertEqual({(r['m'],r['axis']) for r in self.rows(c['candidate'])},
                             {(m,a) for m in range(13) for a in ('X','Y')})

    def test_reuses_A_audit_without_new_signal(self):
        self.assertEqual(H.AUDIT,'c212cee857d639d2400819bbca3afa1dc052a9f4')
        self.assertTrue(all(not r['signal']['new_signal'] for r in self.data['rows']))
        # S2 is missing m2 and m5..m11, per A; Y4/M5 missing m5..m11.
        for name,expected in [('FULL_PF S2_16h_over_7',{2,*range(5,12)}),
                              ('FULL_PF Y4_16h_over_3',set(range(5,12))),
                              ('TRUNCATED_B0I_P7 M5_16h',set(range(5,12)))]:
            self.assertEqual({r['m'] for r in self.rows(name) if r['signal']['status'].startswith('S3')},expected)

    def test_reuses_52_DF_rows_and_26_structural_axes(self):
        df=[r for r in self.data['rows'] if r['candidate'].startswith('PR-DF')]
        self.assertEqual(len(df),52)
        self.assertEqual(sum(r['price_coverage']['structural_expectation'] for r in df),26)
        self.assertTrue(all(r['decoder']['status']=='SOURCE_BOUND_TOTAL_VIEW' for r in df))
        self.assertTrue(all(r['MSE']['total_evaluation_status']=='NOT_EVALUATED' for r in df))
        self.assertTrue(all(r['N']['total_vector_status'] is None for r in df))
        self.assertEqual({r['decoder']['total_configuration_id'] for r in df},
                         {'9c3d392ec5c5ad373aaa1d96a8ba0828ddd268e6fef35da229029beec0229e8b',
                          'e6fe7e6845dd5eae8829f54b8bbae655ccf1c81c320c4d0ee1bfed0399bb5583'})
        self.assertTrue(all(r['physical_candidate']['method']=='PR' for r in df))
        self.assertFalse(any(r['price_coverage']['finite_path_or_cohort'] for r in self.rows('PR-DF LD2')))
        for r in self.rows('PR-DF LD1'):
            self.assertEqual({p['n_paths'] for p in r['price']},{2,4} if r['m'] in (0,6) else set())
            self.assertFalse(r['models']['C1']['qualified'])

    def test_saved_receipt_roles_and_cell_plan(self):
        direct=[r for r in self.data['rows'] if r['price_coverage']['admitted_deterministic']]
        self.assertEqual(len(direct),4)
        self.assertEqual({r['m'] for r in direct},{4})
        self.assertTrue(all(r['price_plan']['role']=='CALIBRATION' for r in direct))
        planned=[r for r in self.data['rows'] if r['price_plan']]
        self.assertEqual(len(planned),36)
        self.assertEqual(sum(r['price_plan']['action']=='NEW_COMPILE' for r in planned),32)
        self.assertTrue(all(r['price_plan']['role']=='HOLDOUT' for r in planned if r['m']==6))

    def test_wiring_price_not_calibration_or_law_mean(self):
        for name,m in [('PR-Pauli LD1',0),('FULL_H kappa2 K2 q=2^m',3)]:
            rr=[r for r in self.rows(name) if r['m']==m]
            for r in rr:
                self.assertEqual(r['price'][0]['n_paths'],1)
                self.assertEqual(r['price'][0]['tier'],'WIRING_SINGLE_PATH_ONLY')
                self.assertIsNone(r['price_plan'])
                self.assertEqual(r['price'][0]['diagnostic']['pilot_role'],'DIRECT_SINGLE_PATH_WIRING')
                self.assertEqual(r['price'][0]['diagnostic']['qualified_expectation_promotion'],'FORBIDDEN')

    def test_N_vectors_are_user_controls_not_sufficient(self):
        for name in ('PR-DF LD1','PR-DF LD2','PR-Pauli LD1','PR-Pauli LD2'):
            x=[r for r in self.rows(name) if r['axis']=='X']
            self.assertEqual([r['N']['used'] for r in x],[64,60,56,52,47,43,38,34,30,25,21,17,12])
            self.assertTrue(all(r['N']['sufficient_N'] is None for r in x))
        self.assertTrue(all(r['MSE']['status'].startswith('SAVED_PARTIAL') for r in self.rows('PR-DF LD1')))
        self.assertTrue(all(r['MSE']['status'].startswith('CONDITIONAL_OPT_IN') for r in self.rows('PR-Pauli LD1')))

    def test_distinct_physical_schedule_not_relabelled(self):
        links=self.data['cross_candidate_links']
        self.assertEqual(links[0]['status'],'DECODER_OR_CANONICAL_BRIDGE_PENDING')
        self.assertTrue(links[0]['physical_round_equal'])
        self.assertTrue(all(l['status']=='PHYSICAL_CANDIDATE_MISMATCH' for l in links[1:]))
        self.assertTrue(all(not l['N_or_price_transfer'] for l in links))

    def test_legacy_locator_does_not_become_accepted_price(self):
        rr=[r for r in self.rows('TRUNCATED_B0I_P7 S2_0.1') if r['m'] in (3,4)]
        self.assertEqual(len(rr),4)
        self.assertTrue(all(r['legacy_price_candidate']['accepted_receipt'] is False for r in rr))
        self.assertTrue(all(r['price']==[] for r in rr))

    def test_full_H_physical_grouping_and_total_events(self):
        k=next(r for r in self.rows('FULL_H kappa2 K2 q=2^m') if r['m']==3 and r['axis']=='X')
        r=next(r for r in self.rows('FULL_H R448 K2 q=2^m (separate candidate)') if r['m']==3 and r['axis']=='X')
        self.assertEqual(k['physical_candidate']['q'],8)
        self.assertEqual(k['physical_candidate']['R_outer'],71)
        self.assertEqual(k['structure']['physical']['allocation'],[71])
        self.assertEqual(k['structure']['canonical']['allocation'],[568])
        self.assertEqual(r['physical_candidate']['R_outer'],448)
        self.assertNotEqual(k['configuration_id'],r['configuration_id'])

    def test_unknowns_and_no_scientific_evaluations(self):
        self.assertEqual(self.data['scientific_calls'],0)
        self.assertEqual(self.data['numerical_reanalysis'],0)
        for r in self.data['rows']:
            self.assertIsNone(r['G']);self.assertIsNone(r['E_C']);self.assertIsNone(r['N']['sufficient_N'])
            for p in r['price']:
                self.assertNotIn('mean',p);self.assertNotIn('metrics',p)
        bad=deepcopy(self.data);bad['rows'][0]['E_C']=1
        with self.assertRaises(ValueError):H.validate(bad)


class SyntheticMetadata(unittest.TestCase):
    def fixture(self):
        physical=dict(id='synthetic',T=1,q=2,R_outer=3,K=2)
        context=dict(physical_candidate=physical,binding=dict(H='toy',state='toy-state'),
                     law=dict(dictionary='toy-dict',law='toy-IID'),split=dict(prefix=1,scalar='toy-scalar'),
                     profile=dict(compiler='toy-compiler',lowering='toy',wrapper='measured',P=0))
        identity=dict(candidate=physical,source='toy-signal-source')
        law=dict(source='toy-law-source')
        p=dict(identity=deepcopy(identity),law_binding=deepcopy(law),physical_context=deepcopy(context),
               axis='X',measurement_axis='cosine',observable='Re(z_raw)',outcome_convention='P0_minus_P1',
               P=0,normalization_is_price_multiplier=False)
        return context,identity,law,physical,p

    def test_valid_metadata_only_match(self):
        c,i,l,ph,p=self.fixture()
        self.assertEqual(H.check_price_link(c,i,l,ph,p,'X'),'MATCHED_EVIDENCE')

    def test_source_law_scalar_lowering_or_axis_mismatch_rejected(self):
        for field,value in [('q',20),('R_outer',30)]:
            c,i,l,ph,p=self.fixture();p['physical_context']['physical_candidate'][field]=value
            with self.assertRaises(ValueError):H.check_price_link(c,i,l,ph,p,'X')
        for key,field in [('binding','H'),('binding','state'),('law','dictionary'),
                          ('split','scalar'),('profile','lowering'),('profile','compiler'),('profile','wrapper')]:
            c,i,l,ph,p=self.fixture();p['physical_context'][key][field]='other'
            with self.assertRaises(ValueError):H.check_price_link(c,i,l,ph,p,'X')
        for field in ('identity','law_binding'):
            c,i,l,ph,p=self.fixture();p[field]['source']='other'
            with self.assertRaises(ValueError):H.check_price_link(c,i,l,ph,p,'X')
        c,i,l,ph,p=self.fixture();p['measurement_axis']='sine'
        with self.assertRaises(ValueError):H.check_price_link(c,i,l,ph,p,'X')

    def test_missing_signal_and_price_stay_missing(self):
        d=H.classify('S3_SAVED_STEP_SPECTRUM_NEW_MEAN_NOT_EXECUTED','PENDING',[])
        self.assertIn('SIGNAL_EVIDENCE_MISSING',d['blockers'])
        self.assertIn('PRICE_EVIDENCE_MISSING',d['blockers'])
        self.assertFalse(d['qualified_for_G'])


if __name__=='__main__':unittest.main(verbosity=2)
