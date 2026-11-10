"""Frozen metadata + artificial mismatch tests. No scientific numerical calls."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import plan_r_screening_price_alignment as m


class AlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=m.p.SavedArtifacts(ROOT)
        cls.packets={s:cls.store.read(dict(commit=m.A,path=m.AP+s+'_METADATA_PROPOSAL.json')) for s in ('H4','H6')}
        cls.aplan=cls.store.read(dict(commit=m.A,path=m.AP+'EXECUTION_PLAN.json'))
        cls.cov=cls.store.read(dict(commit=m.C,path=m.CP+'SAVED_PRICE_COVERAGE.json'))
        cls.fixed=cls.store.read(dict(commit=m.B,path='docs/research/rpe_provider_integration_20261111/fixed_coverage/COVERAGE.json'))
        cls.describer=staticmethod(m.fixed_C_describer(ROOT))
        cls.text=m.blob(ROOT,m.C,m.CP+'MINIMAL_SCIENTIFIC_PLAN.md').decode()
        cls.result=m.build(ROOT)

    def align(self,packets=None,cov=None):
        return m.align(packets or self.packets,self.aplan,cov or self.cov,self.fixed,m.phase1_specs(self.text),self.describer)

    def test_C_rectangle_preserved_with_two_explicit_H4_candidates(self):
        cells=self.result['C_phase1_cells'];self.assertEqual(len(cells),8)
        self.assertEqual(self.result['sets']['C_in_A_auto'],6);self.assertEqual(self.result['sets']['C_explicit_outside_A'],2)
        extra=[v for v in cells if not v['in_A_auto_menu']]
        self.assertEqual({(v['system'],v['q_PF'],v['candidate']['R_outer']) for v in extra},{('H4',8,8),('H4',16,8)})
        self.assertTrue(all(v['intake_lane']=='EXPLICIT_C_CALIBRATION_VALIDATION_ONLY' for v in extra))
        self.assertEqual({v['m'] for v in cells if v['system']=='H6'},{4,5})
        self.assertTrue(all(v['q_RPE']==16*v['q_PF'] for v in cells if v['system']=='H6'))

    def test_24_scalar_and_80_wrapper_units_and_stage_overlap_are_distinct(self):
        result=self.result;self.assertEqual(len(result['A_normalization24']),24)
        self.assertEqual(len(result['sets']['A24_C_overlap']),2)
        self.assertEqual(result['sets']['A_N_pilot_C_overlap'],[])
        self.assertTrue(result['sets']['A_N_pilot_is_subset_of_A24'])
        units=result['C_execution_units']
        self.assertEqual(units['ordinary_wrappers_max'],64);self.assertEqual(units['rare_diagnostic_wrappers_max'],16)
        self.assertEqual(units['all_wrappers_max'],80);self.assertFalse(units['rare_paths_pooled_into_unconditional_cohort'])
        self.assertEqual(sum(v['planned_price_paths'] for v in result['C_phase1_cells'] if v['role']!='SAVED_CALIBRATION'),32)
        for s in ('H4','H6'):
            roles=[v['role'] for v in result['C_phase1_cells'] if v['system']==s]
            self.assertEqual(roles.count('SAVED_CALIBRATION'),2);self.assertEqual(roles.count('INDEPENDENT_VALIDATION'),1)

    def test_saved_N_C_metadata_and_new_science_boundary(self):
        for row in self.result['C_phase1_cells']:
            old=row['role']=='SAVED_CALIBRATION'
            self.assertEqual(row['N_saved'],old);self.assertEqual(bool(row['C_saved_metadata']),old)
            self.assertEqual(len(row['C_selected_calibration_metadata']),1 if old else 0)
            if old:self.assertEqual(row['C_selected_calibration_metadata'][0]['stored_n_paths'],row['planned_price_paths'])
            self.assertFalse(row['new_N_required_for_C_price_pilot'])
            self.assertEqual(row['new_N_required_for_same_point_G'],not old)
            self.assertFalse(row['price_values_exported']);self.assertNotIn('N_X',row);self.assertNotIn('mean',row)
            self.assertFalse(row['long_q_transfer_qualified']);self.assertFalse(row['full_law_expected_price_qualified'])
        self.assertIsNone(self.result['ranking']);self.assertIsNone(self.result['total_resources'])

    def test_no_menu_mutation_shrink_or_R_bound_is_inferred(self):
        before=deepcopy(self.packets);result=self.align()
        self.assertEqual(before,self.packets);self.assertEqual(result['sets']['A_auto_PR'],86)
        self.assertFalse(result['A_screen_scope_automatically_extended']);self.assertFalse(result['A_comparison_menu_automatically_reduced'])
        self.assertIsNone(result['R_upper_bound']);self.assertFalse(result['range_complete'])

    def test_identity_law_recipe_and_saved_C_mismatch_fail_closed(self):
        for mutate in [lambda x:x['H4']['rows'][0].update(delta=.2),
                       lambda x:x['H6']['rows'][0]['request']['law_binding'].update(R_outer=99),
                       lambda x:x['H6']['rows'][0]['structure']['allocation'].__setitem__(0,9),
                       lambda x:x['H4']['menu']['profile'].update(compiler='FOREIGN')]:
            bad=deepcopy(self.packets);mutate(bad)
            with self.assertRaises(ValueError):self.align(packets=bad)
        bad=deepcopy(self.cov);bad['fixed_provider_evidence'][0]['candidate']['K']=0
        with self.assertRaises(ValueError):self.align(cov=bad)
        with self.assertRaisesRegex(ValueError,'C fixed Phase1'):m.phase1_specs(self.text.replace('8 / 16 | 4 / 8','8 / 16 | 4 / 16'))

    def test_law_definition_matches_but_raw_alias_and_sampler_are_not_assumed(self):
        law=self.result['law_definition'];self.assertTrue(law['implementation_bytes_equal'])
        self.assertFalse(law['raw_binding_alias_automatically_applied']);self.assertFalse(law['actual_finite_sampler_distance_qualified'])
        self.assertTrue(all(not v['law_binding_raw_equal'] for v in self.result['C_phase1_cells']))

    def test_metadata_path_never_runs_numerical_entry_or_loads_scientific_stack(self):
        with (patch.object(m.r,'evaluate_menu',side_effect=AssertionError('No G')),
              patch.object(m.b.nmodel,'analyze',side_effect=AssertionError('No N')),
              patch.object(m.b.RMenuBatch,'__call__',side_effect=AssertionError('No N batch')),
              patch.object(m.b.FixedSavedPrices,'__call__',side_effect=AssertionError('No fresh C'))):
            self.align()
        env=os.environ.copy();env['PYTHONPATH']='src:scripts'
        code="import sys;import plan_r_screening_price_alignment as m;m.build();assert not any(n in sys.modules for n in ('numpy','scipy','qiskit'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)


if __name__=='__main__':unittest.main(verbosity=2)
