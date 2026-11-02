"""Invented affine/depth scalars only; never opens registered results/events."""
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('d2_saved_only',ROOT/'scripts/tracks/algorithm_codesign/analyze_d2_final_saved_integration.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def row(name,price,slope):return M.affine_row(1,name,{k:F(price) for k in M.M.METRICS},F(slope))


class SavedOnly(unittest.TestCase):
    def test_three_way_envelope_and_ties(self):
        rows=[row('a',0,4),row('b',3,2),row('c',7,0)]
        v=M.envelopes(rows,'T');regions=v['merged_minimum_envelope_open_regions']
        self.assertEqual([(s['open_lo'],s['open_hi'],s['minimum_arms']) for s in regions],[(0,F(3,2),['r1:a']),(F(3,2),2,['r1:b']),(2,None,['r1:c'])])
        self.assertEqual(v['minimum_switch_endpoint_ties'][0]['minimum_arms'],['r1:a','r1:b'])
    def test_origin_is_not_automatically_pair_intersection(self):
        v=M.envelopes([row('a',1,3),row('b',4,1)],'T')
        self.assertEqual(v['domain_breakpoints_including_origin'],[0,F(3,2)])
        self.assertEqual(v['actual_pair_nonnegative_crossings'],[F(3,2)])
    def test_irrelevant_crossing_merged(self):
        rows=[row('always',0,0),row('a',4,3),row('b',9,1)]
        v=M.envelopes(rows,'T');self.assertEqual(len(v['merged_minimum_envelope_open_regions']),1)
        self.assertEqual(v['merged_minimum_envelope_open_regions'][0]['minimum_arms'],['r1:always'])
    def test_identical_lines_tie_everywhere(self):
        v=M.envelopes([row('a',1,1),row('b',1,1)],'T')
        self.assertEqual(v['merged_minimum_envelope_open_regions'][0]['minimum_arms'],['r1:a','r1:b'])
    def test_input_rows_unchanged(self):
        rows=[row('a',1,2),row('b',4,1)];before=json.dumps(rows,default=str,sort_keys=True)
        M.envelopes(rows,'CX');self.assertEqual(json.dumps(rows,default=str,sort_keys=True),before)
    def test_pareto_independent_coordinates(self):
        a=M.affine_row(1,'a',{'T':1,'CX':9,'1Q':10},2)
        b=M.affine_row(1,'b',{'T':9,'CX':1,'1Q':10},2)
        self.assertFalse(M.dominates_at(a,b,{'T':0,'CX':0,'1Q':0}));self.assertEqual(M.M.pareto_zero([a,b]),['r1:a','r1:b'])
    def test_strict_dominance_excludes_full_tie(self):
        a=row('a',1,1);b=row('b',1,1)
        self.assertFalse(M.dominates_at(a,b,dict(T=0,CX=0,**{'1Q':0})))
    def test_saved_depth_template_consistency_is_not_measurement(self):
        v={'arm':'invented_provider','acceptance_squared':'1','depth_probability_unconditional':{'0':'1/5','1':'1/10','7':'3/10','10':'2/5'},'cancellation_saving_per_whole_trial':{'T':'11/5','CX':'22/5','1Q':'8'}}
        r=M.structure_summary(v,True)
        self.assertTrue(r['provider_template_consistency']['equals_saved_all_coordinates'])
        self.assertFalse(r['provider_template_consistency']['establishes_per_depth_measured_prices'])
        self.assertEqual(r['expected_removed_IR_tokens_unconditional'],F(62,5))
    def test_inconsistent_template_remains_unexplained(self):
        v={'arm':'invented','acceptance_squared':'1','depth_probability_unconditional':{'0':'1/2','7':'1/2'},'cancellation_saving_per_whole_trial':{'T':'3','CX':'4','1Q':'5'}}
        r=M.structure_summary(v,True);self.assertFalse(r['provider_template_consistency']['equals_saved_all_coordinates'])
        self.assertFalse(r['unique_causal_or_gate_attribution'])
    def test_zero_fill_conditional_depth_probability(self):
        v={'arm':'matched_CTS','acceptance_squared':'1/4','depth_probability_unconditional':{'0':'1/8','2':'1/8'},'cancellation_saving_per_whole_trial':{'T':'0','CX':'1/8','1Q':'3/8'}}
        r=M.structure_summary(v,True);self.assertEqual(r['zero_trial_mass'],F(3,4));self.assertEqual(r['cancellation_probability_given_nonzero'],F(1,2))
    def test_zero_T_requires_positive_saved_R_prices(self):
        v={'arm':'matched_CTS','acceptance_squared':'1','depth_probability_unconditional':{'0':'1'},'cancellation_saving_per_whole_trial':{'T':'0','CX':'0','1Q':'0'}}
        self.assertTrue(M.structure_summary(v,True)['no_T_priced_R_removed_follows_nonnegative_cost'])
        self.assertFalse(M.structure_summary(v,False)['no_T_priced_R_removed_follows_nonnegative_cost'])
    def test_inconsistent_depth_mass_rejected(self):
        v={'arm':'matched_CTS','acceptance_squared':'1','depth_probability_unconditional':{'0':'1/2'},'cancellation_saving_per_whole_trial':{'T':'0','CX':'0','1Q':'0'}}
        with self.assertRaises(ArithmeticError):M.structure_summary(v,True)
    def test_exclusive_output(self):
        with tempfile.TemporaryDirectory() as t:
            out=M.Output(Path(t)/'new');out.write('a.json',{'v':F(2,3)})
            with self.assertRaises(FileExistsError):out.write('a.json',{'v':0})
            self.assertEqual(json.loads((Path(t)/'new/a.json').read_text())['v'],'2/3')
    def test_output_512K_cap(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(RuntimeError):M.Output(Path(t)/'new').write('large.json',{'v':'x'*524288})
    def test_unknown_values_not_silently_null(self):
        with self.assertRaises(TypeError):M.fraction_json(object())
    def test_science_calls_static_absent(self):
        text=Path(M.__file__).read_text()
        for denied in ('import numpy','aggregate_pairs(','native_ir(','transpile(','synthesize(','Q_matrices(','.sample('):self.assertNotIn(denied,text)


if __name__=='__main__':unittest.main()
