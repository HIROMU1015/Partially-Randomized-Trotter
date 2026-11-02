"""Synthetic selection-rule checks; no molecular data or scientific kernels."""
import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('event_driver_test',ROOT/'scripts/run_pauli_rte_event_extension.py')
D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
def base():
    return [dict(molecule=m,epsilon_sig=e,group=g,candidate_id=g+'_old',G_exact='250' if g=='Hybrid' else '500',G_SE='1') for m in ['H4','H6'] for e in D.EPS for g in ['Hybrid','RTE']]
def row(m,r,g,bias='0',K=2):
    return dict(candidate_id=f'{m}_p000_q8_r{r}_K{K}',molecule=m,epsilon_sig='0.05',cohort='exploration',r=r,s=8*r,K=K,G_exact=str(g) if g is not None else None,G_SE=1,
       eligibility='CONDITIONAL_GUARD_ELIGIBLE',axes={a:dict(signed_bias_exact=bias,u_enc_exact='0',epsilon_axis_exact='1/10') for a in D.AXES})
UNIVERSE=[dict(candidate_id=f'{m}_p000_q8_r{r}_K{K}') for m,rs in [('H4',[16,32]),('H6',[16,32,64,128])] for r in rs for K in [2,4]]
def proposed(data):return {x['candidate_id'] for x in D.proposal(data,base(),UNIVERSE)['candidates']}
class Checks(unittest.TestCase):
    def test_null_never_zero_or_infinity(self):
        a=row('H6',16,None);b=row('H6',32,100)
        self.assertEqual(D.minimum([a,b]),b);self.assertFalse(D.nearest(a,b))
    def test_H6_upper_boundary_adds_1024(self):
        self.assertIn('H6_p000_q8_r128_K2',proposed([row('H6',16,900),row('H6',32,700),row('H6',64,600)]))
    def test_H6_clear_turning_point_stops_extension(self):
        self.assertNotIn('H6_p000_q8_r128_K2',proposed([row('H6',16,900),row('H6',32,600),row('H6',64,800)]))
    def test_H4_clear_worsening_and_same_rank_stops(self):
        self.assertEqual(proposed([row('H4',16,800)]),set())
    def test_H4_improvement_adds_only_two_allowed(self):
        self.assertEqual(proposed([row('H4',16,400)]),{'H4_p000_q8_r32_K2','H4_p000_q8_r32_K4'})
    def test_material_bias_adds_only_promising_K4(self):
        p=proposed([row('H6',16,600,'1/100'),row('H6',32,900)])
        self.assertEqual(p,{'H6_p000_q8_r16_K4'})
    def test_already_executed_candidate_not_repeated(self):
        p=proposed([row('H6',64,600),row('H6',128,500)])
        self.assertNotIn('H6_p000_q8_r128_K2',p)
if __name__=='__main__':unittest.main()
