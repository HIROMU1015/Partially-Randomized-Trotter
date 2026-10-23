"""Artificial numbers only; never read a real-H4 numerical endpoint or price."""
import copy
from fractions import Fraction as F
import math
import unittest
import sys
import types
import tempfile
from pathlib import Path
pkg=types.ModuleType('h4_cng_private');pkg.__path__=[str(Path(__file__).resolve().parents[1]/'src/trotterlib')]
sys.modules['h4_cng_private']=pkg
from h4_cng_private.h4_conditional_cng import (trans_bounds,scale_bounds,base_axis,shot,
    paired,cost_summary,comparison,rare_law,reuse,AXES,extract)


def axis(s=F(1),u=F(0),A=F(10)):
    return dict(s_exact=str(s),u_enc_exact=str(u),rho_guard_exact=str(s/101-u),
                scale=dict(A_lower=str(A),A_upper=str(A)),domain_nonempty=s>0 and s/101>=u)


class ConditionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixed,cls.stats,cls.join=map(staticmethod,reuse())

    def test_transcendental_enclosure(self):
        lo,hi=trans_bounds('exp',F(0));self.assertLessEqual(lo,1);self.assertGreaterEqual(hi,1)
        lo,hi=trans_bounds('ln',F(1));self.assertLessEqual(lo,0);self.assertGreaterEqual(hi,0)

    def test_outward_input_conversion(self):
        lo,hi=trans_bounds('exp',F(1,3));self.assertLess(float(lo),1.396);self.assertGreater(float(hi),1.395)
        self.assertLessEqual(lo,hi)

    def test_scale_encloses_saved_binary64(self):
        sc=scale_bounds(1.1,.01);self.assertGreaterEqual(F(sc['B_up_upper']),F(1.1))
        self.assertGreaterEqual(F(sc['A_upper']),F(sc['A_lower']))

    def test_guard_closed_boundary(self):
        a=axis();r=shot(a,F(1,101));self.assertIsNotNone(r['N']);self.assertEqual(F(r['u_over_h_exact']),F(1,100))
        self.assertIsNone(shot(a,F(1,101)+F(1,10**9))['N'])

    def test_missing_and_negative(self):
        self.assertIsNone(shot(axis(),None)['N'])
        with self.assertRaises(ValueError):shot(axis(),F(-1))

    def test_headroom_empty(self):
        self.assertIsNone(shot(axis(s=0),0)['N'])
        self.assertFalse(base_axis(.1,.2,'0',scale_bounds(1,0))['domain_nonempty'])

    def test_ceiling_and_sufficient_inequality(self):
        self.assertEqual(shot(axis(A=F(1000001,100000)),0)['N'],11)
        self.assertEqual(shot(axis(A=10),0)['N'],10)
        self.assertEqual(shot(axis(A=F(999999,100000)),0)['N'],10)

    def test_safe_integer(self):
        self.assertIsNone(shot(axis(A=2**52),0)['N'])
        self.assertEqual(shot(axis(A=2**52-1),0)['N'],2**52-1)

    def test_correlated_prices(self):
        p=[dict(cosine=1,sine=9),dict(cosine=9,sine=1)]
        z=paired(p,dict(cosine=3,sine=3),self.stats)
        self.assertEqual(z['mean_exact'],'30');self.assertEqual(z['standard_error'],0)
        c=cost_summary(p,self.stats);self.assertLess(F(c['covariance_exact']['cosine']['sine']),0)

    def test_empirical_not_certificate(self):
        d=self.fixed(epsilon_sig=.1,bias=dict.fromkeys(AXES,0),allowance=dict.fromkeys(AXES,0),
                    log_B_upper=0,evidence_kind='EMPIRICAL',evidence_ref='TOY')
        self.assertFalse(d['certificate_verified']);self.assertIsNotNone(d['N_total'])

    def test_original_guard_reuse(self):
        precision=extract('src/trotterlib/h2_non_df_pilot.py','precision_accounting',
                          dict(fixed_signal_shots=self.fixed,AXES=AXES,Fraction=F,_statistics=self.stats))
        result=dict(signal=dict(signed_bias=dict(real=0,imag=0)),shots=dict(evidence_ref='TOY',axes={a:dict(allowance=.01) for a in AXES}),
                    normalization=dict(log_B_upper=0),circuit_cost=dict(kind='finite_MC_mean',axes={a:dict(mean_exact='1') for a in AXES},samples=[dict(RZ=dict.fromkeys(AXES,1))]))
        r=precision(result,.1,dict(evaluation=dict(alpha_axis=.025),numerical=dict(maximum_u_over_headroom=.01)))
        self.assertIsNone(r['total_RZ'])

    def test_independent_boxes_not_common_theta(self):
        a=dict(case='toy_PF',epsilon_sig='.1',domain_nonempty=True,axes={x:axis(A=10000) for x in AXES})
        b=copy.deepcopy(a);b['case']='toy_PR'
        m=dict(toy_PF=dict.fromkeys(AXES,'1'),toy_PR=dict.fromkeys(AXES,'1001/1000'))
        r=comparison(a,b,m);self.assertEqual(r['empirical_order'],'CEIL_PARTITION_DEPENDENT')
        m['toy_PR']=dict.fromkeys(AXES,'2');r=comparison(a,b,m)
        self.assertEqual(r['empirical_order'],'LEFT_LOWER_THROUGHOUT_INDEPENDENT_BOXES')

    def test_unseen_mass_without_zero_filling_mean(self):
        comps=[dict(coefficient_abs=1,sign=1),dict(coefficient_abs=1,sign=1)]
        model=dict(representation='pauli_jw',cases=dict(PR=dict(components=comps,
                   law=dict(finite_distribution={},order_probabilities=[.9,.09,.01],component_probabilities=[.5,.5]))))
        norm=dict(component_effective_probabilities=['1/2','1/2'],order_effective_probabilities=['9/10','9/100','1/100'],IID_uniform_word_assumption=True)
        event=dict(taylor_order=0,selected_component_ids=['0'],application_sequence=[dict(component_id='0',coefficient_abs=1,coefficient_sign=1)])
        d=rare_law(model,norm,[dict(distribution={},q=8,r=4,events=[event]*32)])
        self.assertGreater(d['implemented_probability']['no4_with_unseen'],0)
        self.assertEqual(d['observed_stratum_count']['no4_with_unseen'],0)
        self.assertIsNone(d['stratum_population_mean']['no4_with_unseen'])
        self.assertIsNone(d['valid_whole_wrapper_upper'])

    def test_corrupt_join_rejected(self):
        with self.assertRaises(ValueError):self.join(dict(trajectory_record_hash='invalid'))

    def test_no_molecular_or_compiler_module_import(self):
        self.assertFalse(any(name.startswith(('trotterlib.chemistry','qiskit','pyscf','openfermion')) for name in sys.modules))

    def test_exclusive_claim_and_nonrefundable_budget(self):
        from h4_cng_private import h4_certificate_control as C
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);C.save(p/'CONSUMED.json',dict(fake='ARTIFICIAL_NOT_A_SCIENCE_GRANT'))
            with self.assertRaises(FileExistsError):C.save(p/'CONSUMED.json',{})
            b=C.CallBudget(p,dict(toy_batch=1));b.take('toy_batch')
            with self.assertRaises(ValueError):b.take('toy_batch')
            with self.assertRaises(ValueError):b.take('sampling')


if __name__=='__main__':unittest.main(verbosity=2)
