import copy
from pathlib import Path
import subprocess
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h246_reversal_evidence as c


def fixture(formula):
    return dict(system='H4',method='PR',formula=formula,prefix=3,q=4,rank=10,hamiltonian='toy_h',state='toy_s',
        compiler='toy_c',native='toy_native',rank_policy='tol_only_1e-8',basis='STO-3G',snapshot='toy_snapshot',
        control='symmetric_directional',T=.8,P=0,
        unit='compiled_RZ_per_full_Hadamard_wrapper_axis',epsilon=.001,cohort='EXPLORATION',
        K=4,r_policy='ABS_TIME_PER_OUTER',r_value=32,reference=dict(real=.1,imag=.2),u=1e-9,
        status='ELIGIBLE',G='fake_literal',N=dict(cosine=2,sine=3),C=dict(cosine=7,sine=8))


class Metadata(unittest.TestCase):
    def test_complete_pair_no_computation(self):
        out=c.classify_control(fixture('2nd'),fixture('4th'))
        self.assertTrue(out['complete_saved_pair']);self.assertFalse(out['new_comparison_computed'])
        self.assertFalse(out['causal_order_only_identified'])

    def test_ineligible_and_missing_cost_never_zero(self):
        for status in ('INELIGIBLE','UNDETERMINED'):
            a=fixture('2nd');a.update(status=status,G=None)
            self.assertFalse(c.classify_control(a,fixture('4th'))['complete_saved_pair'])
        a=fixture('2nd');a['C']['cosine']=None
        self.assertFalse(c.classify_control(a,fixture('4th'))['complete_saved_pair'])

    def test_system_rank_q_prefix_K_cohort_policy_ref_mismatch(self):
        for k,v in [('system','H6'),('rank',19),('q',8),('prefix',5),('K',2),('cohort','CONFIRMATION'),
                    ('r_policy','PER_PIECE'),('r_value',4),('epsilon',.005),('reference',dict(real=.3,imag=.2)),('u',0)]:
            with self.subTest(k=k):
                b=dict(fixture('4th'),**{k:v})
                with self.assertRaises(c.prior.Stop):c.classify_control(fixture('2nd'),b)

    def test_m5_not_yoshida(self):
        with self.assertRaises(c.prior.Stop):c.classify_control(fixture('2nd'),fixture('4th(m5_best)'))

    def test_confirmation_requires_exact_law_and_independent_cohort(self):
        target=fixture('4th');record=dict(target,independent=True,cohort='INDEPENDENT')
        self.assertTrue(c.confirmation_matches(target,record))
        for k,v in [('formula','4th(m5_best)'),('prefix',4),('K',2),('q',1),('independent',False),('cohort','EXPLORATION')]:
            self.assertFalse(c.confirmation_matches(target,dict(record,**{k:v})))

    def test_execution_rejected_before_source_or_input_read(self):
        with self.assertRaises(c.prior.Stop):c.reject_execution(True)
        p=Path(c.__file__)
        r=subprocess.run([sys.executable,'-B',str(p),'--execute','--source','missing_ref'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('NO_SCIENCE_EXECUTION_PATH',r.stderr)


if __name__=='__main__':unittest.main()
