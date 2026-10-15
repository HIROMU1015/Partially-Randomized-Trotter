import importlib.util
from pathlib import Path
import sys
import unittest
from fractions import Fraction as F

R=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,R/path);m=importlib.util.module_from_spec(s)
    sys.modules[name]=m;s.loader.exec_module(m);return m
m=load('_original_model','src/trotterlib/native_df_gate_model.py')
g=load('_grouped_model','src/trotterlib/native_df_grouped_expectation.py')


class GroupedExpectationTests(unittest.TestCase):
    def setUp(self):
        self.a=m.Basis('a','h1',(m.Gate('rz',(0,),.3),m.Gate('xx_plus_yy',(0,1),.4,.2)))
        self.b=m.Basis('b','h2',(m.Gate('rz',(1,),-.2),))
    def compare(self,components,orders,r=2,tau=.3):
        for model in ('A','B'):
            old=m.expected_rte_occurrence(components,orders,r,tau,2,model)
            new=g.expected_rte_grouped(m,components,orders,r,tau,2,model)
            self.assertEqual(old.lower,new.lower);self.assertEqual(old.upper,new.upper)
            self.assertEqual(old.phase_probability_lower,new.phase_probability_lower)
            self.assertEqual(old.phase_probability_upper,new.phase_probability_upper)
    def test_all_orders_and_shared_basis(self):
        c=(m.Component('p',F(1,3),self.a,(0,),1),m.Component('n',F(1,3),self.a,(0,1),-1),m.Component('b',F(1,3),self.b,(1,),1))
        self.compare(c,{0:F(1,2),2:F(1,3),4:F(1,6)})
    def test_identity_barrier_and_phase_bounds(self):
        self.compare((m.Component('a',F(1,2),self.a,(0,)),m.Component('i',F(1,2),None,sign=-1)),{0:F(1,2),2:F(1,2)})
    def test_zero_tau_keeps_explicit_rotations(self):
        self.compare((m.Component('a',F(1),self.a,(0,)),),{0:F(1)},tau=0)
    def test_positive_identity(self):
        self.compare((m.Component('a',F(1,2),self.a,(0,)),m.Component('i',F(1,2),None)),{0:F(1,2),4:F(1,2)})
    def test_rare_order_not_dropped(self):
        self.compare((m.Component('a',F(1),self.a,(0,1),-1),),{0:1-F(1,10**30),4:F(1,10**30)})
    def test_zero_mass_components(self):
        self.compare((m.Component('a',F(1),self.a,(0,)),m.Component('b',F(0),self.b,(1,))),{0:F(1),2:F(0)})
    def test_many_components_one_basis(self):
        c=tuple(m.Component(str(i),F(1,8),self.a,(i%2,),(-1)**i) for i in range(8))
        self.compare(c,{0:F(1,3),2:F(2,3)},r=3)
    def test_distinct_ids_never_cancel_as_same_basis(self):
        b=m.Basis('other-id',self.a.source_hash,self.a.operations)
        self.compare((m.Component('a',F(1,2),self.a,(0,)),m.Component('b',F(1,2),b,(0,))),{0:F(1)},r=3)
    def test_transition_cap_preserved(self):
        with self.assertRaises(RuntimeError):g.expected_rte_grouped(m,(m.Component('a',F(1),self.a,(0,)),),{4:F(1)},2,.3,2,'B',max_transitions=1)
    def test_ancilla_and_law_rejected(self):
        c=(m.Component('a',F(1,2),self.a,(0,)),)
        with self.assertRaises(ValueError):g.expected_rte_grouped(m,c,{0:F(1)},1,.3,2)
    def test_unsupported_model_rejected(self):
        with self.assertRaises(ValueError):g.expected_rte_grouped(m,(m.Component('a',F(1),self.a,(0,)),),{0:F(1)},1,.3,2,'C')


if __name__=='__main__':unittest.main()
