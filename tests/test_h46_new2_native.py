"""Artificial native tests; no saved molecular input and no transpile."""
import unittest
import itertools
import numpy as np
from dataclasses import replace
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from h46_new2_native import gaussian_operations,prepare,make_requests,native_evolution,OccupationOracle,forced_event_request
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.df_rte_circuit import DFRTEEventCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(10817)
        def hermitian():
            x=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
            return .1*(x+x.conj().T)
        cls.ham=DFHamiltonian(.37,hermitian(),np.array([.2,-.3,.1]),
            tuple(hermitian() for _ in range(3)),{})
        cls.state=np.zeros(16,complex);cls.state[3]=1/np.sqrt(2);cls.state[12]=1j/np.sqrt(2)
        cls.prep=prepare(cls.ham,1)
        cls.oracle=OccupationOracle(cls.ham,cls.state)

    def test_gaussian_exterior_minors_complex_all_sectors(self):
        rng=np.random.default_rng(73)
        raw=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));u,_=np.linalg.qr(raw)
        qc=QuantumCircuit(4)
        for g,q in gaussian_operations(u):qc.append(g,q)
        actual=np.asarray(Operator(qc).data);expected=np.zeros((16,16),complex)
        for i in range(16):
            modes_i=[m for m in range(4) if i&(1<<m)]
            for j in range(16):
                modes_j=[m for m in range(4) if j&(1<<m)]
                if len(modes_i)==len(modes_j):
                    expected[i,j]=np.linalg.det(u[np.ix_(modes_i,modes_j)]) if modes_i else 1.
        self.assertLess(np.max(abs(actual-expected)),1e-12)

    def test_twelve_mode_gaussian_one_particle_and_vacuum(self):
        rng=np.random.default_rng(12);u,_=np.linalg.qr(rng.normal(size=(12,12)))
        qc=QuantumCircuit(12)
        for g,q in gaussian_operations(u):qc.append(g,q)
        vacuum=np.zeros(4096);vacuum[0]=1
        self.assertLess(np.linalg.norm(simulate_statevector(qc,vacuum)-vacuum),1e-12)
        for j in (0,5,11):
            v=np.zeros(4096);v[1<<j]=1
            expected=np.zeros(4096,complex)
            for i in range(12):expected[1<<i]=u[i,j]
            self.assertLess(np.linalg.norm(simulate_statevector(qc,v)-expected),1e-12)

    def test_negative_step_composition_full_oracle_and_control(self):
        qstate=np.zeros(16,complex);qstate[self.oracle.qindices]=self.oracle.initial
        low=np.r_[qstate,np.zeros(16)];high=np.r_[np.zeros(16),qstate]
        for label,q in [('2nd',4),('4th',2),('4th(new_2)',1)]:
            requests=make_requests(self.prep,label,q,T=.8,total_steps=32,
                seed_for=lambda a,b:10000+a*10+b)
            evolution=native_evolution(requests,label,q,.8)
            self.assertLess(np.linalg.norm(simulate_statevector(evolution.circuit,low)-low),1e-11)
            actual=simulate_statevector(evolution.circuit,high)
            self.assertLess(np.linalg.norm(actual[:16]),1e-11)
            self.assertLess(np.linalg.norm(actual[16:]-self.oracle.trajectory(requests)),1e-11)

    def test_rare_orders_and_signed_product_phase(self):
        from trotterlib.df_partial_s2 import make_df_partial_s2_step_request
        builder=QiskitDFRTEEventCircuitBuilder(basis_registry=self.prep.rte_preparation.basis_registry)
        components={c.component_id:c for c in self.prep.rte_preparation.symbolic_tail.components}
        qstate=np.zeros(16,complex);qstate[self.oracle.qindices]=self.oracle.initial
        for time in (.2,-.2):
            r=make_df_partial_s2_step_request(self.prep,step_time=time,rte_steps=1,
                finite_taylor_order=4,controlled=True,ancilla_qubit=4)
            for order in (0,2,4):
                e=forced_event_request(r,order)
                req=DFRTEEventCircuitRequest(e,r.rte_occurrence.component_specs,controlled=True,ancilla_qubit=4)
                circuit=builder.build_event(req).circuit
                expected=np.zeros(16,complex);expected[self.oracle.qindices]=self.oracle.event(e,components,self.oracle.initial)
                actual=simulate_statevector(circuit,np.r_[np.zeros(16),qstate])
                self.assertLess(np.linalg.norm(actual[16:]-expected),1e-11)

    def test_nonfinite_nonunitary_and_schedule_mismatch_stop(self):
        for u in (np.array([[float('nan')]]),np.array([[2.]])):
            with self.assertRaises(ValueError):gaussian_operations(u)
        r=make_requests(self.prep,'4th(new_2)',1,T=.8,total_steps=32,seed_for=lambda a,b:b)
        with self.assertRaises(ValueError):native_evolution(tuple(reversed(r[:-1])),'4th(new_2)',1,.8)
        with self.assertRaises(ValueError):native_evolution(r,'4th(new_2)',1,.9)

    def test_sampler_repeat_and_negative_normalization(self):
        a=make_requests(self.prep,'4th(new_2)',1,T=.8,total_steps=32,seed_for=lambda a,b:b+47)
        b=make_requests(self.prep,'4th(new_2)',1,T=.8,total_steps=32,seed_for=lambda a,b:b+47)
        self.assertEqual([[e.to_dict() for e in r.rte_occurrence.events] for r in a],
            [[e.to_dict() for e in r.rte_occurrence.events] for r in b])
        self.assertTrue(any(r.step_time<0 for r in a))
        self.assertTrue(all(r.rte_distribution.exact_finite_distribution>=1 for r in a))

    def test_shots_include_headroom_and_B_squared(self):
        from d0_frozen_shot_rule_v1 import u_aware_shots
        import math
        def budget(bias,B):
            return u_aware_shots(dict(real=bias,imag=bias),dict(real=1e-9,imag=1e-9),
                epsilon=.001,log_B_upper=math.log(B),evidence_kind='EMPIRICAL',evidence_ref='artificial')
        a=budget(0.,1.);b=budget(.0003,1.);c=budget(.0003,2.)
        self.assertGreater(b['N_total'],a['N_total'])
        self.assertLess(abs(c['N_total']-4*b['N_total']),8)
        self.assertEqual(budget(.001,1.)['eligibility_under_declared_allowance'],'INELIGIBLE')

    def test_rounded_coefficients_are_not_strict_fourth_order(self):
        from decimal import Decimal as D
        a,b=D('.42008729'),D('.40899193');w=(b,a,1-2*(a+b),a,b)
        self.assertEqual(sum(w),1)
        self.assertEqual(sum(x**3 for x in w),D('7.346136322761508E-9'))


if __name__=='__main__':unittest.main()
