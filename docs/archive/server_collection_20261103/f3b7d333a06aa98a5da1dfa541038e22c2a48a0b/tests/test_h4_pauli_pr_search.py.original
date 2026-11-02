"""Artificial Pauli search tests: supplied 2/3-qubit models only."""
from dataclasses import replace
from fractions import Fraction as Q
import importlib
import json
import math
from pathlib import Path
import sys
import types
import unittest

pkg=types.ModuleType("pr_search_test");pkg.__path__=[str(Path(__file__).resolve().parents[1]/"src/trotterlib")];sys.modules[pkg.__name__]=pkg
S=importlib.import_module(pkg.__name__+".h4_pauli_pr_search")


class SearchTests(unittest.TestCase):
    def toy(self,n=2):
        from pr_search_test.pauli_signal_adapter import PauliHamiltonian
        return PauliHamiltonian.from_terms([("I"*n,.125),("Z"+"I"*(n-1),.5),
                                           ("X"+"I"*(n-1),-.25),("Y"+"I"*(n-1),.125)])

    def test_four_parameters_and_partition(self):
        for p,q,r,K in ((0,1,2,2),(1,2,4,4),(2,8,8,2)):
            d=S.law_descriptor(self.toy(),S.config(p,q,r,K))
            self.assertEqual((len(d["retained"]),len(d["components"])),(p,3-p))
            self.assertEqual((d["config"]["q"],d["config"]["r"],d["config"]["K"]),(q,r,K))
            self.assertEqual(Q(d["step_time"])*q,Q(.8))
            self.assertEqual(d["orders"],list(range(0,K+1,2)))
        with self.assertRaises(ValueError):S.law_descriptor(self.toy(),S.config(3,1,2,2))
        with self.assertRaises(ValueError):S.candidate_id(True,1,2,2)

    def test_cdf_graph_and_sampler_mass(self):
        import numpy as np
        p=[.1,.2,.7];d=S.probability_masses(p)
        c=np.asarray(p).cumsum();c/=c[-1]
        self.assertEqual(d["cdf"],c.tolist())
        self.assertEqual(sum(map(Q,d["exact_masses"])),1)

    def test_seed_addresses_are_disjoint_reproducible(self):
        discovery=[S.trajectory_seed(1,2,4,4,"exploration",i) for i in range(64)]
        confirmation=[S.trajectory_seed(1,2,4,4,"confirmation",i) for i in range(384)]
        self.assertFalse(set(discovery)&set(confirmation))
        self.assertEqual(discovery[3],S.trajectory_seed(1,2,4,4,"exploration",3))
        self.assertNotEqual(discovery[3],S.trajectory_seed(2,2,4,4,"exploration",3))

    def test_order2_order4_signed_noncommuting_control(self):
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from pr_search_test.pauli_signal_adapter import PauliSignalAdapter,append_pauli_event
        from pr_search_test.rte import _make_event,event_unitary
        from pr_search_test.h4_d2_pauli_control import central_control_body
        a=PauliSignalAdapter(self.toy(),"PR",1,.4,2,4)
        matrices={str(j):c.coefficient_sign*np.asarray(Operator.from_label(a.labels[str(j)]).data)
                  for j,c in enumerate(a.components)}
        for order_index,indices in ((1,[0,0,1]),(2,[0,0,1,0,1])):
            e=_make_event(indices,a.components,a.distribution,order_index)
            b=QuantumCircuit(2);append_pauli_event(b,e,a.labels)
            expected=event_unitary(e,matrices)
            np.testing.assert_allclose(Operator(b).data,expected,atol=2e-14,rtol=0)
            lift=central_control_body(b).body
            oracle=np.eye(8,dtype=complex);oracle[4:,4:]=expected
            np.testing.assert_allclose(Operator(lift).data,oracle,atol=3e-14,rtol=0)
            self.assertEqual(e.phase,(-1)**(e.taylor_order//2))

    def test_paired_wrappers_and_compiled_measurement(self):
        import numpy as np
        from qiskit.quantum_info import Operator,Statevector
        from pr_search_test.fixed_signal_evaluation import default_compiler
        from pr_search_test.rte_compiled_cost import transpile_and_measure_cost
        h=self.toy(3);cfg=S.config(1,2,2,4)
        body,wrappers,sidecar=S.central_trajectory(h,cfg,seed=123)
        raw=np.array([1,1j,.5,-.25j,0,0,0,0],dtype=complex);raw/=np.linalg.norm(raw)
        z=np.vdot(raw,Operator(body).data@raw)
        for axis,key in (("cosine","real"),("sine","imag")):
            cost=transpile_and_measure_cost(wrappers[axis],default_compiler())
            output=cost.transpiled_circuit.remove_final_measurements(inplace=False)
            prob=Statevector(np.r_[raw,np.zeros_like(raw)]).evolve(output).probabilities([3])
            self.assertAlmostEqual(prob[0]-prob[1],getattr(z,key),places=12)
        self.assertEqual(sidecar["phase_owner"],"saved_body_aggregate_only_no_event_phase_readdition")

    def test_candidate_own_B_guard_and_null_price(self):
        from pr_search_test.h4_pauli_b0_execution import precision_decisions
        t=dict(axes={a:dict(signed_bias_exact="0",u_enc_exact="1/1000000000000") for a in ("cosine","sine")})
        d=S.law_descriptor(self.toy(),S.config(1,2,4,4))
        row=S.account(t,dict(cosine=100,sine=110),"0.05",d)
        self.assertEqual(Q(row["G_exact"]),100*row["N_cos"]+110*row["N_sin"])
        self.assertEqual(Q(row["axes"]["cosine"]["scale"]["B_saved_exact"]),Q(d["B_saved_binary64"]))
        self.assertIsNone(S.account(t,dict(cosine=None,sine=None),"0.05",d)["G_exact"])
        self.assertIsNone(row["actual_G"])


def fixtures():
    test=SearchTests();out=[]
    for p,q,r,K in ((0,1,2,2),(1,2,4,4),(2,4,8,2),(1,1,2,4)):
        out.append(dict(law=S.law_descriptor(test.toy(),S.config(p,q,r,K)),
                        raw=[[1.,0.],[0.,1.],[.5,-.25],[-.5,.25]],
                        reference=dict(reference={k:dict(lower="0",upper="0") for k in ("real","imag")})))
    return out


if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--write-artificial-fixtures":
        Path(sys.argv[2]).write_text(json.dumps(fixtures(),indent=2)+"\n")
    else:unittest.main()
