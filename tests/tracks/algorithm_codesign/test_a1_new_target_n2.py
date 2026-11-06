"""Invented inputs only: no saved molecular NPZ/state or N3 processing."""
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import math
import unittest
from unittest.mock import patch
import numpy as np
from trottertracks.algorithm_codesign import a1_new_target_n2 as n2


class N2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Preserve the earlier A1 passive-import test during suite discovery.
        global QuantumCircuit,Statevector,Operator,FermionOperator,get_sparse_operator
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector,Operator
        from openfermion import FermionOperator,get_sparse_operator
        cls.dep=n2.n1.dependencies()

    def certificate(self,a,v):
        d,q=np.linalg.eigh(np.array(a,float));s=n2.spectral_enclosure([[F(x) for x in row] for row in a],d,q)
        return n2.state_certificate([[F(x) for x in row] for row in a],v,s)

    def test_exact_df_assembly_matches_independent_fermion_operator(self):
        h=self.dep.hamiltonian.DFHamiltonian(constant=.125,one_body=np.array([[.5,.25,0],[.25,-.5,0],[0,0,.75]],complex),
            lambdas=np.array([.5,-.25]),g_matrices=(np.array([[1,.5,0],[.5,0,.25],[0,.25,-.5]],complex),np.diag([.5,1.,-.5]).astype(complex)),metadata={})
        sec=self.dep.hamiltonian.PhysicalSector.number_sector(n_qubits=3,n_electrons=2)
        a=n2.exact_sector_matrix(h,sec,self.dep.hamiltonian._build_excitation_table(sec))
        def one(m):return sum((float(m[p,q].real)*FermionOperator(((p,1),(q,0))) for p in range(3) for q in range(3)),FermionOperator())
        op=FermionOperator((),h.constant)+one(h.one_body)
        for lam,g in zip(h.lambdas,h.g_matrices):op+=lam*one(g)*one(g)
        expected=get_sparse_operator(op,n_qubits=3).toarray()[np.ix_(sec.basis_indices,sec.basis_indices)]
        np.testing.assert_array_equal(np.array(a,float),expected.real)

    def test_exact_assembly_refuses_asymmetric_model(self):
        h=self.dep.hamiltonian.DFHamiltonian(0.,np.array([[0,1],[0,0]],complex),np.array([]),(),{})
        sec=self.dep.hamiltonian.PhysicalSector.number_sector(n_qubits=2,n_electrons=1)
        with self.assertRaisesRegex(ValueError,'Hermitian'):n2.exact_sector_matrix(h,sec,self.dep.hamiltonian._build_excitation_table(sec))

    def test_exact_ground_has_zero_error_without_sqrt_roundoff(self):
        c=self.certificate([[0,0],[0,2]],[1,0]);self.assertEqual(F(c['coordinate_state_error_upper']['exact']),0)
        self.assertEqual(F(c['gap_lower']['exact']),2)

    def test_excited_weight_bound_and_complex_normalization(self):
        c=self.certificate([[0,0],[0,2]],[3,4j])
        self.assertEqual(F(c['excited_weight_upper']['exact']),F(16,25));self.assertGreaterEqual(F(c['coordinate_state_error_upper']['exact']),F(8,5))
        self.assertEqual(F(c['norm_squared']['exact']),25)

    def test_residual_zero_excited_state_is_not_ground_certificate(self):
        with self.assertRaisesRegex(ValueError,'ground branch'):self.certificate([[0,0],[0,2]],[0,1])

    def test_degenerate_ground_stops(self):
        with self.assertRaisesRegex(ValueError,'not spectrally separated'):self.certificate([[0,0],[0,0]],[1,0])

    def test_bad_eigenvectors_not_accepted(self):
        with self.assertRaisesRegex(ValueError,'invertibility'):n2.spectral_enclosure([[F(0),F(0)],[F(0),F(2)]],[0,2],np.zeros((2,2)))

    def test_enclosure_charges_inexact_reconstruction(self):
        a=[[F(0),F(0)],[F(0),F(2)]];s=n2.spectral_enclosure(a,[.125,2.125],np.eye(2))
        self.assertGreaterEqual(F(s['matrix_operator_error_upper']),F(1,8))
        for ev,truth in zip(s['eigenvalue_intervals'],[0,2]):self.assertLessEqual(F(ev['lower']),truth);self.assertGreaterEqual(F(ev['upper']),truth)

    def artificial_wrapper(self,circuit,axis):
        # Exercise the existing wrapper body with an invented already-controlled
        # circuit. No DF preparation or repeated/molecular builder is invoked.
        e=NS(circuit=circuit,repetition_count=1,step_time=.125,trajectory_fingerprint='artificial',provenance_fingerprint='artificial',
             circuit_semantics_fingerprint='artificial',constant_phase=.0,extracted_identity_phase=.0,rte_relative_phase=.0)
        request=NS(evolution=e,axis=axis,include_measurement=False)
        return self.dep.hadamard.QiskitRPEHadamardInterrogationBuilder()._build_validated(request,round_index=0,ancilla=1,total_time=.125)

    def test_existing_wrapper_XY_phase_and_bit_mapping(self):
        theta=.37;u=QuantumCircuit(2);u.p(theta,1)
        for axis,expected in [('cosine',math.cos(theta)),('sine',math.sin(theta))]:
            wrapped=self.artificial_wrapper(u,axis);v=Statevector([1,0,0,0]).evolve(wrapped.circuit)
            probs=v.probabilities([1]);self.assertAlmostEqual(probs[0]-probs[1],expected,places=14)
            self.assertEqual(wrapped.bit_value_mapping,((0,1),(1,-1)))
            names=[i.operation.name for i in wrapped.circuit.data]
            self.assertEqual(names[-2:] if axis=='sine' else names[-1:],['sdg','h'] if axis=='sine' else ['h'])

    def test_existing_scalar_identity_phases_are_controlled_relative(self):
        c=QuantumCircuit(2);req=NS(step_time=.125,preparation=NS(constant_coefficient=.5,extracted_identity_coefficient=-.25),controlled=True,ancilla_qubit=1)
        phases=self.dep.partial.QiskitDFPartialS2CircuitBuilder._append_step_phases(c,req)
        expected=np.diag([1,1,np.exp(1j*sum(phases)),np.exp(1j*sum(phases))]);np.testing.assert_allclose(Operator(c).data,expected,atol=2e-16,rtol=0)
        self.assertEqual(float(c.global_phase),0)

    def test_actual_RTE_builder_keeps_n2_event_and_negative_identity_signs(self):
        rte=self.dep.rte;tail=n2.n1.cpu_module('df_rte_tail');interface=n2.n1.cpu_module('df_rte_circuit');builder=n2.n1.cpu_module('df_rte_qiskit')
        component=rte.RTEComponent('invented-I',1.,1.,-1,is_identity=True,diagonal_pauli_support=())
        distribution=rte.finite_rte_distribution(.125,2);event=rte._make_event([0,0,0],(component,),distribution,1)
        spec=interface.DFRTEIdentityCircuitSpec('invented-I',1.,-1,1)
        result=builder.QiskitDFRTEEventCircuitBuilder(basis_registry=tail.DFBasisRegistry()).build_event(
            interface.DFRTEEventCircuitRequest(event,(spec,),controlled=True,ancilla_qubit=1))
        expected=-np.exp(1j*event.rotation_angle)
        np.testing.assert_allclose(Operator(result.circuit).data,np.diag([1,1,expected,expected]),atol=1e-15,rtol=0)

    def test_number_sector_and_all_bit_reversal_positions(self):
        sec=self.dep.hamiltonian.PhysicalSector.number_sector(n_qubits=8,n_electrons=4)
        self.assertEqual(sec.dimension,70);self.assertEqual(set(sec.basis_indices),{i for i in range(256) if i.bit_count()==4})
        for mode in range(4):
            a=np.zeros(16,complex);a[1<<(3-mode)]=1
            b=self.dep.split._to_qiskit_state_order(a,4);self.assertEqual(np.flatnonzero(b).tolist(),[1<<mode])

    def test_exact_Bernoulli_grid_all_small_thresholds(self):
        M=8
        for p in (F(0),F(1),F(1,16),F(1,3),F(1,2),F(7,8)):
            count=sum(F(k,M)<p for k in range(M));self.assertLessEqual(2*abs(F(count,M)-p),F(2,M))

    def test_scope_guard_rejects_later_calls_and_restores(self):
        original=np.random.default_rng
        with n2.n2_only(self.dep):
            for fn,args in [(np.random.default_rng,()),(self.dep.split.split_df_hamiltonian_by_ld,(None,3)),
                            (self.dep.partial.prepare_df_partial_s2,(None,None)),(self.dep.rte.finite_rte_distribution,(1,2)),
                            (self.dep.hadamard.QiskitRPEHadamardInterrogationBuilder().build,(None,))]:
                with self.assertRaisesRegex(RuntimeError,'N2 forbids'):fn(*args)
        self.assertIs(np.random.default_rng,original)

    def test_preinput_failure_preserves_unknown_and_fresh_output(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'output'
            with patch.object(n2,'git',side_effect=lambda root,*a:'fake' if a==('rev-parse','HEAD') else ''),patch.object(n2.n1,'dependencies',return_value=self.dep):
                result=n2.run(out,root/'missing_profile',root)
            self.assertEqual(result['counts']['ground_state_solve'],0);self.assertEqual(n2.read(out/'STATE_RESULT.json')['status'],'UNKNOWN')
            self.assertEqual(n2.read(out/'STOP.json')['status'],'MANDATORY_STOP')
            with patch.object(n2,'git',return_value=''):
                with self.assertRaises(FileExistsError):n2.run(out,root/'missing_profile',root)

if __name__=='__main__':unittest.main()
